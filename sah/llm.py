"""LLM client (Unit: large language models, prompt engineering).

Providers:
  groq    – Groq Cloud (Llama 3.3 70B etc.), OpenAI-compatible REST API (needs GROQ_API_KEY, free tier)
  gemini  – Google Gemini API via the google-genai SDK (needs GEMINI_API_KEY)
  ollama  – a local model served by Ollama (http://localhost:11434)
  offline – no LLM; every feature falls back to an extractive NLP method
If the chosen provider fails, `available` turns False and the app degrades to
offline mode instead of crashing — handy during a live demo with bad Wi-Fi.
"""
from __future__ import annotations

import json
import re

import requests

from . import config


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.model = None
        self.last_error = ""
        self._client = None
        if self.provider == "groq":
            if not config.GROQ_API_KEY:
                self.last_error = "GROQ_API_KEY not set – running offline."
                self.provider = "offline"
            else:
                self.model = config.GROQ_MODEL
        elif self.provider == "gemini":
            if not config.GEMINI_API_KEY:
                self.last_error = "GEMINI_API_KEY not set – running offline."
                self.provider = "offline"
            else:
                from google import genai

                self._client = genai.Client(api_key=config.GEMINI_API_KEY)
                self.model = config.GEMINI_MODEL
        elif self.provider == "ollama":
            self.model = config.OLLAMA_MODEL

    @property
    def available(self) -> bool:
        return self.provider in ("groq", "gemini", "ollama")

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.model else self.provider

    # -------------------------------------------------------------------
    def generate(self, prompt: str, system: str = "", json_mode: bool = False,
                 temperature: float = 0.3) -> str:
        if self.provider == "groq":
            return self._groq(prompt, system, json_mode, temperature)
        if self.provider == "gemini":
            return self._gemini(prompt, system, json_mode, temperature)
        if self.provider == "ollama":
            return self._ollama(prompt, system, json_mode, temperature)
        raise LLMError("No LLM configured (offline mode).")

    def generate_json(self, prompt: str, system: str = "", temperature: float = 0.4):
        raw = self.generate(prompt, system, json_mode=True, temperature=temperature)
        return parse_json(raw)

    # -------------------------------------------------------------------
    def _groq(self, prompt, system, json_mode, temperature) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        candidates = [self.model] + [m for m in config.GROQ_FALLBACK_MODELS if m != self.model]
        last = ""
        for model in candidates:
            body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 2048}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            try:
                r = requests.post(config.GROQ_URL, json=body, timeout=60,
                                  headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"})
            except requests.RequestException as exc:  # network down
                last = str(exc)
                break
            if r.status_code == 200:
                self.model = model  # remember the one that worked
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            # 429 = this model's free-tier limit is used up; 404/400 model errors = retired model.
            # Limits are per model on Groq, so the next model usually still works.
            if r.status_code == 429 or (r.status_code in (400, 404) and "model" in r.text.lower()):
                continue
            break  # bad key (401) or other error: no point trying other models
        self.last_error = f"Groq error: {last}"
        raise LLMError(self.last_error)

    def _gemini(self, prompt, system, json_mode, temperature) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        candidates = [self.model] + [m for m in config.GEMINI_FALLBACK_MODELS if m != self.model]
        last = None
        for model in candidates:
            try:
                resp = self._client.models.generate_content(model=model, contents=prompt, config=cfg)
                self.model = model  # remember the one that worked
                return (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc).lower()
                if "not found" in msg or "404" in msg or "not supported" in msg:
                    continue  # try the next model name
                break
        self.last_error = f"Gemini error: {last}"
        raise LLMError(self.last_error)

    def _ollama(self, prompt, system, json_mode, temperature) -> str:
        body = {"model": self.model, "prompt": prompt, "system": system, "stream": False,
                "options": {"temperature": temperature}}
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{config.OLLAMA_URL}/api/generate", json=body, timeout=180)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Ollama error: {exc}"
            raise LLMError(self.last_error) from exc


def parse_json(raw: str):
    """Parse JSON from a model reply, tolerating ```json fences or chatter."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise
