"""Bridge to the Arduino UNO R4 WiFi status companion.

The Arduino runs a tiny HTTP server on the same Wi-Fi network:
  GET /status?state=<ready|processing|success|error>&msg=<text>  -> sets RGB LED + scrolls msg
  GET /button                                                    -> {"presses": n} since last read
  GET /ping                                                      -> {"ok": true, ...}
Every call is best-effort with a short timeout: if the board is unplugged the
app keeps working and just reports the hardware as offline.
"""
from __future__ import annotations

from contextlib import contextmanager

import time

import requests

from . import config

STATES = ("ready", "processing", "success", "error")


def enabled() -> bool:
    return bool(config.ARDUINO_IP)


_offline_until = 0.0  # circuit breaker: after a failure, skip calls for a while


def _get(path: str, **params):
    global _offline_until
    if not enabled() or time.time() < _offline_until:
        return None
    try:
        r = requests.get(f"http://{config.ARDUINO_IP}{path}", params=params, timeout=config.ARDUINO_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001 – hardware is optional
        _offline_until = time.time() + 8   # don't slow every request while the board is unplugged
        return None


def reset_backoff() -> None:
    """Forget a previous failure – used when the user presses a test button."""
    global _offline_until
    _offline_until = 0.0


def set_status(state: str, msg: str = "") -> bool:
    if state not in STATES:
        raise ValueError(state)
    # LED matrix is 12x8; keep scrolling text short and ASCII-only.
    msg = "".join(ch for ch in msg if 32 <= ord(ch) < 127)[:40]  # matrix font is ASCII only
    return _get("/status", state=state, msg=msg) is not None


def ping() -> dict | None:
    return _get("/ping")


def read_button_presses() -> int:
    data = _get("/button")
    return int(data.get("presses", 0)) if data else 0


@contextmanager
def working(done_msg: str = "Done"):
    """with arduino.working("Quiz"): ...  -> blue + hourglass while running, then green tick
    (or red cross on error), then scrolls `done_msg`."""
    set_status("processing")
    try:
        yield
    except Exception:
        set_status("error", "Error")
        raise
    else:
        set_status("success", done_msg)
