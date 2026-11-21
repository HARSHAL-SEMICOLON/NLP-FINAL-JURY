# 🎓 Smart Academic Help: An AI-Powered Academic Assistant

**Semester VII project for Natural Language Processing with Large Models (23CSE4021)**

**Objective.** Smart Academic Help is an AI-powered academic assistance and management platform built with NLP, Large Language Models and an Arduino UNO R4 WiFi. Students get syllabus-based answers, quizzes, summaries, a study plan and progress tracking in a low-stress, supportive space. Faculty get automation for repetitive work (quiz creation and marking, assignment review drafts, reports, repeated doubts) and a dashboard showing where the class is struggling. AI assists; the teacher keeps control of every mark and decision.

The syllabus itself lists this kind of project: *AI-powered Student Helpdesk Chatbot* (Unit I PBL), *AI Question Generator* (Unit III), *Personalized Study Assistant* and *College Information Chatbot* (Unit V).

It runs in four modes. Set the mode with `LLM_PROVIDER` in `.env`:

| Mode | Needs | What happens |
|---|---|---|
| **Groq** (default) | free `GROQ_API_KEY`, no card | Llama 3.3 70B on Groq's fast inference. On a rate limit it switches to the next model (gpt-oss-120b, gpt-oss-20b, Llama 3.1 8B) |
| **Gemini** | free `GEMINI_API_KEY` | Same features through Google's API |
| **Ollama** | local model, no internet | Same features, runs on your laptop |
| **Offline** | nothing | Extractive NLP fallbacks (TF-IDF ranking, cloze MCQs, pattern-based definitions) |

If the LLM fails during the demo (bad Wi-Fi, quota), the app switches to offline mode by itself. It does not crash.

---

## 1. The seven modules

| # | Module | Student side | Faculty side |
|---|---|---|---|
| 1 | **AI Academic Assistant** | Ask in English/Hindi/Marathi at Beginner→Advanced level; "explain it more simply" | Same assistant to test answers |
| 2 | **Smart RAG Knowledge Base** | Answers with page citations, grounding score, "not in your material" label | Upload notes + syllabus once; units detected automatically; content-gap list |
| 3 | **Student Well-being Support** | Daily check-in (private by default), supportive next steps, exam-readiness meter, deadline reminders, streaks & badges, today's 3–4 small tasks, anonymous doubts | Anonymous class mood totals; names only when a student opts in to share |
| 4 | **Smart Learning & Assessment** | Practice quizzes by unit and Bloom's level, teacher-assigned quizzes, assignments, study planner, progress by unit/Bloom | — |
| 5 | **Faculty Work Automation** | — | Class quizzes generated from notes → review → publish → auto-marked; assignments with rubric-based AI mark suggestions + copy detection (teacher releases final marks); question papers with Bloom's level + CO + answer key; teaching notes; FAQ built from repeated questions; attendance CSV import; weekly + per-student reports with AI summary |
| 6 | **Faculty Analytics Dashboard** | — | Weekly summary, "needs re-teaching" units (draft a practice quiz or get teaching ideas in one click), unit accuracy by week, questions by unit and Bloom's level, students needing attention (low accuracy, open doubts, missing work, attendance < 75%), LDA themes of student questions |
| 7 | **Arduino IoT Integration** | LED matrix shows thinking/done/score, scrolls a near deadline | Status display page |

**The feedback loop:** students struggle with a unit (quiz accuracy, doubts) → it appears under *Needs re-teaching* → the teacher drafts a practice quiz or gets teaching ideas → publishes the quiz to students → *Is understanding improving?* shows that unit's accuracy week by week.

**Well-being, responsibly:** this is academic-stress support, not diagnosis or treatment. Private notes are never stored unless the student shares them. If a note suggests serious distress, the app points the student to a trusted person, the college counsellor and Tele-MANAS (14416 / 1-800-891-4416, free, 24×7).

## 2. Architecture

```
 Student / Faculty ──► Streamlit UI (app.py + ui/) ┐          FastAPI (api.py) ◄── other clients / Arduino
                                             ▼                 │
                         ┌──────────── sah/ engine ────────────┴──────────────┐
                         │ preprocess.py  clean · tokenize · stop words ·     │
                         │                stem · lemmatize                    │
                         │ documents.py   PDF → pages → overlapping chunks    │
                         │ embeddings.py  TF-IDF  |  Sentence-BERT            │
                         │ vectorstore.py persistent vector DB (FAISS/NumPy)  │
                         │ rag.py         retrieve → grounded prompt → LLM    │
                         │ llm.py         Gemini | Ollama | offline           │
                         │ quiz.py · summarize.py · planner.py · evaluation.py│
                         │ syllabus.py    unit parsing · unit tagger · Bloom  │
                         │ classical.py   NER · sentiment · LDA topics        │
                         │ metrics.py     BLEU · ROUGE · perplexity · grounding│
                         │ paper.py       question-paper generator            │
                         │ wellbeing.py   check-ins · readiness · streaks     │
                         │ assignments.py rubric review · copy detection      │
                         │ reports.py · faq.py   reports · attendance · FAQ   │
                         │ db.py          SQLite: users, quizzes, doubts, …   │
                         │ arduino.py     HTTP bridge ──► UNO R4 WiFi         │
                         └────────────────────────────────────────────────────┘
```

**One question, step by step (RAG):** query → preprocess → embed → cosine search over chunk vectors → keep chunks above the relevance threshold → numbered context + rules go into the prompt → LLM answers with `[1]`, `[2]` citations → the app shows the answer, the basis label and the source pages.

## 3. Setup (Windows)

**Quick way: double-click the scripts**

1. **`setup.bat`** (one time, 5–10 min). It creates `.venv`, installs the packages, offers the optional transformer embeddings, creates `.env` and runs the tests. It opens `.env` in Notepad at the end.
2. Paste your free Groq key into `.env` as `GROQ_API_KEY=...` (get one at https://console.groq.com/keys, no card needed), then save.
3. **`run_app.bat`** starts the app and opens http://localhost:8501.
4. **`run_api.bat`** (optional) starts the REST API, with docs at http://localhost:8000/docs.

The NLTK data (Punkt tokenizer, stop words, WordNet, POS tagger) is already in `nltk_data/`, so nothing else needs downloading for it.

**Default course material – no uploading.** At every start the app loads the course syllabus, the sample notes and every PDF/TXT/MD in a `textbooks` folder next to the project (`E:\NLP JURY FINAL\textbooks`) or inside it. It also copies the free Jurafsky & Martin draft from Downloads if it finds it there. Each file is indexed once (about 10 s for a 600-page book) and remembered, so later starts take under a second. Removing a document in the app keeps it removed; set `AUTOLOAD=0` in `.env` to switch this off.

**Manual way (VS Code terminal)**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env                   # then paste your GROQ_API_KEY into .env
streamlit run app.py
```

Recommended for the final demo (better retrieval; about 300 MB including PyTorch):

```powershell
pip install sentence-transformers faiss-cpu
```

Documents you already indexed are re-embedded automatically on the next start.

Other commands:

```powershell
pytest -q                                          # 17 tests, work fully offline
uvicorn api:app --reload                           # REST API, docs at http://localhost:8000/docs
python -m sah.setup_nltk                           # re-download NLTK data if nltk_data/ is deleted
```

## 4. Arduino UNO R4 WiFi (board only, no wiring)

**Hardware:** Arduino UNO R4 WiFi + USB-C data cable. Nothing else. The build uses only the board's built-in 12×8 LED matrix, its on-board LED and its Wi-Fi module.

1. In Arduino IDE, install **Arduino UNO R4 Boards** (Boards Manager) and **ArduinoGraphics** (Library Manager).
2. Open `arduino/smart_academic_help/smart_academic_help.ino` and set `WIFI_SSID` / `WIFI_PASS`. Use a **2.4 GHz** network (the R4 can't join 5 GHz). Your laptop must be on the same network; a phone hotspot works well for the demo.
3. Upload the sketch. Open Serial Monitor at 115200 baud and copy the printed IP into `.env` as `ARDUINO_IP=192.168.x.x`. The IP also scrolls on the matrix.
4. Restart the app. The sidebar shows **Arduino: 🟢 online**. Test each state from the 🔌 Hardware page.

| Event | LED matrix | On-board LED |
|---|---|---|
| Connecting to Wi-Fi | Wi-Fi icon, then scrolls its IP | — |
| Ready | smiley | off |
| Retrieving / generating | animated hourglass | blinking |
| Done | tick, then scrolls "ANSWER READY" / "QUIZ READY" / quiz score | off |
| Error / poor quiz score | cross | off |

The board serves `GET /status?state=…&msg=…` and `GET /ping` (plus `GET /button`, which always returns 0 because this build has no button). If the board is unplugged, every call times out quietly and the software keeps working. Help requests are logged from the app's **😕 Still confused** button.

## 5. Syllabus mapping (23CSE4021, for the viva)

| Unit | Syllabus topics | Where in the project |
|---|---|---|
| **I** Intro to NLP & LLMs | Tokenization, TF-IDF, word embeddings, probability for NLP, transformers, LLMs | `preprocess.tokenize`, `embeddings.TfidfEmbedder`, `SbertEmbedder`, `metrics.BigramLM` (Laplace-smoothed bigram LM, perplexity), NLP Lab |
| **II** Text processing & classical NLP | Cleaning, stop words, stemming, lemmatization, **POS tagging**, **NER**, dependency parsing, **vector databases**, **text classification**, **sentiment**, **topic modelling**, **information retrieval** | `preprocess` (POS-aware lemmatization), `classical.entities` (NLTK maxent NER, spaCy optional for dependencies), `vectorstore` (FAISS), `syllabus.tag_unit` + `bloom_level` (classification), `classical.sentiment` (VADER on feedback), `classical.topics` (LDA on student questions), `evaluation` (IR metrics) |
| **III** Transformers & MT | Role prompting, function-style **structured output**, machine translation, **MT evaluation** | `rag.SYSTEM` + prompts (role prompting), JSON-mode quizzes and papers, Hindi/Marathi answers, NLP Lab translation + back-translation **BLEU** |
| **IV** LLMs | Evaluation: perplexity, **BLEU, ROUGE**, human eval, **bias & hallucination**, responsible AI, privacy, **Edge AI** | `metrics.bleu/rouge_*`, `BigramLM.perplexity`, 👍/👎 ratings (human eval), `metrics.grounding` (hallucination check), "not in your material" labels, local SQLite, Ollama/offline mode, Arduino edge device |
| **V** Building LLM apps | APIs for LLMs, open-source LLMs, **RAG**, embeddings, **document QA**, evaluation metrics, industrial applications | `llm.py` (Groq/Gemini/Ollama), `rag.Assistant`, FastAPI `api.py`, Hit@k/MRR evaluation |

| CO | How the project shows it |
|---|---|
| CO1 apply text representation & basic NLP | TF-IDF / embeddings / tokenization in retrieval and NLP Lab |
| CO2 preprocessing, feature engineering, linguistic analysis | POS, lemmatization, NER, sentiment, LDA |
| CO3 analyse statistical, DL, transformer models & MT | TF-IDF vs SBERT comparison, bigram LM, LLM translation + BLEU |
| CO4 evaluate performance, deployment & ethics | Hit@k/MRR, BLEU/ROUGE, perplexity, grounding score, privacy (local data), edge deployment |
| CO5 develop apps with pretrained LLMs & prompt engineering | The whole app: RAG + prompts + structured output + faculty tools |

### Technique → code reference

| Course concept | Implementation |
|---|---|
| Text preprocessing | `preprocess.clean_text`, `documents.mark_headings` (PDF clean-up) |
| Tokenization | `preprocess.tokenize`, `preprocess.sentences` (NLTK Punkt) |
| Stop-word removal | `preprocess.remove_stopwords`: used for retrieval, never for answer generation |
| POS tagging | `preprocess.pos_tag` (NLTK averaged perceptron, Penn Treebank tags) |
| Stemming & lemmatization | `preprocess.stem` (Porter), `preprocess.lemmatize` (WordNet + POS tag) |
| Semantic similarity | cosine similarity on L2-normalised vectors; `vectorstore.search` |
| Prompt engineering | `rag.build_rag_prompt` (role, rules, numbered context, citation format, difficulty, language), JSON prompts in `quiz.py` / `paper.py` |
| RAG | `rag.Assistant.ask`: overlapping chunks, top-k retrieval, threshold, grounded generation, citations |
| Summarisation | `summarize.extractive_summary` (TF-IDF sentence scoring) vs LLM map-reduce, ROUGE vs source |
| Question generation | `quiz.cloze_mcq` / LLM MCQs at a Bloom's level, `paper.generate` |
| Answer evaluation | `quiz.grade_short`: semantic similarity × keyword coverage |

**Likely viva questions**
- *Why chunk with overlap?* An idea that crosses a chunk boundary can still be retrieved whole from one of the two chunks.
- *Why is TF-IDF's threshold (0.08) lower than SBERT's (0.30)?* Sparse lexical vectors give low cosine scores even for relevant text. Dense embeddings sit higher. Each threshold is set separately.
- *Why does the lemmatizer need POS tags?* WordNet treats every word as a noun by default, so "running" would stay "running". With the tag VBG it becomes "run", and "better" (JJR) becomes "good".
- *Why not remove stop words before the LLM?* "Not", "no" and "without" carry meaning. Stop-word removal only helps lexical matching.
- *How do you reduce hallucination?* Grounded prompt, citation rules, a relevance threshold, and labelling general-knowledge answers as such.
- *How does unit tagging work?* Nearest-centroid text classification: each syllabus unit's topics become a TF-IDF vector (Porter stems), and a question goes to the unit with the highest cosine similarity. Below 0.05 it is "Outside syllabus".
- *How is hallucination detected?* Each answer sentence is checked against the retrieved chunks (unigram recall, like ROUGE-1 recall). Sentences with under 25% support are shown as "not found in your notes".
- *Follow-up questions like "give an example"?* Short queries are combined with the previous question before retrieval (`Assistant.retrieve`).

## 6. Demo script (8 minutes)

1. **Faculty** (PIN `1234`) → Materials & syllabus → load the course syllabus and notes. Five units are detected.
2. Faculty → Class quizzes → generate a Unit I quiz → untick a weak question → **Publish**. Assignments → create one with the sample rubric.
3. **Student** (name + roll) → Home: exam readiness, today's plan, deadlines, badges. Check in as *Stressed* to show the supportive response.
4. Ask *"Compare stemming and lemmatization"* → citations, Unit II · Analyze · grounding tags, LED hourglass → tick. Send the doubt **anonymously**.
5. Quiz → take the teacher's quiz. Assignments → submit an answer.
6. **Faculty** → Class overview: weekly summary, *Needs re-teaching* → **Teaching ideas**. Assignments → **AI review** → adjust → **Release mark**. Doubts → **Reply + add to FAQ**. Reports → **Write AI summary** → download.
7. Student → Ask the same question again → the teacher's FAQ answer appears first.
8. NLP Lab and Evaluation for the theory questions.

## 7. Project structure

```
setup.bat             one-time Windows setup (double-click)
run_app.bat           start the app      ·  run_api.bat  start the REST API
.streamlit/           Streamlit config (headless, theme)
nltk_data/            bundled NLTK data: punkt_tab, stopwords, wordnet, POS tagger, NER chunker, words, VADER
app.py                login + role-based navigation
ui/                   theme.py (design system) · common.py · student.py · faculty.py · ctx.py
api.py                FastAPI REST backend
sah/                  engine package (see architecture)
arduino/…/…ino        UNO R4 WiFi sketch
sample_data/          demo PDF + generator script
tests/test_core.py    offline test suite
data/                 created at runtime: uploads, vector index, SQLite DB (git-ignored)
```

## 8. Future work

Voice input (Whisper), password-based student accounts on a shared server, re-ranking with a cross-encoder, LLM-as-judge for answer faithfulness, and LoRA fine-tuning of a small model on the course notes for fully offline use.
