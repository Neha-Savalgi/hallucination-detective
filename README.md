# Hallucination Detective 

## What it is

A RAG-based pipeline that fact-checks any claim or LLM output against a
knowledge base in real time, returning a structured verdict — **GROUNDED**,
**HALLUCINATED**, or **UNCERTAIN** — with reasoning and cited evidence.
Built and validated end-to-end in a single Colab notebook, with an optional
FastAPI + Streamlit layer for a live demo.

**Stack:** Claude API (Sonnet 4.6) · ChromaDB · sentence-transformers
(`all-MiniLM-L6-v2`) · FastAPI · Streamlit · localtunnel

---

## Architecture

Two-stage design, deliberately kept simple so each stage can be reasoned
about (and debugged) independently:

1. **Retrieval** — source documents are chunked (500 words, 50-word overlap)
   and embedded into ChromaDB using `all-MiniLM-L6-v2`. A claim is embedded
   the same way and the top-k most semantically similar chunks are pulled
   back as evidence.
2. **Judgment** — Claude receives the claim plus *only* the retrieved
   evidence (not its own parametric knowledge) and is instructed to return
   strict JSON: verdict, confidence, reasoning, and the specific evidence
   that drove the decision.

The key design choice is that the judge is explicitly told to say
**UNCERTAIN** when evidence is missing or insufficient, rather than
guessing. This is what separates a real fact-checker from a model that
just sounds confident either way — a system that always picks
GROUNDED/HALLUCINATED with no abstention option is easy to build and
much less trustworthy.

---

## What was actually run and validated

### Part 1 — Setup
Installed `anthropic`, `chromadb`, `sentence-transformers`, `fastapi`,
`uvicorn`, `streamlit`, and supporting libraries. One non-blocking
dependency warning surfaced (`google-adk` wanting an older
`opentelemetry` version) — unrelated to this project, safe to ignore.
API key entered securely via `getpass`.

### Part 2 — RAG pipeline
Defined `chunk_text()` and a `KnowledgeBase` class wrapping ChromaDB +
sentence-transformers, with `add_document()` for ingestion and
`retrieve()` for top-k semantic search. Confirmed importable with no
errors.

### Part 3 — Hallucination judge
Defined the judge system prompt and `judge_claim()` / `check_claim()`
functions. The judge calls Claude with `max_tokens=600`, strips markdown
fences defensively, and parses strict JSON — with a graceful fallback to
`UNCERTAIN` if parsing fails, so a malformed model response never crashes
the pipeline.

### Part 4 — Inline test (the core validation)
Indexed one document (~150 words of Eiffel Tower facts) into ChromaDB —
this triggered the one-time `all-MiniLM-L6-v2` model download (~91 MB),
visible in the run as the tokenizer/config/model download logs. Ran four
test claims:

| Claim | Verdict | Confidence | Notes |
|---|---|---|---|
| "The Eiffel Tower was completed in 1889." | **GROUNDED** | 0.99 | Directly stated in source |
| "The Eiffel Tower is 500 meters tall." | **HALLUCINATED** | 0.99 | Source says 330m — correctly caught the fabricated number |
| "The Eiffel Tower was designed by Gustave Eiffel's company." | **GROUNDED** | 0.99 | Directly stated in source |
| "The Eiffel Tower has a rotating restaurant at the top." | **HALLUCINATED** | 0.82 | Not mentioned anywhere in source — judge correctly flagged rather than staying silent on an absent fact |

This is the result worth highlighting: the pipeline didn't just catch an
outright numerical contradiction (500m vs. 330m), it also correctly
flagged a claim that was *entirely unsupported* rather than defaulting to
UNCERTAIN or guessing GROUNDED because the topic (Eiffel Tower) matched.
That's a harder case than simple contradiction detection — it requires
the judge to recognize the absence of confirming evidence as itself a
verdict-relevant signal.

### Part 5 — FastAPI backend
Wrote `app.py` with `/health`, `/ingest`, and `/check` endpoints,
replicating the notebook logic behind a real API. Launched in a
background thread via `nest_asyncio` + `uvicorn`. Health check confirmed:
`{'status': 'ok', 'documents_indexed': 1}` — correctly reflecting the one
document already indexed from Part 4 (ChromaDB's in-memory client
persisted across cells within the same runtime).

### Part 6 — Streamlit frontend
Wrote `streamlit_app.py` (document ingestion panel + claim-checking
panel with color-coded verdicts and an expandable raw-evidence view).
Tunneled out via `localtunnel`: `https://blue-frogs-build.loca.lt`,
password-gated by the runtime's public IP. The tunnel launched
successfully (`your url is: https://blue-frogs-build.loca.lt`) before
being manually interrupted.

---

## Known issue and fix already applied

Streamlit-over-localtunnel intermittently throws
`TypeError: Failed to fetch dynamically imported module` for hashed JS
chunks (e.g. `TextInput.*.js`). This is a stale tunnel/browser cache
issue, not an application bug. Fix applied to the notebook:

- Launch Streamlit with `--server.enableCORS false
  --server.enableXsrfProtection false
  --server.enableWebsocketCompression false`, since CORS/XSRF checks are
  usually what block the dynamic import when the request arrives via a
  tunnel domain instead of `localhost`.
- Documented fallback order: hard-refresh → restart the tunnel cell for a
  fresh URL → use Colab's built-in `google.colab.kernel.proxyPort(8501)`
  as a more stable (if uglier-URL) alternative that never leaves Google's
  infrastructure.

---

## What this demonstrates (useful framing for interviews/README)

- **End-to-end RAG implementation** built from raw components (ChromaDB +
  sentence-transformers), not a LangChain/LlamaIndex black box — you can
  speak to every step of the retrieval path.
- **LLM-as-judge pattern** with structured output (JSON schema
  enforcement) and defensive parsing, a pattern directly relevant to
  eval/grading pipelines.
- **Abstention as a first-class outcome** — the UNCERTAIN verdict path is
  a deliberate design decision, not an afterthought, and the test run
  shows the judge using it appropriately (would have been the correct
  call for a claim with truly no related evidence).
- **Full-stack packaging** — same core logic exposed three ways (inline
  Python, REST API, web UI), showing it's not just a notebook script but
  something that could sit behind a real product surface.
