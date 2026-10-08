# Verifiable Legal Assistant (HNX26EPS01)

A grounded RAG chatbot for legal documents. Every claim cites one exact source chunk, and a verifier removes
claims the source does not support. If nothing is supported, the system answers "Not found in the provided sources."

## Pipeline
documents -> chunks (doc + page) -> hybrid search (BM25 + embeddings, RRF) -> LLM answers as cited claims
-> verifier (citation exists? source supports claim?) -> show supported claims, or abstain

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then edit it (see below)
```
**LLM options** (any OpenAI-compatible API): in `.env` set `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`.
- Qwen via Ollama: `ollama pull qwen2.5:7b`, then use the defaults in `.env.example`
- No LLM yet? set `LLM_PROVIDER=mock` to test the plumbing (results are meaningless)

## Run
1. Put text-based PDFs (or .txt) in `data/raw/` (if empty, synthetic demo docs in `data/sample/` are used)
2. `python -m backend.ingest`
3. `uvicorn backend.main:app --reload`  ->  open http://127.0.0.1:8000
4. Evaluate: edit `eval/questions.json` for YOUR documents, then `python -m eval.run_eval`

## Variants (the ablation)
| Variant | What it adds |
|---|---|
| baseline | dense-only retrieval, loose prompt, no checks (plain RAG) |
| hybrid | + BM25/embedding hybrid retrieval |
| strict | + "answer only from sources" prompt and abstention |
| full | + verifier (our system) |

## Files
`backend/ingest.py` chunking | `retrieve.py` hybrid search | `answer.py` cited claims | `verify.py` verifier |
`pipeline.py` variants | `main.py` API | `frontend/index.html` UI | `eval/` test set and metrics
