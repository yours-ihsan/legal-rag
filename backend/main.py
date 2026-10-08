"""Step 6: FastAPI app.  Run:  uvicorn backend.main:app --reload"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import config
from .pipeline import VARIANTS, run
from .retrieve import Retriever

app = FastAPI(title="Verifiable Legal Assistant")
state = {}


class Question(BaseModel):
    text: str
    variant: str = "full"


@app.on_event("startup")
def load():
    if not config.CHUNKS_PATH.exists():
        raise RuntimeError("Run `python -m backend.ingest` first.")
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    state["retriever"] = Retriever(chunks)


@app.post("/ask")
def ask(q: Question):
    if q.variant not in VARIANTS:
        raise HTTPException(400, f"variant must be one of {list(VARIANTS)}")
    if not q.text.strip():
        raise HTTPException(400, "empty question")
    try:
        return run(state["retriever"], q.text, q.variant)
    except Exception as e:  # LLM down, bad key, etc.
        raise HTTPException(502, f"LLM call failed: {e}")


@app.get("/health")
def health():
    r = state["retriever"]
    return {"chunks": len(r.chunks), "dense": r.dense_kind, "llm": config.LLM_PROVIDER, "model": config.LLM_MODEL}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="ui")
