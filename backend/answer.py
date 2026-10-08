"""Step 4: ask the LLM to answer as a list of claims, each citing one source id."""
import json
import re
from . import llm

FORMAT = 'Output ONLY JSON: {"claims": [{"text": "<one sentence>", "source": "<source id>"}]}'

STRICT = (
    "TASK:ANSWER\nYou are a careful legal research assistant.\n"
    "Answer ONLY using the SOURCES provided. Never use outside knowledge.\n"
    "Each claim must be ONE sentence and cite exactly one source id, copied exactly from the brackets.\n"
    'If the sources do not contain the answer, return {"claims": []}.\n' + FORMAT
)
LOOSE = (  # used for the baseline: no abstention rule, no "only sources" rule
    "TASK:ANSWER\nYou are a helpful legal assistant. Answer the question using the context. "
    "Cite the source id for each statement.\n" + FORMAT
)


def format_sources(chunks):
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


def parse_claims(raw: str):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    claims = data.get("claims", []) if isinstance(data, dict) else []
    return [
        {"text": str(c["text"]).strip(), "source": str(c["source"]).strip().strip("[]")}
        for c in claims
        if isinstance(c, dict) and c.get("text") and c.get("source")
    ]


def generate_claims(question, chunks, strict=True):
    user = f"SOURCES:\n{format_sources(chunks)}\n\nQUESTION: {question}"
    return parse_claims(llm.chat(STRICT if strict else LOOSE, user))
