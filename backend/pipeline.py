"""The four variants used for the ablation table."""
from . import config
from .answer import generate_claims
from .verify import verify_claims

NOT_FOUND = "Not found in the provided sources."

# variant -> (retrieval mode, strict prompt?, verifier?)
VARIANTS = {
    "baseline": ("dense", False, False),   # naive RAG
    "hybrid": ("hybrid", False, False),    # + hybrid retrieval
    "strict": ("hybrid", True, False),     # + strict grounded prompt / abstention
    "full": ("hybrid", True, True),        # + verifier (our system)
}


def run(retriever, question: str, variant: str = "full", k: int = None):
    mode, strict, verify = VARIANTS[variant]
    chunks = retriever.search(question, k or config.TOP_K, mode=mode)
    claims = generate_claims(question, chunks, strict=strict)
    if verify:
        shown, rejected = verify_claims(claims, chunks)
    else:
        shown, rejected = claims, []
    return {
        "question": question,
        "variant": variant,
        "claims": shown,
        "rejected": rejected,
        "abstained": len(shown) == 0,
        "message": NOT_FOUND if not shown else "",
        "sources": chunks,
    }
