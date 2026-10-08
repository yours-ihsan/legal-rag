"""Step 5: the verifier. Two gates: (1) code check, (2) LLM entailment check."""
from . import llm

VERIFY_SYSTEM = (
    "TASK:VERIFY\nYou check whether a source passage supports a claim.\n"
    "Reply YES only if the passage directly states or clearly implies the claim. Otherwise reply NO.\n"
    "Reply with a single word."
)


def verify_claims(claims, chunks, use_llm=True):
    by_id = {c["id"]: c for c in chunks}
    supported, rejected = [], []
    for cl in claims:
        src = by_id.get(cl["source"])
        if src is None:  # gate 1: citation must be one of the retrieved chunks
            rejected.append({**cl, "reason": "citation is not one of the retrieved sources"})
            continue
        if use_llm:  # gate 2: does the cited text really support the claim?
            verdict = llm.chat(VERIFY_SYSTEM, f"SOURCE:\n{src['text']}\n\nCLAIM:\n{cl['text']}")
            if not verdict.strip().upper().startswith("YES"):
                rejected.append({**cl, "reason": "source does not support the claim"})
                continue
        supported.append(cl)
    return supported, rejected
