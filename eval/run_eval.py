"""Step 8: compare the 4 variants. Run:  python -m eval.run_eval

Metrics
- recall@k       : gold phrase appears in a retrieved chunk (answerable questions)
- groundedness   : % of SHOWN claims whose cited chunk is judged to support them
- fabricated     : shown citations that do not exist anywhere in the corpus (must be 0)
- abstain_ok     : unanswerable questions where the system showed nothing
NOTE: groundedness uses an LLM judge -> also spot-check ~20 claims by hand and report that.
"""
import json
from pathlib import Path
from backend import config, llm
from backend.pipeline import VARIANTS, run
from backend.retrieve import Retriever

JUDGE = ("TASK:JUDGE\nYou are a strict fact-checker. Reply YES only if EVERY detail of the claim is stated in "
         "the passage. Otherwise reply NO. Reply with a single word.")


def main():
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in chunks}
    retriever = Retriever(chunks)
    questions = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
    rows = []
    for variant in VARIANTS:
        hit = n_ans = claims_total = claims_ok = fabricated = abst_ok = n_unans = 0
        for item in questions:
            res = run(retriever, item["q"], variant)
            if item["answerable"]:
                n_ans += 1
                hit += any(item["gold_phrase"].lower() in c["text"].lower() for c in res["sources"])
            else:
                n_unans += 1
                abst_ok += res["abstained"]
            for cl in res["claims"]:
                claims_total += 1
                src = by_id.get(cl["source"])
                if src is None:
                    fabricated += 1
                    continue
                claims_ok += llm.chat(JUDGE, f"SOURCE:\n{src['text']}\n\nCLAIM:\n{cl['text']}").strip().upper().startswith("YES")
        rows.append((variant, hit / max(n_ans, 1), claims_ok / max(claims_total, 1), fabricated, abst_ok / max(n_unans, 1), claims_total))
    out = ["| Variant | Recall@k | Groundedness | Fabricated citations | Abstains when unanswerable | Claims shown |",
           "|---|---|---|---|---|---|"]
    out += [f"| {v} | {r:.0%} | {g:.0%} | {f} | {a:.0%} | {n} |" for v, r, g, f, a, n in rows]
    text = "\n".join(out)
    print(f"Dense retriever: {retriever.dense_kind} | LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}\n")
    print(text)
    (Path(__file__).parent / "results.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
