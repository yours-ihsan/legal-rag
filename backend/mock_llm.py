"""A fake LLM for testing the PLUMBING only (no API needed). Not for real results."""
import json
import re

STOP = set("the a an of to in is are be and or for on by with that this it as at what which who when how can does do if".split())


def words(t):
    return {w for w in re.findall(r"\w+", t.lower()) if w not in STOP}


def chat(system: str, user: str) -> str:
    if system.startswith("TASK:ANSWER"):
        q = user.split("QUESTION:", 1)[1]
        best, best_score = None, 0
        for cid, text in re.findall(r"\[([^\]]+)\] (.*?)(?=\n\n\[|\Z)", user.split("QUESTION:")[0], re.S):
            score = len(words(q) & words(text))
            if score > best_score:
                best, best_score = (cid, text), score
        if not best or best_score < 2:
            return '{"claims": []}'
        sentences = re.split(r"(?<=[.;])\s", best[1])
        sentence = max(sentences, key=lambda x: len(words(q) & words(x)))
        return json.dumps({"claims": [{"text": sentence, "source": best[0]}]})
    # VERIFY / JUDGE: YES if most claim words appear in the source
    src = user.split("SOURCE:", 1)[1].split("CLAIM:")[0]
    claim = user.split("CLAIM:", 1)[1]
    cw = words(claim)
    return "YES" if cw and len(cw & words(src)) / len(cw) >= 0.6 else "NO"
