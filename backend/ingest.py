"""Step 2: documents -> chunks. Every chunk remembers its document and page.

Run:  python -m backend.ingest
"""
import json
import re
from pathlib import Path
from pypdf import PdfReader
from . import config


def read_pages(path: Path):
    """Return a list of (page_number, text)."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(i + 1, t) for i, t in enumerate(text.split("\f"))]  # \f = page break


LEGAL_START = re.compile(r"^(Section|Sec\.|Article|Rule)\s+\d+|^\d+\.\s")


def chunk_text(text: str, max_words: int = 180, overlap: int = 30):
    """Greedy paragraph packing; very long paragraphs are split with overlap."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]
    chunks, current = [], []

    def flush():
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for p in paras:
        words = p.split()
        if LEGAL_START.match(p):  # a new Section / numbered paragraph starts a new chunk
            flush()
        if len(words) > max_words:  # oversize paragraph
            flush()
            step = max_words - overlap
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + max_words]))
            continue
        if len(" ".join(current).split()) + len(words) > max_words:
            flush()
        current.append(p)
    flush()
    return chunks


def build_chunks(folder: Path):
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".pdf", ".txt"})
    out = []
    for f in files:
        for page, text in read_pages(f):
            for n, piece in enumerate(chunk_text(text), start=1):
                out.append({"id": f"{f.stem}_p{page}_c{n}", "doc": f.name, "page": page, "text": piece})
    return out


def main():
    folder = config.RAW_DIR
    if not any(folder.glob("*.pdf")) and not any(folder.glob("*.txt")):
        print(f"[!] {folder} is empty -> using SYNTHETIC sample docs in {config.SAMPLE_DIR}")
        folder = config.SAMPLE_DIR
    chunks = build_chunks(folder)
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    docs = {c["doc"] for c in chunks}
    print(f"Wrote {len(chunks)} chunks from {len(docs)} documents -> {config.CHUNKS_PATH}")
    empty = [d for d in docs if not any(c["doc"] == d and c["text"] for c in chunks)]
    if empty:
        print("[!] No text extracted from:", empty, "(scanned PDF? needs OCR)")


if __name__ == "__main__":
    main()
