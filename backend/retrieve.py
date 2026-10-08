"""Step 3: hybrid retrieval = BM25 (keywords) + dense (meaning), merged with RRF."""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from . import config


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self._init_dense()

    def _init_dense(self):
        texts = [c["text"] for c in self.chunks]
        try:  # best option: real embeddings
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.EMBED_MODEL)
            self.matrix = self.model.encode(texts, normalize_embeddings=True)
            self.dense_kind = "sentence-transformers"
        except Exception:  # fallback: TF-IDF with word pairs (no download needed)
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self.matrix = self.model.fit_transform(texts)
            self.dense_kind = "tfidf-fallback"

    def _rank_bm25(self, q):
        scores = self.bm25.get_scores(tokenize(q))
        return list(np.argsort(-scores))

    def _rank_dense(self, q):
        if self.dense_kind == "sentence-transformers":
            qv = self.model.encode([q], normalize_embeddings=True)[0]
            scores = self.matrix @ qv
        else:
            scores = (self.matrix @ self.model.transform([q]).T).toarray().ravel()
        return list(np.argsort(-scores))

    def search(self, query: str, k: int = None, mode: str = "hybrid"):
        """mode: 'bm25' | 'dense' | 'hybrid'. Returns top-k chunks (dicts)."""
        k = k or config.TOP_K
        if mode == "bm25":
            order = self._rank_bm25(query)[:k]
        elif mode == "dense":
            order = self._rank_dense(query)[:k]
        else:  # Reciprocal Rank Fusion: score = sum 1/(60 + rank)
            fused = {}
            for ranking in (self._rank_bm25(query)[:30], self._rank_dense(query)[:30]):
                for rank, idx in enumerate(ranking):
                    fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)
            order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [dict(self.chunks[i]) for i in order]
