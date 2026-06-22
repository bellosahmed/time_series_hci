"""
v2 RAG layer — hybrid retrieval (lexical BM25 + semantic LSA), fully offline.

Two retrievers over a small trade knowledge base, merged and re-ranked:
  * BM25  - classic lexical/sparse scoring (good at exact terms; pure-python, no deps).
  * LSA   - TF-IDF -> TruncatedSVD dense vectors (semantic: matches meaning/paraphrase; sklearn).
  * Hybrid- min-max normalise each, weighted-combine, re-rank (best of both).

Why LSA and not neural embeddings? The installed Ollama models don't support embeddings and we want
zero new downloads. LSA gives an offline semantic signal now; swap in Ollama/`nomic-embed-text`
later behind the same `search()` interface.

RAG retrieves CONTEXT only (descriptions, known events). It NEVER provides numbers — those always
come from the query engine. Answers stay trust-aware: 🔢 computed · 📚 retrieved · 🧠 inferred.
"""
from __future__ import annotations
import math
import re
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------- chunking
def chunk_text(text: str, max_words: int = 40, overlap: int = 10) -> list[str]:
    """Split a long document into overlapping, sentence-aware chunks.

    Needed only for *long* sources (reports, articles); our KB docs are already atomic.
    `overlap` repeats the last few words so meaning isn't cut at a boundary.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur: list[str] = []
    for sent in sentences:
        words = sent.split()
        if cur and len(cur) + len(words) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []      # carry overlap into the next chunk
        cur.extend(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# --------------------------------------------------------------------------- knowledge base
def knowledge_base() -> list[tuple[str, str]]:
    """Small KB of (doc_id, text). Context only — no exact numbers live here."""
    return [
        # --- countries (qualitative descriptions) ---
        ("DE", "Germany (DE) is the European Union's largest exporter and trading nation. It "
               "consistently runs a large trade surplus, exporting cars, machinery and chemicals."),
        ("NL", "The Netherlands (NL) is a major trade hub with the port of Rotterdam. It re-exports "
               "many goods and usually records a trade surplus."),
        ("FR", "France (FR) is a large economy that typically runs a trade deficit, importing more "
               "goods than it exports."),
        ("IT", "Italy (IT) is a major manufacturing exporter of machinery, fashion and food."),
        ("PL", "Poland (PL) is a fast-growing central European economy whose trade has expanded "
               "rapidly since joining the EU in 2004."),
        ("IE", "Ireland (IE) has volatile trade figures distorted by multinational pharmaceutical "
               "and tech companies booking activity there."),
        # --- trade concepts ---
        ("exports", "Exports are goods a country sells to the rest of the world; money flows in."),
        ("imports", "Imports are goods a country buys from abroad; money flows out."),
        ("balance", "Trade balance equals exports minus imports. Positive is a surplus, negative is "
                    "a deficit."),
        ("surplus", "A trade surplus means a country exports more than it imports."),
        ("deficit", "A trade deficit means a country imports more than it exports."),
        # --- known real-world events (context for anomalies) ---
        ("event_2008", "The 2008 global financial crisis caused a sharp collapse in world trade in "
                       "late 2008 and early 2009, with exports falling around 25-30 percent."),
        ("event_covid", "The COVID-19 pandemic in early 2020 triggered lockdowns that halted global "
                        "trade; exports crashed about 30 percent in April 2020 before rebounding."),
        ("event_2022", "The 2022 energy crisis following Russia's invasion of Ukraine disrupted "
                       "trade, spiked energy prices and collapsed EU imports from Russia."),
        ("event_rebound", "After the COVID-19 collapse, trade rebounded strongly in 2021, producing "
                          "unusually large year-on-year increases."),
        # --- methods ---
        ("stl", "STL decomposition splits a time series into trend, seasonal and residual parts; "
                "anomalies show up in the residual."),
        ("zscore", "A z-score measures how many standard deviations a value is from normal; large "
                   "absolute z-scores flag anomalies."),
        ("isoforest", "Isolation Forest is an unsupervised machine-learning anomaly detector that "
                      "isolates unusual points; used here across many countries for systemic shocks."),
        ("sarima", "SARIMA is a seasonal forecasting model; it predicts the expected range, and "
                   "values outside its prediction band are anomalies."),
        ("seasonality", "Trade is seasonal, with regular dips in summer and December; this yearly "
                        "pattern must be removed before detecting anomalies."),
    ]


# --------------------------------------------------------------------------- BM25 (lexical)
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[tuple[str, str]]):
        self.ids = [d[0] for d in docs]
        corpus = [tokenize(d[1]) for d in docs]
        self.tf = [Counter(c) for c in corpus]
        self.doc_len = [len(c) for c in corpus]
        self.N = len(corpus)
        self.avgdl = sum(self.doc_len) / self.N
        df = Counter()
        for c in corpus:
            df.update(set(c))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        return self

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores = []
        for i in range(self.N):
            s = 0.0
            for w in q:
                f = self.tf[i].get(w, 0)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
            scores.append((self.ids[i], s))
        return sorted(scores, key=lambda x: -x[1])[:k]


# --------------------------------------------------------------------------- LSA (semantic)
class SemanticLSA:
    def __init__(self, n_components: int = 64):
        self.n_components = n_components

    def fit(self, docs: list[tuple[str, str]]):
        self.ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        self.vec = TfidfVectorizer(stop_words="english")
        X = self.vec.fit_transform(texts)
        n = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n, random_state=0)
        self.emb = normalize(self.svd.fit_transform(X))
        return self

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = normalize(self.svd.transform(self.vec.transform([query])))
        sims = (self.emb @ q.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]


# --------------------------------------------------------------------------- BERT (neural)
class BertRetriever:
    """Dense neural retrieval via sentence-transformers (a distilled BERT). Drop-in for SemanticLSA.

    Requires `sentence-transformers` (downloads a small model once). Best on large/varied corpora;
    for our tiny KB it is comparable to LSA but captures paraphrase more strongly.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, docs: list[tuple[str, str]] | None = None):
        docs = docs or knowledge_base()
        self.ids = [d[0] for d in docs]
        self.docs = dict(docs)
        self.emb = self.model.encode([d[1] for d in docs], normalize_embeddings=True)
        return self

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.emb @ q
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def retrieve_texts(self, query: str, k: int = 2) -> list[tuple[str, str]]:
        return [(i, self.docs[i]) for i, _ in self.search(query, k)]


def _semantic_backend(name: str):
    """Pick the semantic retriever: 'lsa' (offline) or 'bert' (neural)."""
    return BertRetriever() if name == "bert" else SemanticLSA()


# --------------------------------------------------------------------------- hybrid
def _minmax(pairs: dict[str, float]) -> dict[str, float]:
    if not pairs:
        return {}
    vals = list(pairs.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in pairs}
    return {k: (v - lo) / (hi - lo) for k, v in pairs.items()}


class HybridRetriever:
    """Combine BM25 + LSA: score = alpha*lexical + (1-alpha)*semantic, then re-rank."""

    def __init__(self, alpha: float = 0.5, semantic: str = "lsa"):
        self.alpha = alpha
        self.semantic = semantic                          # 'lsa' (offline) or 'bert' (neural)

    def fit(self, docs: list[tuple[str, str]] | None = None):
        docs = docs or knowledge_base()
        self.docs = dict(docs)
        self.bm25 = BM25().fit(docs)
        self.sem = _semantic_backend(self.semantic).fit(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        n = len(self.docs)
        bn = _minmax(dict(self.bm25.search(query, k=n)))
        sn = _minmax(dict(self.sem.search(query, k=n)))
        combined = {i: self.alpha * bn.get(i, 0.0) + (1 - self.alpha) * sn.get(i, 0.0)
                    for i in self.docs}
        return sorted(combined.items(), key=lambda x: -x[1])[:k]

    def retrieve_texts(self, query: str, k: int = 2) -> list[tuple[str, str]]:
        return [(i, self.docs[i]) for i, _ in self.search(query, k)]


# --------------------------------------------------------------------------- trust-aware explain
def explain(computed: str, retrieved: list[tuple[str, str]] | None = None,
            inferred: str | None = None) -> str:
    """Assemble a trust-labelled answer: 🔢 computed (truth) · 📚 retrieved · 🧠 inferred."""
    lines = [f"🔢 {computed}"]
    for _id, text in (retrieved or []):
        lines.append(f"📚 {text}")
    if inferred:
        lines.append(f"🧠 {inferred}")
    return "\n".join(lines)
