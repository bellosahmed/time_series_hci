"""RAG layer: local context retrieval + (template) explanation.

Two parts, both fully local and offline:

1. **Retriever** — a TF-IDF search over a small knowledge base of sensor
   descriptions. This is the "R" in RAG; it grounds answers in real metadata
   without any external service. (A neural embedding store can replace TF-IDF
   later; the interface stays the same.)
2. **Explainer** — turns a computed result into natural language with an explicit
   **trust label** (computed / retrieved / inferred). Today it uses deterministic
   templates; a local LLM (Ollama) can be dropped in later to phrase the *same
   grounded facts* — it must never change the numbers.

Trust labels make the core principle visible to the user:
    🔢 computed  = exact, from the query engine (the truth)
    📚 retrieved = context pulled from the knowledge base
    🧠 inferred  = reasoning on top of computed values
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"  # a small local Ollama embedding model (`ollama pull nomic-embed-text`)
ST_MODEL = "all-MiniLM-L6-v2"     # the module's recommended local dense encoder (sentence-transformers)

# --- Knowledge base: one short document per sensor (Jena Climate) -------------
# Each doc pairs the technical definition with everyday synonyms and "how it feels" language, so
# retrieval (especially the dense encoder) can match colloquial/paraphrase queries. Enriched after
# the retrieval evaluation showed the original one-line docs missed queries like "muggy" / "breezy"
# (see docs/RETRIEVAL_EVALUATION.md); the effect is re-measured on the same query sets.
SENSOR_DOCS: dict[str, str] = {
    "T (degC)": "T (degC) is the air temperature in degrees Celsius — how hot or cold it is outside. "
                "Covers heat, warmth, chill and freezing conditions: scorching, hot, warm, mild, cool, "
                "cold, and icy days. Typical range about -23 to 37 C, with strong seasonal and daily cycles.",
    "p (mbar)": "p (mbar) is atmospheric or barometric pressure in millibars — how high or low the "
                "barometer reads. Higher pressure often means settled, calm, fair weather; low pressure "
                "brings unsettled, stormy weather.",
    "Tpot (K)": "Tpot (K) is potential temperature in Kelvin, a temperature measure that tracks the air "
                "temperature (heat/warmth) closely once altitude effects are removed.",
    "Tdew (degC)": "Tdew (degC) is the dew-point temperature — the temperature at which the air becomes "
                   "saturated and dew, mist or condensation forms. A high dew point makes the air feel "
                   "muggy, clammy and humid.",
    "rh (%)": "rh (%) is relative humidity in percent — how much moisture or water vapour is in the air, "
              "i.e. how muggy, sticky, damp, humid, clammy or close it feels versus dry. It tends to rise "
              "when temperature falls.",
    "VPmax (mbar)": "VPmax (mbar) is the saturation vapour pressure — the most water vapour the air could "
                    "hold at its current temperature. It increases sharply as the air gets hotter.",
    "VPact (mbar)": "VPact (mbar) is the actual vapour pressure — the real amount of water vapour or "
                    "moisture currently in the air.",
    "VPdef (mbar)": "VPdef (mbar) is the vapour-pressure deficit — how far the air is from saturation. "
                    "High values mean dry, thirsty, parched air; low values mean damp, moist air.",
    "sh (g/kg)": "sh (g/kg) is specific humidity — the mass of water vapour (moisture) per kilogram of "
                 "air, another measure of how humid or damp the air is.",
    "H2OC (mmol/mol)": "H2OC (mmol/mol) is the water-vapour concentration — how much water vapour or "
                       "moisture is present in the air.",
    "rho (g/m**3)": "rho (g/m**3) is air density — how heavy, thick, dense or thin the air is. Cold air "
                    "is heavier and denser, so density moves opposite to temperature.",
    "wv (m/s)": "wv (m/s) is wind speed — how windy, breezy, blowy, gusty or calm it is, i.e. the "
                "strength of the wind. Occasional -9999 readings are sensor errors and are cleaned out.",
    "max. wv (m/s)": "max. wv (m/s) is the maximum wind gust — the strongest sudden blast, burst, gust "
                     "or squall of wind in the interval.",
    "wd (deg)": "wd (deg) is wind direction in degrees — which way the wind blows (its compass bearing), "
                "e.g. a northerly, southerly, easterly or westerly wind.",
}


@dataclass
class Snippet:
    sensor: str
    text: str
    score: float


class _SensorDocsBase:
    """Shared sensor-doc storage + direct-lookup, common to every retriever implementation below.

    Each retriever's `__init__` calls `super().__init__(docs)` for this part, then builds its own
    index (TF-IDF matrix, BM25 index, embedding matrix, ...) on top of `self.docs`/`self.keys`.
    """

    def __init__(self, docs: dict[str, str] | None = None):
        self.docs = docs or SENSOR_DOCS
        self.keys = list(self.docs.keys())

    def for_sensor(self, sensor: str) -> Snippet | None:
        """Direct lookup of a sensor's description (no search)."""
        if sensor in self.docs:
            return Snippet(sensor, self.docs[sensor], 1.0)
        return None


class Retriever(_SensorDocsBase):
    """TF-IDF retriever over the sensor knowledge base (local, offline)."""

    def __init__(self, docs: dict[str, str] | None = None):
        super().__init__(docs)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self.docs.values())

    def retrieve(self, query: str, k: int = 2) -> list[Snippet]:
        """Return the top-k most relevant sensor docs for a free-text query."""
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [Snippet(self.keys[i], self.docs[self.keys[i]], float(sims[i]))
                for i in order if sims[i] > 0]


class LSARetriever(_SensorDocsBase):
    """Latent Semantic Analysis retriever (the LSI method from the module).

    Builds TF-IDF vectors, then reduces them with **truncated SVD** to a small latent-topic
    space and ranks by cosine similarity there. Because related terms collapse onto shared
    latent dimensions, it can match on *meaning* — a query and a doc that share topics but not
    exact keywords can still rank together — which pure keyword TF-IDF cannot. Fully local
    (scikit-learn only, no external service) and deterministic (`random_state=0`). Same
    interface as ``Retriever`` (drop-in). ``n_components`` = number of latent dimensions
    (defaults to one below the number of docs, the maximum SVD allows here).
    """

    def __init__(self, docs: dict[str, str] | None = None, n_components: int | None = None):
        super().__init__(docs)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = self._vectorizer.fit_transform(self.docs.values())
        # SVD rank is bounded by both dimensions of the term-document matrix.
        max_k = max(1, min(tfidf.shape[0], tfidf.shape[1]) - 1)
        k = min(n_components or max_k, max_k)
        self.n_components = k
        self._svd = TruncatedSVD(n_components=k, random_state=0)
        self._matrix = normalize(self._svd.fit_transform(tfidf))  # unit rows -> dot == cosine

    def retrieve(self, query: str, k: int = 2) -> list[Snippet]:
        """Top-k docs by cosine similarity in the latent-topic space."""
        q = normalize(self._svd.transform(self._vectorizer.transform([query])))
        sims = (self._matrix @ q.T).ravel()
        order = sims.argsort()[::-1][:k]
        return [Snippet(self.keys[i], self.docs[self.keys[i]], float(sims[i]))
                for i in order if sims[i] > 0]


class BM25Retriever(_SensorDocsBase):
    """BM25 (Okapi) retriever — the probabilistic ranking baseline (from the IR material).

    BM25 scores a query–document match with **term-frequency saturation** (extra occurrences
    help less and less) and **document-length normalisation**, which usually ranks short queries
    better than raw TF-IDF. It is still keyword-based, so it shares the vocabulary-mismatch
    limitation; it is included as a second sparse baseline alongside TF-IDF in the retriever
    comparison. Same interface as ``Retriever``. Requires the optional ``rank_bm25`` dependency.
    """

    def __init__(self, docs: dict[str, str] | None = None):
        from rank_bm25 import BM25Okapi

        super().__init__(docs)
        self._bm25 = BM25Okapi([self._tokenize(d) for d in self.docs.values()])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Drop English stop words so BM25 uses the SAME preprocessing as the TF-IDF baseline
        # (TfidfVectorizer(stop_words="english")). Without this, BM25 matches on common words
        # like "it"/"is", giving an unfair, artefactual comparison on paraphrase queries.
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]

    def retrieve(self, query: str, k: int = 2) -> list[Snippet]:
        """Top-k docs by BM25 score."""
        scores = self._bm25.get_scores(self._tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [Snippet(self.keys[i], self.docs[self.keys[i]], float(scores[i]))
                for i in order if scores[i] > 0]


def ollama_embed(text: str, model: str = EMBED_MODEL, url: str = OLLAMA_URL,
                 timeout: float = 10.0) -> list[float]:
    """Embed one string with a local Ollama embedding model. Raises if unreachable."""
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(f"{url}/api/embeddings", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["embedding"]


class EmbeddingRetriever(_SensorDocsBase):
    """Optional *semantic* retriever over the same sensor knowledge base.

    Embeds each doc with a local embedding model (Ollama, default ``nomic-embed-text``) and
    ranks by cosine similarity, so it matches on *meaning* rather than keyword overlap — a
    differently-phrased question ("how muggy does it get?") can still retrieve the humidity
    doc. It exposes the exact same interface as ``Retriever`` (``retrieve`` / ``for_sensor``)
    so it is a drop-in swap.

    It is always **safe to use**: if no embedding backend is reachable it transparently falls
    back to the TF-IDF ``Retriever`` (``semantic`` is then False). ``embed_fn`` is injectable
    for tests or an alternative backend. TF-IDF remains the default in the interface; this is
    the recommended-but-optional upgrade.
    """

    def __init__(self, docs: dict[str, str] | None = None, embed_fn=None):
        super().__init__(docs)
        self._embed = embed_fn or ollama_embed
        self._fallback: Retriever | None = None
        self._matrix: np.ndarray | None = None
        try:
            mat = np.asarray([self._embed(t) for t in self.docs.values()], dtype=float)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self._matrix = mat / np.clip(norms, 1e-12, None)  # unit rows -> cosine == dot
        except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError):
            self._fallback = Retriever(self.docs)  # semantic backend unavailable -> keyword search

    @property
    def semantic(self) -> bool:
        """True if the embedding backend is live; False when falling back to TF-IDF."""
        return self._matrix is not None

    def retrieve(self, query: str, k: int = 2) -> list[Snippet]:
        """Top-k docs by embedding cosine similarity (or TF-IDF if no backend)."""
        if self._fallback is not None:
            return self._fallback.retrieve(query, k)
        try:
            q = np.asarray(self._embed(query), dtype=float)
        except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError):
            return Retriever(self.docs).retrieve(query, k)  # backend died mid-session
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._matrix @ q
        order = np.argsort(sims)[::-1][:k]
        return [Snippet(self.keys[i], self.docs[self.keys[i]], float(sims[i]))
                for i in order if sims[i] > 0]


def sentence_transformer_embedder(model_name: str = ST_MODEL):
    """Build an ``embed_fn`` backed by a local **sentence-transformers** model (the module's
    recommended dense encoder, e.g. ``all-MiniLM-L6-v2``).

    Unlike keyword TF-IDF/BM25/LSA, a sentence encoder maps meaning, so it can match a
    paraphrase to its sensor even with **no shared words** ("how hot" → temperature). Downloads
    the model once (then cached and offline). Raises ``ImportError``/``OSError`` if the package
    or weights are unavailable, so callers (or ``EmbeddingRetriever``) can fall back to TF-IDF.
    Use as: ``EmbeddingRetriever(embed_fn=sentence_transformer_embedder())`` — or ``dense_retriever()``.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def _embed(text: str) -> list[float]:
        return model.encode(text, normalize_embeddings=False).tolist()

    return _embed


def dense_retriever(docs: dict[str, str] | None = None, model_name: str = ST_MODEL):
    """A dense semantic retriever (sentence-transformers) with a safe TF-IDF fallback.

    Returns an ``EmbeddingRetriever`` backed by the sentence encoder if it loads, else a plain
    TF-IDF ``Retriever`` — so it is always usable. Both expose the same interface.
    """
    try:
        return EmbeddingRetriever(docs, embed_fn=sentence_transformer_embedder(model_name))
    except Exception:
        return Retriever(docs)


def explain(headline: str, *, computed: str, context: str | None = None,
            inferred: str | None = None) -> str:
    """Compose a trust-aware answer from labelled parts.

    `computed` is the exact result (truth), `context` is retrieved metadata,
    `inferred` is optional reasoning. A local LLM could later rewrite this text,
    but only the wording — never the computed values.
    """
    lines = [headline, f"🔢 Computed: {computed}"]
    if context:
        lines.append(f"📚 Context: {context}")
    if inferred:
        lines.append(f"🧠 Inferred: {inferred}")
    return "\n".join(lines)
