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

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Knowledge base: one short document per sensor (Jena Climate) -------------
SENSOR_DOCS: dict[str, str] = {
    "T (degC)": "T (degC) is the air temperature in degrees Celsius. Typical range about -23 to 37 C, with strong seasonal and daily cycles.",
    "p (mbar)": "p (mbar) is atmospheric pressure in millibars. Higher pressure often means settled weather.",
    "Tpot (K)": "Tpot (K) is potential temperature in Kelvin; it tracks air temperature closely.",
    "Tdew (degC)": "Tdew (degC) is the dew-point temperature; the temperature at which air becomes saturated.",
    "rh (%)": "rh (%) is relative humidity in percent. It tends to rise when temperature falls.",
    "VPmax (mbar)": "VPmax (mbar) is saturation vapour pressure; it increases sharply with temperature.",
    "VPact (mbar)": "VPact (mbar) is actual vapour pressure, the real water-vapour content of the air.",
    "VPdef (mbar)": "VPdef (mbar) is vapour-pressure deficit; high values mean dry air.",
    "sh (g/kg)": "sh (g/kg) is specific humidity, mass of water vapour per kilogram of air.",
    "H2OC (mmol/mol)": "H2OC (mmol/mol) is water-vapour concentration.",
    "rho (g/m**3)": "rho (g/m**3) is air density. Cold air is denser, so density moves opposite to temperature.",
    "wv (m/s)": "wv (m/s) is wind speed. Occasional -9999 readings are sensor errors and are cleaned out.",
    "max. wv (m/s)": "max. wv (m/s) is the maximum wind gust in the interval.",
    "wd (deg)": "wd (deg) is wind direction in degrees.",
}


@dataclass
class Snippet:
    sensor: str
    text: str
    score: float


class Retriever:
    """TF-IDF retriever over the sensor knowledge base (local, offline)."""

    def __init__(self, docs: dict[str, str] | None = None):
        self.docs = docs or SENSOR_DOCS
        self.keys = list(self.docs.keys())
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self.docs.values())

    def retrieve(self, query: str, k: int = 2) -> list[Snippet]:
        """Return the top-k most relevant sensor docs for a free-text query."""
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [Snippet(self.keys[i], self.docs[self.keys[i]], float(sims[i]))
                for i in order if sims[i] > 0]

    def for_sensor(self, sensor: str) -> Snippet | None:
        """Direct lookup of a sensor's description (no search)."""
        if sensor in self.docs:
            return Snippet(sensor, self.docs[sensor], 1.0)
        return None


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
