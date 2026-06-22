"""
v2 natural-language interface.

Turns a plain-English question into a call to the **query engine** (the truth layer), then returns
a trust-aware answer. The local LLM (if available) may only *phrase* the computed facts — the
numbers always come from `query_engine`, never from the model.

Pipeline:  question -> detect intent + country/flow/dates -> query_engine computes -> answer.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

import pandas as pd

from . import query_engine as qe
from . import reasoning as rz
from . import preprocess as pp
from . import llm

# country name -> Eurostat code
COUNTRY_ALIASES = {
    "austria": "AT", "belgium": "BE", "bulgaria": "BG", "cyprus": "CY", "czechia": "CZ",
    "czech republic": "CZ", "germany": "DE", "denmark": "DK", "estonia": "EE", "greece": "EL",
    "spain": "ES", "finland": "FI", "france": "FR", "croatia": "HR", "hungary": "HU",
    "ireland": "IE", "italy": "IT", "lithuania": "LT", "luxembourg": "LU", "latvia": "LV",
    "malta": "MT", "netherlands": "NL", "holland": "NL", "poland": "PL", "portugal": "PT",
    "romania": "RO", "sweden": "SE", "slovenia": "SI", "slovakia": "SK",
}

INTENTS = {
    "max":         ["highest", "most", "maximum", "peak", "largest", "biggest export", "top month"],
    "min":         ["lowest", "least", "minimum", "smallest", "weakest"],
    "mean":        ["average", "mean", "typical", "usually"],
    "correlation": ["correlation", "correlated", "move together", "related", "connected", "linked"],
    "surplus":     ["surplus"],
    "deficit":     ["deficit"],
    "balance":     ["balance"],
    "growth":      ["grow", "growth", "increase", "increased", "change", "rise", "rose"],
    "drop":        ["worst", "biggest drop", "crash", "collapse", "fall", "plunge", "drop"],
    "rank":        ["top exporters", "rank", "ranking", "biggest exporters", "largest exporters", "who exports"],
    "summary":     ["summary", "summarise", "summarize", "overview", "tell me about", "describe"],
    "shock":       ["shock", "anomaly", "abnormal", "which countries", "crisis", "systemic"],
}


@dataclass
class Answer:
    question: str
    intent: str
    computed: str                       # the exact, code-computed facts (the truth)
    inferred: str | None = None         # labelled interpretation, if any
    phrased: str | None = None          # optional LLM rephrasing of `computed`
    error: str | None = None
    trust: str = "computed"

    def __str__(self) -> str:
        if self.error:
            return f"⚠️ {self.error}"
        out = [f"🔢 {self.computed}"]
        if self.inferred:
            out.append(f"🧠 {self.inferred}")
        if self.phrased:
            out.append(f"💬 {self.phrased}")
        return "\n".join(out)


# --------------------------------------------------------------------------- parsing
def detect_intent(q: str) -> str | None:
    ql = q.lower()
    # word-boundary match so e.g. "rise" doesn't fire inside "summarise"
    for intent, kws in INTENTS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", ql) for k in kws):
            return intent
    return None


def resolve_countries(q: str) -> list[str]:
    ql = q.lower()
    found = []
    for name, code in COUNTRY_ALIASES.items():
        if re.search(rf"\b{name}\b", ql) and code not in found:
            found.append(code)
    # also accept bare 2-letter codes (DE, FR ...)
    for code in set(COUNTRY_ALIASES.values()):
        if re.search(rf"\b{code}\b", q) and code not in found:
            found.append(code)
    return found


def resolve_flow(q: str) -> str:
    return "IMP" if re.search(r"\bimport", q.lower()) else "EXP"


def resolve_months(q: str, df: pd.DataFrame) -> list[str]:
    """Find YYYY-MM or bare YYYY tokens and map them to a real month in the data."""
    months = re.findall(r"\b(19|20)\d\d-\d\d\b", q)
    months = re.findall(r"\b(?:19|20)\d\d-\d\d\b", q)
    out = [f"{m}-01" for m in months]
    for yr in re.findall(r"\b(?:19|20)\d\d\b", q):
        cand = [d for d in df.index if d.year == int(yr)]
        if cand:
            out.append(str(cand[-1].date()))           # last available month of that year
    # de-dup preserving order
    seen, res = set(), []
    for m in out:
        if m not in seen:
            seen.add(m); res.append(m)
    return res


# --------------------------------------------------------------------------- main entry
def ask(df: pd.DataFrame, question: str, use_llm: bool = False, model: str = llm.DEFAULT_MODEL) -> Answer:
    intent = detect_intent(question)
    flow = resolve_flow(question)
    cc = resolve_countries(question)
    months = resolve_months(question, df)
    latest = str(df.index[-1].date())

    try:
        if intent in ("max", "min"):
            if not cc:
                return Answer(question, intent, "", error="Which country? (e.g. 'France').")
            e = qe.get_max(df, cc[0], flow) if intent == "max" else qe.get_min(df, cc[0], flow)
            computed = f"{cc[0]} {flow}: {'highest' if intent=='max' else 'lowest'} was {e.value:,.0f} EUR m in {e.month}."

        elif intent == "mean":
            if not cc:
                return Answer(question, intent, "", error="Which country?")
            m, s = qe.get_mean_std(df, cc[0], flow)
            computed = f"{cc[0]} {flow}: mean {m:,.0f} EUR m (std {s:,.0f}) over {len(df)} months."

        elif intent == "correlation":
            if len(cc) < 2:
                return Answer(question, intent, "", error="Name two countries to correlate.")
            c = qe.get_correlation(df, cc[0], cc[1], flow)
            computed = f"Correlation of {cc[0]} and {cc[1]} {flow} (monthly growth) = {c:.2f}."

        elif intent in ("surplus", "deficit", "balance"):
            month = months[0] if months else latest
            if cc:
                b = qe.trade_balance(df, cc[0], month)
                word = "surplus" if b >= 0 else "deficit"
                computed = f"{cc[0]} trade balance in {month}: {b:,.0f} EUR m ({word})."
            else:
                e = qe.biggest_deficit(df, month) if intent == "deficit" else qe.biggest_surplus(df, month)
                computed = f"Biggest {intent} in {month}: {e.country} at {e.value:,.0f} EUR m."

        elif intent == "growth":
            if not cc:
                return Answer(question, intent, "", error="Which country?")
            if len(months) >= 2:
                a, b = months[0], months[1]
            else:
                a, b = str(df.index[0].date()), latest
            g = qe.growth(df, cc[0], flow, a, b)
            computed = f"{cc[0]} {flow} grew {g:+.0f}% from {a} to {b}."

        elif intent == "drop":
            if not cc:
                return Answer(question, intent, "", error="Which country?")
            e = qe.worst_drop(df, cc[0], flow)
            computed = f"{cc[0]} {flow}: worst single-month fall was {e.value:.1f}% in {e.month}."

        elif intent == "rank":
            month = months[0] if months else latest
            r = qe.rank_countries(df, flow, month, n=5)
            items = ", ".join(f"{k} {v:,.0f}" for k, v in r.items())
            computed = f"Top {flow} in {month} (EUR m): {items}."

        elif intent == "summary":
            if not cc:
                return Answer(question, intent, "", error="Which country?")
            s = qe.country_summary(df, cc[0])
            computed = (f"{cc[0]}: avg exports {s['exp_mean']:,.0f}, avg imports {s['imp_mean']:,.0f} EUR m; "
                        f"in surplus {s['surplus_share']:.0%} of months; latest balance {s['latest_balance']:,.0f}.")

        elif intent == "shock":
            month = months[0] if months else "2020-04-01"
            logged = pp.log_transform(df)
            cz = rz.cross_section_z(logged, "_EXP")
            rep = rz.explain_month(cz, month)
            computed = rep.sentence()
            inferred = None
            if rep.kind == "shared shock":
                inferred = "A region-wide cause is likely (e.g. a global crisis), not a single-country issue."
            ans = Answer(question, intent, computed, inferred=inferred)
            if use_llm:
                ans.phrased = llm.explain(question, computed, inferred, model=model)
            return ans

        else:
            return Answer(question, "unknown", "",
                          error="I can answer: highest/lowest, average, correlation, surplus/deficit, "
                                "growth, worst drop, top exporters, summary, or shocks.")

        ans = Answer(question, intent, computed)
        if use_llm:
            ans.phrased = llm.explain(question, computed, model=model)
        return ans

    except (KeyError, ValueError, IndexError) as exc:
        return Answer(question, intent or "error", "", error=f"Could not compute: {exc}")


SUGGESTIONS = [
    "What was France's highest export month?",
    "How much did Poland's exports grow from 2010 to 2024?",
    "Are Germany and France exports correlated?",
    "Who had the biggest trade surplus in 2024?",
    "What was Spain's worst monthly export drop?",
    "Top exporters in 2024?",
    "Summarise the Netherlands.",
    "Which countries were abnormal in 2020-04?",
]
