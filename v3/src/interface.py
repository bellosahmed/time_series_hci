"""NL interface for v3 — the HCI layer.

Routes a plain-English question to the exact `query_engine` and returns a **trust-labelled** answer:
🔢 *computed* (the exact figure from the data), 📚 *context* (background about the domain), 🧠 *inferred*
(a cautious interpretation). Routing is deterministic; the numbers come only from the query engine.
An optional local LLM (M5) will reword the computed facts without ever changing a number.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import interface_parsing as ip
from . import llm
from . import query_engine as qe

DAY_NAMES = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
             "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}

# Small knowledge base — fixed facts about the attack types (context, never computed).
ATTACK_KB = {
    "modification": "Modification changes the temperature payload to an out-of-pattern value.",
    "duplication": "Duplication replays messages, sending more than the scheduled rate.",
    "interception": "Interception drops messages, so readings arrive less often than every 5 minutes.",
}


@dataclass
class Answer:
    question: str
    intent: str | None
    text: str
    value: object = None
    computed: str = ""
    context: str | None = None
    inferred: str | None = None
    used_llm: bool = False
    chart: object = None       # small {kind, x, y, title, ...} payload for the web UI


class Interface:
    """Answer natural-language questions about the DAD MESSAGES frame."""

    def __init__(self, df: pd.DataFrame, use_llm: bool = False):
        self.df = df
        self.use_llm = use_llm

    def ask(self, question: str) -> Answer:
        intent = ip.detect_intent(question)
        if intent is None:
            return Answer(question, None,
                          "I couldn't map that to a known query. Try asking about attacks, attack "
                          "types, the attacker IP, temperatures, or traffic.")
        day = ip.resolve_day(question)
        atype = ip.resolve_attack_type(question)
        computed, value, context, inferred = self._route(intent, question, day, atype)
        text, used = self._phrase(question, computed, context, inferred)
        chart = self._chart_for(intent, day)
        return Answer(question, intent, text, value, computed, context, inferred,
                      used_llm=used, chart=chart)

    def _chart_for(self, intent, day):
        """Build a small presentation-only chart payload for the web UI (numbers already computed)."""
        if intent in ("count", "when", "attacker", "summary", "anomalies"):
            byday = qe.attacks_by_day(self.df)
            return {"kind": "bar", "title": "Attack packets per day",
                    "x": [DAY_NAMES[d][:3] for d in byday], "y": list(byday.values()),
                    "highlight": DAY_NAMES[day][:3] if day else None}
        if intent == "sensors":
            pub = self.df[self.df["temperature"].notna()]
            counts = pub.groupby("clientid").size().sort_index()
            return {"kind": "bar", "title": "Temperature readings per sensor",
                    "x": [str(int(c)) for c in counts.index], "y": [int(v) for v in counts.values],
                    "highlight": None}
        if intent == "breakdown":
            bd = qe.attack_breakdown(self.df)
            return {"kind": "bar", "title": "Attack packets by type",
                    "x": list(bd.keys()), "y": list(bd.values()), "highlight": None}
        if intent == "temperature":
            sub = self.df[self.df["day"] == day] if day else self.df
            s = sub["temperature"].dropna().resample("1h").mean().dropna()
            return {"kind": "line", "title": "Temperature (hourly mean)", "ylabel": "°C",
                    "x": [t.isoformat() for t in s.index], "y": [round(float(v), 2) for v in s.values]}
        if intent == "traffic":
            sub = self.df[self.df["day"] == day] if day else self.df
            r = sub.resample("1h").size()
            return {"kind": "line", "title": "Messages per hour", "ylabel": "messages",
                    "x": [t.isoformat() for t in r.index], "y": [int(v) for v in r.values]}
        return None

    def _phrase(self, question, computed, context, inferred) -> tuple[str, bool]:
        """LLM prose if enabled+available AND it invents no number; otherwise the exact template."""
        if self.use_llm and llm.is_available():
            try:
                out = llm.explain(question, computed, context, inferred)
                facts = " ".join(x for x in (computed, inferred) if x)
                if out and not llm.faithfulness(facts, out)["invented"]:
                    return out, True
            except Exception:
                pass          # any LLM failure → fall back to the exact template
        return computed, False

    # ---- routing: each branch is a thin call to the deterministic truth layer ----
    def _route(self, intent, question, day, atype):
        where = f" on {DAY_NAMES[day]}" if day else ""
        if intent == "attacker":
            ips = qe.attacker_ips(self.df)
            return (f"The attack traffic comes from {', '.join(ips)}.", ips,
                    "In DAD all attacks originate from a single injected host.", None)

        if intent == "count":
            n = qe.count_attacks(self.df, day=day, attack_type=atype)
            scope = where + (f" ({atype})" if atype else "")
            return (f"There were {n} labelled attack packets{scope}.", n,
                    ATTACK_KB.get(atype), None)

        if intent == "breakdown":
            bd = qe.attack_breakdown(self.df)
            listed = ", ".join(f"{k}: {v}" for k, v in bd.items())
            return (f"Labelled attack packets by type — {listed}.", bd,
                    "; ".join(ATTACK_KB[k] for k in bd if k in ATTACK_KB) or None,
                    "'mixed' windows interleave all three types, so most attacks are only typeable "
                    "at day level.")

        if intent == "anomalies":
            total = qe.count_attacks(self.df)
            bd = qe.attack_breakdown(self.df)
            days = [DAY_NAMES[d] for d, c in qe.attacks_by_day(self.df).items() if c > 0]
            listed = ", ".join(f"{k} {v}" for k, v in bd.items())
            return (f"There are {total} labelled anomalies (network attacks) across {len(days)} days: "
                    f"{', '.join(days)}.", total,
                    f"By type — {listed}.",
                    "Anomalies here are attacks on the MQTT messages (duplication / interception / "
                    "modification), not just temperature spikes.")

        if intent == "sensors":
            ids = [int(x) for x in qe.sensor_ids(self.df)]
            return (f"The dataset has {len(ids)} sensors (MQTT client ids): {ids}.", ids,
                    "Client ids encode the cooling unit and probe — e.g. 131 = InRow 13, probe 1 "
                    "(4 InRow units × 4 probes: supply-air, return-air, entering-fluid, leaving-fluid).",
                    None)

        if intent == "temperature":
            stat = ip.resolve_stat(question)
            val = qe.temperature_stat(self.df, stat, day=day)
            word = {"max": "highest", "min": "lowest", "mean": "average"}[stat]
            return (f"The {word} reported temperature{where} was {val:.2f} °C.", val, None, None)

        if intent == "traffic":
            if "busiest" in question.lower():
                ts, n = qe.busiest_hour(self.df)
                return (f"The busiest hour was {ts:%Y-%m-%d %H:%M} UTC with {n} messages.", (ts, n),
                        None, None)
            total = qe.total_messages(self.df, day=day)
            return (f"There were {total:,} MQTT/network messages{where}.", total, None, None)

        if intent == "when":
            byday = qe.attacks_by_day(self.df)
            days = [DAY_NAMES[d] for d, c in byday.items() if c > 0]
            return (f"Labelled attacks occurred on: {', '.join(days)}.", days,
                    "Monday, Tuesday and Sunday carry no labelled attacks.", None)

        # summary
        n = qe.count_attacks(self.df, day=day)
        total = qe.total_messages(self.df, day=day)
        tmin = qe.temperature_stat(self.df, "min", day=day)
        tmax = qe.temperature_stat(self.df, "max", day=day)
        scope = DAY_NAMES[day] if day else "the whole week"
        return (f"{scope}: {total:,} messages, {n} labelled attack packets, "
                f"temperatures {tmin:.1f}–{tmax:.1f} °C.",
                {"messages": total, "attacks": n}, None, None)
