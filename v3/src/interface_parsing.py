"""Deterministic NL parsing for v3 — turn a plain-English question into (intent, day, attack type,
stat) with keyword/regex matching only. No model, no data — independently testable, and it keeps the
*routing* deterministic so the right numbers never depend on an LLM.
"""
from __future__ import annotations

import re

# Intents, checked in order so the specific ones win over the generic "attack"/"count".
INTENTS = {
    "attacker":   ["attacker", "which ip", "who is attacking", "malicious ip", "source of the attack"],
    "explain":    ["why", "explain", "what caused", "what happened", "reason for"],
    "forecast":   ["forecast", "predict", "next hour", "next day", "future temperature"],
    "day-type":   ["day-type", "day type", "kind of day", "kinds of day", "similar days",
                   "group the days", "cluster the days"],
    "anomalies":  ["anomaly", "anomalies", "unusual", "abnormal", "suspicious", "outlier", "outliers"],
    "breakdown":  ["type of attack", "types of attack", "kind of attack", "kinds of attack",
                   "what type", "what kind", "breakdown", "categor"],
    "temperature": ["temperature", "temp", "degrees", "hottest", "coldest", "warmest", "coolest"],
    "sensors":    ["sensor", "probe", "device", "client id", "clientid"],
    "traffic":    ["how many message", "messages", "traffic", "busiest", "packets", "volume",
                   "message rate", "how busy"],
    "when":       ["when", "what time", "which day"],
    "count":      ["how many attack", "number of attack", "count", "attack"],
    "summary":    ["summary", "summarise", "summarize", "overview", "tell me about", "report", "describe"],
}

DAY_WORDS = {
    "monday": "mon", "mon": "mon", "tuesday": "tue", "tue": "tue", "wednesday": "wed", "wed": "wed",
    "thursday": "thu", "thu": "thu", "friday": "fri", "fri": "fri", "saturday": "sat", "sat": "sat",
    "sunday": "sun", "sun": "sun",
}

ATTACK_WORDS = {
    "modification": "modification", "modify": "modification", "modif": "modification",
    "duplication": "duplication", "duplicate": "duplication", "replay": "duplication",
    "interception": "interception", "intercept": "interception",
}


def detect_intent(question: str) -> str | None:
    q = question.lower()
    for intent, keywords in INTENTS.items():
        # trailing `s?` so a keyword matches its plural too ("attack" → "attacks", "message"→"messages")
        if any(re.search(rf"\b{re.escape(kw)}s?\b", q) for kw in keywords):
            return intent
    return None


def resolve_day(question: str) -> str | None:
    q = question.lower()
    for word, day in DAY_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return day
    return None


def resolve_attack_type(question: str) -> str | None:
    q = question.lower()
    for word, kind in ATTACK_WORDS.items():
        if re.search(rf"\b{word}s?\b", q):        # tolerate plurals (modifications, duplicates)
            return kind
    return None


def resolve_stat(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ("highest", "hottest", "warmest", "max", "peak")):
        return "max"
    if any(w in q for w in ("lowest", "coldest", "coolest", "min")):
        return "min"
    return "mean"
