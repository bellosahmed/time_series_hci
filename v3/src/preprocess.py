"""Preprocessing for v3 — parse the real DAD pcaps into a labelled per-message table.

The DAD dataset ships 7 daily MQTT captures (`.pcap`) plus per-packet attack-label CSVs. We parse
each pcap with scapy (temperature lives in the MQTT PUBLISH payload as a plain float string, e.g.
``b"15.54"``) and join the authoritative label / client-id / msg-id from the matching
``*_labels.csv`` on ``frame.number`` — which equals the 1-based packet index in the capture.

The output ``MESSAGES`` frame (one row per packet, datetime-indexed) is the single source of truth
for the rest of v3: features → detection → evaluation, and query-engine → interface → LLM.

Deterministic and offline; no model, no network.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Targeted scapy imports (NOT `scapy.all`) so scapy does not noisily auto-load optional
# layers (smb/kerberos/tls) that need a newer cryptography build than is installed here.
logging.getLogger("scapy").setLevel(logging.CRITICAL)
from scapy.contrib.mqtt import MQTT, MQTTPublish  # noqa: E402,F401  (MQTT registers the layer)
from scapy.layers.inet import IP, TCP  # noqa: E402
from scapy.utils import PcapReader  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dad"
ATTACKER_IP = "10.6.56.41"

# capture_<YYMMDD>0000_<day>.pcap  +  capture_<YYMMDD>0000_labels.csv
DAY_STAMPS = {
    "mon": "1910210000", "tue": "1910220000", "wed": "1910230000", "thu": "1910240000",
    "fri": "1910250000", "sat": "1910260000", "sun": "1910270000",
}
# Coarse (day-level) attack type from the paper's schedule — the shipped labels are binary, so this
# is the finest attribution the dataset supports (Fri/Sat interleave all three types → "mixed").
DAY_ATTACK_TYPE = {
    "tue": "interception",   # packets removed → not labelled, but recorded for completeness
    "wed": "modification", "thu": "duplication", "fri": "mixed", "sat": "mixed",
}


def _parse_temperature(value) -> float:
    """MQTT PUBLISH payload → float °C, or NaN. Payloads are plain ASCII floats (b'15.54')."""
    if value is None:
        return np.nan
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    try:
        return float(str(value).strip().strip("'\""))
    except (ValueError, TypeError):
        return np.nan


def _decode(x):
    return x.decode(errors="ignore") if isinstance(x, bytes) else x


def parse_pcap(pcap_path) -> pd.DataFrame:
    """One row per packet: frame (1-based), ts (epoch), src_ip, src_port, topic, temperature."""
    rows = []
    with PcapReader(str(pcap_path)) as reader:
        for frame, pkt in enumerate(reader, start=1):
            rec = {"frame": frame, "ts": float(pkt.time), "src_ip": None, "src_port": None,
                   "topic": None, "temperature": np.nan}
            if pkt.haslayer(IP):
                rec["src_ip"] = pkt[IP].src
            if pkt.haslayer(TCP):
                rec["src_port"] = int(pkt[TCP].sport)
            if pkt.haslayer(MQTTPublish):                 # temperature is only in PUBLISH payloads
                p = pkt[MQTTPublish]
                rec["topic"] = _decode(p.topic)
                rec["temperature"] = _parse_temperature(p.value)
            rows.append(rec)
    return pd.DataFrame(rows)


def _labels(labels_csv) -> pd.DataFrame:
    lab = pd.read_csv(labels_csv, sep=";")
    return lab.rename(columns={"frame.number": "frame", "mqtt.clientid": "clientid",
                               "mqtt.msgid": "msgid"})[["frame", "clientid", "msgid", "label"]]


def load_day(day: str) -> pd.DataFrame:
    """Parse one day's pcap and join its labels on frame.number → labelled message table."""
    stamp = DAY_STAMPS[day]
    pcap = DATA_DIR / f"capture_{stamp}_{day}.pcap"
    csv = DATA_DIR / f"capture_{stamp}_labels.csv"
    df = parse_pcap(pcap).merge(_labels(csv), on="frame", how="left")
    df["label"] = df["label"].fillna(0).astype("int8")
    df["day"] = day
    return df


def load_week(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Concatenate the 7 days into one datetime-indexed MESSAGES frame with an attack_type column."""
    frames = [load_day(day) for day in DAY_STAMPS]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df["attack_type"] = np.where(
        df["label"] == 1, df["day"].map(DAY_ATTACK_TYPE).fillna("mixed"), "none"
    )
    return df.set_index("ts").sort_index()
