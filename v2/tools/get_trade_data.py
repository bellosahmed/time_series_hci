"""
Fetch a multivariate monthly international-trade time series from Eurostat.

Dataset: ei_eteu27_2020_m  (EU27 international trade, monthly, 2002-2026).
We keep it simple and trustworthy: raw trade VALUE in million EUR, not
seasonally adjusted (so real seasonality + the 2008 / 2020 shocks survive),
total goods, partner = WORLD. Each reporting country's monthly EXPORTS and
IMPORTS become one column ("channel"), analogous to a sensor in v1.

Output: code/data/eu_trade_monthly.csv  (index = month, columns = CC_EXP / CC_IMP).

Offline note: this is the ONLY step that needs the internet. Once the CSV is
saved, the rest of the pipeline runs fully locally.
"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_eteu27_2020_m"
# Fixed slice: total goods, partner = world, raw value (million EUR, not seasonally adjusted).
FIXED = "format=JSON&indic=ET-T&partner=WORLD&unit=MIO-EUR-NSA"
# Region/aggregate codes to drop (we want individual countries only).
AGGREGATES = {"EU", "EA"}

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "data" / "eu_trade_monthly.csv"


def fetch_flow(flow: str) -> pd.DataFrame:
    """Return a wide DataFrame (index=month, columns=country) for one flow (EXP/IMP)."""
    url = f"{BASE}?{FIXED}&stk_flow={flow}"
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)

    geo = d["dimension"]["geo"]["category"]["index"]
    time = d["dimension"]["time"]["category"]["index"]
    geo_inv = {v: k for k, v in geo.items()}
    time_inv = {v: k for k, v in time.items()}
    n_time = len(time)

    # JSON-stat flattens dims in d['id'] order; here only geo & time vary.
    # flat index = geo_pos * n_time + time_pos
    records: dict[str, dict[str, float]] = {}
    for flat, val in d["value"].items():
        flat = int(flat)
        geo_pos, time_pos = divmod(flat, n_time)
        cc = geo_inv[geo_pos]
        if len(cc) != 2 or cc in AGGREGATES:
            continue  # keep individual countries only
        records.setdefault(cc, {})[time_inv[time_pos]] = val

    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df.index, format="%Y-%m")
    df = df.sort_index()
    df.columns = [f"{c}_{flow}" for c in df.columns]
    return df


def main() -> None:
    print("Fetching Eurostat monthly trade (this needs the internet)...")
    exp = fetch_flow("EXP")
    imp = fetch_flow("IMP")
    df = exp.join(imp, how="outer").sort_index()
    df.index.name = "month"
    df = df.reindex(sorted(df.columns), axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)

    print(f"\nSaved: {OUT}")
    print(f"Shape: {df.shape[0]} months x {df.shape[1]} country-flow columns")
    print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Countries: {sorted({c.split('_')[0] for c in df.columns})}")
    miss = df.isna().mean().mul(100).round(1)
    print(f"Columns with most missing %:\n{miss.sort_values(ascending=False).head(6).to_string()}")


if __name__ == "__main__":
    main()
