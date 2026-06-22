"""
Bilateral trade DEMO — EU trade *by partner country*.

This is a small illustrative dataset to understand country-to-country (bilateral) trade.
Dataset: Eurostat `ext_lt_maineu` (Extra-EU trade by partner), ANNUAL 2002-2025.

Important: the *reporter* here is the EU as a whole (EU27_2020), and the breakdown is by
PARTNER country. So a value answers: "how much did the EU export to / import from <country>
that year". True individual pairs (e.g. France->Belgium) need Comext (DS-018995), which the
public API doesn't serve.

Output: code/data/eu_trade_by_partner.csv (index = year, columns = <PARTNER>_EXP / _IMP, EUR m).
"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path
import pandas as pd

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ext_lt_maineu"
PARTNERS = {  # code -> friendly name
    "US": "USA", "CN_X_HK": "China", "UK": "UK", "RU": "Russia",
    "CH": "Switzerland", "JP": "Japan", "TR": "Turkiye", "IN": "India",
    "NO": "Norway", "KR": "SouthKorea", "BR": "Brazil",
}
INDIC = {"MIO_EXP_VAL": "EXP", "MIO_IMP_VAL": "IMP"}

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "eu_trade_by_partner.csv"


def fetch(indic: str) -> pd.DataFrame:
    url = f"{BASE}?format=JSON&geo=EU27_2020&sitc06=TOTAL&indic_et={indic}"
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    part = d["dimension"]["partner"]["category"]["index"]
    time = d["dimension"]["time"]["category"]["index"]
    part_inv = {v: k for k, v in part.items()}
    time_inv = {v: k for k, v in time.items()}
    n_time = len(time)
    rec: dict[str, dict[str, float]] = {}
    for flat, val in d["value"].items():
        flat = int(flat)
        p_pos, t_pos = divmod(flat, n_time)
        code = part_inv[p_pos]
        if code not in PARTNERS:
            continue
        rec.setdefault(PARTNERS[code], {})[time_inv[t_pos]] = val
    df = pd.DataFrame(rec)
    df.index = pd.to_datetime(df.index, format="%Y")
    df = df.sort_index()
    df.columns = [f"{c}_{INDIC[indic]}" for c in df.columns]
    return df


def main() -> None:
    print("Fetching EU trade by partner (annual)...")
    df = fetch("MIO_EXP_VAL").join(fetch("MIO_IMP_VAL"), how="outer").sort_index()
    df.index.name = "year"
    df = df.reindex(sorted(df.columns), axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print(f"Saved: {OUT}")
    print(f"Shape: {df.shape[0]} years x {df.shape[1]} partner-flow columns")
    print(f"Range: {df.index.min().year} -> {df.index.max().year}\n")

    # Illustrate real, explainable bilateral events
    yrs = [2013, 2019, 2021, 2022, 2023, 2024]
    show = df.loc[df.index.year.isin(yrs), ["China_EXP", "USA_EXP", "UK_EXP", "Russia_EXP", "Russia_IMP"]]
    show.index = show.index.year
    print("EU exports/imports to key partners (EUR millions):")
    print(show.round(0).to_string())
    rus = df["Russia_IMP"]
    print(f"\nEU imports from Russia: {rus.loc['2021'].iloc[0]:,.0f} (2021) -> {rus.loc['2024'].iloc[0]:,.0f} (2024)  "
          f"= {(rus.loc['2024'].iloc[0]/rus.loc['2021'].iloc[0]-1)*100:+.0f}% after sanctions.")


if __name__ == "__main__":
    main()
