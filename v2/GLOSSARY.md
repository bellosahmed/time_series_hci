# v2 Dataset Glossary — EU Monthly Trade

Plain-English meaning of every column in `code/data/eu_trade_monthly.csv`.

## Naming convention

Each column is `CC_FLOW`:

- **`CC`** = 2-letter country code (Eurostat / ISO-3166).
- **`FLOW`** = direction of trade:
  - **`_EXP`** = **Exports** — goods that country *sold to* the rest of the world (money coming in).
  - **`_IMP`** = **Imports** — goods that country *bought from* the rest of the world (money going out).

**Example (your style):**
- `DE_EXP` = **Germany exports** (€ millions of goods Germany sold to the world that month)
- `DE_IMP` = **Germany imports** (€ millions of goods Germany bought from the world that month)

> Values are **trade value in millions of euro (€m)**, partner = whole world, total goods,
> not seasonally adjusted. One row = one month.

## Country codes in this dataset (27 EU members)

| Code | Country | Export column | Import column |
| --- | --- | --- | --- |
| AT | Austria | `AT_EXP` | `AT_IMP` |
| BE | Belgium | `BE_EXP` | `BE_IMP` |
| BG | Bulgaria | `BG_EXP` | `BG_IMP` |
| CY | Cyprus | `CY_EXP` | `CY_IMP` |
| CZ | Czechia (Czech Republic) | `CZ_EXP` | `CZ_IMP` |
| DE | Germany | `DE_EXP` | `DE_IMP` |
| DK | Denmark | `DK_EXP` | `DK_IMP` |
| EE | Estonia | `EE_EXP` | `EE_IMP` |
| EL | Greece | `EL_EXP` | `EL_IMP` |
| ES | Spain | `ES_EXP` | `ES_IMP` |
| FI | Finland | `FI_EXP` | `FI_IMP` |
| FR | France | `FR_EXP` | `FR_IMP` |
| HR | Croatia | `HR_EXP` | `HR_IMP` |
| HU | Hungary | `HU_EXP` | `HU_IMP` |
| IE | Ireland | `IE_EXP` | `IE_IMP` |
| IT | Italy | `IT_EXP` | `IT_IMP` |
| LT | Lithuania | `LT_EXP` | `LT_IMP` |
| LU | Luxembourg | `LU_EXP` | `LU_IMP` |
| LV | Latvia | `LV_EXP` | `LV_IMP` |
| MT | Malta | `MT_EXP` | `MT_IMP` |
| NL | Netherlands | `NL_EXP` | `NL_IMP` |
| PL | Poland | `PL_EXP` | `PL_IMP` |
| PT | Portugal | `PT_EXP` | `PT_IMP` |
| RO | Romania | `RO_EXP` | `RO_IMP` |
| SE | Sweden | `SE_EXP` | `SE_IMP` |
| SI | Slovenia | `SI_EXP` | `SI_IMP` |
| SK | Slovakia | `SK_EXP` | `SK_IMP` |

That's **27 countries × 2 flows = 54 columns.**

## Codes you might expect but are NOT here

Eurostat lists these as reporters, but they returned no data for this exact slice
(world partner, total goods, monthly value), so they're not in the CSV:

- **`UK` / `GB`** = United Kingdom (your "GB_exp = England exports" example) — left the EU; not in this series.
- **`US`** = United States, **`JP`** = Japan, **`CH`** = Switzerland, **`NO`** = Norway, **`TR`** = Türkiye.

If we want any of these later, we'd pull them from a different Eurostat partner dataset or another source.

## Handy derived quantities (not stored, computed on demand)

- **Trade balance** = `CC_EXP − CC_IMP` (positive = surplus / sells more than it buys; negative = deficit).
- **Total trade** = `CC_EXP + CC_IMP` (overall trade activity).

## Quick reference

- Units: **€ millions** (e.g. `DE_EXP = 50330` → Germany exported €50.33 **billion** that month).
- Frequency: **monthly**, Jan 2002 → Apr 2026 (292 months).
- Missing values: **none**.
- Source: Eurostat `ei_eteu27_2020_m` via API → saved locally (see `tools/get_trade_data.py`).
