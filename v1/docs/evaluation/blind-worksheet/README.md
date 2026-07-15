# Blind-labelling worksheet — instructions for the second rater

This is the real gold-standard mitigation for self-authored ground truth (see
`docs/evaluation/06-query-routing-coverage.md`, §3). `worksheet.csv` has 72 real questions from
this project's query-routing benchmark, with **no answers shown** — please fill in your own
independent judgement of each one, without discussing it with the person who built the system
first.

## What to fill in, per row

- **`your_intent`** — which ONE of the following best describes what the question is asking for,
  or leave blank if none really fits:
  `max, min, mean, spread, correlation, missing, forecast, trend, compare, anomaly, weather,
  daytype, summary`
  (`spread` = "how much does it vary"; `missing` = "how much data is missing"; `weather` = a
  general snapshot of conditions; `daytype` = classifying what kind of day it was; `summary` = a
  general overview when no specific statistic is implied.)
- **`your_sensors`** — comma-separated column name(s) the question names or clearly implies, from:
  `p (mbar), T (degC), Tpot (K), Tdew (degC), rh (%), VPmax (mbar), VPact (mbar), VPdef (mbar),
  sh (g/kg), H2OC (mmol/mol), rho (g/m**3), wv (m/s), max. wv (m/s), wd (deg)`
  (leave blank if no specific sensor is named/implied — e.g. "what was the weather like on 3
  March?" doesn't name one).
- **`your_has_scope`** — `yes` if the question names a specific time window (a date, month, year,
  season, or a relative phrase like "last month"/"this week"), otherwise `no`.

## Once filled in

Save the file (keep the same 72 rows, same order, don't add/remove rows), then run:

```bash
python3.12 -c "
from src import query_coverage as qc
cases = qc.GROUND_TRUTH + qc.HELD_OUT_GROUND_TRUTH
result = qc.score_blind_labels(cases, 'docs/evaluation/blind-worksheet/worksheet.csv')
print(f\"intent agreement:  {result['intent_agreement_pct']:.1f}%\")
print(f\"sensors agreement: {result['sensors_agreement_pct']:.1f}%\")
print(f\"scope agreement:   {result['scope_agreement_pct']:.1f}%\")
print(f'{len(result[\"disagreements\"])} disagreements:')
for d in result['disagreements']:
    print('-', d['question'])
    print('   original:    ', d['original'])
    print('   second rater:', d['second_rater'])
"
```

This gives genuine human inter-rater agreement — the real substitute the LLM-judge cross-check in
`06-query-routing-coverage.md` §3 could only partially stand in for.
