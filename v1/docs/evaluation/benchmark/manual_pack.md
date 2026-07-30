# Manual benchmark pack — commercial LLMs as contestants

These tools can't be called from the project environment (no API keys), so run them by hand and record what they say. **Do this identically for each tool** so the comparison is fair.

## Procedure (do it the way a real user would)

1. Upload the dataset file `data/jena_climate_2009_2016.csv` to the tool (ChatGPT, Claude, Kimi and GLM all accept file uploads + can run code on them).
2. Paste this framing once:

   > This CSV has a `Date Time` column (format `dd.mm.yyyy HH:MM:SS`, 10-minute readings) and weather sensor columns. Some rows use `-9999` as a missing-value code — treat those as missing. Answer each question with just the number, rounded to 2 decimals.

3. Ask the questions below **verbatim, one at a time**, and write the number each tool returns into `answers_template.csv` (one column per tool).

## Questions

1. what was the highest temperature on 4 July 2015?
2. what was the lowest temperature on 4 July 2015?
3. what was the average temperature on 4 July 2015?
4. what was the maximum humidity on 2 October 2012?
5. what was the minimum pressure on 15 January 2014?
6. what was the highest wind speed on 10 February 2016?
7. what was the highest dew point on 20 August 2011?
8. what was the average humidity on 2 October 2012?

## Notes to watch for (these are the interesting failure modes)

- Does the tool silently average to hourly first (biasing max/min)? Ours computes on the native 10-minute grid.
- Does it keep the dataset's **duplicate timestamps** (183 in 2014 alone)? Ours drops them.
- Does it invent a number when the code errors, instead of saying it failed?

When the sheet is filled: `python3.12 docs/evaluation/benchmark/score_manual.py docs/evaluation/benchmark/answers_template.csv`
