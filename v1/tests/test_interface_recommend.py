"""Tests for the "recommend" intent: directional ranking ("hot days in June") and analog
similarity search ("dates similar to 13 February 2013"), both routed through Interface.ask.
"""

import numpy as np
import pandas as pd

from src import interface


def _seasonal_df(years=(2012, 2013, 2014, 2015), seed=0):
    """Multi-year hourly data with a real seasonal cycle (summer hot, winter cold), so
    "hottest in June" and "December isn't genuinely hot" have unambiguous, real answers.
    """
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    rng = np.random.default_rng(seed)
    seasonal = 10 + 15 * np.sin((idx.dayofyear - 80) / 365 * 2 * np.pi)
    return pd.DataFrame(
        {"T (degC)": seasonal + rng.normal(0, 1.5, len(idx)),
         "p (mbar)": 1000 + rng.normal(0, 5, len(idx))},
        index=idx,
    )


def test_recommend_analog_mode_finds_similar_day_and_excludes_the_target_itself():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend other dates similar to the weather of 13 february 2013")
    assert ans.intent == "recommend"
    candidates = [c["when"] for c in ans.value["candidates"]]
    assert len(candidates) == 5
    assert pd.Timestamp("2013-02-13") not in candidates
    assert "similar weather" in ans.computed.lower()


def test_recommend_analog_mode_year_granularity():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend a similar year to 2013")
    assert ans.intent == "recommend"
    assert all(c["when"].year != 2013 for c in ans.value["candidates"])
    assert "2013" in ans.computed  # names the target year, not a truncated date


def test_recommend_analog_mode_without_any_scope_asks_for_one():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend some similar dates")
    assert ans.intent == "recommend"
    assert "specific day, month, or year" in ans.text


def test_recommend_directional_mode_hottest_days_in_named_month_and_year():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend the hottest days in July 2013")
    assert ans.intent == "recommend"
    dates = [c["date"] for c in ans.value["candidates"]]
    assert all(d.month == 7 and d.year == 2013 for d in dates)
    # ranked descending by temperature
    values = [c["value"] for c in ans.value["candidates"]]
    assert values == sorted(values, reverse=True)


def test_recommend_directional_mode_recurring_month_spans_all_years():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend hot days in june")   # no year -> every June across the record
    assert ans.intent == "recommend"
    dates = [c["date"] for c in ans.value["candidates"]]
    assert all(d.month == 6 for d in dates)
    assert len({d.year for d in dates}) >= 1
    assert "hottest" in ans.computed.lower()
    assert ans.value["meets_threshold"] is True   # June genuinely gets hot in this record


def test_recommend_directional_mode_gives_honest_caveat_when_scope_never_meets_bar():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend hot days in december")
    assert ans.intent == "recommend"
    assert ans.value["meets_threshold"] is False
    assert "isn't genuinely hot" in ans.computed
    assert "not a real hot recommendation" in ans.computed
    # still returns the closest candidates, not an empty/error answer
    assert len(ans.value["candidates"]) == 5


def test_recommend_directional_mode_coldest_uses_correct_superlative_spelling():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend the coldest days in january")
    assert ans.intent == "recommend"
    assert "coldest days" in ans.computed
    assert "coldest" in ans.computed and "coldst" not in ans.computed


def test_recommend_directional_mode_works_for_a_sensor_with_no_adjective():
    # "pressure" has no colloquial hot/cold-style word -- must fall back to the generic
    # highest/lowest resolver so all 14 sensors are reachable, not just T/rh/wv.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend the highest pressure days in 2013")
    assert ans.intent == "recommend"
    assert ans.sensors == ["p (mbar)"]
    dates = [c["date"] for c in ans.value["candidates"]]
    assert all(d.year == 2013 for d in dates)
    values = [c["value"] for c in ans.value["candidates"]]
    assert values == sorted(values, reverse=True)
    assert "highest p (mbar)" in ans.computed


def test_recommend_directional_mode_forward_looking_gives_recurring_pattern_not_a_fake_date():
    # this is a fixed historical record with no real "future" beyond its last row -- "when next
    # will it be hot?" must not pretend to point at a real future date; it should honestly say
    # so and answer with the recurring seasonal pattern instead.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("when next will it be really hot?")
    assert ans.intent == "recommend"
    assert "no real 'next'" in ans.computed
    assert "happen most around" in ans.computed
    assert "hottest on record" in ans.computed.lower()


def test_recommend_directional_mode_forward_looking_combines_with_honesty_caveat():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("how soon will conditions be hot in december?")
    assert "no real 'next'" in ans.computed
    assert "never gets genuinely hot" in ans.computed


def test_recommend_directional_mode_names_the_best_pick():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend the hottest days in july 2013")
    best = ans.value["candidates"][0]
    assert "Best pick" in ans.computed
    assert "go with that one" in ans.computed
    assert f"{best['value']:.1f}" in ans.computed


def test_recommend_analog_mode_names_a_closest_match_to_go_with():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend other dates similar to 13 february 2013")
    assert "Closest match" in ans.computed
    assert "day to go with" in ans.computed


def test_recommend_analog_mode_restricted_scope_stays_within_the_named_month():
    # the reported bug: naming the month twice ("6 April 2013 ... in the month of April 2013")
    # must restrict candidates to April 2013, not search the whole multi-year record.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend days that have similar weather condition to the 6 April 2013 "
                     "in the month of april 2013")
    assert ans.intent == "recommend"
    dates = [c["when"] for c in ans.value["candidates"]]
    assert dates                                     # April 2013 has other days to compare against
    assert all(d.year == 2013 and d.month == 4 for d in dates)
    assert "searched within April 2013 only" in ans.computed


def test_recommend_analog_mode_bare_date_still_searches_the_whole_record():
    # the un-restricted case must be unaffected by the restriction feature -- naming the date
    # once (no repeated month, no "within"/"staying" phrase) searches every year, as designed
    # and confirmed working before this fix.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend other dates similar to 13 february 2013")
    dates = [c["when"] for c in ans.value["candidates"]]
    assert "searched within" not in ans.computed
    assert len({d.year for d in dates}) > 1           # candidates span more than one year


def test_recommend_analog_mode_restriction_phrase_without_repeating_the_month():
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend days similar to 6 april 2013 staying within that month")
    dates = [c["when"] for c in ans.value["candidates"]]
    assert all(d.year == 2013 and d.month == 4 for d in dates)
    # "staying WITHIN" must not be misread as the sensor alias "thin" (-> air density) -- this
    # silently ranked on rho alone until resolve_sensors got word-boundary matching.
    assert ans.sensors == ["T (degC)", "p (mbar)"]


def test_recommend_analog_mode_restricted_to_a_different_month_than_the_target():
    # the reported bug: "5 September 2012 in November 2012" must restrict candidates to
    # November, not September (the target's own month) and not the whole record.
    iface = interface.Interface(_seasonal_df(years=(2011, 2012, 2013)))
    ans = iface.ask("recommend a day to go out in 5 september 2012 in november 2012")
    assert ans.intent == "recommend"
    dates = [c["when"] for c in ans.value["candidates"]]
    assert dates
    assert all(d.year == 2012 and d.month == 11 for d in dates)
    assert "searched within November 2012 only" in ans.computed


def test_recommend_analog_mode_restricted_to_a_different_month_and_year():
    iface = interface.Interface(_seasonal_df(years=(2011, 2012, 2013, 2014, 2015, 2016)))
    ans = iface.ask("recommend a day to go out in 5 september 2012 in june 2016")
    dates = [c["when"] for c in ans.value["candidates"]]
    assert dates
    assert all(d.year == 2016 and d.month == 6 for d in dates)
    assert "searched within June 2016 only" in ans.computed


def test_recommend_analog_mode_similar_to_marker_makes_the_second_month_the_target():
    # "a day to go out in January 2010, similar to June 2010": June is the TARGET (the weather
    # wanted), January is the RESTRICTION (where to search) -- the reverse of what plain dict-
    # order period parsing would give (January sorts before June). Neither month has a day
    # number, so `resolve_day` can't disambiguate -- only the "similar to" marker can.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend a day to go out in january 2013 similar to june 2013")
    assert ans.intent == "recommend"
    assert "similar weather to June 2013" in ans.computed
    assert "searched within January 2013 only" in ans.computed
    dates = [c["when"] for c in ans.value["candidates"]]
    assert dates
    assert all(d.year == 2013 and d.month == 1 for d in dates)   # candidates are January days


def test_recommend_analog_mode_honest_caveat_when_restriction_never_gets_close_to_target():
    # January genuinely never gets close to June's heat -- must say so, with real numbers, not
    # just silently present January's coldest-available day as if it were a good match.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend a day to go out in january 2013 similar to june 2013")
    assert "Honest caveat" in ans.computed
    assert "don't count on January 2013" in ans.computed
    best = ans.value["candidates"][0]
    target_value = ans.value["target"]["T (degC)"]
    assert f"{target_value:.1f}" in ans.computed
    assert f"{best['T (degC)']:.1f}" in ans.computed


def test_recommend_analog_mode_no_caveat_when_restriction_is_a_genuinely_close_match():
    # adjacent winter months (January vs February) ARE close -- no honest caveat should fire.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend a day to go out in january 2013 similar to february 2013")
    assert "Honest caveat" not in ans.computed


def _mismatched_second_sensor_df(years=(2012, 2013, 2014, 2015), seed=0):
    # T (degC), the FIRST sensor in the default set, has no seasonal signal at all -- it's a
    # genuinely close match everywhere. p (mbar), checked second, carries the whole seasonal
    # mismatch instead -- so a check that only looked at the first sensor would wrongly report
    # no caveat at all.
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    rng = np.random.default_rng(seed)
    seasonal_p = 1000 + 15 * np.sin((idx.dayofyear - 80) / 365 * 2 * np.pi)
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 1.5, len(idx)), "p (mbar)": seasonal_p + rng.normal(0, 1.5, len(idx))},
        index=idx,
    )


def test_recommend_honesty_caveat_catches_a_mismatch_in_a_sensor_other_than_the_first():
    iface = interface.Interface(_mismatched_second_sensor_df())
    ans = iface.ask("recommend a day to go out in january 2013 similar to june 2013")
    assert "Honest caveat" in ans.computed
    assert "p (mbar)" in ans.computed        # the sensor that's actually mismatched
    # T never contributes a mismatch line -- confirm it wasn't (wrongly) flagged instead
    assert "T (degC) — " not in ans.computed


def test_recommend_forward_looking_restricts_to_near_term_window_by_default():
    # "when next do we get similar conditions" means SOON -- candidates must be strictly after
    # the target date and within the default 3-month window, not scattered across other years.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("I liked the weather on 13 february 2013, when next do we get similar conditions?")
    assert ans.intent == "recommend"
    target = pd.Timestamp("2013-02-13")
    dates = [c["when"] for c in ans.value["candidates"]]
    assert dates
    assert all(target < d <= target + pd.DateOffset(months=3) for d in dates)
    assert "The next time conditions look like 13 February 2013" in ans.computed


def test_recommend_forward_looking_with_explicit_restriction_stays_forward_and_in_scope():
    # naming an explicit month ("in November") alongside "when next" must respect BOTH: only
    # November candidates, AND only ones after the target date.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("when next in november 2013 will conditions be similar to 5 september 2013?")
    target = pd.Timestamp("2013-09-05")
    dates = [c["when"] for c in ans.value["candidates"]]
    assert dates
    assert all(d.year == 2013 and d.month == 11 and d > target for d in dates)


def test_recommend_non_forward_looking_question_is_unaffected():
    # plain "similar to" (no "when next"/"how soon"/...) must keep searching the whole record,
    # not silently start restricting to a 3-month forward window.
    iface = interface.Interface(_seasonal_df())
    ans = iface.ask("recommend other dates similar to 13 february 2013")
    dates = [c["when"] for c in ans.value["candidates"]]
    assert len({d.year for d in dates}) > 1
    assert "The next time" not in ans.computed
