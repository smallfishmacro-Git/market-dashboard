#!/usr/bin/env python3
"""
check_freshness.py
------------------
Freshness gate for the daily data pipeline. Runs as the LAST step before the
commit step in update_data.yml.

Using the NYSE calendar (XNYS via exchange_calendars), it computes the most
recent trading session that has fully completed (close + 15-min settle) as of
*now* on the New-York clock, then asserts that the freshly-written JSONs carry
that same session as their most recent date:

    - data/datasets/market_risk.json     -> max(thm.dates)
    - data/datasets/btd_indicators.json  -> max(composite.dates)

On mismatch it prints expected vs actual and exits non-zero, turning the Action
red so stale data (the BD-2 bug) is never silently published.

No scheduled run time is special-cased — the expectation is derived purely from
the calendar and the current clock, so each run self-adjusts:
    * post-close run  -> today's session has settled  -> expect today
    * mid-session run -> today hasn't closed          -> expect previous session
    * pre-market run  -> ditto                         -> expect previous session
"""

import json
import os
import sys
from datetime import timezone

import pandas as pd
import exchange_calendars as xcals

from market_time import now_et, SETTLE_BUFFER_MINUTES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(BASE_DIR, "data", "datasets")

MARKET_RISK_JSON = os.path.join(DATASETS, "market_risk.json")
BTD_JSON = os.path.join(DATASETS, "btd_indicators.json")


def expected_last_session(now_et_dt) -> str:
    """Most recent NYSE session whose close + settle buffer is <= now (ET)."""
    cal = xcals.get_calendar("XNYS")
    now_utc = pd.Timestamp(now_et_dt.astimezone(timezone.utc))
    buffer = pd.Timedelta(minutes=SETTLE_BUFFER_MINUTES)

    # A two-week window comfortably spans any holiday cluster back to a session.
    start = pd.Timestamp(now_et_dt.date()) - pd.Timedelta(days=14)
    end = pd.Timestamp(now_et_dt.date())
    sessions = cal.sessions_in_range(start, end)

    completed = [s for s in sessions if cal.session_close(s) + buffer <= now_utc]
    if not completed:
        raise RuntimeError("No completed NYSE session found in the last 14 days")
    return completed[-1].strftime("%Y-%m-%d")


def _max_date(path: str, *keys: str) -> str:
    """Load a JSON file and return max() of the date list at data[k1][k2]..."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    node = data
    for k in keys:
        node = node[k]
    if not node:
        raise ValueError(f"{os.path.basename(path)}: empty date list at {'/'.join(keys)}")
    return max(node)


def main() -> int:
    now = now_et()
    expected = expected_last_session(now)
    print(f"check_freshness: now={now.isoformat()}  expected last NYSE session={expected}")

    checks = [
        ("market_risk.json", MARKET_RISK_JSON, ("thm", "dates")),
        ("btd_indicators.json", BTD_JSON, ("composite", "dates")),
    ]

    failures = []
    for label, path, keys in checks:
        if not os.path.exists(path):
            print(f"  ❌  {label}: file not found at {path}")
            failures.append(label)
            continue
        try:
            actual = _max_date(path, *keys)
        except Exception as e:
            print(f"  ❌  {label}: could not read date — {e}")
            failures.append(label)
            continue
        ok = actual == expected
        mark = "✅" if ok else "❌"
        print(f"  {mark}  {label}: expected {expected}, actual {actual}")
        if not ok:
            failures.append(label)

    if failures:
        print(f"check_freshness: STALE — {len(failures)} file(s) behind expected session "
              f"{expected}: {', '.join(failures)}")
        return 1

    print(f"check_freshness: OK — all datasets current through {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
