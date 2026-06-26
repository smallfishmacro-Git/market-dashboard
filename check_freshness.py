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

BARCHART = os.path.join(BASE_DIR, "data", "barchart")

# Raw Barchart CSVs that feed live indicators and must stay current. Checked at
# the per-symbol level because the JSON checks above can pass on a single stale
# symbol when the aggregate date is carried by other inputs (exactly how the
# Cumulative A/D Line froze unnoticed for months).
CRITICAL_BARCHART = [
    "NYSE_Advancing_Stocks_$NSHU.csv",
    "NYSE_Declining_Stocks_$NSHD.csv",
    "NYSE_Advancing_Volume_$NVLU.csv",
    "NYSE_Declining_Volume_$DVCN.csv",
    "NASD_Advancing_Stocks_$QSHU.csv",
    "NASD_Declining_Stocks_$QSHD.csv",
    "S&P_500_Index_$SPX.csv",
    "CBOE_Volatility_Index_$VIX.csv",
]
# Allowed lag, in completed NYSE sessions, before a symbol is flagged. Default
# tolerates the ~1-day settle lag on internals plus weekend/holiday slack.
BARCHART_STALE_TOLERANCE = int(os.getenv("BARCHART_STALE_TOLERANCE", "3"))
# Default: a stale Barchart symbol is a LOUD WARNING only — a single dead symbol
# must NOT block committing everything else. Set BARCHART_STALE_FAIL=1 to escalate
# to a red Action.
BARCHART_STALE_FAIL = os.getenv("BARCHART_STALE_FAIL", "0") == "1"


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


def check_barchart_freshness(now_et_dt) -> list:
    """Warn if any CRITICAL_BARCHART CSV's last populated Last is too old. Returns
    the list of stale filenames; never raises."""
    cal = xcals.get_calendar("XNYS")
    expected = expected_last_session(now_et_dt)
    window = cal.sessions_in_range(
        pd.Timestamp(expected) - pd.Timedelta(days=20), pd.Timestamp(expected)
    )
    if len(window) == 0:
        return []
    oldest_ok = (window[-(BARCHART_STALE_TOLERANCE + 1)]
                 if len(window) > BARCHART_STALE_TOLERANCE else window[0])
    stale = []
    for fn in CRITICAL_BARCHART:
        path = os.path.join(BARCHART, fn)
        if not os.path.exists(path):
            print(f"  ⚠️  barchart: {fn} not found"); stale.append(fn); continue
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        last = pd.to_numeric(
            df["Last"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ).dropna()
        if last.empty:
            print(f"  ⚠️  barchart: {fn} has no valid Last"); stale.append(fn); continue
        last_date = last.index.max().normalize()
        if last_date < oldest_ok:
            print(f"  ⚠️  barchart STALE: {fn} last valid Last {last_date.date()} "
                  f"(< {oldest_ok.date()}, expected ~{expected})")
            stale.append(fn)
    if not stale:
        print(f"check_barchart_freshness: OK — all critical symbols within "
              f"{BARCHART_STALE_TOLERANCE} sessions of {expected}")
    return stale


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

    stale_bc = check_barchart_freshness(now)
    if stale_bc:
        print(f"check_barchart_freshness: ⚠️  {len(stale_bc)} stale Barchart symbol(s): "
              f"{', '.join(stale_bc)}")
        if BARCHART_STALE_FAIL:
            failures.extend(stale_bc)

    if failures:
        print(f"check_freshness: STALE — {len(failures)} file(s) behind expected session "
              f"{expected}: {', '.join(failures)}")
        return 1

    print(f"check_freshness: OK — all datasets current through {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
