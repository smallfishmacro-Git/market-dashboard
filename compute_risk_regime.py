#!/usr/bin/env python3
"""
Risk Regime raw-series builder for the Market Risk dashboard.
Fetches FRED inflation / real-yield / credit series, aligns them to business
days, writes a compact JSON. The growth x inflation regime is computed
client-side (so the lookback window stays interactive). Run after data_updater.py.
"""
import json, os
from datetime import datetime, timezone
import pandas as pd
from fredapi import Fred

FRED_KEY = os.environ.get("FRED_KEY", "5ccedb95e2418de2e5b7bae928c4e406")
START = "2017-01-01"
OUT = "data/datasets/risk_regime.json"
SERIES = {
    "be5":     "T5YIE",         # 5Y breakeven inflation
    "be10":    "T10YIE",        # 10Y breakeven inflation
    "real5":   "DFII5",         # 5Y real yield (TIPS)
    "real10":  "DFII10",        # 10Y real yield (TIPS)
    "fwd5y5y": "T5YIFR",        # 5Y5Y forward inflation
    "hy_oas":  "BAMLH0A0HYM2",  # ICE BofA US HY OAS
    "ig_oas":  "BAMLC0A0CM",    # ICE BofA US IG OAS
}

def main():
    fred = Fred(api_key=FRED_KEY)
    cols = {}
    for key, tick in SERIES.items():
        print(f"  Fetching {key} ({tick})...")
        cols[key] = fred.get_series(tick, START)
    df = pd.DataFrame(cols).sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("B").ffill()
    core = ["be5", "be10", "real5", "hy_oas", "ig_oas"]
    df = df[df[core].notna().all(axis=1)]
    df = df[df.index >= pd.Timestamp(START)]
    out = {"updated": datetime.now(timezone.utc).isoformat(),
           "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
    for key in SERIES:
        out[key] = [None if pd.isna(v) else round(float(v), 4) for v in df[key]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  risk_regime.json saved ({len(df)} rows, {os.path.getsize(OUT)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
