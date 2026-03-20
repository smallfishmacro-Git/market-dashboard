"""
compute_market_risk_json.py
---------------------------
Reads the 3 pre-computed market risk CSVs and outputs a single JSON file
for the Vercel Market Risk app.

Run after data_updater.py in the GitHub Actions pipeline:
    python compute_market_risk_json.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(BASE_DIR, "data", "datasets")

CSV_LT  = os.path.join(DATASETS, "market_risk_lt_composite.csv")
CSV_THM = os.path.join(DATASETS, "market_risk_health_model.csv")
CSV_IND = os.path.join(DATASETS, "market_risk_indicators.csv")

# Indicator metadata: (display_name, csv_column, group, color)
INDICATORS = [
    ("OECD CLI Diffusion Index",       "OECD_CLI",       "lt", "#f9ca24"),
    ("Nasdaq 100 Cumulative Hi-Lo",    "N100_HiLo",      "lt", "#4ecdc4"),
    ("Credit Spreads (FRED)",          "Credit_Spreads",  "lt", "#ff6b6b"),
    ("Volatility Regime (MOVE/VIX)",   "Vol_Regime",      "thm", "#ff9f43"),
    ("NYSE 52-Week Hi-Lo Trend",       "HiLo_52W",       "thm", "#f9ca24"),
    ("Canary Model",                   "Canary",          "thm", "#4ecdc4"),
    ("% Stocks Above 200-Day SMA",     "Pct_200SMA",     "thm", "#ff9f43"),
    ("ACWI 200 SMA Breadth",           "ACWI_200",       "thm", "#4ecdc4"),
    ("Cumulative A/D Line",            "AD_Line",         "thm", "#4ecdc4"),
    ("In/Out Indicator",               "InOut",           "thm", "#ff9f43"),
    ("VIX Term Structure",             "VIX_TS",          "thm", "#f9ca24"),
    ("HMM Regime Indicator",           "HMM",             "thm", "#00ff88"),
    ("Quad 1 & 2",                     "Quad",            "thm", "#4ecdc4"),
    ("Bitcoin Liquidity Proxy",        "BTC",             "thm", "#f9ca24"),
    ("SuperTrend Long Term",           "ST_LT",           "thm", "#00ff88"),
    ("SuperTrend Medium Term",         "ST_MT",           "thm", "#00ff88"),
    ("SuperTrend Short Term",          "ST_ST",           "thm", "#00ff88"),
    ("VIX Term Structure x HMM",       "VIX_HMM",        "thm", "#ff6b6b"),
]


def downsample(dates, values_dict, max_points=8000):
    n = len(dates)
    if n <= max_points:
        return dates, values_dict
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    new_dates = [dates[i] for i in idx]
    new_vals = {}
    for k, v in values_dict.items():
        new_vals[k] = [v[i] if i < len(v) else None for i in idx]
    return new_dates, new_vals


def last_change_date(series):
    """Find the last date where the series changed value."""
    s = series.dropna()
    if len(s) < 2:
        return None
    changed = s[s.diff().abs() > 0]
    return changed.index[-1].strftime("%Y-%m-%d") if len(changed) else None


def main():
    print("Loading Market Risk CSVs...")

    for path in [CSV_LT, CSV_THM, CSV_IND]:
        if not os.path.exists(path):
            print(f"  ERROR: Missing {path}")
            return

    # Load CSVs
    df_lt = pd.read_csv(CSV_LT, index_col=0, parse_dates=True)
    df_lt.index = pd.to_datetime(df_lt.index, errors="coerce")
    df_lt = df_lt[df_lt.index.notna()].sort_index()

    df_thm = pd.read_csv(CSV_THM, index_col=0, parse_dates=True)
    df_thm.index = pd.to_datetime(df_thm.index, errors="coerce")
    df_thm = df_thm[df_thm.index.notna()].sort_index()

    df_ind = pd.read_csv(CSV_IND, index_col=0, parse_dates=True)
    df_ind.index = pd.to_datetime(df_ind.index, errors="coerce")
    df_ind = df_ind[df_ind.index.notna()].sort_index()

    # ── Trend Health Model (THM) ──────────────────────────────────────────
    print("Processing THM...")
    thm_start = "2005-01-01"
    thm = df_thm[df_thm.index >= thm_start].copy()
    thm_dates = [d.strftime("%Y-%m-%d") for d in thm.index]
    thm_spx = thm["S&P500"].ffill().tolist()
    thm_composite = thm["Trend_Composite"].ffill().tolist()
    thm_trend = thm["Trend"].fillna(0).astype(int).tolist()

    thm_dates, thm_sampled = downsample(thm_dates, {
        "spx": thm_spx, "composite": thm_composite, "trend": thm_trend
    })

    # ── Long Term Composite (LT) ─────────────────────────────────────────
    print("Processing LT...")
    lt_start = "1997-01-01"
    lt = df_lt[df_lt.index >= lt_start].copy()
    lt_dates = [d.strftime("%Y-%m-%d") for d in lt.index]
    lt_spx = lt["S&P500"].ffill().tolist()
    lt_composite = lt["Composite"].ffill().tolist()
    lt_trend = lt["Trend"].fillna(0).astype(int).tolist()

    lt_cols = [c for c in lt.columns if c not in ("S&P500", "Composite", "Trend")]
    lt_ind_data = {}
    for c in lt_cols:
        lt_ind_data[c] = lt[c].ffill().fillna(0).astype(int).tolist()

    lt_dates, lt_sampled = downsample(lt_dates, {
        "spx": lt_spx, "composite": lt_composite, "trend": lt_trend,
        **lt_ind_data
    })

    # ── Individual indicators ─────────────────────────────────────────────
    print("Processing individual indicators...")
    indicators = []
    for name, col, group, color in INDICATORS:
        if col not in df_ind.columns:
            indicators.append({
                "name": name, "col": col, "group": group, "color": color,
                "status": None, "lastChange": None, "lastUpdate": None,
            })
            continue

        s = df_ind[col].dropna()
        status = int(s.iloc[-1]) if len(s) else None
        lc = last_change_date(s)
        lu = s.index[-1].strftime("%Y-%m-%d") if len(s) else None

        # Build the indicator's trend series aligned with SPX
        spx_s = df_thm["S&P500"].dropna()
        ind_s = s.copy()
        # Align to SPX dates
        common_start = max(spx_s.index[0], ind_s.index[0])
        spx_slice = spx_s[spx_s.index >= common_start]
        ind_aligned = ind_s.reindex(spx_slice.index, method="ffill").fillna(0).astype(int)

        ind_dates = [d.strftime("%Y-%m-%d") for d in spx_slice.index]
        ind_spx = spx_slice.tolist()
        ind_trend = ind_aligned.tolist()

        ind_dates, ind_sampled = downsample(ind_dates, {
            "spx": ind_spx, "trend": ind_trend
        }, max_points=4000)

        indicators.append({
            "name": name, "col": col, "group": group, "color": color,
            "status": status, "lastChange": lc, "lastUpdate": lu,
            "dates": ind_dates, "spx": ind_sampled["spx"],
            "trend": ind_sampled["trend"],
        })

    # ── Metrics ───────────────────────────────────────────────────────────
    print("Computing metrics...")
    thm_score = float(df_thm["Trend_Composite"].dropna().iloc[-1])
    thm_is_bull = thm_score > 55
    thm_trend_s = df_thm["Trend"].dropna()
    thm_chg = thm_trend_s[thm_trend_s.diff().abs() > 0]
    thm_regime_date = thm_chg.index[-1].strftime("%Y-%m-%d") if len(thm_chg) else None

    lt_score = int(df_lt["Composite"].dropna().iloc[-1])
    lt_cols_count = len([c for c in df_lt.columns if c not in ("S&P500", "Composite", "Trend")])
    lt_is_bull = df_lt["Trend"].dropna().iloc[-1] == 1
    lt_trend_s = df_lt["Trend"].dropna()
    lt_chg = lt_trend_s[lt_trend_s.diff().abs() > 0]
    lt_regime_date = lt_chg.index[-1].strftime("%Y-%m-%d") if len(lt_chg) else None

    spx_s = df_thm["S&P500"].dropna()
    spx_price = float(spx_s.iloc[-1])
    spx_chg = float((spx_s.iloc[-1] / spx_s.iloc[-2] - 1) * 100) if len(spx_s) >= 2 else 0

    bullish_count = sum(1 for i in indicators if i["status"] == 1)
    total_count = sum(1 for i in indicators if i["status"] is not None)

    metrics = {
        "thmScore": round(thm_score, 1),
        "thmBull": thm_is_bull,
        "thmRegimeDate": thm_regime_date,
        "ltScore": lt_score,
        "ltTotal": lt_cols_count,
        "ltBull": lt_is_bull,
        "ltRegimeDate": lt_regime_date,
        "spxPrice": round(spx_price, 2),
        "spxChg": round(spx_chg, 2),
        "bullishCount": bullish_count,
        "totalCount": total_count,
    }

    # ── Assemble response ─────────────────────────────────────────────────
    response = {
        "thm": {
            "dates": thm_dates,
            "spx": thm_sampled["spx"],
            "composite": thm_sampled["composite"],
            "trend": thm_sampled["trend"],
        },
        "lt": {
            "dates": lt_dates,
            "spx": lt_sampled["spx"],
            "composite": lt_sampled["composite"],
            "trend": lt_sampled["trend"],
        },
        "indicators": indicators,
        "metrics": metrics,
        "computedAt": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_path = os.path.join(DATASETS, "market_risk.json")

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            v = convert(obj)
            if v is not obj:
                return v
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(response, f, separators=(",", ":"), cls=NpEncoder)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✅  market_risk.json saved ({size_mb:.1f} MB)")
    print(f"      THM Score: {metrics['thmScore']}% ({'BULL' if metrics['thmBull'] else 'BEAR'})")
    print(f"      LT Score: {metrics['ltScore']}/{metrics['ltTotal']} ({'BULL' if metrics['ltBull'] else 'BEAR'})")
    print(f"      Indicators: {metrics['bullishCount']}/{metrics['totalCount']} bullish")


if __name__ == "__main__":
    main()
