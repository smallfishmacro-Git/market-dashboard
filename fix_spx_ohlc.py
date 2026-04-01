"""
fix_spx_ohlc.py
---------------
Repairs S&P_500_Index_$SPX.csv rows where Barchart returned empty OHLC fields.
Uses yfinance (^GSPC) as the authoritative fallback for price data.

Usage:
    python fix_spx_ohlc.py          # one-time repair + ongoing fallback
    Called automatically by GitHub Actions after data_updater.py runs.
"""

import os
import sys
import io
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Fix Windows console encoding ───────────────────────────────────────────────
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPX_PATH = os.path.join(BASE_DIR, "data", "barchart", "S&P_500_Index_$SPX.csv")


def fix_spx_ohlc(log_fn=print):
    """
    Scan $SPX CSV for rows with missing OHLC data.
    Fill them from yfinance ^GSPC.
    Also recalculate Change / Change% if missing.
    """
    if not os.path.exists(SPX_PATH):
        log_fn(f"  ⚠️ SPX CSV not found at {SPX_PATH}")
        return False

    # Load existing CSV
    df = pd.read_csv(SPX_PATH, parse_dates=True, index_col=0, date_format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()

    # Coerce OHLC columns to numeric (handles commas, whitespace, empty strings)
    ohlc_cols = ["Open", "High", "Low", "Last"]
    for col in ohlc_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip().replace("", pd.NA),
                errors="coerce"
            )

    # Find rows where ALL four OHLC values are missing
    mask = df[ohlc_cols].isna().all(axis=1)
    broken_dates = df.index[mask]

    if broken_dates.empty:
        log_fn("  ✅ SPX OHLC — no missing rows found, all good.")
        return True

    log_fn(f"  🔧 SPX OHLC — found {len(broken_dates)} rows with missing prices: "
           f"{broken_dates[0].strftime('%Y-%m-%d')} to {broken_dates[-1].strftime('%Y-%m-%d')}")

    # Fetch yfinance data covering the broken range (with buffer)
    start = (broken_dates[0] - timedelta(days=5)).strftime("%Y-%m-%d")
    end = (broken_dates[-1] + timedelta(days=2)).strftime("%Y-%m-%d")

    log_fn(f"  📡 Fetching ^GSPC from yfinance ({start} to {end})...")
    yf_df = yf.download("^GSPC", start=start, end=end, progress=False, timeout=30)

    if yf_df.empty:
        log_fn("  ❌ yfinance returned no data for ^GSPC — cannot repair.")
        return False

    # Flatten multi-level columns if present (yfinance sometimes returns MultiIndex)
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = yf_df.columns.get_level_values(0)

    yf_df.index = pd.to_datetime(yf_df.index)

    repaired = 0
    for date in broken_dates:
        if date in yf_df.index:
            row = yf_df.loc[date]
            df.at[date, "Open"]  = round(float(row["Open"]), 2)
            df.at[date, "High"]  = round(float(row["High"]), 2)
            df.at[date, "Low"]   = round(float(row["Low"]), 2)
            df.at[date, "Last"]  = round(float(row["Close"]), 2)

            # Recalculate Change and Change% from previous row
            prev_idx = df.index.get_loc(date)
            if prev_idx > 0:
                prev_close = df.iloc[prev_idx - 1]["Last"]
                if pd.notna(prev_close) and prev_close != 0:
                    change = round(float(row["Close"]) - prev_close, 2)
                    pct = round(change / prev_close * 100, 2)
                    df.at[date, "Change"]   = change
                    df.at[date, "Change%"]  = f"{pct}%"

            # Fill volume from yfinance if our volume is 0 or missing
            vol = df.at[date, "Volume"] if "Volume" in df.columns else 0
            if pd.isna(vol) or vol == 0:
                df.at[date, "Volume"] = int(row.get("Volume", 0))

            repaired += 1
            log_fn(f"    ✓ {date.strftime('%Y-%m-%d')}: "
                   f"O={df.at[date, 'Open']:.2f}  H={df.at[date, 'High']:.2f}  "
                   f"L={df.at[date, 'Low']:.2f}  C={df.at[date, 'Last']:.2f}")
        else:
            # Date not in yfinance = likely not a trading day (weekend/holiday)
            # Remove the row entirely — it shouldn't be in the CSV
            df = df.drop(date)
            log_fn(f"    ✗ {date.strftime('%Y-%m-%d')}: not a trading day — removed row")
            repaired += 1

    # Save repaired CSV
    df.to_csv(SPX_PATH)
    log_fn(f"  ✅ SPX OHLC — repaired {repaired} rows, saved to CSV.")
    return True


if __name__ == "__main__":
    print(f"[fix_spx_ohlc] Starting at {datetime.now().isoformat()}")
    fix_spx_ohlc()
    print("[fix_spx_ohlc] Done.")
