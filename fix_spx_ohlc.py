"""
fix_spx_ohlc.py - Repairs SPX CSV: fixes empty OHLC rows AND adds missing trading days.
Uses yfinance (^GSPC) as fallback. Runs after data_updater.py in GitHub Actions.
"""
import os, sys, io, pandas as pd, yfinance as yf
from datetime import datetime, timedelta

if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() not in ("utf-8","utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPX_PATH = os.path.join(BASE_DIR, "data", "barchart", "S&P_500_Index_$SPX.csv")

def fix_spx_ohlc(log_fn=print):
    if not os.path.exists(SPX_PATH):
        log_fn(f"  Warning: SPX CSV not found at {SPX_PATH}"); return False
    df = pd.read_csv(SPX_PATH, parse_dates=True, index_col=0, date_format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    ohlc_cols = ["Open","High","Low","Last"]
    for col in ohlc_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",","").str.strip().replace("",pd.NA), errors="coerce")

    # --- PART 1: Fix rows with empty OHLC ---
    mask = df[ohlc_cols].isna().all(axis=1)
    broken_dates = df.index[mask]
    repaired = 0

    if not broken_dates.empty:
        log_fn(f"  Found {len(broken_dates)} rows with missing OHLC")
        start = (broken_dates[0] - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (broken_dates[-1] + timedelta(days=2)).strftime("%Y-%m-%d")
        yf_df = yf.download("^GSPC", start=start, end=end, progress=False, timeout=30)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df.index = pd.to_datetime(yf_df.index)
            for date in broken_dates:
                if date in yf_df.index:
                    row = yf_df.loc[date]
                    df.at[date,"Open"] = round(float(row["Open"]),2)
                    df.at[date,"High"] = round(float(row["High"]),2)
                    df.at[date,"Low"] = round(float(row["Low"]),2)
                    df.at[date,"Last"] = round(float(row["Close"]),2)
                    prev_idx = df.index.get_loc(date)
                    if prev_idx > 0:
                        prev_close = df.iloc[prev_idx-1]["Last"]
                        if pd.notna(prev_close) and prev_close != 0:
                            change = round(float(row["Close"]) - prev_close, 2)
                            df.at[date,"Change"] = change
                            df.at[date,"Change%"] = f"{round(change/prev_close*100,2)}%"
                    vol = df.at[date,"Volume"] if "Volume" in df.columns else 0
                    if pd.isna(vol) or vol == 0:
                        df.at[date,"Volume"] = int(row.get("Volume",0))
                    repaired += 1
                    log_fn(f"    Fixed {date.strftime('%Y-%m-%d')}: O={df.at[date,'Open']:.2f} H={df.at[date,'High']:.2f} L={df.at[date,'Low']:.2f} C={df.at[date,'Last']:.2f}")
                else:
                    df = df.drop(date)
                    log_fn(f"    {date.strftime('%Y-%m-%d')}: not a trading day - removed")
                    repaired += 1

    # --- PART 2: Add missing recent trading days ---
    log_fn("  Checking for missing recent trading days...")
    fetch_start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
    fetch_end = datetime.now().strftime("%Y-%m-%d")
    yf_recent = yf.download("^GSPC", start=fetch_start, end=fetch_end, progress=False, timeout=30)
    added = 0
    if not yf_recent.empty:
        if isinstance(yf_recent.columns, pd.MultiIndex):
            yf_recent.columns = yf_recent.columns.get_level_values(0)
        yf_recent.index = pd.to_datetime(yf_recent.index)
        today = pd.Timestamp(datetime.now().date())
        for date in yf_recent.index:
            if date >= today:
                continue
            if date not in df.index:
                row = yf_recent.loc[date]
                close_val = round(float(row["Close"]),2)
                new_row = {col: 0 for col in df.columns}
                new_row["Open"] = round(float(row["Open"]),2)
                new_row["High"] = round(float(row["High"]),2)
                new_row["Low"] = round(float(row["Low"]),2)
                new_row["Last"] = close_val
                new_row["Volume"] = int(row.get("Volume",0))
                prev_dates = df.index[df.index < date]
                if len(prev_dates) > 0:
                    prev_close = df.at[prev_dates[-1], "Last"]
                    if pd.notna(prev_close) and prev_close != 0:
                        change = round(close_val - prev_close, 2)
                        new_row["Change"] = change
                        new_row["Change%"] = f"{round(change/prev_close*100,2)}%"
                df.loc[date] = new_row
                added += 1
                log_fn(f"    Added {date.strftime('%Y-%m-%d')}: O={new_row['Open']:.2f} H={new_row['High']:.2f} L={new_row['Low']:.2f} C={new_row['Last']:.2f}")
    df = df.sort_index()
    df.to_csv(SPX_PATH)
    log_fn(f"  SPX OHLC done - repaired {repaired}, added {added} missing days.")
    return True

if __name__ == "__main__":
    print(f"[fix_spx_ohlc] Starting at {datetime.now().isoformat()}")
    fix_spx_ohlc()
    print("[fix_spx_ohlc] Done.")
