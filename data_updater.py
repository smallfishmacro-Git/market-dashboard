"""
data_updater.py
---------------
Updates all 49 Barchart CSVs + CNN Fear & Greed CSV.

Run from the market-dashboard folder:
    python data_updater.py

Or triggered from within the Streamlit dashboard via the Update Data button.
"""

import io
import os
import sys
import requests  # type: ignore[import-untyped]
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import unquote
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Fix Windows cp1252 console so emoji log lines don't crash ──────────────────
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if isinstance(sys.stderr, io.TextIOWrapper) and sys.stderr.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BARCHART = os.path.join(BASE_DIR, "data", "barchart")
DATASETS = os.path.join(BASE_DIR, "data", "datasets")

# ── Master list of all 50 symbols ───────────────────────────────────────────────
SYMBOLS = [
    ("$MMFD",     "Percent_of_Stocks_Above_5-Day_Average_$MMFD.csv"),
    ("$R3FD",     "Russell_3000_Stocks_Above_5-Day_Average_$R3FD.csv"),
    ("$MMTW",     "Percent_of_Stocks_Above_20-Day_Average_$MMTW.csv"),
    ("$MMFI",     "Percent_of_Stocks_Above_50-Day_Average_$MMFI.csv"),
    ("$MMOH",     "Percent_of_Stocks_Above_100-Day_Average_$MMOH.csv"),
    ("$MMTH",     "Percent_of_Stocks_Above_200-Day Average_$MMTH.csv"),
    ("$CPCS",     "Equity_PutCall_Ratio_$CPCS.csv"),
    ("$CPC",      "Total_PutCall_Ratio_$CPC.csv"),
    ("$CPCB",     "Bond_PutCall_Ratio_$CPCB.csv"),
    ("$NSHU",     "NYSE_Advancing_Stocks_$NSHU.csv"),
    ("$QSHU",     "NASD_Advancing_Stocks_$QSHU.csv"),
    ("$NSHD",     "NYSE_Declining_Stocks_$NSHD.csv"),
    ("$QSHD",     "NASD_Declining_Stocks_$QSHD.csv"),
    ("$NVLU",     "NYSE_Advancing_Volume_$NVLU.csv"),
    ("$DVCN",     "NYSE_Declining_Volume_$DVCN.csv"),
    ("$M1HN",     "1-Month_Highs_NYSE_$M1HN.csv"),
    ("$M1LN",     "1-Month_Lows_NYSE_$M1LN.csv"),
    ("$UNCN",     "NYSE_Unchanged_Stocks_$UNCN.csv"),
    ("$MAHN",     "52-Week_Highs_NYSE_$MAHN.csv"),
    ("$MAHP",     "S&P_500_52-Week_Highs_$MAHP.csv"),
    ("$NAHC",     "Nasdaq_100_52-Week_Highs_$NAHC.csv"),
    ("$MALN",     "52-Week_Lows_NYSE_$MALN.csv"),
    ("$MALP",     "S&P_500_52-Week_Lows_$MALP.csv"),
    ("$NALC",     "Nasdaq_100_52-Week_Lows_$NALC.csv"),
    ("$M3HN",     "3-Month_Highs_NYSE_$M3HN.csv"),
    ("$M3LN",     "3-Month_Lows_NYSE_$M3LN.csv"),
    ("$VXMT",     "CBOE_6-Month_VIX_$VXMT.csv"),
    ("$VXV",      "CBOE_3-Month_VIX_$VXV.csv"),
    ("$VXST",     "CBOE_9-Day_VIX_$VXST.csv"),
    ("$VIX",      "CBOE_Volatility_Index_$VIX.csv"),
    ("$VVIX",     "CBOE_Vvolatility_$VVIX.csv"),
    ("$CRBO",     "CRB_Fats_&_Oils_Sub_Index_$CRBO.csv"),
    ("$CRBF",     "CRB_Foodstuff_Sub_Index_$CRBF.csv"),
    ("$CRBL",     "CRB_Livestock_Sub_Index_$CRBL.csv"),
    ("$CRBM",     "CRB_Metals_Sub_Index_$CRBM.csv"),
    ("$CRBR",     "CRB_Raw_Industrials_Sub_Index_$CRBR.csv"),
    ("$CRBS",     "CRB_Spot_Index_$CRBS.csv"),
    ("$CRBT",     "CRB_Textiles_Sub_Index_$CRBT.csv"),
    ("$OVX",      "Crude_Oil_VIX_$OVX.csv"),
    ("$EVZ",      "Euro_FX_VIX_$EVZ.csv"),
    ("$GVZ",      "Gold_VIX_$GVZ.csv"),
    ("$MOVE",     "Move_Index_$MOVE.csv"),
    ("$VXN",      "Nasdaq_100_VIX_$VXN.csv"),
    ("$S5FD",     "S&P_500_Stocks_ Above_ 5-Day_Average_$S5FD.csv"),
    ("$S5TW",     "S&P_500_Stocks_Above_20-Day_Average_$S5TW.csv"),
    ("$S5FI",     "S&P_500_Stocks_Above_50-Day_Average_$S5FI.csv"),
    ("$S5OH",     "S&P_500_Stocks_Above_100-Day_Average_$S5OH.csv"),
    ("$S5TH",     "S&P_500_Stocks_Above_200-Day_Average_$S5TH.csv"),
    ("$SPX",      "S&P_500_Index_$SPX.csv"),
    ("ISMMFG.RT", "ISM_Manufacturing_Index.csv"),
]

# ── Core Barchart update function ───────────────────────────────────────────────
def update_symbol(symbol, filename, session, log_fn=print):
    file_path = os.path.join(BARCHART, filename)

    if not os.path.exists(file_path):
        log_fn(f"  ⚠️  File not found, skipping: {filename}")
        return False

    # Load existing CSV
    df = pd.read_csv(file_path, parse_dates=True, index_col=0, date_format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    if len(df) > 1:
        df = df.iloc[:-1]

    # Step 1: GET the page to obtain XSRF token
    get_url = f"https://www.barchart.com/stocks/quotes/{symbol}/price-history/"
    get_headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "max-age=0",
        "upgrade-insecure-requests": "1",
        "referer": f"https://www.barchart.com/stocks/quotes/{symbol}/price-history/historical",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36",
    }
    session.get(get_url, headers=get_headers, timeout=15)

    # Step 2: Call the API
    api_url = "https://www.barchart.com/proxies/core-api/v1/historical/get"
    api_headers = {
        "accept": "application/json",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
        "x-xsrf-token": unquote(unquote(session.cookies.get_dict().get("XSRF-TOKEN", ""))),
    }
    payload = {
        "symbol": symbol,
        "fields": "tradeTime.format(m/d/Y),openPrice,highPrice,lowPrice,lastPrice,priceChange,percentChange,volume,openInterest,impliedVolatility,symbolCode,symbolType",
        "type": "eod",
        "orderBy": "tradeTime",
        "orderDir": "desc",
        "limit": 65,
        "raw": "1",
    }
    response = session.get(api_url, params=payload, headers=api_headers, timeout=15)
    data = response.json()

    if "data" not in data or not data["data"]:
        log_fn(f"  ⚠️  No data returned for {symbol}")
        return False

    # Step 3: Process new data
    df1 = pd.DataFrame(data["data"])
    df1.set_index(df1.iloc[:, 0].name, inplace=True)
    df1.index.rename("Time", inplace=True)
    df1.index = pd.to_datetime(df1.index, errors="coerce")
    df1 = df1[df1.index.notna()].sort_index()
    df1 = df1.iloc[:, :7]
    df1.columns = df.columns

    for col in ["Open", "High", "Low", "Last"]:
        if col in df1.columns:
            if df1[col].dtype == object:
                df1[col] = df1[col].str.replace(",", "", regex=False).str.strip()
            df1[col] = pd.to_numeric(df1[col], errors="coerce")

    # Step 4: Append only new rows
    new_data = df1.loc[df.index[-1] + timedelta(days=1):]
    df = pd.concat([df, new_data])

    # Drop today if incomplete
    if df.index[-1].date() == datetime.today().date():
        df = df.iloc[:-1]

    # Step 5: Save
    df.to_csv(file_path)
    log_fn(f"  ✅  {symbol:15s} — {len(new_data)} new row(s) added  →  {filename}")
    return True


# ── CNN Fear & Greed update ─────────────────────────────────────────────────────
def update_cnn_fear_greed(log_fn=print):
    file_path = os.path.join(DATASETS, "cnn_fear_greed.csv")

    if not os.path.exists(file_path):
        log_fn("  ⚠️  cnn_fear_greed.csv not found, skipping.")
        return False

    cnn_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2020-09-06"
    cnn_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0"}

    response = requests.get(cnn_url, headers=cnn_headers, timeout=15, verify=False)
    response.raise_for_status()
    data = response.json()

    new_df = pd.DataFrame(data["fear_and_greed_historical"]["data"])
    new_df["x"] = pd.to_datetime(new_df["x"], unit="ms")
    new_df = new_df.rename(columns={"x": "Date", "y": "Fear_Greed"})
    new_df.set_index("Date", inplace=True)
    new_df = new_df[["Fear_Greed"]].resample("D").last()

    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()

    new_rows = new_df.loc[df.index[-1] + timedelta(days=1):]
    updated = pd.concat([df, new_rows])
    updated.to_csv(file_path)

    log_fn(f"  ✅  CNN Fear & Greed   — {len(new_rows)} new row(s) added  →  cnn_fear_greed.csv")
    return True


# ── Main runner ─────────────────────────────────────────────────────────────────
def run_update(log_fn=print):
    log_fn("=" * 60)
    log_fn(f"Update started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_fn("=" * 60)

    session = requests.Session()
    success = 0
    failed  = 0

    for symbol, filename in SYMBOLS:
        try:
            ok = update_symbol(symbol, filename, session, log_fn)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            log_fn(f"  ❌  {symbol} — ERROR: {e}")
            failed += 1

    # CNN Fear & Greed
    try:
        ok = update_cnn_fear_greed(log_fn)
        if ok:
            success += 1
        else:
            failed += 1
    except Exception as e:
        log_fn(f"  ❌  CNN Fear & Greed — ERROR: {e}")
        failed += 1

    log_fn("=" * 60)
    log_fn(f"Done. {success} updated, {failed} skipped/failed.")
    log_fn("=" * 60)
    return success, failed


# ── Run directly from command line ──────────────────────────────────────────────
if __name__ == "__main__":
    run_update()
