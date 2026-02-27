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


# ── BTD signal computation ───────────────────────────────────────────────────────
def compute_btd_signals(log_fn=print):
    """
    Replicates tab_buy_the_dip.build_composite() without Streamlit.
    Saves data/datasets/btd_signals.csv with all 9 signal columns + Composite.
    """
    import numpy as np

    def _load(filename):
        path = os.path.join(BARCHART, filename)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()].sort_index()
        df["Last"] = pd.to_numeric(df["Last"].astype(str).str.replace(",", ""), errors="coerce")
        return df

    def _compute_signal(df, condition):
        signal = pd.Series(0, index=df.index)
        last_signal_date = None
        signal_active_until = None
        signal_level = None
        for i in range(len(df)):
            spx_current = df["S&P500"].iloc[i]
            if condition(df, i):
                if last_signal_date is None or (df.index[i] - last_signal_date).days > 5:
                    signal.iloc[i] = 1
                    last_signal_date = df.index[i]
                    signal_active_until = df.index[i] + pd.Timedelta(days=10)
                    signal_level = spx_current
            if signal_active_until is not None and df.index[i] <= signal_active_until:
                signal.iloc[i] = 1
                if spx_current >= signal_level * 1.04:
                    signal_active_until = None
                    last_signal_date = None
        return signal

    log_fn("  Computing BTD signals...")
    try:
        spx   = _load("S&P_500_Index_$SPX.csv")
        r3fd  = _load("Russell_3000_Stocks_Above_5-Day_Average_$R3FD.csv")
        adv   = _load("NYSE_Advancing_Stocks_$NSHU.csv")
        dec   = _load("NYSE_Declining_Stocks_$NSHD.csv")
        adv_v = _load("NYSE_Advancing_Volume_$NVLU.csv")
        dec_v = _load("NYSE_Declining_Volume_$DVCN.csv")
        pc    = _load("Equity_PutCall_Ratio_$CPCS.csv")
        vix   = _load("CBOE_Volatility_Index_$VIX.csv")
        vxv   = _load("CBOE_3-Month_VIX_$VXV.csv")
        highs = _load("S&P_500_52-Week_Highs_$MAHP.csv")

        if any(x is None for x in [spx, r3fd, adv, dec, adv_v, dec_v, pc, vix, vxv, highs]):
            log_fn("  ⚠️  BTD signals: one or more barchart CSVs missing, skipping.")
            return False

        fg_path   = os.path.join(DATASETS, "cnn_fear_greed.csv")
        acwi_path = os.path.join(DATASETS, "acwi_oscillator.csv")

        if not os.path.exists(fg_path):
            log_fn("  ⚠️  BTD signals: cnn_fear_greed.csv missing, skipping.")
            return False
        if not os.path.exists(acwi_path):
            log_fn("  ⚠️  BTD signals: acwi_oscillator.csv missing, skipping.")
            return False

        fg = pd.read_csv(fg_path, parse_dates=True, index_col=0)
        fg.index = pd.to_datetime(fg.index, errors="coerce")
        fg = fg[fg.index.notna()].sort_index()

        acwi_df = pd.read_csv(acwi_path, parse_dates=True, index_col=0)
        acwi_df.index = pd.to_datetime(acwi_df.index, errors="coerce")
        acwi_df = acwi_df[acwi_df.index.notna()].sort_index()
        acwi = acwi_df["Percentage"]

        # Derived indicators
        rana   = (adv["Last"] - dec["Last"]) / (adv["Last"] + dec["Last"]) * 1000
        mco    = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
        pc_sma = pc["Last"].rolling(5).mean()
        pc_z   = (pc_sma.rolling(52).mean() - pc_sma) / pc_sma.rolling(52).std()
        zweig  = (adv["Last"] / (adv["Last"] + dec["Last"])).ewm(span=10, adjust=False).mean()
        total_s   = adv["Last"] + dec["Last"]
        total_v   = adv_v["Last"] + dec_v["Last"]
        dec_pct_s = (dec["Last"] / total_s) * 100
        dec_pct_v = (dec_v["Last"] / total_v) * 100
        lowry  = ((dec_pct_s >= 90).astype(int) +
                  ((dec_pct_s >= 80) & (dec_pct_s < 90)).astype(int) +
                  (dec_pct_v >= 90).astype(int) +
                  ((dec_pct_v >= 80) & (dec_pct_v < 90)).astype(int)).rolling(6).sum()
        vc = vxv["Last"] / vix["Last"] - 1

        series = {
            "Percent Above 5DMA":   r3fd["Last"],
            "ACWI Oscillator":      acwi,
            "McClellan Oscillator": mco,
            "Equity PC Zscore":     pc_z,
            "CNN Fear Greed":       fg["Fear_Greed"],
            "Lowry Panic":          lowry,
            "Zweig Breadth":        zweig,
            "Volatility Curve":     vc,
            "52W Highs":            highs["Last"],
            "S&P500":               spx["Last"],
        }
        for k, s in series.items():
            s.index = pd.to_datetime(s.index)
            series[k] = s[~s.index.duplicated(keep="last")].sort_index()

        df = pd.concat(series.values(), axis=1)
        df.columns = list(series.keys())
        df = df.resample("D").last().ffill()
        df = df[df.index >= "2000-01-01"].dropna(subset=["S&P500"])

        conditions = {
            "Percent Above 5DMA":   lambda df, i: df["Percent Above 5DMA"].iloc[i] < 10,
            "ACWI Oscillator":      lambda df, i: df["ACWI Oscillator"].iloc[i] == 0,
            "McClellan Oscillator": lambda df, i: df["McClellan Oscillator"].shift(1).iloc[i] >= -80 and df["McClellan Oscillator"].iloc[i] < -80,
            "Equity PC Zscore":     lambda df, i: df["Equity PC Zscore"].iloc[i] < -2.5,
            "CNN Fear Greed":       lambda df, i: df["CNN Fear Greed"].iloc[i] < 25,
            "Lowry Panic":          lambda df, i: df["Lowry Panic"].iloc[i] >= 4 and df["Lowry Panic"].shift(1).iloc[i] < 4,
            "Zweig Breadth":        lambda df, i: df["Zweig Breadth"].shift(1).iloc[i] > 0.35 and df["Zweig Breadth"].iloc[i] <= 0.35,
            "Volatility Curve":     lambda df, i: df["Volatility Curve"].iloc[i] >= 0 and df["Volatility Curve"].shift(1).iloc[i] < 0,
            "52W Highs":            lambda df, i: df["52W Highs"].iloc[i] < 1,
        }
        for name, cond in conditions.items():
            df[f"{name} Signal"] = _compute_signal(df, cond)

        sig_cols = [f"{n} Signal" for n in conditions]
        df["Composite"] = df[sig_cols].sum(axis=1)

        out = df[sig_cols + ["Composite"]]
        out_path = os.path.join(DATASETS, "btd_signals.csv")
        out.to_csv(out_path)
        log_fn(f"  ✅  btd_signals.csv saved ({len(out)} rows, latest: {out.index[-1].date()})")
        return True

    except Exception as e:
        log_fn(f"  ❌  BTD signals — ERROR: {e}")
        return False


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

    # BTD signals
    try:
        ok = compute_btd_signals(log_fn)
        if ok:
            success += 1
        else:
            failed += 1
    except Exception as e:
        log_fn(f"  ❌  BTD signals — ERROR: {e}")
        failed += 1

    log_fn("=" * 60)
    log_fn(f"Done. {success} updated, {failed} skipped/failed.")
    log_fn("=" * 60)
    return success, failed


# ── Run directly from command line ──────────────────────────────────────────────
if __name__ == "__main__":
    run_update()
