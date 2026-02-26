"""
tab_market_risk.py  —  Market Risk / Regime tab  [v2 - CSV-cached, fast loading]
==================================================================================
Architecture: ALL heavy computation is pre-saved to CSV files.
The tab itself only reads CSVs — zero network calls, zero long loops on render.

CSV files (auto-created in datasets/ on first "Compute" button press):
  market_risk_lt_composite.csv   — Long Term Composite + S&P500
  market_risk_health_model.csv   — Trend Health Model composite + S&P500
  market_risk_indicators.csv     — All individual indicator signals

Run compute_and_save_all() once via the UI button or data_updater.py.
"""

import os, io, warnings
import pandas as pd
import numpy as np
import requests  # type: ignore[import-untyped]
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta, datetime

warnings.filterwarnings("ignore", category=FutureWarning)

_COLAB_DRIVE = "/content/drive/MyDrive/Python"
_LOCAL_DIR   = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(_COLAB_DRIVE):
    BASE_DIR = _COLAB_DRIVE
else:
    BASE_DIR = _LOCAL_DIR
BARCHART = os.path.join(BASE_DIR, "data", "barchart")
DATASETS = os.path.join(BASE_DIR, "data", "datasets")
FRED_KEY = "5ccedb95e2418de2e5b7bae928c4e406"

CSV_LT  = os.path.join(DATASETS, "market_risk_lt_composite.csv")
CSV_THM = os.path.join(DATASETS, "market_risk_health_model.csv")
CSV_IND = os.path.join(DATASETS, "market_risk_indicators.csv")

# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = "#0a0a0a"
BG_PLOT = "#0e0e0e"
GRID    = "rgba(255,255,255,0.04)"
WHITE   = "#e8e8e8"
DIM     = "#888888"
LIME    = "#00ff88"
RED     = "#ff4444"
ORANGE  = "#ff9f43"
BORDER  = "#222222"
RB      = [dict(bounds=["sat", "mon"])]

def _xax():
    return dict(showgrid=False, zeroline=False, tickfont=dict(color=DIM),
                linecolor=BORDER, rangebreaks=RB)

def _yax(color=WHITE, log=False, grid=False):
    return dict(tickfont=dict(color=DIM), title_font=dict(color=color, size=11),
                showgrid=grid, gridcolor=GRID, zeroline=False, linecolor=BORDER,
                type="log" if log else "linear")

def _layout(height=580):
    return dict(
        plot_bgcolor=BG_PLOT, paper_bgcolor=BG, height=height,
        margin=dict(l=60, r=40, t=50, b=40),
        font=dict(color=WHITE, family="Inter, Arial, sans-serif"),
        legend=dict(orientation="v", x=1.01, y=0.99,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#cccccc", size=11)),
    )

# ── CSV helpers ───────────────────────────────────────────────────────────────
def _load_bc(filename):
    path = os.path.join(BARCHART, filename)
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df.columns = df.columns.str.strip()
    for c in df.columns:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df

def _load_spx():
    df = _load_bc("S&P_500_Index_$SPX.csv")
    df.columns = df.columns.str.lower()
    rn = {}
    for c in df.columns:
        if c in ("close", "price"):       rn[c] = "last"
        elif c in ("high", "hi"):         rn[c] = "high"
        elif c in ("low", "lo"):          rn[c] = "low"
    return df.rename(columns=rn)

def _read_cache(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df[df.index.notna()].sort_index()

# ── Stats ─────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR COMPUTATIONS  (only called from compute_and_save_all)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_oecd_cli(log=print):
    url = ("https://sdmx.oecd.org/public/rest/data/"
           "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/"
           "ZAF+IDN+IND+CHN+BRA+GBR+TUR+ESP+MEX+KOR+JPN+ITA+DEU+FRA+CAN+AUS+USA"
           ".M.LI...AA...H?dimensionAtObservation=AllDimensions&format=csvfilewithlabels")
    log("  Fetching OECD CLI...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    cli  = df.pivot_table(index="TIME_PERIOD", columns="REF_AREA",
                          values="OBS_VALUE", aggfunc="first")
    diff = cli.pct_change().gt(0).sum(axis=1) / cli.count(axis=1) * 100
    out  = pd.DataFrame({"G20_CLI": cli.mean(axis=1), "Diffusion": diff})
    daily = out.resample("D").ffill()
    daily.index = daily.index + pd.DateOffset(days=33)
    idx   = pd.date_range(start=out.index.min(), end=datetime.today(), freq="D")
    daily = daily.reindex(idx, method="ffill").ffill()
    daily["Trend"] = (daily["Diffusion"] > 50).astype(int)
    log("  OECD CLI done.")
    return daily

def _compute_n100_hilo():
    hi = _load_bc("Nasdaq_100_52-Week_Highs_$NAHC.csv")["Last"].rename("H")
    lo = _load_bc("Nasdaq_100_52-Week_Lows_$NALC.csv")["Last"].rename("L")
    df = pd.concat([hi, lo], axis=1).fillna(0)
    df["Cum"]   = (df["H"] - df["L"]).cumsum()
    df["SMA"]   = df["Cum"].rolling(200).mean()
    df["Trend"] = (df["Cum"] > df["SMA"]).astype(int)
    return df

def _compute_credit_spreads(log=print):
    from fredapi import Fred
    log("  Fetching FRED credit spreads...")
    fred  = Fred(api_key=FRED_KEY)
    ticks = ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLEMCBPIOAS",
             "BAMLEMHBHYCRPIOAS", "BAMLHE00EHYIOAS"]
    df = pd.DataFrame({t: fred.get_series(t, "1997-01-01") for t in ticks})
    def z_chl(s):
        s = s.dropna()
        hi5, lo5 = s.rolling(5).max(), s.rolling(5).min()
        flag = np.where(s == hi5, 1, np.where(s == lo5, -1, 0))
        cum  = pd.Series(np.cumsum(flag), index=s.index)
        return (cum - cum.rolling(252).mean()) / cum.rolling(252).std()
    z = pd.DataFrame({t: z_chl(df[t]) for t in ticks})
    z["Summed"] = z.sum(axis=1)
    idx = pd.date_range(start=z.index.min(),
                        end=datetime.today() - timedelta(days=1), freq="B")
    z   = z.reindex(idx, method="ffill")
    z["Trend"] = (z["Summed"] < 0).astype(int)
    log("  Credit spreads done.")
    return z

def _compute_vol_regime():
    vix  = _load_bc("CBOE_Volatility_Index_$VIX.csv")["Last"].rename("VIX")
    move = _load_bc("Move_Index_$MOVE.csv")["Last"].rename("MOVE")
    df   = pd.concat([vix, move], axis=1).ffill().dropna()
    w    = 126
    def trend_band(s, lo, hi):
        out, sig = [], False
        for v, l, h in zip(s, lo, hi):
            if not sig and v > h:  sig = True
            elif sig and v < l:    sig = False
            out.append(1 if sig else 0)
        return out
    for c in ["VIX", "MOVE"]:
        df[f"{c}_lo"] = df[c].rolling(w).quantile(0.1)
        df[f"{c}_hi"] = df[c].rolling(w).quantile(0.9)
    df = df.dropna()
    df["tv"]    = trend_band(df["VIX"],  df["VIX_lo"],  df["VIX_hi"])
    df["tm"]    = trend_band(df["MOVE"], df["MOVE_lo"], df["MOVE_hi"])
    df["Trend"] = np.where((df["tv"] == 0) & (df["tm"] == 0), 1, 0)
    return df

def _compute_52w_hilo():
    hi = _load_bc("52-Week_Highs_NYSE_$MAHN.csv")["Last"].rename("H")
    lo = _load_bc("52-Week_Lows_NYSE_$MALN.csv")["Last"].rename("L")
    df = pd.concat([hi, lo], axis=1).ffill()
    df["ratio"] = df["H"] / (df["H"] + df["L"]) * 100
    df["Trend"] = (df["ratio"].rolling(20).mean() > 60).astype(int)
    return df

def _compute_canary(log=print):
    import yfinance as yf
    log("  Downloading Canary (SPY/BND/EEM/EFA)...")
    tickers = ["SPY", "BND", "EEM", "EFA"]
    data = yf.download(tickers, start="2003-01-01", progress=False)
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data.xs("Close", axis=1, level=0))
    di = sum(prices.pct_change(d) * m for d, m in [(21,12),(62,6),(126,2),(252,1)])
    avail = [t for t in tickers if t in di.columns]
    result = pd.DataFrame({f"{t}_t": (di[t] > 0).astype(int) for t in avail},
                          index=di.index)
    result["Trend"] = (result.sum(axis=1) == len(avail)).astype(int)
    log("  Canary done.")
    return result

def _compute_pct_above_200():
    df = _load_bc("Percent_of_Stocks_Above_200-Day Average_$MMTH.csv")
    df["Trend"] = (df["Last"] > 50).astype(int)
    return df

def _compute_acwi_200sma(log=print):
    path = os.path.join(DATASETS, "acwi_oscillator.csv")
    if os.path.exists(path):
        log("  ACWI 200 SMA: reading from acwi_oscillator.csv")
        osc = pd.read_csv(path, parse_dates=True, index_col=0).squeeze()
        osc.index = pd.to_datetime(osc.index, errors="coerce")
        osc = osc[osc.index.notna()].sort_index()
        return pd.DataFrame({"Pct": osc, "Trend": (osc > 50).astype(int)})
    import yfinance as yf
    log("  Downloading ACWI 200 SMA (46 ETFs)...")
    etfs = ["EWA","EWO","EWK","EWC","EDEN","EFNL","EWQ","EWG","EWH","EIRL",
            "EIS","EWI","EWJ","EWN","ENZL","NORW","PGAL","EWS","EWP","EWD",
            "EWL","EWU","SPY","EWZ","ECH","MCHI","GXG","CEZ","EGPT","GREK",
            "INDA","EIDO","EWY","KWT","EWM","EWW","EPU","EPHE","EPOL","QAT",
            "KSA","EZA","EWT","THD","TUR","UAE"]
    data   = yf.download(etfs, start="2007-01-01", progress=False)
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data.xs("Close", axis=1, level=0))
    sma200 = prices.rolling(200, min_periods=200).mean()
    osc    = (prices > sma200).sum(axis=1) / prices.notna().sum(axis=1) * 100
    log("  ACWI 200 SMA done.")
    return pd.DataFrame({"Pct": osc, "Trend": (osc > 50).astype(int)})

def _compute_adl():
    adv = _load_bc("NYSE_Advancing_Stocks_$NSHU.csv")["Last"]
    dec = _load_bc("NYSE_Declining_Stocks_$NSHD.csv")["Last"]
    cum = (adv - dec).dropna().cumsum()
    sma = cum.rolling(200, min_periods=200).mean()
    return pd.DataFrame({"CumAD": cum, "Trend": (cum > sma).astype(int)})

def _compute_inout(log=print):
    import yfinance as yf
    log("  Downloading In/Out tickers...")
    tickers = ["QQQ","XLI","DBB","IGE","SHY","UUP","GLD","SLV","XLU"]
    data = yf.download(tickers, start="2002-01-01", progress=False)
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data.xs("Close", axis=1, level=0))
    rets = prices / prices.rolling(11, center=True).mean().shift(60) - 1
    if "UUP" in rets: rets["UUP"] *= -1
    if "GLD" in rets and "SLV" in rets:
        rets["G_S"] = -(rets["GLD"] - rets["SLV"])
    if "XLU" in rets and "XLI" in rets:
        rets["U_I"] = -(rets["XLU"] - rets["XLI"])
    ext = rets < rets.rolling(252 * 5).quantile(0.05)
    if "SHY" in ext.columns and "DBB" in rets.columns:
        above_med = rets > rets.rolling(1250).median()
        ext["SHY"] = np.where(ext["SHY"] & above_med["DBB"], False, ext["SHY"])
    for c in ["GLD", "QQQ", "SLV", "XLU"]:
        if c in ext.columns:
            ext = ext.drop(columns=[c])
    for c in ext.columns:
        ext[c] = (ext[c] | ext[c].shift(1).fillna(False)
                  .rolling(21, min_periods=1).max().astype(bool))
    log("  In/Out done.")
    return pd.DataFrame({"Trend": (ext.sum(axis=1) == 0).astype(int)})

def _compute_vix_ts():
    v3 = _load_bc("CBOE_3-Month_VIX_$VXV.csv")["Last"].rename("VIX3M")
    v1 = _load_bc("CBOE_Volatility_Index_$VIX.csv")["Last"].rename("VIX1M")
    df = pd.concat([v3, v1], axis=1).ffill().dropna()
    df["ratio"] = df["VIX1M"] / df["VIX3M"]
    df["MA1"]   = df["ratio"].ewm(span=7,  adjust=False).mean()
    df["MA2"]   = df["ratio"].ewm(span=12, adjust=False).mean()
    df["Trend"] = (df["MA1"] < df["MA2"]).astype(int)
    return df

def _compute_hmm(log=print):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        log("  hmmlearn not installed — skipping HMM")
        return None
    log("  Training HMM (~30s)...")
    spx = _load_spx()[["last"]].dropna().sort_index()
    spx["ret"] = spx["last"].pct_change()
    train = spx["ret"].dropna().values.reshape(-1, 1)
    hmm   = GaussianHMM(n_components=2, covariance_type="full",
                        n_iter=300, random_state=17, tol=1e-3)
    hmm.fit(train)
    bull  = int(np.argmax(hmm.means_.ravel()))
    regs  = hmm.predict(train)
    spx.loc[spx["ret"].dropna().index, "Regime"] = regs
    spx["Trend"] = (spx["Regime"] == bull).astype(float).shift(1).fillna(0)
    log("  HMM done.")
    return spx

def _compute_quad(log=print):
    import yfinance as yf
    log("  Downloading Quad 1&2 tickers...")
    tickers = ["XLF","QQQ","SMH","BTC-USD","XLK","KRE","IWM"]
    data = yf.download(tickers, start="2014-01-01",
                       auto_adjust=True, progress=False)
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data["Close"])
    sma   = prices.rolling(62).mean()
    trend = ((prices > sma).sum(axis=1) > len(prices.columns) / 2).astype(int)
    log("  Quad done.")
    return pd.DataFrame({"Trend": trend})

def _compute_btc(log=print):
    import yfinance as yf
    log("  Downloading BTC/SPY...")
    data = yf.download(["BTC-USD", "SPY"], start="2014-01-01",
                       auto_adjust=True, progress=False)
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data["Close"])
    try:
        prices.index = prices.index.tz_localize(None)
    except Exception:
        pass
    rets = prices.pct_change(21).shift(1)
    cond = (rets["BTC-USD"] > 0) & ((rets["BTC-USD"] - rets["SPY"]) > 0)
    log("  BTC done.")
    return pd.DataFrame({"Trend": cond.fillna(False).astype(int)})

def _compute_supertrend(atr_period, multiplier, log=print):
    log(f"  SuperTrend ATR={atr_period} x{multiplier}...")
    spx = _load_spx()
    rn = {}
    for c in spx.columns:
        cl = c.lower()
        if "last" in cl or "close" in cl: rn[c] = "last"
        elif "high" in cl or cl == "hi":  rn[c] = "high"
        elif "low"  in cl or cl == "lo":  rn[c] = "low"
    spx = spx.rename(columns=rn)
    if not {"last","high","low"}.issubset(set(spx.columns)):
        log(f"  SuperTrend skipped — missing high/low columns")
        return None
    df  = spx[["last","high","low"]].dropna().copy()
    hl2 = (df["high"] + df["low"]) / 2
    tr  = pd.concat([df["high"] - df["low"],
                     (df["high"] - df["last"].shift()).abs(),
                     (df["low"]  - df["last"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    df["upper"] = hl2 + multiplier * atr
    df["lower"] = hl2 - multiplier * atr
    # Vectorised-friendly loop using numpy arrays
    n         = len(df)
    direction = np.ones(n, dtype=np.int8)
    upper_a   = df["upper"].values.copy()
    lower_a   = df["lower"].values.copy()
    close_a   = df["last"].values
    # Tight loop — numpy arrays avoid pandas overhead
    for i in range(1, n):
        pu = upper_a[i-1]; pl = lower_a[i-1]; pd_ = direction[i-1]
        if   close_a[i] > pu: direction[i] = 1
        elif close_a[i] < pl: direction[i] = -1
        else:
            direction[i] = pd_
            if pd_ ==  1 and lower_a[i] < pl: lower_a[i] = pl
            if pd_ == -1 and upper_a[i] > pu: upper_a[i] = pu
    df["dir"]   = direction
    trend_arr   = np.empty(n, dtype=np.int8)
    trend_arr[0] = 1
    trend_arr[1:] = (direction[:-1] == 1).astype(np.int8)
    df["Trend"] = trend_arr
    log(f"  SuperTrend ATR={atr_period} done.")
    return df

def _compute_vix_hmm_combined(vts_trend, hmm_trend):
    if vts_trend is None or hmm_trend is None:
        return None
    combined = (vts_trend.shift(1).fillna(0) *
                (1 - hmm_trend).shift(1).fillna(0)).clip(0, 1).astype(int)
    return pd.DataFrame({"Trend": combined})

# ══════════════════════════════════════════════════════════════════════════════
#  MASTER COMPUTE + SAVE  (run once, slow — triggered by button)
# ══════════════════════════════════════════════════════════════════════════════
def compute_and_save_all(log=print):
    """
    Computes all indicators and saves 3 CSV files to datasets/.
    Takes ~3-5 minutes first time (HMM + yfinance downloads).
    After that, tab loads instantly from CSV.
    """
    log("=== Market Risk: starting full computation ===")
    spx   = _load_spx()
    spx_s = spx["last"].dropna()
    res   = {}  # name -> Trend Series

    def try_compute(name, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if result is not None and "Trend" in result.columns:
                res[name] = result["Trend"]
        except Exception as e:
            log(f"  {name} failed: {e}")

    try_compute("OECD_CLI",       _compute_oecd_cli,       log)
    try_compute("N100_HiLo",      _compute_n100_hilo)
    try_compute("Credit_Spreads", _compute_credit_spreads, log)
    try_compute("Vol_Regime",     _compute_vol_regime)
    try_compute("HiLo_52W",       _compute_52w_hilo)
    try_compute("Canary",         _compute_canary,         log)
    try_compute("Pct_200SMA",     _compute_pct_above_200)
    try_compute("ACWI_200",       _compute_acwi_200sma,    log)
    try_compute("AD_Line",        _compute_adl)
    try_compute("InOut",          _compute_inout,          log)
    try_compute("VIX_TS",         _compute_vix_ts)
    try_compute("HMM",            _compute_hmm,            log)
    try_compute("Quad",           _compute_quad,           log)
    try_compute("BTC",            _compute_btc,            log)
    try_compute("ST_LT",          _compute_supertrend, 252, 12, log)
    try_compute("ST_MT",          _compute_supertrend,  63,  9, log)
    try_compute("ST_ST",          _compute_supertrend,  63,  5, log)

    # VIX x HMM requires both
    if "VIX_TS" in res and "HMM" in res:
        try:
            vh = _compute_vix_hmm_combined(res["VIX_TS"], res["HMM"])
            if vh is not None:
                res["VIX_HMM"] = vh["Trend"]
        except Exception as e:
            log(f"  VIX×HMM failed: {e}")

    # ── Save individual signals ───────────────────────────────────────────────
    ind_df = pd.DataFrame(res)
    ind_df.index = pd.to_datetime(ind_df.index, errors="coerce")
    ind_df = ind_df[ind_df.index.notna()].sort_index()
    ind_df.to_csv(CSV_IND)
    log(f"  Saved indicators → {CSV_IND}")

    def _build_composite(spx_s, cols, threshold_fn, label):
        avail = [c for c in cols if c in res]
        df = pd.DataFrame({"S&P500": spx_s,
                           **{c: res[c] for c in avail}})
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()].sort_index()
        df = df.resample("D").last().ffill().dropna(subset=["S&P500"])
        df["Composite"] = df[avail].sum(axis=1)
        df["Trend"]     = threshold_fn(df, avail).astype(int)
        return df

    # Long Term Composite
    lt_cols  = ["OECD_CLI", "N100_HiLo", "Credit_Spreads"]
    lt_avail = [c for c in lt_cols if c in res]
    lt_df = pd.DataFrame({"S&P500": spx_s, **{c: res[c] for c in lt_avail}})
    lt_df.index = pd.to_datetime(lt_df.index, errors="coerce")
    lt_df = lt_df[lt_df.index.notna()].sort_index()
    lt_df = lt_df.resample("D").last().dropna(subset=["S&P500"])
    for c in lt_avail:
        lt_df[c] = lt_df[c].ffill()
    first_data = lt_df[lt_avail].first_valid_index()
    lt_df = lt_df[lt_df.index >= first_data]
    lt_df["Composite"] = lt_df[lt_avail].sum(axis=1)
    lt_df["Trend"]     = (lt_df["Composite"] >= 2).astype(int)
    lt_df.to_csv(CSV_LT)
    log(f"  Saved LT composite → {CSV_LT}")

    # Trend Health Model — all indicators
    all_cols = list(res.keys())
    thm_df   = pd.DataFrame({"S&P500": spx_s,
                              **{c: res[c] for c in all_cols}})
    thm_df.index = pd.to_datetime(thm_df.index, errors="coerce")
    thm_df = thm_df[thm_df.index.notna()].sort_index()
    thm_df = thm_df.resample("D").last().ffill().dropna(subset=["S&P500"])
    ind_c  = [c for c in thm_df.columns if c != "S&P500"]
    thm_df["Trend_Composite"] = (thm_df[ind_c].gt(0).sum(axis=1) /
                                  thm_df[ind_c].notna().sum(axis=1)) * 100
    thm_df["Trend"] = (thm_df["Trend_Composite"] > 55).astype(int)
    thm_df.to_csv(CSV_THM)
    log(f"  Saved Health Model → {CSV_THM}")

    log("=== Market Risk: computation complete ===")
    return True

# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS  (read-only, zero computation)
# ══════════════════════════════════════════════════════════════════════════════
def _regime_chart(title, spx, signal, indicator=None, ind_name="",
                  ind_color=ORANGE, height=520,
                  start_date=None):
    has_ind = indicator is not None

    # Slice to correct start date
    spx = spx.dropna()
    if start_date:
        spx = spx[spx.index >= pd.Timestamp(start_date)]
        if indicator is not None:
            indicator = indicator[indicator.index >= pd.Timestamp(start_date)]

    # Resample SPX to business days so no weekend gaps exist in the index
    spx = spx.resample("B").last().ffill()
    # Align signal, forward-fill across weekends, back-fill start
    sig_r = signal.reindex(spx.index).ffill().bfill().fillna(0).astype(int)

    bull_mask = sig_r == 1
    bear_mask = sig_r == 0

    fig = make_subplots(
        rows=2 if has_ind else 1, cols=1,
        shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.62, 0.38] if has_ind else [1.0])

    # White base line — always visible beneath coloured overlays
    fig.add_trace(go.Scatter(x=spx.index, y=spx, name="S&P 500",
                             mode="lines", line=dict(color=WHITE, width=1.5),
                             connectgaps=False), row=1, col=1)
    # Coloured overlays — slightly thicker so they fully cover the white base
    fig.add_trace(go.Scatter(x=spx.index, y=spx.where(bull_mask), name="Bull",
                             mode="lines", line=dict(color=LIME, width=2.2),
                             connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=spx.index, y=spx.where(bear_mask), name="Bear",
                             mode="lines", line=dict(color=RED, width=2.2),
                             connectgaps=False), row=1, col=1)

    if has_ind:
        fig.add_trace(go.Scatter(x=indicator.index, y=indicator,
                                 mode="lines", name=ind_name,
                                 line=dict(color=ind_color, width=1.3)),
                      row=2, col=1)
        fig.update_yaxes(**_yax(ind_color), title_text=ind_name, row=2, col=1)
        fig.update_xaxes(**_xax(), row=2, col=1)

    layout = _layout(height)
    layout["title"] = dict(text=title, font=dict(color=WHITE, size=13))
    fig.update_layout(**layout)
    fig.update_xaxes(**_xax(), row=1, col=1)
    fig.update_yaxes(**_yax(WHITE, log=True, grid=True),
                     title_text="S&P 500 (log)", row=1, col=1)
    return fig

def _chart_lt(df):
    spx_full   = df["S&P500"].dropna()
    # Normalize composite to account for fewer indicators before ~1999
    _lt_keys = [c for c in df.columns if c not in ("S&P500", "Composite", "Trend")]
    _avail = df[_lt_keys].notna().sum(axis=1).replace(0, np.nan)
    _n_total = max(len(_lt_keys), 1)
    comp_full = (df["Composite"] / _avail * _n_total).fillna(df["Composite"])
    # Resample to business days to eliminate weekend gaps
    spx_full = spx_full.resample("B").last().ffill()
    sig_r_full = df["Trend"].reindex(spx_full.index).ffill().bfill().fillna(0).astype(int).bfill().fillna(0).astype(int)

    comp_start = comp_full.first_valid_index() or spx_full.index[0]
    spx   = spx_full[spx_full.index >= comp_start]
    comp  = comp_full[comp_full.index >= comp_start]
    sig_r = sig_r_full[sig_r_full.index >= comp_start]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.60, 0.40],
                        subplot_titles=("S&P 500 Price with Trend Overlay (Log Scale)",
                                        "Long Term Trend Composite"))
    fig.add_trace(go.Scatter(x=spx.index, y=spx, name="S&P 500",
                             mode="lines", line=dict(color=WHITE, width=1.2),
                             connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=spx.index, y=spx.where(sig_r >= 1), name="Bull",
                             mode="lines", line=dict(color=LIME, width=2.0),
                             connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=spx.index, y=spx.where(sig_r == 0), name="Bear",
                             mode="lines", line=dict(color=RED, width=2.0),
                             connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=comp.index, y=comp,
                             name="Long Term Composite", mode="lines",
                             line=dict(color=ORANGE, width=2, shape="hv"),
                             connectgaps=True), row=2, col=1)
    fig.add_hline(y=2, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)", line_width=1.5, row=2, col=1)
    layout = _layout(600)
    layout["title"] = dict(text="S&P 500 vs Long Term Trend Composite with Overlay",
                           font=dict(color=WHITE, size=13))
    fig.update_layout(**layout)
    fig.update_xaxes(**_xax(), row=1, col=1)
    fig.update_xaxes(**_xax(), row=2, col=1)
    fig.update_yaxes(**_yax(WHITE, log=True, grid=True),
                     title_text="S&P 500 (Log Scale)", row=1, col=1)
    fig.update_yaxes(**_yax(ORANGE), title_text="Long Term Composite",
                     row=2, col=1, range=[-0.2, 3.5])
    return fig


def _chart_thm(df):
    spx_full   = df["S&P500"].dropna()
    comp_full  = df["Trend_Composite"]
    # Resample to business days to eliminate weekend gaps
    spx_full = spx_full.resample("B").last().ffill()
    sig_r_full = df["Trend"].reindex(spx_full.index).ffill().bfill().fillna(0).astype(int).bfill().fillna(0).astype(int)

    comp_start = comp_full.first_valid_index() or spx_full.index[0]
    spx   = spx_full[spx_full.index >= comp_start]
    comp  = comp_full[comp_full.index >= comp_start]
    sig_r = sig_r_full[sig_r_full.index >= comp_start]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.60, 0.40],
                        subplot_titles=("S&P 500 Price with Trend Overlay (Log Scale)",
                                        "Trend Health Model"))
    fig.add_trace(go.Scatter(x=spx.index, y=spx, name="S&P 500",
                             mode="lines", line=dict(color=WHITE, width=1.2),
                             connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=spx.index, y=spx.where(sig_r == 1), name="Bullish",
                             mode="lines", line=dict(color=LIME, width=2.0),
                             connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=spx.index, y=spx.where(sig_r == 0), name="Bearish",
                             mode="lines", line=dict(color=RED, width=2.0),
                             connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=comp.index, y=comp,
                             name="Trend Health Model", mode="lines",
                             line=dict(color=ORANGE, width=1.8),
                             connectgaps=True), row=2, col=1)
    fig.add_hline(y=55, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)", line_width=1.5, row=2, col=1)
    layout = _layout(620)
    layout["title"] = dict(text="S&P 500 vs Trend Health Model with Overlay",
                           font=dict(color=WHITE, size=13))
    fig.update_layout(**layout)
    fig.update_xaxes(**_xax(), row=1, col=1)
    fig.update_xaxes(**_xax(), row=2, col=1)
    fig.update_yaxes(**_yax(WHITE, log=True, grid=True),
                     title_text="S&P 500 (Log Scale)", row=1, col=1)
    fig.update_yaxes(**_yax(ORANGE), title_text="Trend Health Model (%)",
                     row=2, col=1, range=[-2, 105])
    return fig


def _ind_chart(title, ind_df, col, spx_s,
               ind_series=None, ind_name="", ind_color=ORANGE, label="Strategy",
               start_date=None):
    if ind_df is None:
        return None, "ind_df is None"
    if col not in ind_df.columns:
        return None, f"Column not in CSV. Available: {list(ind_df.columns)}"
    try:
        trend = ind_df[col].dropna()
        fig = _regime_chart(title, spx_s, trend,
                            ind_series, ind_name, ind_color,
                            start_date=start_date)
        return fig, None
    except Exception as e:
        return None, str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render():
    st.subheader("📊 Market Risk — Regime Indicators")
    st.caption("🟢 Lime = Bull regime  |  🔴 Red = Bear regime  "
               "|  Long Term: Bull if ≥ 2/3 signals  "
               "|  Health Model: Bull if > 55%")

    csvs_ready = all(os.path.exists(p) for p in [CSV_LT, CSV_THM, CSV_IND])

    # ── First-run: show compute button ────────────────────────────────────────
    if not csvs_ready:
        st.warning(
            "⚠️ Market Risk CSVs not found. Run the **💾 Save Dashboard CSVs** "
            "cell in the Colab notebook first, then click Refresh."
        )
        st.code(f"Looking in:\n  {CSV_LT}\n  {CSV_THM}\n  {CSV_IND}")
        if st.button("🔄 Refresh from CSVs"):
            st.rerun()
        return

    # ── Load cached data (instant) ────────────────────────────────────────────
    df_lt  = _read_cache(CSV_LT)
    df_thm = _read_cache(CSV_THM)
    ind_df = _read_cache(CSV_IND)

    # Use ind_df for SPX — it has the longest history
    if ind_df is not None and "S&P500" in ind_df.columns:
        spx_s = ind_df["S&P500"].dropna()
    elif df_thm is not None and "S&P500" in df_thm.columns:
        spx_s = df_thm["S&P500"].dropna()
    else:
        spx_s = None

    if df_lt is None or df_thm is None or ind_df is None or spx_s is None:
        st.error(f"Cached data missing or corrupted.")
        st.code(f"CSV_LT:  {'✅' if df_lt  is not None else '❌'} {CSV_LT}\n"
                f"CSV_THM: {'✅' if df_thm is not None else '❌'} {CSV_THM}\n"
                f"CSV_IND: {'✅' if ind_df is not None else '❌'} {CSV_IND}")
        return

    # Show available columns for debugging
    with st.expander("🔍 Debug: CSV column check", expanded=False):
        st.write("**ind_df columns:**", list(ind_df.columns))
        st.write("**df_thm columns:**", list(df_thm.columns))
        st.write("**df_lt columns:**",  list(df_lt.columns))

    # Refresh button — just clears cache and reloads from CSVs (fast)
    if st.button("🔄 Refresh from CSVs"):
        st.cache_data.clear()
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — Long Term Trend Composite
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🌍 Long Term Trend Composite")
    st.plotly_chart(_chart_lt(df_lt), width='stretch')

    lt_ind_cols = [c for c in df_lt.columns
                   if c not in ("S&P500", "Composite", "Trend")]
    latest_lt   = int(df_lt["Composite"].dropna().iloc[-1])
    c1, c2, c3  = st.columns(3)
    c1.metric("Composite Score", f"{latest_lt} / {len(lt_ind_cols)}",
              delta="🟢 BULL" if df_lt["Trend"].dropna().iloc[-1] == 1 else "🔴 BEAR")
    c2.metric("OECD CLI",
              "BULL 🟢" if "OECD_CLI" in df_lt.columns
              and df_lt["OECD_CLI"].iloc[-1] == 1 else "BEAR 🔴")
    c3.metric("Credit Spreads",
              "BULL 🟢" if "Credit_Spreads" in df_lt.columns
              and df_lt["Credit_Spreads"].iloc[-1] == 1
              else "BEAR 🔴" if "Credit_Spreads" in df_lt.columns else "N/A ⚠️")

    st.markdown("#### Individual Long Term Indicators")
    for title, col, ind_name, ind_col, label, start in [
        ("1. OECD CLI Diffusion Index",    "OECD_CLI",       "", "#f9ca24", "OECD CLI Strategy",     "1997-01-01"),
        ("2. Nasdaq 100 Cumulative Hi-Lo", "N100_HiLo",      "", "#4ecdc4", "N100 Hi-Lo Strategy",   "1999-01-01"),
        ("3. Credit Spreads (FRED)",       "Credit_Spreads", "", "#ff6b6b", "Credit Spread Strategy", "1997-01-01"),
    ]:
        key = f"show_{col}"
        with st.expander(title, expanded=False):
            if key not in st.session_state:
                st.session_state[key] = False
            if not st.session_state[key]:
                if st.button("📊 Load chart", key=f"btn_{col}"):
                    st.session_state[key] = True
                    st.rerun()
            else:
                fig, err = _ind_chart(title, ind_df, col, spx_s,
                                       None, ind_name, ind_col, label, start_date=start)
                if fig:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning(f"Chart unavailable: {err}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — Trend Health Model
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🩺 Trend Health Model")
    st.plotly_chart(_chart_thm(df_thm), width='stretch')

    ind_thm      = [col for col in df_thm.columns
                    if col not in ("S&P500", "Trend_Composite", "Trend")]
    lt_ind_cols2 = [col for col in df_lt.columns
                    if col not in ("S&P500", "Composite", "Trend")]
    comp_now   = df_thm["Trend_Composite"].dropna().iloc[-1]
    _thm_last  = df_thm[ind_thm].iloc[-1]
    _lt_last   = df_lt[lt_ind_cols2].reindex(df_thm.index).ffill().iloc[-1]
    bull_count = int(_thm_last.gt(0).sum()) + int(_lt_last.gt(0).sum())
    total_ind  = int(_thm_last.notna().sum()) + int(_lt_last.notna().sum())
    regime_chg = df_thm.index[df_thm["Trend"].diff().abs() > 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Health Score", f"{comp_now:.0f}%",
              delta="🟢 BULL" if df_thm["Trend"].dropna().iloc[-1] == 1 else "🔴 BEAR")
    c2.metric("Bullish Indicators", f"{bull_count} / {total_ind}")
    c3.metric("Last Regime Change",
              str(regime_chg[-1].date()) if len(regime_chg) > 0 else "N/A")

    st.markdown("#### Individual Health Model Indicators")
    for title, col, ind_col, label, start in [
        ("4.  Volatility Regime (MOVE/VIX)",       "Vol_Regime", ORANGE,    "Vol Regime",         "2008-01-01"),
        ("5.  NYSE 52-Week Hi-Lo Trend",            "HiLo_52W",  "#f9ca24", "Hi-Lo Strategy",     "1990-01-01"),
        ("6.  Canary Model",                        "Canary",    "#4ecdc4", "Canary Strategy",    "2003-01-01"),
        ("7.  % Stocks Above 200-Day SMA",          "Pct_200SMA",ORANGE,    "200 SMA Strategy",   "2002-01-01"),
        ("8.  ACWI 200 SMA Breadth",                "ACWI_200",  "#4ecdc4", "ACWI Strategy",      "2007-01-01"),
        ("9.  Cumulative A/D Line",                 "AD_Line",   "#4ecdc4", "A/D Strategy",       "2005-03-01"),
        ("10. In/Out Indicator",                    "InOut",     ORANGE,    "In/Out Strategy",    "2002-01-01"),
        ("11. VIX Term Structure",                  "VIX_TS",    "#f9ca24", "VIX TS Strategy",    "2007-01-01"),
        ("12. HMM Regime Indicator",                "HMM",       LIME,      "HMM Strategy",       "1960-01-01"),
        ("13. Quad 1 & 2",                          "Quad",      "#4ecdc4", "Quad Strategy",      "2014-01-01"),
        ("14. Bitcoin Liquidity Proxy",             "BTC",       "#f9ca24", "BTC Strategy",       "2014-01-01"),
        ("15. SuperTrend Long Term (ATR 252×12)",   "ST_LT",     LIME,      "ST LT Strategy",     "1990-01-01"),
        ("16. SuperTrend Medium Term (ATR 63×9)",   "ST_MT",     LIME,      "ST MT Strategy",     "1990-01-01"),
        ("17. SuperTrend Short Term (ATR 63×5)",    "ST_ST",     LIME,      "ST ST Strategy",     "1990-01-01"),
        ("18. VIX Term Structure × HMM",            "VIX_HMM",   "#ff6b6b", "VIX×HMM Strategy",  "2007-01-01"),
    ]:
        key = f"show_{col}"
        with st.expander(title, expanded=False):
            if key not in st.session_state:
                st.session_state[key] = False
            if not st.session_state[key]:
                if st.button("📊 Load chart", key=f"btn_{col}"):
                    st.session_state[key] = True
                    st.rerun()
            else:
                fig, err = _ind_chart(title, ind_df, col, spx_s,
                                       None, "", ind_col, label, start_date=start)
                if fig:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning(f"Chart unavailable: {err}")
