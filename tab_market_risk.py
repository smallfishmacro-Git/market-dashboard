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
try:
    FRED_KEY = st.secrets.get("FRED_KEY", "5ccedb95e2418de2e5b7bae928c4e406")
except Exception:
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

@st.cache_data
def _read_cache(path, file_mtime=None):   # file_mtime busts cache when CSV changes
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df[df.index.notna()].sort_index()

# ── Timeframe selector helpers ─────────────────────────────────────────────────
_TF_OPTIONS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "ALL"]

def _tf_cutoff(tf):
    today = pd.Timestamp.now().normalize()
    return {"1M":  today - pd.DateOffset(months=1),
            "3M":  today - pd.DateOffset(months=3),
            "6M":  today - pd.DateOffset(months=6),
            "YTD": pd.Timestamp(today.year, 1, 1),
            "1Y":  today - pd.DateOffset(years=1),
            "2Y":  today - pd.DateOffset(years=2),
            "5Y":  today - pd.DateOffset(years=5),
            "10Y": today - pd.DateOffset(years=10),
            "ALL": None}.get(tf)

def _tf_widget(key, default="2Y"):
    _, ctrl = st.columns([2, 3])
    with ctrl:
        return st.segmented_control(
            "Timeframe", _TF_OPTIONS,
            default=default, key=key,
            label_visibility="collapsed")

def _apply_tf(fig, tf):
    if tf == "ALL" or tf is None:
        return fig
    cutoff = _tf_cutoff(tf)
    if cutoff is None:
        return fig
    today_str = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    fig.update_xaxes(range=[cutoff.strftime("%Y-%m-%d"), today_str])
    return fig


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
    def calculate_trend(series, low_band, high_band):
        values, signal = [], False
        for value, low, high in zip(series, low_band, high_band):
            if not signal and value > high:
                values.append(1)
                signal = True
            elif not signal and value <= high:
                values.append(0)
            elif signal and value < low:
                values.append(0)
                signal = False
            elif signal and value >= low:
                values.append(1)
            else:
                values.append(0)
        return values
    for c in ["VIX", "MOVE"]:
        df[f"{c}_lo"] = df[c].rolling(w).quantile(0.1)
        df[f"{c}_hi"] = df[c].rolling(w).quantile(0.9)
    df = df.dropna()
    df["tv"]    = calculate_trend(df["VIX"],  df["VIX_lo"],  df["VIX_hi"])
    df["tm"]    = calculate_trend(df["MOVE"], df["MOVE_lo"], df["MOVE_hi"])
    # Vol_Regime = 1 - regime1; regime1 = both VIX and MOVE elevated
    df["Trend"] = np.where((df["tv"] == 1) & (df["tm"] == 1), 0, 1)
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
    data = yf.download(tickers, start="2003-01-01", progress=False, timeout=30)
    if data.empty:
        raise ValueError("yfinance returned empty dataframe for Canary tickers")
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
    import yfinance as yf
    path = os.path.join(DATASETS, "acwi_oscillator.csv")
    log("  Downloading ACWI 200 SMA (46 ETFs)...")
    etfs = ["EWA","EWO","EWK","EWC","EDEN","EFNL","EWQ","EWG","EWH","EIRL",
            "EIS","EWI","EWJ","EWN","ENZL","NORW","PGAL","EWS","EWP","EWD",
            "EWL","EWU","SPY","EWZ","ECH","MCHI","GXG","CEZ","EGPT","GREK",
            "INDA","EIDO","EWY","KWT","EWM","EWW","EPU","EPHE","EPOL","QAT",
            "KSA","EZA","EWT","THD","TUR","UAE"]
    data   = yf.download(etfs, start="2007-01-01", progress=False, timeout=30)
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data.xs("Close", axis=1, level=0))
    valid  = prices.notna().sum(axis=1)

    # 10-day MA — for BTD oscillator (acwi_oscillator.csv, signals when 0%)
    sma10  = prices.rolling(10, min_periods=10).mean()
    osc10  = ((prices > sma10).sum(axis=1) / valid * 100).dropna()

    # 200-day SMA — for ACWI_200 market risk indicator (Trend = osc > 50%)
    sma200 = prices.rolling(200, min_periods=200).mean()
    osc200 = ((prices > sma200).sum(axis=1) / valid * 100).dropna()

    if len(osc10) == 0:
        log("  ACWI: no data returned (SSL/network issue), keeping existing CSV")
        return None  # baseline in compute_and_save_all preserves existing ACWI_200
    # Save 10-day MA oscillator to acwi_oscillator.csv (BTD tab reads "Percentage")
    pd.DataFrame({"Percentage": osc10}).to_csv(path)
    log(f"  ACWI done. Saved {len(osc10)} rows (10-day MA) → {path}")
    return pd.DataFrame({"Pct": osc200, "Trend": (osc200 > 50).astype(int)})

def _compute_adl():
    START_DATE = "2005-03-01"
    adv = _load_bc("NYSE_Advancing_Stocks_$NSHU.csv")["Last"].dropna()
    dec = _load_bc("NYSE_Declining_Stocks_$NSHD.csv")["Last"].dropna()
    # Filter to START_DATE first, then align on inner join dates (Colab cell 9)
    adv = adv[adv.index >= START_DATE]
    dec = dec[dec.index >= START_DATE]
    common = adv.index.intersection(dec.index)
    daily_net = adv.reindex(common) - dec.reindex(common)
    cumulative_ad_line = daily_net.cumsum()
    cumulative_ad_line_100ema = cumulative_ad_line.ewm(span=100, adjust=False).mean()
    adl_indicator = (cumulative_ad_line > cumulative_ad_line_100ema).astype(int)
    return pd.DataFrame({"CumAD": cumulative_ad_line, "Trend": adl_indicator})

def _compute_inout(log=print):
    import yfinance as yf
    log("  Downloading In/Out tickers...")
    tickers = ["QQQ","XLI","DBB","IGE","SHY","UUP","GLD","SLV","XLU"]
    data = yf.download(tickers, start="2002-01-01", progress=False, timeout=30)
    if data.empty:
        raise ValueError("yfinance returned empty dataframe for InOut tickers")
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
                       auto_adjust=True, progress=False, timeout=30)
    if data.empty:
        raise ValueError("yfinance returned empty dataframe for Quad tickers")
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data["Close"])
    # Reindex to equity market dates: drop weekends/holidays where only BTC-USD has data
    equity_cols = [t for t in tickers if t != "BTC-USD" and t in prices.columns]
    prices = prices.dropna(subset=equity_cols)
    sma   = prices.rolling(62).mean()
    trend = ((prices > sma).sum(axis=1) > len(prices.columns) / 2).astype(int)
    log("  Quad done.")
    return pd.DataFrame({"Trend": trend})

def _compute_btc(log=print):
    import yfinance as yf
    log("  Downloading BTC/SPY...")
    data = yf.download(["BTC-USD", "SPY"], start="2014-01-01",
                       auto_adjust=True, progress=False, timeout=30)
    if data.empty:
        raise ValueError("yfinance returned empty dataframe for BTC/SPY")
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
    # Accept either a DataFrame (with "Trend" column) or a plain Series
    vts = (vts_trend["Trend"] if isinstance(vts_trend, pd.DataFrame) else vts_trend).shift(1).fillna(0)
    hmm = (hmm_trend["Trend"] if isinstance(hmm_trend, pd.DataFrame) else hmm_trend).shift(1).fillna(0)
    combined = (vts + hmm).clip(0, 1).astype(int)
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

    # Seed res from existing CSV so failed indicators keep historical data
    res = {}
    _existing_ind = os.path.join(DATASETS, "market_risk_indicators.csv")
    if os.path.exists(_existing_ind):
        try:
            _base = pd.read_csv(_existing_ind, index_col=0, parse_dates=True)
            _base.index = pd.to_datetime(_base.index, errors="coerce")
            _base = _base[_base.index.notna()].sort_index()
            for _col in _base.columns:
                res[_col] = _base[_col].dropna()
            log(f"  Loaded {len(_base.columns)} existing indicator columns as baseline")
        except Exception as _e:
            log(f"  Could not load existing indicators CSV: {_e}")

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

    # ── Normalize timezones (yfinance returns tz-aware; Barchart is tz-naive) ──
    def _strip_tz(s):
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        return s
    res   = {k: _strip_tz(v) for k, v in res.items()}
    spx_s = _strip_tz(spx_s)

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
    thm_df = thm_df[thm_df.index <= spx_s.dropna().index[-1]]  # no dates past last market close
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
    START = "1999-01-01"
    spx_full  = df["S&P500"].dropna()
    _lt_keys  = [c for c in df.columns if c not in ("S&P500", "Composite", "Trend")]
    _avail    = df[_lt_keys].notna().sum(axis=1).replace(0, np.nan)
    _n_total  = max(len(_lt_keys), 1)
    comp_full = (df["Composite"] / _avail * _n_total).fillna(df["Composite"])
    spx_full  = spx_full.resample("B").last().ffill()
    sig_r_full = (df["Trend"].reindex(spx_full.index)
                  .ffill().bfill().fillna(0).astype(int))

    spx   = spx_full[spx_full.index >= START]
    comp  = comp_full[comp_full.index >= START]
    sig_r = sig_r_full[sig_r_full.index >= START]

    _C = "#141414"   # chart bg
    _P = "#0f0f0f"   # paper bg
    _G = "#1e1e1e"   # gridlines
    _T = "#888888"   # text / ticks
    _B = "#222222"   # axis lines

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.65, 0.35],
    )

    # ── Row 1: S&P 500 with bull/bear colouring ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx, name="S&P 500",
        mode="lines", line=dict(color="#444444", width=1.2),
        connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.where(sig_r >= 1), name="Bull",
        mode="lines", line=dict(color="#00c896", width=2.0),
        connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.where(sig_r == 0), name="Bear",
        mode="lines", line=dict(color="#ff4444", width=2.0),
        connectgaps=False), row=1, col=1)

    # ── Row 2: composite line + filled area ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=comp.index, y=comp.values,
        mode="lines",
        line=dict(color="#ff6600", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 102, 0, 0.15)",
        name="Composite",
        showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[comp.index[0], comp.index[-1]], y=[2, 2],
        mode="lines",
        line=dict(color="#333333", dash="dot", width=1),
        showlegend=False), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        plot_bgcolor=_C, paper_bgcolor=_P, height=700,
        margin=dict(l=60, r=40, t=40, b=40),
        font=dict(color=_T, family="Inter, Arial, sans-serif", size=11),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=_T, size=11)),
    )

    # ── Axes ──────────────────────────────────────────────────────────────────
    _xax_style = dict(showgrid=False, zeroline=False,
                      tickfont=dict(color=_T, size=11),
                      linecolor=_B, rangebreaks=RB)
    fig.update_xaxes(**_xax_style, row=1, col=1)
    fig.update_xaxes(**_xax_style, row=2, col=1)

    fig.update_yaxes(
        tickfont=dict(color=_T, size=11),
        type="log", showgrid=True, gridcolor=_G,
        zeroline=False, linecolor=_B,
        row=1, col=1)
    fig.update_yaxes(
        tickfont=dict(color=_T, size=11),
        showgrid=True, gridcolor=_G, dtick=1,
        zeroline=False, linecolor=_B, range=[0, 3],
        row=2, col=1)
    return fig


def _chart_thm(df):
    START = "1999-01-01"
    spx_full   = df["S&P500"].dropna()
    comp_full  = df["Trend_Composite"]
    # Resample both to business days to eliminate weekend gaps
    spx_full  = spx_full.resample("B").last().ffill()
    comp_full = comp_full.resample("B").last().ffill()
    sig_r_full = (df["Trend"].reindex(spx_full.index)
                  .ffill().bfill().fillna(0).astype(int))

    comp_start = comp_full.first_valid_index() or spx_full.index[0]
    start = max(pd.Timestamp(START), comp_start)
    spx   = spx_full[spx_full.index >= start]
    comp  = comp_full[comp_full.index >= start]
    sig_r = sig_r_full[sig_r_full.index >= start]

    _C = "#141414"   # chart bg
    _P = "#0f0f0f"   # paper bg
    _G = "#1e1e1e"   # gridlines
    _T = "#888888"   # text / ticks
    _B = "#222222"   # axis lines

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.65, 0.35],
    )

    # Row 1: S&P 500 with bull/bear colouring
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx, name="S&P 500",
        mode="lines", line=dict(color="#444444", width=1.2),
        connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.where(sig_r >= 1), name="Bull",
        mode="lines", line=dict(color="#00c896", width=2.0),
        connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.where(sig_r == 0), name="Bear",
        mode="lines", line=dict(color="#ff4444", width=2.0),
        connectgaps=False), row=1, col=1)

    # Row 2: health model line + filled area
    fig.add_trace(go.Scatter(
        x=comp.index, y=comp.values,
        mode="lines",
        line=dict(color="#ff6600", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 102, 0, 0.15)",
        name="Health Model",
        showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[comp.index[0], comp.index[-1]], y=[55, 55],
        mode="lines",
        line=dict(color="#333333", dash="dot", width=1),
        showlegend=False), row=2, col=1)

    # Layout
    fig.update_layout(
        plot_bgcolor=_C, paper_bgcolor=_P, height=700,
        margin=dict(l=60, r=40, t=40, b=40),
        font=dict(color=_T, family="Inter, Arial, sans-serif", size=11),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=_T, size=11)),
    )

    # Axes
    _xax_style = dict(showgrid=False, zeroline=False,
                      tickfont=dict(color=_T, size=11),
                      linecolor=_B, rangebreaks=RB)
    fig.update_xaxes(**_xax_style, row=1, col=1)
    fig.update_xaxes(**_xax_style, row=2, col=1)

    fig.update_yaxes(
        tickfont=dict(color=_T, size=11),
        type="log", showgrid=True, gridcolor=_G,
        zeroline=False, linecolor=_B,
        row=1, col=1)
    fig.update_yaxes(
        tickfont=dict(color=_T, size=11),
        showgrid=True, gridcolor=_G,
        zeroline=False, linecolor=_B, range=[-2, 105],
        row=2, col=1)
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
    csvs_ready = all(os.path.exists(p) for p in [CSV_LT, CSV_THM, CSV_IND])

    # ── First-run: show compute button ────────────────────────────────────────
    if not csvs_ready:
        st.warning(
            "Market Risk CSVs not found. Run the **Save Dashboard CSVs** "
            "cell in the Colab notebook first, then click Refresh."
        )
        st.code(f"Looking in:\n  {CSV_LT}\n  {CSV_THM}\n  {CSV_IND}")
        if st.button("↻  Refresh"):
            st.rerun()
        return

    # ── Load cached data — cache busts automatically when any CSV changes ─────
    df_lt  = _read_cache(CSV_LT,  file_mtime=os.path.getmtime(CSV_LT)  if os.path.exists(CSV_LT)  else None)
    df_thm = _read_cache(CSV_THM, file_mtime=os.path.getmtime(CSV_THM) if os.path.exists(CSV_THM) else None)
    ind_df = _read_cache(CSV_IND, file_mtime=os.path.getmtime(CSV_IND) if os.path.exists(CSV_IND) else None)

    # Use ind_df for SPX — it has the longest history
    if ind_df is not None and "S&P500" in ind_df.columns:
        spx_s = ind_df["S&P500"].dropna()
    elif df_thm is not None and "S&P500" in df_thm.columns:
        spx_s = df_thm["S&P500"].dropna()
    else:
        spx_s = None

    if df_lt is None or df_thm is None or ind_df is None or spx_s is None:
        st.error(f"Cached data missing or corrupted.")
        st.code(f"CSV_LT:  {'OK' if df_lt  is not None else 'ERR'} {CSV_LT}\n"
                f"CSV_THM: {'OK' if df_thm is not None else 'ERR'} {CSV_THM}\n"
                f"CSV_IND: {'OK' if ind_df is not None else 'ERR'} {CSV_IND}")
        return

    # Refresh button — just clears cache and reloads from CSVs (fast)
    if st.button("↻  Refresh"):
        st.cache_data.clear()
        st.rerun()


    # ══════════════════════════════════════════════════════════════════════════
    #  INDICATOR TABLES
    # ══════════════════════════════════════════════════════════════════════════

    def _last_change_date(df, col):
        if col not in df.columns:
            return "—"
        s = df[col].dropna()
        if len(s) < 2:
            return "—"
        changed = s[s.diff().abs() > 0]
        return changed.index[-1].strftime("%b %d, %Y") if len(changed) else "—"

    def _last_update_date(df, col):
        if col not in df.columns:
            return "—"
        s = df[col].dropna()
        return s.index[-1].strftime("%b %d, %Y") if len(s) else "—"

    def _get_signal(df, col):
        if col not in df.columns:
            return None
        s = df[col].dropna()
        return int(s.iloc[-1]) if len(s) else None

    def _pill(val):
        if val == 1:
            return ("<span style='background:#00c89626;border:1px solid #00c896;"
                    "color:#00c896;border-radius:4px;padding:3px 12px;"
                    "font-size:0.7rem;font-weight:600;'>Long</span>")
        elif val == 0:
            return ("<span style='background:#ff444426;border:1px solid #ff4444;"
                    "color:#ff4444;border-radius:4px;padding:3px 12px;"
                    "font-size:0.7rem;font-weight:600;'>Short</span>")
        return ("<span style='border:1px solid #444;color:#555;"
                "border-radius:4px;padding:3px 12px;"
                "font-size:0.7rem;font-weight:600;'>—</span>")

    def _build_table(section_title, rows, title_top_margin="0"):
        hover_css = ("<style>.ind-tbl tr:hover td"
                     "{background:#161616 !important;}</style>")
        th_s = ("background:#0a0a0a;text-align:left;padding:10px 20px;"
                "font-family:Inter,sans-serif;font-size:0.68rem;"
                "font-weight:500;color:#555555;text-transform:uppercase;"
                "letter-spacing:0.1em;white-space:nowrap;")
        th_r = ("background:#0a0a0a;text-align:right;padding:10px 20px;"
                "font-family:Inter,sans-serif;font-size:0.68rem;"
                "font-weight:500;color:#555555;text-transform:uppercase;"
                "letter-spacing:0.1em;white-space:nowrap;")
        thead = (f"<thead><tr style='border-bottom:1px solid #1e1e1e;'>"
                 f"<th style='{th_s}width:38%;'>Name</th>"
                 f"<th style='{th_s}width:20%;'>Last Change</th>"
                 f"<th style='{th_s}width:20%;'>Last Update</th>"
                 f"<th style='{th_r}width:22%;'>Status</th>"
                 f"</tr></thead>")
        tbody = ""
        for idx, (name, col, df_src) in enumerate(rows):
            row_bg = "#0a0a0a" if idx % 2 == 0 else "#111111"
            lc  = _last_change_date(df_src, col)
            lu  = _last_update_date(df_src, col)
            val = _get_signal(df_src, col)
            tbody += (
                f"<tr>"
                f"<td style='background:{row_bg};padding:14px 20px;"
                f"font-family:Inter,sans-serif;font-size:0.85rem;"
                f"color:#e0e0e0;font-weight:400;'>{name}</td>"
                f"<td style='background:{row_bg};padding:14px 20px;"
                f"font-family:Inter,sans-serif;font-size:0.8rem;"
                f"color:#888888;'>{lc}</td>"
                f"<td style='background:{row_bg};padding:14px 20px;"
                f"font-family:Inter,sans-serif;font-size:0.8rem;"
                f"color:#888888;'>{lu}</td>"
                f"<td style='background:{row_bg};padding:14px 20px;"
                f"text-align:right;'>{_pill(val)}</td></tr>"
            )
        title_html = (
            f"<p style='font-family:Inter,sans-serif;font-size:0.75rem;"
            f"font-weight:600;color:#ff6600;text-transform:uppercase;"
            f"letter-spacing:0.12em;margin:{title_top_margin} 0 12px 0;'>"
            f"{section_title}</p>"
        )
        table_html = (
            f"<table class='ind-tbl' style='width:100%;"
            f"border-collapse:collapse;'>"
            f"{thead}<tbody>{tbody}</tbody></table>"
        )
        card_html = (
            f"<div style='background:#0f0f0f;border:1px solid #1e1e1e;"
            f"border-radius:12px;overflow:hidden;margin-bottom:24px;'>"
            f"{table_html}</div>"
        )
        return f"{hover_css}{title_html}{card_html}"

    _lt_rows = [
        ("OECD CLI Diffusion Index",    "OECD_CLI",
         ind_df if "OECD_CLI" in ind_df.columns else df_lt),
        ("Nasdaq 100 Cumulative Hi-Lo", "N100_HiLo",
         ind_df if "N100_HiLo" in ind_df.columns else df_lt),
        ("Credit Spreads (FRED)",       "Credit_Spreads",
         ind_df if "Credit_Spreads" in ind_df.columns else df_lt),
    ]
    _thm_rows = [
        ("Volatility Regime (MOVE/VIX)",   "Vol_Regime",  ind_df),
        ("NYSE 52-Week Hi-Lo Trend",        "HiLo_52W",   ind_df),
        ("Canary Model",                    "Canary",     ind_df),
        ("% Stocks Above 200-Day SMA",      "Pct_200SMA", ind_df),
        ("ACWI 200 SMA Breadth",            "ACWI_200",   ind_df),
        ("Cumulative A/D Line",             "AD_Line",    ind_df),
        ("In/Out Indicator",                "InOut",      ind_df),
        ("VIX Term Structure",              "VIX_TS",     ind_df),
        ("HMM Regime Indicator",            "HMM",        ind_df),
        ("Quad 1 & 2",                      "Quad",       ind_df),
        ("Bitcoin Liquidity Proxy",         "BTC",        ind_df),
        ("SuperTrend Long Term",            "ST_LT",      ind_df),
        ("SuperTrend Medium Term",          "ST_MT",      ind_df),
        ("SuperTrend Short Term",           "ST_ST",      ind_df),
        ("VIX Term Structure x HMM",        "VIX_HMM",   ind_df),
    ]
    st.markdown(_build_table("Long Term Indicators", _lt_rows),  unsafe_allow_html=True)
    st.markdown(_build_table("Health Model Indicators", _thm_rows, title_top_margin="32px"), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — Long Term Trend Composite
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 12px 0;">Long Term Composite</p>', unsafe_allow_html=True)
    tf_lt = _tf_widget("tf_mr_lt", "ALL")
    st.plotly_chart(_apply_tf(_chart_lt(df_lt), tf_lt), width='stretch')

    # LT info cards
    lt_ind_cols   = [c for c in df_lt.columns if c not in ("S&P500", "Composite", "Trend")]
    latest_lt     = int(df_lt["Composite"].dropna().iloc[-1])
    lt_trend_s    = df_lt["Trend"].dropna()
    lt_is_bull    = lt_trend_s.iloc[-1] == 1
    lt_chg_dates  = lt_trend_s[lt_trend_s.diff().abs() > 0]
    lt_regime_str = lt_chg_dates.index[-1].strftime("%b %d, %Y") if len(lt_chg_dates) else "—"
    lt_update_str = df_lt.dropna(subset=["Composite"]).index[-1].strftime("%b %d, %Y")
    lt_spx_s      = df_lt["S&P500"].dropna()
    lt_spx_price  = lt_spx_s.iloc[-1]
    lt_spx_chg    = (lt_spx_s.iloc[-1] / lt_spx_s.iloc[-2] - 1) * 100 if len(lt_spx_s) >= 2 else 0
    _lt_accent    = "#ff6600" if lt_is_bull else "#ff4444"
    _lt_bull_col  = "#00c896" if lt_is_bull else "#ff4444"
    _lt_bull_txt  = "BULL" if lt_is_bull else "BEAR"
    _lt_chg_col   = "#00c896" if lt_spx_chg > 0 else ("#ff4444" if lt_spx_chg < 0 else "#888888")
    _card = "background:#141414;border-radius:12px;border:1px solid #2a2a2a;padding:20px 24px;flex:1;position:relative;overflow:hidden;"
    _lbl  = "font-family:Inter,sans-serif;font-size:0.7rem;color:#888888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;"
    _val  = "font-family:Inter,sans-serif;font-size:1.8rem;font-weight:600;color:#ffffff;line-height:1;"
    _sub  = "font-family:Inter,sans-serif;font-size:0.75rem;margin-top:6px;font-weight:600;"
    _acc  = "position:absolute;bottom:0;left:0;right:0;height:2px;"
    st.markdown(f"""
<div style='display:flex;gap:16px;margin-bottom:24px;'>
<div style='{_card}'><div style='{_lbl}'>COMPOSITE SCORE</div><div style='{_val}'>{latest_lt} / {len(lt_ind_cols)}</div><div style='{_sub}color:{_lt_bull_col};'>{_lt_bull_txt}</div><div style='{_acc}background:{_lt_accent};'></div></div>
<div style='{_card}'><div style='{_lbl}'>LAST REGIME CHANGE</div><div style='{_val}'>{lt_regime_str}</div><div style='{_acc}background:#2a2a2a;'></div></div>
<div style='{_card}'><div style='{_lbl}'>LAST UPDATE</div><div style='{_val}'>{lt_update_str}</div><div style='{_acc}background:#2a2a2a;'></div></div>
<div style='{_card}'><div style='{_lbl}'>S&P 500</div><div style='{_val}'>{lt_spx_price:,.2f}</div><div style='{_sub}color:{_lt_chg_col};'>{lt_spx_chg:+.2f}%</div><div style='{_acc}background:#2a2a2a;'></div></div>
</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — Trend Health Model
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 12px 0;">Trend Health Model</p>', unsafe_allow_html=True)
    tf_thm = _tf_widget("tf_mr_thm", "ALL")
    st.plotly_chart(_apply_tf(_chart_thm(df_thm), tf_thm), width='stretch')

    # THM info cards
    thm_comp       = df_thm["Trend_Composite"].dropna().iloc[-1]
    thm_is_bull    = thm_comp > 55
    thm_trend_s    = df_thm["Trend"].dropna()
    thm_chg_dates  = thm_trend_s[thm_trend_s.diff().abs() > 0]
    thm_regime_str = thm_chg_dates.index[-1].strftime("%b %d, %Y") if len(thm_chg_dates) else "—"
    thm_update_str = df_thm.dropna(subset=["Trend_Composite"]).index[-1].strftime("%b %d, %Y")
    thm_spx_s      = df_thm["S&P500"].dropna()
    thm_spx_price  = thm_spx_s.iloc[-1]
    thm_spx_chg    = (thm_spx_s.iloc[-1] / thm_spx_s.iloc[-2] - 1) * 100 if len(thm_spx_s) >= 2 else 0
    _thm_accent    = "#ff6600" if thm_is_bull else "#ff4444"
    _thm_bull_col  = "#00c896" if thm_is_bull else "#ff4444"
    _thm_bull_txt  = "BULL" if thm_is_bull else "BEAR"
    _thm_chg_col   = "#00c896" if thm_spx_chg > 0 else ("#ff4444" if thm_spx_chg < 0 else "#888888")
    st.markdown(f"""
<div style='display:flex;gap:16px;margin-bottom:32px;'>
<div style='{_card}'><div style='{_lbl}'>HEALTH SCORE</div><div style='{_val}'>{thm_comp:.0f}%</div><div style='{_sub}color:{_thm_bull_col};'>{_thm_bull_txt}</div><div style='{_acc}background:{_thm_accent};'></div></div>
<div style='{_card}'><div style='{_lbl}'>LAST REGIME CHANGE</div><div style='{_val}'>{thm_regime_str}</div><div style='{_acc}background:#2a2a2a;'></div></div>
<div style='{_card}'><div style='{_lbl}'>LAST UPDATE</div><div style='{_val}'>{thm_update_str}</div><div style='{_acc}background:#2a2a2a;'></div></div>
<div style='{_card}'><div style='{_lbl}'>S&P 500</div><div style='{_val}'>{thm_spx_price:,.2f}</div><div style='{_sub}color:{_thm_chg_col};'>{thm_spx_chg:+.2f}%</div><div style='{_acc}background:#2a2a2a;'></div></div>
</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    #  INDIVIDUAL INDICATORS
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 12px 0;">Individual Indicators</p>', unsafe_allow_html=True)
    for _i, (title, col, ind_name, ind_col, label, start) in enumerate([
        ("1. OECD CLI Diffusion Index",    "OECD_CLI",       "", "#f9ca24", "OECD CLI Strategy",     "1997-01-01"),
        ("2. Nasdaq 100 Cumulative Hi-Lo", "N100_HiLo",      "", "#4ecdc4", "N100 Hi-Lo Strategy",   "1999-01-01"),
        ("3. Credit Spreads (FRED)",       "Credit_Spreads", "", "#ff6b6b", "Credit Spread Strategy", "1997-01-01"),
    ]):
        with st.expander(title, expanded=False):
            tf = _tf_widget(f"tf_mr_lt_ind_{_i}", "2Y")
            fig, err = _ind_chart(title, ind_df, col, spx_s,
                                   None, ind_name, ind_col, label, start_date=start)
            if fig:
                st.plotly_chart(_apply_tf(fig, tf), width='stretch')
            else:
                st.warning(f"Chart unavailable: {err}")
    for _i, (title, col, ind_col, label, start) in enumerate([
        ("4.  Volatility Regime (MOVE/VIX)",       "Vol_Regime", ORANGE,    "Vol Regime",        "2008-01-01"),
        ("5.  NYSE 52-Week Hi-Lo Trend",            "HiLo_52W",  "#f9ca24", "Hi-Lo Strategy",    "1990-01-01"),
        ("6.  Canary Model",                        "Canary",    "#4ecdc4", "Canary Strategy",   "2003-01-01"),
        ("7.  % Stocks Above 200-Day SMA",          "Pct_200SMA",ORANGE,    "200 SMA Strategy",  "2002-01-01"),
        ("8.  ACWI 200 SMA Breadth",                "ACWI_200",  "#4ecdc4", "ACWI Strategy",     "2007-01-01"),
        ("9.  Cumulative A/D Line",                 "AD_Line",   "#4ecdc4", "A/D Strategy",      "2005-03-01"),
        ("10. In/Out Indicator",                    "InOut",     ORANGE,    "In/Out Strategy",   "2002-01-01"),
        ("11. VIX Term Structure",                  "VIX_TS",    "#f9ca24", "VIX TS Strategy",   "2007-01-01"),
        ("12. HMM Regime Indicator",                "HMM",       LIME,      "HMM Strategy",      "1960-01-01"),
        ("13. Quad 1 & 2",                          "Quad",      "#4ecdc4", "Quad Strategy",     "2014-01-01"),
        ("14. Bitcoin Liquidity Proxy",             "BTC",       "#f9ca24", "BTC Strategy",      "2014-01-01"),
        ("15. SuperTrend Long Term (ATR 252×12)",   "ST_LT",     LIME,      "ST LT Strategy",    "1990-01-01"),
        ("16. SuperTrend Medium Term (ATR 63×9)",   "ST_MT",     LIME,      "ST MT Strategy",    "1990-01-01"),
        ("17. SuperTrend Short Term (ATR 63×5)",    "ST_ST",     LIME,      "ST ST Strategy",    "1990-01-01"),
        ("18. VIX Term Structure × HMM",            "VIX_HMM",   "#ff6b6b", "VIX×HMM Strategy",  "2007-01-01"),
    ]):
        with st.expander(title, expanded=False):
            tf = _tf_widget(f"tf_mr_thm_ind_{_i}", "2Y")
            fig, err = _ind_chart(title, ind_df, col, spx_s,
                                   None, "", ind_col, label, start_date=start)
            if fig:
                st.plotly_chart(_apply_tf(fig, tf), width='stretch')
            else:
                st.warning(f"Chart unavailable: {err}")
