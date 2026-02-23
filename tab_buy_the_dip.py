"""
tab_buy_the_dip.py
------------------
All Buy the Dip charts for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st
import os
import ssl
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BARCHART = os.path.join(BASE_DIR, "datasets", "barchart")
DATASETS = os.path.join(BASE_DIR, "datasets")

# ── Fix SSL for yfinance on Windows corporate networks ──────────────────────────
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    import yfinance.base as _yfbase
    import requests as _req
    _orig_session = _yfbase.TickerBase._get_session if hasattr(_yfbase.TickerBase, '_get_session') else None
except Exception:
    pass

def _download_no_ssl(tickers, **kwargs):
    """yfinance download with SSL verification disabled."""
    import requests
    session = requests.Session()
    session.verify = False
    return yf.download(tickers, session=session, **kwargs)

# ── Professional dark theme constants ───────────────────────────────────────────
BG      = "#0a0a0a"
BG_PLOT = "#0e0e0e"
GRID    = "rgba(255,255,255,0.04)"
WHITE   = "#e8e8e8"
DIM     = "#888888"
ACCENT  = "#00ff88"
ORANGE  = "#ff9f43"
BORDER  = "#222222"

def pro_layout(height=500):
    return dict(
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG,
        height=height,
        margin=dict(l=60, r=40, t=45, b=40),
        font=dict(color=WHITE, family="Inter, Arial, sans-serif"),
        legend=dict(
            orientation="h", x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cccccc", size=11),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(color=DIM),
            linecolor=BORDER,
            rangebreaks=[dict(bounds=["sat", "mon"])],  # remove weekends
        ),
    )

def yax(color=WHITE, log=False, grid=False):
    return dict(
        tickfont=dict(color=DIM),
        title_font=dict(color=color, size=11),
        showgrid=grid,
        gridcolor=GRID,
        zeroline=False,
        linecolor=BORDER,
        type="log" if log else "linear",
    )

# ── Helpers ──────────────────────────────────────────────────────────────────────
def load_bc(filename):
    path = os.path.join(BARCHART, filename)
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df["Last"] = pd.to_numeric(df["Last"].astype(str).str.replace(",", ""), errors="coerce")
    return df


def dual_chart(title, spx, indicator, indicator_name,
               ind_color=ORANGE, signal_dates=None,
               spx_log=False, hline=None, hline_color="rgba(255,80,80,0.5)"):
    """Professional dual-subplot chart: SPX on top, indicator on bottom."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.58, 0.42],
    )

    # SPX line
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx,
        mode="lines", name="S&P 500",
        line=dict(color=WHITE, width=1.4),
        connectgaps=True
    ), row=1, col=1)

    # Indicator line
    fig.add_trace(go.Scatter(
        x=indicator.index, y=indicator,
        mode="lines", name=indicator_name,
        line=dict(color=ind_color, width=1.4)
    ), row=2, col=1)

    # Buy signal markers on SPX panel
    if signal_dates is not None and len(signal_dates) > 0:
        valid = signal_dates[signal_dates.isin(spx.index)]
        if len(valid) > 0:
            fig.add_trace(go.Scatter(
                x=valid, y=spx.loc[valid],
                mode="markers", name="Buy Signal",
                marker=dict(
                    color=ACCENT, size=9, symbol="triangle-up",
                    line=dict(color="#003322", width=1)
                )
            ), row=1, col=1)

    # Threshold line on indicator panel
    if hline is not None:
        fig.add_hline(
            y=hline, line_dash="dot",
            line_color=hline_color, line_width=1.5,
            row=2, col=1
        )

    layout = pro_layout(height=520)
    # Add second xaxis rangebreak for lower panel
    layout["xaxis2"] = dict(
        showgrid=False, zeroline=False,
        tickfont=dict(color=DIM),
        linecolor=BORDER,
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )
    fig.update_layout(title=dict(text=title, font=dict(color=WHITE, size=13)), **layout)

    fig.update_yaxes(title_text="S&P 500", **yax(WHITE, log=spx_log, grid=True), row=1, col=1)
    fig.update_yaxes(title_text=indicator_name, **yax(ind_color), row=2, col=1)

    return fig


# ── Compute signal utility ───────────────────────────────────────────────────────
def compute_signal(df, condition):
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


# ── Individual chart functions ───────────────────────────────────────────────────
def chart_1_r3fd(spx):
    r3fd = load_bc("Russell_3000_Stocks_Above_5-Day_Average_$R3FD.csv")
    r3fd = r3fd[r3fd.index >= "1999-01-04"]
    spx_a = spx.reindex(r3fd.index, method="ffill")
    signals = r3fd[r3fd["Last"] < 10].index
    return dual_chart("S&P 500 vs % Russell 3000 Stocks Above 5-Day MA",
                      spx_a["Last"], r3fd["Last"], "% Above 5DMA",
                      signal_dates=signals, hline=10)


@st.cache_data(ttl=3600)
def get_acwi_data():
    path = os.path.join(DATASETS, "acwi_oscillator.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    return df["Percentage"]


def chart_2_acwi(spx):
    with st.spinner("Loading ACWI ETF data..."):
        oscillator = get_acwi_data()
    spx_a = spx.reindex(oscillator.index, method="ffill")
    signals = oscillator[oscillator == 0].index
    return dual_chart("S&P 500 vs ACWI ETF Oscillator (% Above 10-Day MA)",
                      spx_a["Last"], oscillator, "ACWI Oscillator (%)",
                      signal_dates=signals)


def chart_3_mcclellan(spx):
    adv = load_bc("NYSE_Advancing_Stocks_$NSHU.csv")
    dec = load_bc("NYSE_Declining_Stocks_$NSHD.csv")
    rana = (adv["Last"] - dec["Last"]) / (adv["Last"] + dec["Last"]) * 1000
    mco = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    mco.index = pd.to_datetime(mco.index)
    spx_a = spx.reindex(mco.index, method="ffill")
    signals = mco.index[(mco.shift(1) >= -80) & (mco < -80)]
    return dual_chart("S&P 500 vs McClellan Oscillator",
                      spx_a["Last"], mco, "McClellan Oscillator",
                      signal_dates=signals, spx_log=True, hline=-80)


def chart_4_putcall(spx):
    pc = load_bc("Equity_PutCall_Ratio_$CPCS.csv")
    pc["5D_SMA"] = pc["Last"].rolling(5).mean()
    rm = pc["5D_SMA"].rolling(52).mean()
    rs = pc["5D_SMA"].rolling(52).std()
    pc["Zscore"] = (rm - pc["5D_SMA"]) / rs
    spx_a = spx.reindex(pc.index, method="ffill")
    signals = pc[pc["Zscore"] < -2.5].index
    return dual_chart("S&P 500 vs Equity Put/Call Ratio Z-Score",
                      spx_a["Last"], pc["Zscore"], "Put/Call Z-Score",
                      signal_dates=signals, hline=-2.5)


def chart_5_cnn(spx):
    path = os.path.join(DATASETS, "cnn_fear_greed.csv")
    fg = pd.read_csv(path, parse_dates=True, index_col=0)
    fg.index = pd.to_datetime(fg.index, errors="coerce")
    fg = fg[fg.index.notna()].sort_index()
    spx_a = spx.reindex(fg.index, method="ffill")
    signals = fg[fg["Fear_Greed"] < 25].index
    return dual_chart("S&P 500 vs CNN Fear & Greed Index",
                      spx_a["Last"], fg["Fear_Greed"], "Fear & Greed Index",
                      signal_dates=signals, hline=25)


def chart_6_lowry(spx):
    adv_s = load_bc("NYSE_Advancing_Stocks_$NSHU.csv")
    dec_s = load_bc("NYSE_Declining_Stocks_$NSHD.csv")
    adv_v = load_bc("NYSE_Advancing_Volume_$NVLU.csv")
    dec_v = load_bc("NYSE_Declining_Volume_$DVCN.csv")
    total_s = adv_s["Last"] + dec_s["Last"]
    total_v = adv_v["Last"] + dec_v["Last"]
    dec_pct_s = (dec_s["Last"] / total_s) * 100
    dec_pct_v = (dec_v["Last"] / total_v) * 100
    cdf = pd.DataFrame({
        "dec_s_90": (dec_pct_s >= 90).astype(int),
        "dec_s_80": ((dec_pct_s >= 80) & (dec_pct_s < 90)).astype(int),
        "dec_v_90": (dec_pct_v >= 90).astype(int),
        "dec_v_80": ((dec_pct_v >= 80) & (dec_pct_v < 90)).astype(int),
    })
    cdf["roll6"] = cdf.sum(axis=1).rolling(6).sum()
    cdf = cdf[cdf.index >= "1999-01-04"]
    spx_a = spx.reindex(cdf.index, method="ffill")
    signals = cdf.index[(cdf["roll6"] >= 4) & (cdf["roll6"].shift(1) < 4)]
    return dual_chart("S&P 500 vs Lowry Panic Indicator",
                      spx_a["Last"], cdf["roll6"], "Lowry Panic (6-Day Sum)",
                      signal_dates=signals, hline=4)


def chart_7_zweig(spx):
    adv = load_bc("NYSE_Advancing_Stocks_$NSHU.csv")
    dec = load_bc("NYSE_Declining_Stocks_$NSHD.csv")
    zweig = (adv["Last"] / (adv["Last"] + dec["Last"])).ewm(span=10, adjust=False).mean()
    zweig.index = pd.to_datetime(zweig.index)
    spx_a = spx.reindex(zweig.index, method="ffill")
    signals = zweig.index[(zweig.shift(1) > 0.35) & (zweig <= 0.35)]
    return dual_chart("S&P 500 vs Zweig Breadth Indicator",
                      spx_a["Last"], zweig, "Zweig Breadth",
                      signal_dates=signals, spx_log=True, hline=0.35)


def chart_8_volcurve(spx):
    vix = load_bc("CBOE_Volatility_Index_$VIX.csv")
    vxv = load_bc("CBOE_3-Month_VIX_$VXV.csv")
    vc = (vxv["Last"] / vix["Last"] - 1).rename("Volatility Curve")
    vc.index = pd.to_datetime(vc.index)
    spx_a = spx.reindex(vc.index, method="ffill")
    signals = vc.index[(vc >= 0) & (vc.shift(1) < 0)]
    return dual_chart("S&P 500 vs Volatility Curve (VXV/VIX - 1)",
                      spx_a["Last"], vc, "Volatility Curve",
                      signal_dates=signals, hline=0)


def chart_9_52wh(spx):
    highs = load_bc("S&P_500_52-Week_Highs_$MAHP.csv")
    highs = highs[highs.index >= "2000-01-01"]
    spx_a = spx.reindex(highs.index, method="ffill")
    signals = highs[highs["Last"] < 1].index
    return dual_chart("S&P 500 vs S&P 500 52-Week New Highs",
                      spx_a["Last"], highs["Last"], "S&P 500 52-Week Highs",
                      signal_dates=signals, hline=1)


# ── Composite builder ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def build_composite():
    spx   = load_bc("S&P_500_Index_$SPX.csv")
    r3fd  = load_bc("Russell_3000_Stocks_Above_5-Day_Average_$R3FD.csv")
    adv   = load_bc("NYSE_Advancing_Stocks_$NSHU.csv")
    dec   = load_bc("NYSE_Declining_Stocks_$NSHD.csv")
    adv_v = load_bc("NYSE_Advancing_Volume_$NVLU.csv")
    dec_v = load_bc("NYSE_Declining_Volume_$DVCN.csv")
    pc    = load_bc("Equity_PutCall_Ratio_$CPCS.csv")
    vix   = load_bc("CBOE_Volatility_Index_$VIX.csv")
    vxv   = load_bc("CBOE_3-Month_VIX_$VXV.csv")
    highs = load_bc("S&P_500_52-Week_Highs_$MAHP.csv")
    fg_path = os.path.join(DATASETS, "cnn_fear_greed.csv")
    fg = pd.read_csv(fg_path, parse_dates=True, index_col=0)
    fg.index = pd.to_datetime(fg.index, errors="coerce")
    fg = fg[fg.index.notna()].sort_index()

    rana = (adv["Last"] - dec["Last"]) / (adv["Last"] + dec["Last"]) * 1000
    mco  = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    pc["5D_SMA"] = pc["Last"].rolling(5).mean()
    pc["Zscore"] = (pc["5D_SMA"].rolling(52).mean() - pc["5D_SMA"]) / pc["5D_SMA"].rolling(52).std()
    zweig = (adv["Last"] / (adv["Last"] + dec["Last"])).ewm(span=10, adjust=False).mean()
    total_s   = adv["Last"] + dec["Last"]
    total_v   = adv_v["Last"] + dec_v["Last"]
    dec_pct_s = (dec["Last"] / total_s) * 100
    dec_pct_v = (dec_v["Last"] / total_v) * 100
    lowry = ((dec_pct_s >= 90).astype(int) +
             ((dec_pct_s >= 80) & (dec_pct_s < 90)).astype(int) +
             (dec_pct_v >= 90).astype(int) +
             ((dec_pct_v >= 80) & (dec_pct_v < 90)).astype(int)).rolling(6).sum()
    vc = vxv["Last"] / vix["Last"] - 1

    # ACWI — use cached version with SSL fix
    acwi = get_acwi_data()

    series = {
        "Percent Above 5DMA":   r3fd["Last"],
        "ACWI Oscillator":      acwi,
        "McClellan Oscillator": mco,
        "Equity PC Zscore":     pc["Zscore"],
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
        df[f"{name} Signal"] = compute_signal(df, cond)

    sig_cols = [f"{n} Signal" for n in conditions]
    df["Composite"] = df[sig_cols].sum(axis=1)
    df["Composite MA2"] = df["Composite"].rolling(2).mean()
    return df


# ── Composite chart ──────────────────────────────────────────────────────────────
def chart_composite(df):
    trigger = df.index[(df["Composite"] > 2) & (df["Composite"] > df["Composite MA2"])]

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df.index, y=df["S&P500"],
        mode="lines", name="S&P 500",
        line=dict(color=WHITE, width=1.5),
        connectgaps=True
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Composite"],
        mode="none", name="Signals Active",
        fill="tozeroy",
        fillcolor="rgba(0, 200, 80, 0.18)",
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=trigger, y=df.loc[trigger, "S&P500"],
        mode="markers", name="Buy Trigger",
        marker=dict(
            color=ACCENT, size=9, symbol="triangle-up",
            line=dict(color="#003322", width=1)
        )
    ), row=1, col=1, secondary_y=False)

    layout = pro_layout(height=560)
    fig.update_layout(
        title=dict(text="Buy The Dip — Composite Signal (9 Indicators)",
                   font=dict(color=WHITE, size=13)),
        **layout
    )

    fig.update_yaxes(
        title_text="S&P 500  (log)",
        title_font=dict(color=DIM, size=11),
        tickfont=dict(color=DIM),
        showgrid=True, gridcolor=GRID,
        zeroline=False, type="log",
        linecolor=BORDER,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Signals Active  (0–9)",
        title_font=dict(color="#00cc66", size=11),
        tickfont=dict(color="#00cc66"),
        showgrid=False, zeroline=False,
        range=[0, 11], fixedrange=True,
        linecolor=BORDER,
        secondary_y=True,
    )
    for lvl in [3, 6]:
        fig.add_hline(y=lvl, line_dash="dot",
                      line_color="rgba(0,200,80,0.25)",
                      line_width=1, secondary_y=True)
    return fig


# ── Main render ──────────────────────────────────────────────────────────────────
def render():
    st.subheader("🎯 Buy the Dip — Oversold Signals")
    st.caption("🟢 Lime triangles = buy signal triggered  |  Charts exclude weekends & holidays")

    spx = load_bc("S&P_500_Index_$SPX.csv")

    st.markdown("### 🏆 Composite Signal — All 9 Indicators")
    with st.spinner("Building composite signal (downloading ETF data first time)..."):
        df_comp = build_composite()
    st.plotly_chart(chart_composite(df_comp), use_container_width=True)

    latest = int(df_comp["Composite"].iloc[-1])
    col1, col2, col3 = st.columns(3)
    col1.metric("Signals Active Today", f"{latest} / 9",
                delta="🔥 Elevated" if latest >= 3 else ("⚠️ Moderate" if latest >= 1 else "Normal"))
    active_dates = df_comp.index[df_comp["Composite"] > 0]
    col2.metric("Last Signal Date", str(active_dates[-1].date()) if len(active_dates) > 0 else "N/A")
    trigger_dates = df_comp.index[(df_comp["Composite"] > 2) & (df_comp["Composite"] > df_comp["Composite MA2"])]
    col3.metric("Last Trigger Date", str(trigger_dates[-1].date()) if len(trigger_dates) > 0 else "N/A")

    st.divider()
    st.markdown("### Individual Indicators")

    charts = [
        ("1. % Russell 3000 Above 5-Day MA",       lambda: chart_1_r3fd(spx)),
        ("2. ACWI ETF Oscillator (% Above 10DMA)", lambda: chart_2_acwi(spx)),
        ("3. McClellan Oscillator",                 lambda: chart_3_mcclellan(spx)),
        ("4. Equity Put/Call Ratio Z-Score",        lambda: chart_4_putcall(spx)),
        ("5. CNN Fear & Greed Index",               lambda: chart_5_cnn(spx)),
        ("6. Lowry Panic Indicator",                lambda: chart_6_lowry(spx)),
        ("7. Zweig Breadth Indicator",              lambda: chart_7_zweig(spx)),
        ("8. Volatility Curve (VXV/VIX)",           lambda: chart_8_volcurve(spx)),
        ("9. S&P 500 52-Week New Highs",            lambda: chart_9_52wh(spx)),
    ]
    for title, fn in charts:
        with st.expander(title, expanded=False):
            try:
                st.plotly_chart(fn(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not render chart: {e}")
