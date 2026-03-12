import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmallFish Macro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Dark theme styling ─────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Apply font only to text-bearing elements, NOT icon elements ── */
body, .stApp,
p, li, td, th, label, caption,
h1, h2, h3, h4, h5, h6,
.stMarkdown p, .stMarkdown li,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] > div,
[data-testid="stCaptionContainer"],
.stTabs [data-baseweb="tab"],
input, select, textarea,
div[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em;
}

/* ── Background ── */
.stApp { background-color: #0a0a0a; color: #cccccc; }

/* ── Headings orange ── */
h1, h2, h3 { color: #ff6600 !important; text-transform: uppercase; letter-spacing: 0.12em; }
.stSubheader > div > div { color: #ff6600 !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background-color: #111111; border-bottom: 1px solid #222; }
.stTabs [data-baseweb="tab"] { color: #555555; font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; }
.stTabs [aria-selected="true"] { color: #ff6600 !important; border-bottom: 2px solid #ff6600 !important; }

/* ── Metrics ── */
div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; color: #ffffff; }
div[data-testid="stMetricLabel"] { color: #666666; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.14em; }
div[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* ── Expanders — text only, arrow untouched ── */
.stExpander { border: 1px solid #1e1e1e !important; background-color: #111111 !important; }
details summary p { color: #aaaaaa; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── Subtle refresh button ── */
div[data-testid="stButton"] > button {
    background-color: #1a1a1a !important;
    color: #444444 !important;
    border: 1px solid #2a2a2a !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    padding: 1px 8px !important;
    border-radius: 3px !important;
    box-shadow: none !important;
    text-transform: uppercase;
}
div[data-testid="stButton"] > button:hover {
    background-color: #222222 !important; color: #888888 !important; border-color: #444444 !important;
}

/* ── Misc ── */
hr { border-color: #222222; }
.stAlert { font-size: 0.78rem; }
.stCaption { color: #444444 !important; font-size: 0.68rem; text-transform: uppercase; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>""", unsafe_allow_html=True)

# ── Paths — works on Colab and local ──────────────────────────────────────────
_COLAB_DRIVE = "/content/drive/MyDrive/Python"
_LOCAL_DIR   = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(_COLAB_DRIVE):
    BASE_DIR = _COLAB_DRIVE
else:
    BASE_DIR = _LOCAL_DIR

BARCHART = os.path.join(BASE_DIR, "data", "barchart")
DATASETS = os.path.join(BASE_DIR, "data", "datasets")

# ── Helper: load a barchart CSV ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_bc(filename):
    path = os.path.join(BARCHART, filename)
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df["Last"] = pd.to_numeric(df["Last"], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_btd_signals():
    path = os.path.join(DATASETS, "btd_signals.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df[df.index.notna()].sort_index()

def last_val(df): return df["Last"].dropna().iloc[-1]
def delta(df):
    s = df["Last"].dropna()
    return s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else 0

# ── Header ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;padding:14px 0 10px 0;'>
  <span style='font-family:Inter,sans-serif;font-weight:700;font-size:1.6rem;color:#ff6600;text-transform:uppercase;letter-spacing:0.06em;'>SMALLFISH MACRO</span>
  <span style='font-family:Inter,sans-serif;font-weight:300;font-size:1.6rem;color:#333333;margin:0 16px;'>|</span>
  <span style='font-family:Inter,sans-serif;font-weight:300;font-size:1.6rem;color:#666666;text-transform:uppercase;letter-spacing:0.06em;'>MARKET TERMINAL</span>
</div>
""", unsafe_allow_html=True)

# ── Refresh button ─────────────────────────────────────────────────────────────
if st.button("↻  Refresh", type="primary"):
    import data_updater
    total = len(data_updater.SYMBOLS)
    log_lines = []
    progress_bar = st.progress(0, text="Starting update...")
    status = st.empty()
    def log_fn(msg):
        log_lines.append(msg)
        completed = sum(1 for l in log_lines)
        if completed > 0 and completed <= total:
            pct = int((completed / total) * 100)
            progress_bar.progress(pct, text=f"Updating... {completed}/{total} ({pct}%)")
            status.caption(msg)
    success, failed = data_updater.run_update(log_fn=log_fn)
    progress_bar.progress(100, text="Update complete!")
    status.empty()
    st.success(f"{success} files updated, {failed} skipped.")
    with st.expander("View update log"):
        st.code("\n".join(log_lines))
    st.session_state.pop("btd_composite", None)
    st.cache_data.clear()
    st.rerun()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
# Their render() is only called when the user is on that tab.
# This prevents WebSocket timeouts caused by running all tabs simultaneously.

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Market Breadth",
    "Volatility",
    "Market Risk",
    "Macro / Regime",
    "Buy the Dip",
    "Strategy Map",
])

# ── TAB 1: OVERVIEW ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:8px 0 12px 0;">Market Snapshot</p>', unsafe_allow_html=True)
    try:
        spx  = load_bc("S&P_500_Index_$SPX.csv")
        vix  = load_bc("CBOE_Volatility_Index_$VIX.csv")
        mmfi = load_bc("Percent_of_Stocks_Above_50-Day_Average_$MMFI.csv")
        mmth = load_bc("Percent_of_Stocks_Above_200-Day Average_$MMTH.csv")

        col1, col2, col3, col4 = st.columns(4)
        _snap_items = [
            (col1, "S&P 500",         f"{last_val(spx):,.0f}",  delta(spx)),
            (col2, "VIX",             f"{last_val(vix):.1f}",   delta(vix)),
            (col3, "% Above 50D MA",  f"{last_val(mmfi):.1f}%", delta(mmfi)),
            (col4, "% Above 200D MA", f"{last_val(mmth):.1f}%", delta(mmth)),
        ]
        for _scol, _slbl, _sval, _sd in _snap_items:
            _sdc = "#00c896" if _sd > 0 else ("#ff4444" if _sd < 0 else "#888888")
            _sds = f"+{_sd:.1f}" if _sd > 0 else f"{_sd:.1f}"
            with _scol:
                st.markdown(f"<div style='background:#141414;border-radius:12px;border:1px solid #2a2a2a;border-bottom:2px solid #ff6600;padding:20px 24px;'><div style='font-size:0.7rem;font-weight:500;color:#888888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>{_slbl}</div><div style='font-size:1.8rem;font-weight:600;color:#ffffff;line-height:1.2;'>{_sval}</div><div style='font-size:0.75rem;color:{_sdc};margin-top:6px;font-weight:500;'>{_sds}</div></div>", unsafe_allow_html=True)

        st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:20px 0 12px 0;">S&P 500 — 2 Years</p>', unsafe_allow_html=True)
        spx2 = spx.last("730D")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spx2.index, y=spx2["Last"],
                                 mode="lines", name="SPX",
                                 line=dict(color="#ff6600", width=2)))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0a0a0a",
                          paper_bgcolor="#0a0a0a", height=400,
                          margin=dict(l=0, r=0, t=20, b=0),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig, width='stretch')

        # ── Regime Scores from Market Risk CSVs ──────────────────────────────
        st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:20px 0 12px 0;">Regime Scores</p>', unsafe_allow_html=True)
        try:
            import os, pandas as pd
            _base = "/content/drive/MyDrive/Python" if os.path.exists("/content/drive/MyDrive/Python") else os.path.dirname(os.path.abspath(__file__))
            _ds   = os.path.join(_base, "data", "datasets")
            _lt   = pd.read_csv(os.path.join(_ds, "market_risk_lt_composite.csv"),  index_col=0, parse_dates=True)
            _thm  = pd.read_csv(os.path.join(_ds, "market_risk_health_model.csv"),   index_col=0, parse_dates=True)

            _lt_score   = int(_lt["Composite"].dropna().iloc[-1])
            _lt_bull    = _lt["Trend"].dropna().iloc[-1] == 1
            _thm_score  = float(_thm["Trend_Composite"].dropna().iloc[-1])
            _thm_bull   = _thm["Trend"].dropna().iloc[-1] == 1

            rc1, rc2, rc3 = st.columns(3)
            _lt_accent  = "#ff6600" if _lt_bull  else "#ff4444"
            _lt_sub_c   = "#00c896" if _lt_bull  else "#ff4444"
            _lt_sub     = "BULL" if _lt_bull  else "BEAR"
            _thm_accent = "#ff6600" if _thm_bull else "#ff4444"
            _thm_sub_c  = "#00c896" if _thm_bull else "#ff4444"
            _thm_sub    = "BULL" if _thm_bull else "BEAR"
            with rc1:
                st.markdown(f"<div style='background:#141414;border-radius:12px;border:1px solid #2a2a2a;border-bottom:2px solid {_lt_accent};padding:20px 24px;'><div style='font-size:0.7rem;font-weight:500;color:#888888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>LT Score</div><div style='font-size:1.8rem;font-weight:600;color:#ffffff;line-height:1.2;'>{_lt_score} / 3</div><div style='font-size:0.75rem;color:{_lt_sub_c};margin-top:6px;font-weight:500;'>{_lt_sub}</div></div>", unsafe_allow_html=True)
            with rc2:
                st.markdown(f"<div style='background:#141414;border-radius:12px;border:1px solid #2a2a2a;border-bottom:2px solid {_thm_accent};padding:20px 24px;'><div style='font-size:0.7rem;font-weight:500;color:#888888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>Health Score</div><div style='font-size:1.8rem;font-weight:600;color:#ffffff;line-height:1.2;'>{_thm_score:.0f}%</div><div style='font-size:0.75rem;color:{_thm_sub_c};margin-top:6px;font-weight:500;'>{_thm_sub}</div></div>", unsafe_allow_html=True)
        except Exception as _re:
            st.warning(f"Regime scores unavailable: {_re}")

        # ── BTD Score from pre-computed btd_signals.csv ───────────────────────
        try:
            _df_btd = load_btd_signals()
            if _df_btd is not None and "Composite" in _df_btd.columns:
                _btd_now    = int(_df_btd.dropna(how='all')["Composite"].iloc[-1])
                _btd_accent = "#ff6600" if _btd_now >= 3 else ("#ff6600" if _btd_now >= 1 else "#2a2a2a")
                _btd_sub_c  = "#00c896" if _btd_now >= 3 else ("#888888" if _btd_now >= 1 else "#555555")
                _btd_sub    = "ELEVATED" if _btd_now >= 3 else ("MODERATE" if _btd_now >= 1 else "NORMAL")
                with rc3:
                    st.markdown(f"<div style='background:#141414;border-radius:12px;border:1px solid #2a2a2a;border-bottom:2px solid {_btd_accent};padding:20px 24px;'><div style='font-size:0.7rem;font-weight:500;color:#888888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>BTD Score</div><div style='font-size:1.8rem;font-weight:600;color:#ffffff;line-height:1.2;'>{_btd_now} / 9</div><div style='font-size:0.75rem;color:{_btd_sub_c};margin-top:6px;font-weight:500;'>{_btd_sub}</div></div>", unsafe_allow_html=True)
            else:
                with rc3:
                    st.markdown("<div style='background:#141414;border-radius:12px;border:1px solid #2a2a2a;border-bottom:2px solid #2a2a2a;padding:20px 24px;'><div style='font-size:0.7rem;font-weight:500;color:#888888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>BTD Score</div><div style='font-size:1.8rem;font-weight:600;color:#555555;line-height:1.2;'>— / 9</div><div style='font-size:0.75rem;color:#555555;margin-top:6px;'>No data</div></div>", unsafe_allow_html=True)
        except Exception:
            pass

        # ── Market Risk Indicators Grid ───────────────────────────────────────
        st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:20px 0 12px 0;">Market Risk Indicators</p>', unsafe_allow_html=True)
        try:
            _ind = pd.read_csv(os.path.join(DATASETS, "market_risk_indicators.csv"), index_col=0, parse_dates=True)
            _ind = _ind[_ind.index.notna()].sort_index()
            _ind_labels = {
                "OECD_CLI":       "OECD CLI",
                "N100_HiLo":      "NDX Hi-Lo",
                "Credit_Spreads": "Credit Spreads",
                "Vol_Regime":     "Vol Regime",
                "HiLo_52W":       "52W Hi-Lo",
                "Canary":         "Canary",
                "Pct_200SMA":     "% Above 200D",
                "ACWI_200":       "ACWI 200",
                "AD_Line":        "A/D Line",
                "InOut":          "In/Out",
                "VIX_TS":         "VIX Term Str.",
                "HMM":            "HMM",
                "Quad":           "Quad",
                "BTC":            "Bitcoin",
                "ST_LT":          "ST Long",
                "ST_MT":          "ST Medium",
                "ST_ST":          "ST Short",
                "VIX_HMM":        "VIX x HMM",
            }
            _ind_cols = [c for c in _ind.columns if c != "S&P500"]
            for _i in range(0, len(_ind_cols), 6):
                if _i > 0:
                    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                _row = st.columns(6)
                for _j, _cn in enumerate(_ind_cols[_i:_i+6]):
                    _s = _ind[_cn].dropna()
                    _v = int(_s.iloc[-1]) if len(_s) else None
                    _lbl = _ind_labels.get(_cn, _cn)
                    with _row[_j]:
                        if _v == 1:
                            st.markdown(f"<div style='background:#141414;border:1px solid #00c896;border-radius:10px;padding:16px 8px;text-align:center;margin-bottom:10px;'><div style='font-size:2rem;color:#00c896;font-weight:bold;line-height:1.2;'>&#10003;</div><div style='font-size:0.65rem;color:#00c896;font-weight:600;letter-spacing:0.1em;margin-top:6px;text-transform:uppercase;'>BULL</div><div style='font-size:0.7rem;color:#666666;margin-top:4px;'>{_lbl}</div></div>", unsafe_allow_html=True)
                        elif _v == 0:
                            st.markdown(f"<div style='background:#141414;border:1px solid #ff4444;border-radius:10px;padding:16px 8px;text-align:center;margin-bottom:10px;'><div style='font-size:2rem;color:#ff4444;font-weight:bold;line-height:1.2;'>&#10007;</div><div style='font-size:0.65rem;color:#ff4444;font-weight:600;letter-spacing:0.1em;margin-top:6px;text-transform:uppercase;'>BEAR</div><div style='font-size:0.7rem;color:#666666;margin-top:4px;'>{_lbl}</div></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:16px 8px;text-align:center;margin-bottom:10px;'><div style='font-size:2rem;color:#444;line-height:1.2;'>—</div><div style='font-size:0.65rem;color:#444;letter-spacing:0.1em;'>N/A</div><div style='font-size:0.7rem;color:#666666;margin-top:4px;'>{_lbl}</div></div>", unsafe_allow_html=True)
        except Exception as _e:
            st.warning(f"Market risk indicators unavailable: {_e}")

        # ── Buy The Dip Signals Grid ──────────────────────────────────────────
        st.markdown('<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:600;color:#ff6600;text-transform:uppercase;letter-spacing:0.12em;margin:20px 0 12px 0;">Buy The Dip Signals</p>', unsafe_allow_html=True)
        _btd_map = [
            ("R3000_5D",   "% R3000 > 5D MA"),
            ("ACWI_Osc",   "ACWI Oscillator"),
            ("AD_Ratio",   "A/D Ratio"),
            ("PutCall",    "Put/Call Ratio"),
            ("FearGreed",  "Fear & Greed"),
            ("Lowry",      "Lowry Panic"),
            ("Zweig",      "Zweig Breadth"),
            ("VIX_TS",     "VIX Term Structure"),
            ("SP500_Highs","S&P 52W Highs"),
        ]
        _df_btd2 = load_btd_signals()
        if _df_btd2 is not None:
            _btd_valid = _df_btd2.dropna(how='all')
            _btd_last  = _btd_valid.iloc[-1] if len(_btd_valid) > 0 else None
            for _i in range(0, len(_btd_map), 6):
                if _i > 0:
                    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                _row = st.columns(6)
                for _j, (_sc, _lbl) in enumerate(_btd_map[_i:_i+6]):
                    _v = (int(_btd_last[_sc]) if _btd_last is not None
                          and _sc in _btd_last.index
                          and pd.notna(_btd_last[_sc]) else None)
                    with _row[_j]:
                        if _v == 1:
                            st.markdown(f"<div style='background:#141414;border:1px solid #00c896;border-radius:10px;padding:16px 8px;text-align:center;margin-bottom:10px;'><div style='font-size:2rem;color:#00c896;font-weight:bold;line-height:1.2;'>&#10003;</div><div style='font-size:0.65rem;color:#00c896;font-weight:600;letter-spacing:0.1em;margin-top:6px;text-transform:uppercase;'>SIGNAL</div><div style='font-size:0.7rem;color:#666666;margin-top:4px;'>{_lbl}</div></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background:#141414;border:1px solid #2a2a2a;border-radius:10px;padding:16px 8px;text-align:center;margin-bottom:10px;'><div style='font-size:2rem;color:#444;line-height:1.2;'>—</div><div style='font-size:0.65rem;color:#555;letter-spacing:0.1em;text-transform:uppercase;'>IDLE</div><div style='font-size:0.7rem;color:#666666;margin-top:4px;'>{_lbl}</div></div>", unsafe_allow_html=True)
        else:
            st.caption("Run data_updater.py to generate BTD signals.")

    except Exception as e:
        st.error(f"Could not load overview data: {e}")

# ── TAB 2: MARKET BREADTH ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Market Breadth")
    st.info("Charts coming soon.")

# ── TAB 3: VOLATILITY ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Volatility")
    st.info("Charts coming soon.")

# ── TAB 4: MARKET RISK ────────────────────────────────────────────────────────
with tab4:
    import tab_market_risk
    tab_market_risk.render()

# ── TAB 5: MACRO / REGIME ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Macro / Market Regime")
    st.info("Charts coming soon.")

# ── TAB 6: BUY THE DIP (lazy loaded) ─────────────────────────────────────────
with tab6:
    import tab_buy_the_dip
    tab_buy_the_dip.render()

# ── TAB 7: STRATEGY MAP ────────────────────────────────────────────────────────
with tab7:
    strategy_map_html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8"/>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.9/babel.min.js"></script>
      <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:#070710; }
        button { cursor:pointer; outline:none; }
      </style>
    </head>
    <body>
      <div id="root"></div>
      <script type="text/babel">
        const { useState } = React;

        const STRATEGIES = [
          { group:"Select Smallcaps", name:"Susanno", method:"Classic", norm:"Rank", region:"US", weight:20, system:"BenjAImin SmallMicro v4", buy:["sector \u2260 FINANCIAL","AvgDailyTot(20) < 1M"], sell:["Full Rebal"], sell2:[] },
          { group:"Select Smallcaps", name:"Red Hawk", method:"AI - FND", norm:"Rank", region:"US", weight:20, system:"FND_3MTot_190F_Rank_US_LGBM", buy:["MedianDailyTot(20) > 25K","Price > 1"], sell:["Rank < 98.5"], sell2:[] },
          { group:"Select Smallcaps", name:"Kong Gun", method:"AI - FND", norm:"Zscore", region:"US", weight:20, system:"FND_3MTot_190F_Zscore_US_LGBM", buy:["MedianDailyTot(20) > 25K","Price > 1","AvgDailyTot(20) < 1M"], sell:["Rank < 98"], sell2:[] },
          { group:"Select Smallcaps", name:"Asura", method:"AI - FND", norm:"Rank", region:"North America", weight:20, system:"FND_3MTot_191F_Rank_NAM_LGBM", buy:["MedianDailyTot(20) > 25K","Price > 1",'CurQEPSMean >= CurQEPS13WkAgo OR FRank(\\"ActualGr%PQ(#EPS)\\",#all,#DESC,#ExclNA) > 80'], sell:["Rank < 97"], sell2:[] },
          { group:"Select Smallcaps", name:"Nitoryu", method:"AI - FND", norm:"Zscore", region:"North Atlantic", weight:20, system:"FND_3MTot_191F_Zscore_NATL_LGBM", buy:["MedianDailyTot(20) > 50K","Price > 0.5","AvgDailyTot(20) < 1M"], sell:["RankPos > 30"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Amaterasu", method:"Classic", norm:"Rank", region:"US", weight:10, system:"BenjAImin SmallMicro v4", buy:["sector \u2260 FINANCIAL","AvgDailyTot(20) < 1M"], sell:["Full Rebal"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Gear Fifth", method:"AI - FND", norm:"Rank", region:"US", weight:10, system:"FND_3MTot_190F_Rank_US_LGBM", buy:["MedianDailyTot(20) > 25K","Price > 1"], sell:["Rank < 99"], sell2:[] },
          { group:"Alpha Smallcaps", name:"King Cobra", method:"AI - FND", norm:"Zscore", region:"US", weight:10, system:"FND_3MTot_190F_Zscore_US_LGBM", buy:["MedianDailyTot(20) > 25K","Price > 1","AvgDailyTot(20) < 1M"], sell:["Rank < 99"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Jigoku", method:"AI - FND", norm:"Rank", region:"North America", weight:10, system:"FND_3MTot_191F_Rank_NAM_LGBM", buy:["MedianDailyTot(20) > 50K","Price > 1",'CurQEPSMean >= CurQEPS13WkAgo OR FRank(\\"ActualGr%PQ(#EPS)\\",#all,#DESC,#ExclNA) > 80'], sell:["Rank < 98"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Santoryu", method:"AI - FND", norm:"Zscore", region:"North Atlantic", weight:10, system:"FND_3MTot_191F_Zscore_NATL_LGBM", buy:["MedianDailyTot(20) > 25K","Price > 0.5","AvgDailyTot(20) < 1M"], sell:["RankPos > 20"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Rengoku", method:"AI - SM", norm:"Rank", region:"US", weight:10, system:"SM_3MRel_179F_Rank_US_LGBM", buy:["MedianDailyTot(20) > 50K","Price > 1","ActualGr%PQ(#EPS) > 0","ActualGr%PQ(#EPS) > 0"], sell:["Rank < 97"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Shiranui", method:"AI - SM", norm:"Zscore", region:"US", weight:10, system:"SM_3MRel_179F_Zscore_US_LGBM", buy:["MedianDailyTot(20) > 50K","Rank > 99","CurQEPSMean > CurQEPS1WkAgo","CurQEPSMean > CurQEPS1WkAgo"], sell:["Rank < 98"], sell2:["CurQEPSMean < CurQEPS1WkAgo"] },
          { group:"Alpha Smallcaps", name:"Nobori", method:"AI - SM", norm:"Rank", region:"US", weight:10, system:"SM_3MRel_179F_Rank_US_LGBM", buy:["MedianDailyTot(20) > 50K","Price > 1","Rank > 99","%(CurQEPSMean, CurQEPS1WkAgo) > 0"], sell:["Rank < 99"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Mizu", method:"AI - SM", norm:"Zscore", region:"US", weight:10, system:"SM_3MTot_179F_Zscore_US_LGBM", buy:["MedianDailyTot(20) > 50K","Price > 1","Rank > 99","CurQEPSMean > CurQEPS1WkAgo","ActualGr%PQ(#EPS) > 0"], sell:["Rank < 99"], sell2:[] },
          { group:"Alpha Smallcaps", name:"Nagi", method:"AI - SM", norm:"Rank", region:"US", weight:10, system:"SM_3MTot_179F_Rank_US_LGBM", buy:["MedianDailyTot(20) > 50K","Price > 1","Rank > 99","CurQEPSMean > CurQEPS1WkAgo"], sell:["Rank < 99"], sell2:[] },
        ];

        const RC = { "US":"#f97316", "North America":"#3b82f6", "North Atlantic":"#a855f7" };
        const NC = { "Rank":"#22d3ee", "Zscore":"#86efac" };
        const MC = { "Classic":"#fbbf24", "AI - FND":"#38bdf8", "AI - SM":"#f472b6" };

        function Tag({ label, color }) {
          return <span style={{ background:color+"22", color, border:"1px solid "+color+"55", borderRadius:3, padding:"2px 8px", fontSize:12, fontFamily:"monospace", fontWeight:700, whiteSpace:"nowrap" }}>{label}</span>;
        }

        function Chip({ text, type }) {
          const c = { buy:{bg:"#052e16",border:"#166534",text:"#4ade80"}, sell:{bg:"#2d0a0a",border:"#7f1d1d",text:"#f87171"} }[type];
          return <span style={{ background:c.bg, border:"1px solid "+c.border, color:c.text, borderRadius:3, padding:"3px 8px", fontSize:12, fontFamily:"monospace", whiteSpace:"nowrap", display:"inline-block", margin:"2px 2px" }}>{text}</span>;
        }

        function Card({ s, selected, onClick }) {
          const rc = RC[s.region]||"#888";
          const mc = MC[s.method]||"#888";
          return (
            <div onClick={onClick} style={{ background:selected?"#14142a":"#0c0c14", border:"1px solid "+(selected?"#f97316":"#1e1e30"), borderRadius:8, padding:"12px 14px 10px 16px", cursor:"pointer", position:"relative", overflow:"hidden" }}>
              <div style={{ position:"absolute", left:0, top:0, bottom:0, width:3, background:rc }} />
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:6 }}>
                <div>
                  <div style={{ fontWeight:700, fontSize:17, color:"#f0f0f0", fontFamily:"sans-serif" }}>{s.name}</div>
                  <div style={{ color:"#3a3a55", fontSize:12, fontFamily:"monospace", marginTop:2 }}>{s.system}</div>
                </div>
                <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:4 }}>
                  <span style={{ background:"#f97316", color:"#000", fontFamily:"monospace", fontWeight:700, fontSize:13, padding:"3px 9px", borderRadius:3 }}>{s.weight}%</span>
                  <div style={{ display:"flex", gap:3, flexWrap:"wrap", justifyContent:"flex-end" }}>
                    <Tag label={s.method} color={mc} />
                    <Tag label={s.region} color={rc} />
                    <Tag label={s.norm} color={NC[s.norm]} />
                  </div>
                </div>
              </div>
              <div style={{ display:"flex", flexWrap:"wrap" }}>
                {s.buy.map((b,i)=><Chip key={"b"+i} text={b} type="buy"/>)}
                {s.sell.map((b,i)=><Chip key={"s"+i} text={b} type="sell"/>)}
                {s.sell2.map((b,i)=><Chip key={"s2"+i} text={b} type="sell"/>)}
              </div>
            </div>
          );
        }

        function App() {
          const [sel, setSel] = useState(null);
          const [fR, setFR] = useState("All");
          const [fN, setFN] = useState("All");
          const [fM, setFM] = useState("All");
          const regions = ["All","US","North America","North Atlantic"];
          const norms = ["All","Rank","Zscore"];
          const methods = ["All","Classic","AI - FND","AI - SM"];
          const groups = ["Select Smallcaps","Alpha Smallcaps"];
          const filtered = STRATEGIES.filter(s=>(fR==="All"||s.region===fR)&&(fN==="All"||s.norm===fN)&&(fM==="All"||s.method===fM));
          const gw = g => STRATEGIES.filter(s=>s.group===g).reduce((a,s)=>a+s.weight,0);
          const btn = a => ({ background:a?"#f97316":"#0e0e1a", color:a?"#000":"#555", border:"1px solid "+(a?"#f97316":"#1e1e30"), borderRadius:4, padding:"5px 14px", fontSize:13, fontFamily:"monospace", cursor:"pointer", fontWeight:a?700:400 });
          return (
            <div style={{ background:"#070710", minHeight:"100vh", color:"#f0f0f0", padding:"24px 28px", fontFamily:"monospace" }}>
              <div style={{ marginBottom:20 }}>
                <h1 style={{ margin:0, fontSize:24, fontWeight:800, color:"#f97316", fontFamily:"sans-serif", letterSpacing:"0.04em" }}>PORTFOLIO STRATEGY MAP</h1>
                <div style={{ marginTop:6, fontSize:13, color:"#333" }}>
                  {groups.map(g=><span key={g} style={{ marginRight:20 }}><span style={{color:"#444"}}>{g}:</span> <span style={{color:"#f97316"}}>{gw(g)}%</span><span style={{color:"#2a2a40"}}> \u00b7 {STRATEGIES.filter(s=>s.group===g).length} strategies</span></span>)}
                  <span style={{ marginLeft:10 }}><span style={{color:"#444"}}>Methods:</span>{" "}
                    <span style={{color:MC["Classic"]}}>{STRATEGIES.filter(s=>s.method==="Classic").length}\u00d7 Classic</span>{" "}
                    <span style={{color:MC["AI - FND"]}}>{STRATEGIES.filter(s=>s.method==="AI - FND").length}\u00d7 FND</span>{" "}
                    <span style={{color:MC["AI - SM"]}}>{STRATEGIES.filter(s=>s.method==="AI - SM").length}\u00d7 SM</span>
                  </span>
                </div>
              </div>
              <div style={{ display:"flex", gap:20, marginBottom:16, flexWrap:"wrap", alignItems:"center" }}>
                <div style={{ display:"flex", gap:5, alignItems:"center" }}>
                  <span style={{ color:"#333", fontSize:12, marginRight:4 }}>REGION</span>
                  {regions.map(r=><button key={r} style={btn(fR===r)} onClick={()=>setFR(r)}>{r}</button>)}
                </div>
                <div style={{ display:"flex", gap:5, alignItems:"center" }}>
                  <span style={{ color:"#333", fontSize:12, marginRight:4 }}>NORM</span>
                  {norms.map(n=><button key={n} style={btn(fN===n)} onClick={()=>setFN(n)}>{n}</button>)}
                </div>
                <div style={{ display:"flex", gap:5, alignItems:"center" }}>
                  <span style={{ color:"#333", fontSize:12, marginRight:4 }}>METHOD</span>
                  {methods.map(m=><button key={m} style={btn(fM===m)} onClick={()=>setFM(m)}>{m}</button>)}
                </div>
                <div style={{ marginLeft:"auto", display:"flex", gap:16 }}>
                  {[["#4ade80","Buy"],["#f87171","Sell"]].map(([c,l])=>(
                    <span key={l} style={{ display:"flex", alignItems:"center", gap:5, fontSize:12, color:"#444" }}>
                      <span style={{ width:8, height:8, background:c, borderRadius:2, display:"inline-block", opacity:0.8 }}/> {l}
                    </span>
                  ))}
                </div>
              </div>
              <div style={{ display:"flex", height:5, borderRadius:3, overflow:"hidden", marginBottom:24, gap:1 }}>
                {filtered.map(s=><div key={s.name} title={s.name+": "+s.weight+"%"} style={{ flex:s.weight, background:RC[s.region]||"#555", opacity:0.65 }}/>)}
              </div>
              {groups.map(g=>{
                const gs=filtered.filter(s=>s.group===g);
                if(!gs.length) return null;
                return (
                  <div key={g} style={{ marginBottom:30 }}>
                    <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:12 }}>
                      <span style={{ fontSize:13, fontWeight:700, color:"#f97316", letterSpacing:"0.14em" }}>{g.toUpperCase()}</span>
                      <div style={{ flex:1, height:1, background:"#12122a" }}/>
                      <span style={{ fontSize:12, color:"#2a2a40" }}>{gw(g)}% \u00b7 {STRATEGIES.filter(s=>s.group===g).length}\u00d7 {STRATEGIES.filter(s=>s.group===g)[0]?.weight}% each</span>
                    </div>
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(310px, 1fr))", gap:10 }}>
                      {gs.map(s=><Card key={s.name} s={s} selected={sel===s.name} onClick={()=>setSel(sel===s.name?null:s.name)}/>)}
                    </div>
                  </div>
                );
              })}
              <div style={{ borderTop:"1px solid #10101e", paddingTop:14, display:"flex", gap:20, flexWrap:"wrap" }}>
                {Object.entries(RC).map(([r,c])=><span key={r} style={{ display:"flex", alignItems:"center", gap:6, fontSize:12, color:"#333" }}><span style={{ width:12, height:3, background:c, borderRadius:2, display:"inline-block" }}/>{r}</span>)}
                {Object.entries(NC).map(([n,c])=><span key={n} style={{ display:"flex", alignItems:"center", gap:6, fontSize:12, color:"#333" }}><span style={{ width:8, height:8, borderRadius:"50%", border:"1px solid "+c, display:"inline-block" }}/>{n}</span>)}
                {Object.entries(MC).map(([m,c])=><span key={m} style={{ display:"flex", alignItems:"center", gap:6, fontSize:12, color:"#333" }}><span style={{ width:8, height:4, background:c, borderRadius:1, display:"inline-block" }}/>{m}</span>)}
              </div>
            </div>
          );
        }

        ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
      </script>
    </body>
    </html>
    """
    st.components.v1.html(strategy_map_html, height=950, scrolling=True)
