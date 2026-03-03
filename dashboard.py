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
    st.cache_data.clear()
    st.rerun()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
# Their render() is only called when the user is on that tab.
# This prevents WebSocket timeouts caused by running all tabs simultaneously.

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Market Breadth",
    "Volatility",
    "Market Risk",
    "Macro / Regime",
    "Buy the Dip",
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
                _btd_now    = int(_df_btd["Composite"].iloc[-1])
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
            for _i in range(0, len(_btd_map), 6):
                if _i > 0:
                    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                _row = st.columns(6)
                for _j, (_sc, _lbl) in enumerate(_btd_map[_i:_i+6]):
                    _s = _df_btd2[_sc].dropna() if _sc in _df_btd2.columns else pd.Series(dtype=float)
                    _v = int(_s.iloc[-1]) if len(_s) else None
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
