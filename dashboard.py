import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Dark theme styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1a1a; }
    .stTabs [data-baseweb="tab"] { color: #aaaaaa; }
    .stTabs [aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #00ff88; }
    .metric-card {
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BARCHART   = os.path.join(BASE_DIR, "datasets", "barchart")
DATASETS   = os.path.join(BASE_DIR, "datasets")

# ── Helper: load a barchart CSV ─────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_bc(filename):
    path = os.path.join(BARCHART, filename)
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df["Last"] = pd.to_numeric(df["Last"], errors="coerce")
    return df

# ── Helper: last value + delta ──────────────────────────────────────────────────
def last_val(df):
    return df["Last"].dropna().iloc[-1]

def delta(df):
    s = df["Last"].dropna()
    return s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else 0

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("📈 Market Dashboard")
st.caption("smallfishmacro — personal market indicators")

# ── Refresh button ───────────────────────────────────────────────────────────────
if st.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🧭 Market Breadth",
    "⚡ Volatility",
    "💹 Relative Strength",
    "🌍 Macro / Regime",
    "🎯 Buy the Dip"
])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Market Snapshot")

    try:
        spx  = load_bc("S&P_500_Index_$SPX.csv")
        vix  = load_bc("CBOE_Volatility_Index_$VIX.csv")
        mmfi = load_bc("Percent_of_Stocks_Above_50-Day_Average_$MMFI.csv")
        mmth = load_bc("Percent_of_Stocks_Above_200-Day Average_$MMTH.csv")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("S&P 500", f"{last_val(spx):,.0f}", f"{delta(spx):+.1f}")
        with col2:
            st.metric("VIX", f"{last_val(vix):.1f}", f"{delta(vix):+.1f}")
        with col3:
            st.metric("% Above 50D MA", f"{last_val(mmfi):.1f}%", f"{delta(mmfi):+.1f}")
        with col4:
            st.metric("% Above 200D MA", f"{last_val(mmth):.1f}%", f"{delta(mmth):+.1f}")

        # SPX chart
        st.subheader("S&P 500 — 2 Years")
        spx2 = spx.last("730D")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spx2.index, y=spx2["Last"],
            mode="lines", name="SPX",
            line=dict(color="#00ff88", width=2)
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0e0e0e",
            paper_bgcolor="#0e0e0e",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load overview data: {e}")

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET BREADTH
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Market Breadth")
    st.info("Charts coming soon — breadth indicators will appear here.")

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Volatility")
    st.info("Charts coming soon — VIX term structure and volatility indicators will appear here.")

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Relative Strength")
    st.info("Charts coming soon — ratio charts and relative strength indicators will appear here.")

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 5 — MACRO / REGIME
# ══════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Macro / Market Regime")
    st.info("Charts coming soon — OECD CLI regime and macro indicators will appear here.")

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 6 — BUY THE DIP
# ══════════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Buy the Dip — Oversold Signals")
    st.info("Charts coming soon — oversold signals and mean reversion indicators will appear here.")
