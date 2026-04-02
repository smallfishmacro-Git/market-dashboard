"""
compute_volatility_json.py
--------------------------
Pre-computes all data for the VOLATILITY tab on the Vercel dashboard.
Outputs: data/datasets/volatility_signals.json

Run from the market-dashboard folder:
    python compute_volatility_json.py

Or called from data_updater.py via compute_volatility_signals()

Two strategies are computed:
  1. Part 3 Combined Rule (8 binary VIX signals, majority vote, 2-day persistence)
  2. ML LogReg (Logistic Regression, ≥5% drawdown in 20d target, p≥0.50 threshold)

The JSON contains everything the frontend needs to render:
  - Equity curves (strategy vs S&P500)
  - Current signals (risk-on/off, probability, individual indicator values)
  - Transaction history (last 2 years of entries/exits)
  - Time series for signal charts
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BARCHART = os.path.join(BASE_DIR, "data", "barchart")
DATASETS = os.path.join(BASE_DIR, "data", "datasets")


def load_barchart(filename):
    """Load a barchart CSV → Series of 'Last' prices."""
    path = os.path.join(BARCHART, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    s = df["Last"].copy()
    if s.dtype == object:
        s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").dropna()


def compute_volatility_signals(log_fn=print):
    """
    Main computation. Returns True on success, False on failure.
    Saves volatility_signals.json to data/datasets/.
    """
    log_fn("  Computing Volatility signals...")

    try:
        # ── 1. Load data ────────────────────────────────────────────────
        spx   = load_barchart("S&P_500_Index_$SPX.csv")
        vix   = load_barchart("CBOE_Volatility_Index_$VIX.csv")
        vix3m = load_barchart("CBOE_3-Month_VIX_$VXV.csv")
        vx1   = load_barchart("VIX_Front_Month_Futures_VI1.csv")
        vx2   = load_barchart("VIX_Second_Month_Futures_VI2.csv")

        missing = []
        if spx is None:   missing.append("SPX")
        if vix is None:   missing.append("VIX")
        if vix3m is None: missing.append("VIX3M")
        if vx1 is None:   missing.append("VX1")
        if vx2 is None:   missing.append("VX2")
        if missing:
            log_fn(f"  ⚠️ Volatility: missing {missing}, skipping.")
            return False

        df = pd.DataFrame({
            "SPX": spx, "VIX": vix, "VIX3M": vix3m, "VX1": vx1, "VX2": vx2
        }).dropna().sort_index()

        # ── 2. Compute 8 features ──────────────────────────────────────
        df["ret"]   = df["SPX"].pct_change()
        df["HV5"]   = df["ret"].rolling(5).std() * np.sqrt(252) * 100
        df["HV10"]  = df["ret"].rolling(10).std() * np.sqrt(252) * 100
        df["EMA7"]  = df["VIX"].ewm(span=7, adjust=False).mean()
        df["SMA50"] = df["VIX"].rolling(50).mean()

        df["F1_VRatio"]       = df["VIX3M"] / df["VIX"]
        df["F2_Contango"]     = df["VX2"] / df["VX1"] - 1
        df["F3_ContangoRoll"] = df["VX2"] / df["VIX"] - 1
        df["F4_VRP"]          = df["VIX"] - df["HV10"]
        df["F5_FVRP"]         = df["EMA7"] - df["HV5"]
        df["F6_VolMom"]       = df["SMA50"] - df["VIX"]
        df["F7_VIX"]          = df["VIX"]
        df["F8_VIX3M"]        = df["VIX3M"]

        fcols = ["F1_VRatio", "F2_Contango", "F3_ContangoRoll", "F4_VRP",
                 "F5_FVRP", "F6_VolMom", "F7_VIX", "F8_VIX3M"]

        # Drop warmup (50 days for SMA50)
        df = df.iloc[50:].copy()
        df = df.dropna(subset=fcols)

        # ── 3. Part 3 Combined Rule ────────────────────────────────────
        df["s1"] = (df["F1_VRatio"] > 1).astype(int)
        df["s2"] = (df["F2_Contango"] > -0.05).astype(int)
        df["s3"] = (df["F3_ContangoRoll"] > 0.10).astype(int)
        df["s4"] = (df["F4_VRP"] > 0).astype(int)
        df["s5"] = (df["F5_FVRP"] > 0).astype(int)
        df["s6"] = (df["F6_VolMom"] > 0).astype(int)
        df["s7"] = ((df["F7_VIX"] > 12) & (df["F7_VIX"] < 20)).astype(int)
        df["s8"] = ((df["F8_VIX3M"] > 12) & (df["F8_VIX3M"] < 20)).astype(int)

        sig_cols = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        df["sig_sum"] = df[sig_cols].sum(axis=1)
        raw_on = (df["sig_sum"] >= 4).astype(int)
        df["rule_on"] = ((raw_on == 1) & (raw_on.shift(1) == 1)).astype(int)

        # Part 3 equity curve
        df["rule_ret"] = df["ret"] * df["rule_on"]
        df["rule_cum"] = (1 + df["rule_ret"]).cumprod()
        df["spx_cum"]  = (1 + df["ret"]).cumprod()

        # ── 4. ML LogReg — expanding-window training ───────────────────
        # Train on expanding window, predict forward (no look-ahead)
        # Target: ≥5% drawdown in next 20 trading days
        sv = df["SPX"].values
        tgt = np.full(len(df), np.nan)
        for i in range(len(df) - 20):
            w = sv[i+1:i+21]
            tgt[i] = 1 if (sv[i] - w.min()) / sv[i] * 100 >= 5 else 0
        df["target"] = tgt

        # Expanding-window predictions
        # Retrain every RETRAIN_FREQ days for speed; reuse model between retrains
        MIN_TRAIN = 504  # ~2 years minimum training
        RETRAIN_FREQ = 20  # retrain monthly
        X_all = df[fcols].values
        y_all = df["target"].values

        ml_prob = np.full(len(df), np.nan)
        current_model = None
        current_scaler = None

        for i in range(MIN_TRAIN, len(df)):
            # Retrain periodically or on first iteration
            if current_model is None or i % RETRAIN_FREQ == 0:
                train_mask = ~np.isnan(y_all[:i])
                if train_mask.sum() < 100:
                    continue
                X_train = X_all[:i][train_mask]
                y_train = y_all[:i][train_mask].astype(int)
                if len(np.unique(y_train)) < 2:
                    continue

                current_scaler = StandardScaler()
                X_train_s = current_scaler.fit_transform(X_train)
                current_model = LogisticRegression(
                    max_iter=2000, class_weight="balanced", C=0.1, random_state=42
                )
                current_model.fit(X_train_s, y_train)

            if current_model is not None:
                X_today_s = current_scaler.transform(X_all[i:i+1])
                ml_prob[i] = current_model.predict_proba(X_today_s)[:, 1][0]

        df["ml_prob"] = ml_prob
        df["ml_signal"] = (df["ml_prob"] >= 0.50).astype(int)  # 1 = danger
        df["ml_invested"] = 1 - df["ml_signal"]

        # ML equity curve
        df["ml_ret"] = df["ret"] * df["ml_invested"]
        df["ml_cum"] = (1 + df["ml_ret"]).cumprod()

        # ── 5. Build transaction histories ─────────────────────────────
        def get_transactions(signal_series, spx_series, label, years=2):
            """Extract entry/exit transactions from a binary signal."""
            cutoff = signal_series.index[-1] - pd.DateOffset(years=years)
            sig = signal_series[signal_series.index >= cutoff].copy()
            spx = spx_series.reindex(sig.index)

            transactions = []
            prev = None
            for date, val in sig.items():
                if prev is not None and val != prev:
                    action = "ENTER" if val == 1 else "EXIT"
                    transactions.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "action": action,
                        "spx": round(float(spx.loc[date]), 2) if pd.notna(spx.loc[date]) else None,
                        "signal": label
                    })
                prev = val

            return transactions

        rule_transactions = get_transactions(df["rule_on"], df["SPX"], "Part 3 Rule")
        ml_transactions = get_transactions(df["ml_invested"], df["SPX"], "ML LogReg")

        # ── 6. Current signals ─────────────────────────────────────────
        latest = df.iloc[-1]
        latest_date = df.index[-1].strftime("%Y-%m-%d")

        current_signals = {
            "date": latest_date,
            "part3_rule": {
                "signal": "RISK-ON" if int(latest["rule_on"]) == 1 else "RISK-OFF",
                "signal_value": int(latest["rule_on"]),
                "signals_on": int(latest["sig_sum"]),
                "signals_required": 4,
                "individual_signals": {
                    "VRatio (VIX3M/VIX > 1)": bool(latest["s1"]),
                    "Contango (VX2/VX1-1 > -5%)": bool(latest["s2"]),
                    "Contango Roll (VX2/VIX-1 > 10%)": bool(latest["s3"]),
                    "VRP (VIX-HV10 > 0)": bool(latest["s4"]),
                    "Fast VRP (EMA7-HV5 > 0)": bool(latest["s5"]),
                    "Vol Momentum (SMA50-VIX > 0)": bool(latest["s6"]),
                    "VIX Mean Rev (12 < VIX < 20)": bool(latest["s7"]),
                    "VIX3M Mean Rev (12 < VIX3M < 20)": bool(latest["s8"]),
                },
                "feature_values": {
                    "VRatio": round(float(latest["F1_VRatio"]), 3),
                    "Contango": round(float(latest["F2_Contango"]) * 100, 2),
                    "Contango Roll": round(float(latest["F3_ContangoRoll"]) * 100, 2),
                    "VRP": round(float(latest["F4_VRP"]), 2),
                    "FVRP": round(float(latest["F5_FVRP"]), 2),
                    "Vol Momentum": round(float(latest["F6_VolMom"]), 2),
                    "VIX": round(float(latest["F7_VIX"]), 2),
                    "VIX3M": round(float(latest["F8_VIX3M"]), 2),
                },
            },
            "ml_logreg": {
                "signal": "RISK-ON" if int(latest.get("ml_invested", 0)) == 1 else "RISK-OFF",
                "signal_value": int(latest.get("ml_invested", 0)),
                "probability": round(float(latest.get("ml_prob", 0)), 4),
                "threshold": 0.50,
                "target": "≥5% drawdown in 20 trading days",
            },
        }

        # ── 7. Time series for charts (downsample to weekly for JSON size)
        # Full daily for last 2 years, weekly before that
        two_years_ago = df.index[-1] - pd.DateOffset(years=2)

        def build_chart_series(daily_series, name):
            """Build a time series suitable for charting."""
            old = daily_series[daily_series.index < two_years_ago].resample("W-FRI").last()
            recent = daily_series[daily_series.index >= two_years_ago]
            combined = pd.concat([old, recent]).dropna()
            return [
                {"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 4)}
                for d, v in combined.items()
            ]

        # Part 3 Rule signal chart (daily, last 2 years only for signal)
        rule_signal_2y = df["rule_on"][df.index >= two_years_ago]
        rule_signal_chart = [
            {"d": d.strftime("%Y-%m-%d"), "v": int(v)}
            for d, v in rule_signal_2y.items()
        ]

        # ML probability chart (daily, last 2 years)
        ml_prob_2y = df["ml_prob"][df.index >= two_years_ago].dropna()
        ml_prob_chart = [
            {"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 4)}
            for d, v in ml_prob_2y.items()
        ]

        # ── 8. Performance metrics ─────────────────────────────────────
        def calc_perf(cum_series, ret_series, label):
            ny = len(ret_series.dropna()) / 252
            if ny <= 0 or cum_series.iloc[-1] <= 0:
                return {}
            cagr = cum_series.iloc[-1] ** (1 / ny) - 1
            vol = ret_series.dropna().std() * np.sqrt(252)
            sharpe = cagr / vol if vol > 0 else 0
            maxdd = ((cum_series / cum_series.cummax()) - 1).min()
            return {
                "label": label,
                "cagr": round(float(cagr * 100), 2),
                "volatility": round(float(vol * 100), 2),
                "sharpe": round(float(sharpe), 2),
                "max_drawdown": round(float(maxdd * 100), 2),
                "total_return": round(float((cum_series.iloc[-1] - 1) * 100), 2),
            }

        perf_bh = calc_perf(df["spx_cum"], df["ret"], "Buy & Hold")
        perf_rule = calc_perf(df["rule_cum"], df["rule_ret"], "Part 3 Rule")

        ml_valid = df["ml_cum"].dropna()
        ml_ret_valid = df["ml_ret"].loc[ml_valid.index]
        perf_ml = calc_perf(ml_valid, ml_ret_valid, "ML LogReg (p≥0.50)")

        # ── 9. Assemble JSON ───────────────────────────────────────────
        output = {
            "last_updated": latest_date,
            "data_range": {
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end": latest_date,
                "trading_days": len(df),
            },
            "current_signals": current_signals,
            "performance": {
                "buy_hold": perf_bh,
                "part3_rule": perf_rule,
                "ml_logreg": perf_ml,
            },
            "equity_curves": {
                "spx": build_chart_series(df["spx_cum"], "S&P 500"),
                "part3_rule": build_chart_series(df["rule_cum"], "Part 3 Rule"),
                "ml_logreg": build_chart_series(
                    df["ml_cum"].dropna(), "ML LogReg"
                ),
            },
            "signal_charts": {
                "part3_rule": rule_signal_chart,
                "ml_probability": ml_prob_chart,
            },
            "transactions": {
                "part3_rule": rule_transactions,
                "ml_logreg": ml_transactions,
            },
        }

        # ── 10. Save ───────────────────────────────────────────────────
        out_path = os.path.join(DATASETS, "volatility_signals.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        log_fn(f"  ✅ Volatility signals saved ({len(df)} days, "
               f"Rule={current_signals['part3_rule']['signal']}, "
               f"ML={current_signals['ml_logreg']['signal']} "
               f"p={current_signals['ml_logreg']['probability']:.2f})")
        return True

    except Exception as e:
        log_fn(f"  ❌ Volatility signals — ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Run directly ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()
    ok = compute_volatility_signals()
    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    if ok:
        path = os.path.join(DATASETS, "volatility_signals.json")
        size_kb = os.path.getsize(path) / 1024
        print(f"  JSON size: {size_kb:.0f} KB")

        with open(path) as f:
            data = json.load(f)
        print(f"\n  Current signals:")
        cs = data["current_signals"]
        print(f"    Date:       {cs['date']}")
        print(f"    Part 3:     {cs['part3_rule']['signal']} "
              f"({cs['part3_rule']['signals_on']}/8 signals on)")
        print(f"    ML LogReg:  {cs['ml_logreg']['signal']} "
              f"(p={cs['ml_logreg']['probability']:.3f}, "
              f"threshold={cs['ml_logreg']['threshold']})")
        print(f"\n  Performance:")
        for k, v in data["performance"].items():
            print(f"    {v['label']:25s}: CAGR={v['cagr']:.1f}%, "
                  f"Sharpe={v['sharpe']:.2f}, MaxDD={v['max_drawdown']:.1f}%")
        print(f"\n  Transactions (last 2yr):")
        print(f"    Part 3 Rule: {len(data['transactions']['part3_rule'])} entries/exits")
        print(f"    ML LogReg:   {len(data['transactions']['ml_logreg'])} entries/exits")
