"""
compute_btd_json.py
-------------------
Reads local Barchart CSVs + datasets, computes all 9 BTD indicators,
composite signal, and metrics.  Writes data/datasets/btd_indicators.json
— the exact JSON shape that the Vercel React frontend expects.

Run after data_updater.py in the GitHub Actions pipeline:
    python compute_btd_json.py
"""

import os
import json
import math
import csv
import io
from datetime import datetime, timedelta

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BARCHART = os.path.join(BASE_DIR, "data", "barchart")
DATASETS = os.path.join(BASE_DIR, "data", "datasets")

CSV_FILES = {
    "spx":    os.path.join(BARCHART, "S&P_500_Index_$SPX.csv"),
    "r3fd":   os.path.join(BARCHART, "Russell_3000_Stocks_Above_5-Day_Average_$R3FD.csv"),
    "nshu":   os.path.join(BARCHART, "NYSE_Advancing_Stocks_$NSHU.csv"),
    "nshd":   os.path.join(BARCHART, "NYSE_Declining_Stocks_$NSHD.csv"),
    "nvlu":   os.path.join(BARCHART, "NYSE_Advancing_Volume_$NVLU.csv"),
    "dvcn":   os.path.join(BARCHART, "NYSE_Declining_Volume_$DVCN.csv"),
    "cpcs":   os.path.join(BARCHART, "Equity_PutCall_Ratio_$CPCS.csv"),
    "vix":    os.path.join(BARCHART, "CBOE_Volatility_Index_$VIX.csv"),
    "vxv":    os.path.join(BARCHART, "CBOE_3-Month_VIX_$VXV.csv"),
    "mahp":   os.path.join(BARCHART, "S&P_500_52-Week_Highs_$MAHP.csv"),
    "fg":     os.path.join(DATASETS, "cnn_fear_greed.csv"),
    "acwi":   os.path.join(DATASETS, "acwi_oscillator.csv"),
}


# ── CSV loading ──────────────────────────────────────────────────────────────
def load_csv(key):
    """Load a local CSV and return (dates[], values[])."""
    path = CSV_FILES[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        dates, values = [], []
        for row in reader:
            date_str = (row.get("Time") or row.get("Date") or "").strip()[:10]
            if not date_str or date_str < "1990-01-01":
                continue
            val_str = (row.get("Last") or row.get("Fear_Greed") or
                       row.get("Percentage") or "")
            val_str = str(val_str).replace(",", "").strip()
            try:
                val = float(val_str)
            except (ValueError, TypeError):
                continue
            dates.append(date_str)
            values.append(val)
    return dates, values


def load_all():
    """Load all CSVs into a dict."""
    data = {}
    for key in CSV_FILES:
        try:
            d, v = load_csv(key)
            data[key] = {"dates": d, "values": v}
        except Exception as e:
            data[key] = {"dates": [], "values": [], "error": str(e)}
    return data


# ── Math helpers (identical to the Vercel data.py) ───────────────────────────
def ewma(values, span):
    alpha = 2.0 / (span + 1)
    result = [values[0]] if values else []
    for i in range(1, len(values)):
        result.append(alpha * values[i] + (1 - alpha) * result[i - 1])
    return result


def rolling_mean(values, window):
    result = [None] * len(values)
    for i in range(window - 1, len(values)):
        result[i] = sum(values[i - window + 1:i + 1]) / window
    return result


def rolling_std(values, window):
    result = [None] * len(values)
    for i in range(window - 1, len(values)):
        subset = values[i - window + 1:i + 1]
        m = sum(subset) / window
        var = sum((x - m) ** 2 for x in subset) / window
        result[i] = math.sqrt(var) if var > 0 else 0.0001
    return result


def rolling_sum(values, window):
    result = [None] * len(values)
    for i in range(window - 1, len(values)):
        result[i] = sum(v for v in values[i - window + 1:i + 1] if v is not None)
    return result


def align_series(dates_a, values_a, dates_b, values_b):
    map_a = dict(zip(dates_a, values_a))
    map_b = dict(zip(dates_b, values_b))
    all_dates = sorted(set(dates_a) | set(dates_b))
    d, va, vb = [], [], []
    last_a, last_b = None, None
    for dt in all_dates:
        a = map_a.get(dt, last_a)
        b = map_b.get(dt, last_b)
        if a is not None:
            last_a = a
        if b is not None:
            last_b = b
        if last_a is not None and last_b is not None:
            d.append(dt)
            va.append(last_a)
            vb.append(last_b)
    return d, va, vb


# ── Indicator computations (identical to Vercel data.py) ─────────────────────
def compute_indicators(raw):
    indicators = {}
    spx_d, spx_v = raw["spx"]["dates"], raw["spx"]["values"]

    # 1. % Russell 3000 Above 5-Day MA
    r3fd = raw["r3fd"]
    d, spx_a, r3fd_v = align_series(spx_d, spx_v, r3fd["dates"], r3fd["values"])
    signals = [d[i] for i in range(len(r3fd_v)) if r3fd_v[i] < 10]
    indicators["r3fd"] = {
        "name": "% Russell 3000 Above 5-Day MA",
        "dates": d, "spx": spx_a, "values": r3fd_v,
        "signals": signals, "threshold": 10, "thresholdDir": "below",
    }

    # 2. ACWI Oscillator
    acwi = raw["acwi"]
    d, spx_a, acwi_v = align_series(spx_d, spx_v, acwi["dates"], acwi["values"])
    signals = [d[i] for i in range(len(acwi_v)) if acwi_v[i] == 0]
    indicators["acwi"] = {
        "name": "ACWI ETF Oscillator (% Above 10DMA)",
        "dates": d, "spx": spx_a, "values": acwi_v,
        "signals": signals, "threshold": 0, "thresholdDir": "at",
    }

    # 3. McClellan Oscillator
    nshu = raw["nshu"]
    nshd = raw["nshd"]
    d_raw, adv_v, dec_v = align_series(nshu["dates"], nshu["values"],
                                        nshd["dates"], nshd["values"])
    rana = [(adv_v[i] - dec_v[i]) / max(adv_v[i] + dec_v[i], 1) * 1000
            for i in range(len(adv_v))]
    rana = [max(-1000, min(1000, v)) for v in rana]
    ema19 = ewma(rana, 19)
    ema39 = ewma(rana, 39)
    mco = [ema19[i] - ema39[i] for i in range(len(rana))]
    mco = [max(-500, min(500, v)) for v in mco]
    d, spx_a, mco_a = align_series(spx_d, spx_v, d_raw, mco)
    signals = [d[i] for i in range(1, len(mco_a))
               if mco_a[i] < -80 and mco_a[i - 1] >= -80]
    indicators["mcclellan"] = {
        "name": "McClellan Oscillator",
        "dates": d, "spx": spx_a, "values": mco_a,
        "signals": signals, "threshold": -80, "thresholdDir": "crossBelow",
    }

    # 4. Equity Put/Call Ratio Z-Score
    cpcs = raw["cpcs"]
    pc_d, pc_v = cpcs["dates"], cpcs["values"]
    sma5 = rolling_mean(pc_v, 5)
    rm52 = rolling_mean([x for x in sma5 if x is not None], 52)
    rs52 = rolling_std([x for x in sma5 if x is not None], 52)
    offset = sum(1 for x in sma5 if x is None)
    zscore_raw = [None] * len(pc_v)
    for i in range(len(pc_v)):
        s5 = sma5[i]
        idx = i - offset
        if (s5 is not None and 0 <= idx < len(rm52)
                and rm52[idx] is not None and rs52[idx] is not None and rs52[idx] > 0):
            zscore_raw[i] = (rm52[idx] - s5) / rs52[idx]
    pc_dates_clean = [pc_d[i] for i in range(len(pc_d)) if zscore_raw[i] is not None]
    zscore_clean = [zscore_raw[i] for i in range(len(pc_d)) if zscore_raw[i] is not None]
    d, spx_a, zs_a = align_series(spx_d, spx_v, pc_dates_clean, zscore_clean)
    signals = [d[i] for i in range(len(zs_a)) if zs_a[i] is not None and zs_a[i] < -2.5]
    indicators["putcall"] = {
        "name": "Equity Put/Call Z-Score",
        "dates": d, "spx": spx_a, "values": zs_a,
        "signals": signals, "threshold": -2.5, "thresholdDir": "below",
    }

    # 5. CNN Fear & Greed
    fg = raw["fg"]
    d, spx_a, fg_v = align_series(spx_d, spx_v, fg["dates"], fg["values"])
    signals = [d[i] for i in range(len(fg_v)) if fg_v[i] < 25]
    indicators["feargreed"] = {
        "name": "CNN Fear & Greed Index",
        "dates": d, "spx": spx_a, "values": fg_v,
        "signals": signals, "threshold": 25, "thresholdDir": "below",
    }

    # 6. Lowry Panic Indicator
    nvlu = raw["nvlu"]
    dvcn = raw["dvcn"]
    d_adv_s, adv_s_v, dec_s_v = align_series(nshu["dates"], nshu["values"],
                                               nshd["dates"], nshd["values"])
    d_adv_v, adv_v_v, dec_v_v = align_series(nvlu["dates"], nvlu["values"],
                                               dvcn["dates"], dvcn["values"])
    d_all, ts_v, tv_v = align_series(d_adv_s, adv_s_v, d_adv_v, adv_v_v)
    dec_s_map = dict(zip(d_adv_s, dec_s_v))
    dec_v_map = dict(zip(d_adv_v, dec_v_v))
    scores = []
    for i, dt in enumerate(d_all):
        total_s = ts_v[i] + (dec_s_map.get(dt, 0) or 0)
        total_v = tv_v[i] + (dec_v_map.get(dt, 0) or 0)
        ds = dec_s_map.get(dt, 0) or 0
        dv = dec_v_map.get(dt, 0) or 0
        dec_pct_s = (ds / total_s * 100) if total_s > 0 else 0
        dec_pct_v = (dv / total_v * 100) if total_v > 0 else 0
        score = (int(dec_pct_s >= 90) + int(80 <= dec_pct_s < 90) +
                 int(dec_pct_v >= 90) + int(80 <= dec_pct_v < 90))
        scores.append(score)
    roll6 = rolling_sum(scores, 6)
    lowry_d = [d_all[i] for i in range(len(roll6)) if roll6[i] is not None]
    lowry_v = [roll6[i] for i in range(len(roll6)) if roll6[i] is not None]
    d, spx_a, lowry_a = align_series(spx_d, spx_v, lowry_d, lowry_v)
    signals = [d[i] for i in range(1, len(lowry_a))
               if lowry_a[i] >= 4 and lowry_a[i - 1] < 4]
    indicators["lowry"] = {
        "name": "Lowry Panic Indicator",
        "dates": d, "spx": spx_a, "values": lowry_a,
        "signals": signals, "threshold": 4, "thresholdDir": "crossAbove",
    }

    # 7. Zweig Breadth
    d_zw, adv_zw, dec_zw = align_series(nshu["dates"], nshu["values"],
                                          nshd["dates"], nshd["values"])
    ratio = [adv_zw[i] / max(adv_zw[i] + dec_zw[i], 1) for i in range(len(adv_zw))]
    ratio = [max(0, min(1, v)) for v in ratio]
    zweig = ewma(ratio, 10)
    zweig = [max(0, min(1, v)) for v in zweig]
    d, spx_a, zw_a = align_series(spx_d, spx_v, d_zw, zweig)
    signals = [d[i] for i in range(1, len(zw_a))
               if zw_a[i] <= 0.35 and zw_a[i - 1] > 0.35]
    indicators["zweig"] = {
        "name": "Zweig Breadth Indicator",
        "dates": d, "spx": spx_a, "values": zw_a,
        "signals": signals, "threshold": 0.35, "thresholdDir": "crossBelow",
    }

    # 8. Volatility Curve (VXV/VIX - 1)
    d_vc, vxv_v, vix_v = align_series(raw["vxv"]["dates"], raw["vxv"]["values"],
                                        raw["vix"]["dates"], raw["vix"]["values"])
    vc = [(vxv_v[i] / vix_v[i] - 1) if vix_v[i] > 0 else 0 for i in range(len(vxv_v))]
    d, spx_a, vc_a = align_series(spx_d, spx_v, d_vc, vc)
    signals = [d[i] for i in range(1, len(vc_a))
               if vc_a[i] >= 0 and vc_a[i - 1] < 0]
    indicators["volcurve"] = {
        "name": "Volatility Curve (VXV/VIX)",
        "dates": d, "spx": spx_a, "values": vc_a,
        "signals": signals, "threshold": 0, "thresholdDir": "crossAbove",
    }

    # 9. S&P 500 52-Week New Highs
    mahp = raw["mahp"]
    d, spx_a, highs_v = align_series(spx_d, spx_v, mahp["dates"], mahp["values"])
    signals = [d[i] for i in range(len(highs_v)) if highs_v[i] < 1]
    indicators["highs52w"] = {
        "name": "S&P 500 52-Week New Highs",
        "dates": d, "spx": spx_a, "values": highs_v,
        "signals": signals, "threshold": 1, "thresholdDir": "below",
    }

    return indicators


def _sub_days(date_str, days):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - timedelta(days=days)).strftime("%Y-%m-%d")


def compute_composite(indicators):
    all_signal_sets = {}
    for key, ind in indicators.items():
        all_signal_sets[key] = set(ind["signals"])

    spx_dates = indicators["r3fd"]["dates"]
    spx_values = indicators["r3fd"]["spx"]

    composite_dates = []
    composite_scores = []
    for i, dt in enumerate(spx_dates):
        if dt < "2000-01-01":
            continue
        score = 0
        for key in all_signal_sets:
            for sig_dt in all_signal_sets[key]:
                if sig_dt <= dt and sig_dt >= _sub_days(dt, 10):
                    score += 1
                    break
        composite_dates.append(dt)
        composite_scores.append(score)

    ma2 = rolling_mean(composite_scores, 2)

    triggers = []
    for i in range(len(composite_scores)):
        if (composite_scores[i] > 2 and ma2[i] is not None
                and composite_scores[i] > ma2[i]):
            triggers.append(composite_dates[i])

    spx_map = dict(zip(spx_dates, spx_values))
    comp_spx = [spx_map.get(d, None) for d in composite_dates]

    return {
        "dates": composite_dates,
        "scores": composite_scores,
        "ma2": [x if x is not None else 0 for x in ma2],
        "triggers": triggers,
        "spx": comp_spx,
    }


def compute_metrics(composite, indicators):
    latest_score = composite["scores"][-1] if composite["scores"] else 0
    last_signal = None
    for ind in indicators.values():
        if ind["signals"]:
            s = ind["signals"][-1]
            if last_signal is None or s > last_signal:
                last_signal = s
    last_trigger = composite["triggers"][-1] if composite["triggers"] else None
    return {
        "btdScore": latest_score,
        "lastSignalDate": last_signal,
        "lastTriggerDate": last_trigger,
    }


def downsample(dates, values_dict, max_points=8000):
    n = len(dates)
    if n <= max_points:
        return dates, values_dict
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    new_dates = [dates[i] for i in idx]
    new_vals = {}
    for k, v in values_dict.items():
        new_vals[k] = [v[i] if i < len(v) else None for i in idx]
    return new_dates, new_vals


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading CSVs...")
    raw = load_all()

    errors = {k: v.get("error") for k, v in raw.items() if v.get("error")}
    if errors:
        print(f"  Warnings: {errors}")
    if len(errors) > 3:
        raise ValueError(f"Too many CSV load errors: {errors}")

    print("Computing 9 indicators...")
    indicators = compute_indicators(raw)

    # Downsample each indicator
    for key in indicators:
        ind = indicators[key]
        ind["dates"], sampled = downsample(
            ind["dates"],
            {"spx": ind["spx"], "values": ind["values"]},
            max_points=8000,
        )
        ind["spx"] = sampled["spx"]
        ind["values"] = sampled["values"]

    print("Computing composite signal...")
    composite = compute_composite(indicators)

    print("Computing metrics...")
    metrics = compute_metrics(composite, indicators)

    # Downsample composite
    composite["dates"], comp_sampled = downsample(
        composite["dates"],
        {"scores": composite["scores"], "ma2": composite["ma2"],
         "spx": composite["spx"]},
        max_points=8000,
    )
    composite["scores"] = comp_sampled["scores"]
    composite["ma2"] = comp_sampled["ma2"]
    composite["spx"] = comp_sampled["spx"]

    response = {
        "indicators": indicators,
        "composite": composite,
        "metrics": metrics,
        "errors": errors if errors else None,
        "computedAt": datetime.now(tz=__import__('datetime').timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_path = os.path.join(DATASETS, "btd_indicators.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(response, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✅  btd_indicators.json saved ({size_mb:.1f} MB)")
    print(f"      BTD Score: {metrics['btdScore']}/9")
    print(f"      Last signal: {metrics['lastSignalDate']}")
    print(f"      Last trigger: {metrics['lastTriggerDate']}")


if __name__ == "__main__":
    main()
