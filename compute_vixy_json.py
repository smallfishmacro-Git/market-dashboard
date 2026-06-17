#!/usr/bin/env python3
"""
VIXY tail-hedge model JSON builder for the Market Risk dashboard.

Replaces the on-demand Vercel function (api/vixy.py) that fetched everything
live from Yahoo Finance on each request. Yahoo throttles/lags index tickers
(notably ^VIX3M) for Vercel datacenter IPs, and because the model intersects all
tickers on common dates, a single lagging series pinned the whole "AS OF" date
(observed stuck at 2026-06-12 while the market had traded through 06-16).

Signal inputs come from the committed, fresh Barchart CSVs (refreshed via
Barchart's own API in data_updater.py); only the ETF legs use yfinance.

  Signal inputs (VIX, VIX3M, realized-vol-of-SPX) -> Barchart CSVs  (always fresh)
  Strategy legs   (SPY, VIXY, SDS, TZA)            -> yfinance       (Actions IP)

CURRENT SIGNALS are computed on the Barchart-only axis, so the AS OF date and
signal flags can never be pinned by an ETF lag. Run AFTER data_updater.py.
Output schema is byte-compatible with the old /api/vixy response.
"""
import os
import json
import math
import time
from datetime import datetime, timezone

import pandas as pd

# ════════════════════════════════════════════════════════════════════════════
# TUNABLE CONFIG (env-overridable so you can iterate without editing code)
# ════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BARCHART = os.environ.get("VIXY_BARCHART_DIR", os.path.join(BASE_DIR, "data", "barchart"))
OUT      = os.environ.get("VIXY_OUT", os.path.join(BASE_DIR, "data", "datasets", "vixy.json"))

VIX_CSV   = os.environ.get("VIXY_VIX_CSV",   "CBOE_Volatility_Index_$VIX.csv")
VIX3M_CSV = os.environ.get("VIXY_VIX3M_CSV", "CBOE_3-Month_VIX_$VXV.csv")
SPX_CSV   = os.environ.get("VIXY_SPX_CSV",   "S&P_500_Index_$SPX.csv")

HEDGE_TICKERS = {"vixy": "VIXY", "sds": "SDS", "tza": "TZA"}
SPY_TICKER    = os.environ.get("VIXY_SPY_TICKER", "SPY")
START         = os.environ.get("VIXY_START", "2007-11-01")

TRADING_DAYS = int(os.environ.get("VIXY_TRADING_DAYS", "252"))
RV_SHORT     = int(os.environ.get("VIXY_RV_SHORT", "5"))
RV_LONG      = int(os.environ.get("VIXY_RV_LONG", "10"))
MA_WINDOW    = int(os.environ.get("VIXY_MA_WINDOW", "30"))
SPY_W        = float(os.environ.get("VIXY_SPY_W", "0.80"))
VIXY_W       = float(os.environ.get("VIXY_FIXED_W", "0.20"))
MIN_DAYS     = int(os.environ.get("VIXY_MIN_DAYS", "100"))

# ════════════════════════════════════════════════════════════════════════════
# MATH — ported verbatim from api/vixy.py (unchanged, already validated vs TV)
# ════════════════════════════════════════════════════════════════════════════
def _ret(prices):
    r = [0.0]
    for i in range(1, len(prices)):
        r.append(math.log(prices[i] / prices[i - 1]) if prices[i - 1] > 0 else 0.0)
    return r


def _rvol(rets, w):
    n = len(rets)
    rv = [None] * n
    for i in range(w - 1, n):
        sl = rets[i - w + 1 : i + 1]
        m = sum(sl) / w
        v = sum((x - m) ** 2 for x in sl) / (w - 1) if w > 1 else 0
        rv[i] = math.sqrt(v) * math.sqrt(TRADING_DAYS) * 100
    return rv


def _ma(vals, w):
    n = len(vals)
    r = [None] * n
    for i in range(w - 1, n):
        s = [v for v in vals[i - w + 1 : i + 1] if v is not None]
        r[i] = sum(s) / len(s) if s else None
    return r


def _stats(equity, port_ret):
    n = len(equity)
    peak = mx_dd = 0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > mx_dd:
            mx_dd = dd
    tr = equity[-1] / equity[0] - 1 if equity[0] > 0 else 0
    yrs = n / TRADING_DAYS
    cagr = (equity[-1] / equity[0]) ** (1 / yrs) - 1 if yrs > 0 and equity[0] > 0 else 0
    m_r = sum(port_ret) / n if n > 0 else 0
    var_r = sum((x - m_r) ** 2 for x in port_ret) / (n - 1) if n > 1 else 0
    vol = math.sqrt(var_r) * math.sqrt(TRADING_DAYS)
    sh = cagr / vol if vol > 0 else 0
    return dict(
        cagr=round(cagr * 100, 2),
        vol=round(vol * 100, 2),
        sharpe=round(sh, 2),
        max_dd=round(mx_dd * 100, 2),
        total_return=round(tr * 100, 2),
    )


def _strat(spy_close, vixy_close, spy_open, vixy_open, signal, vix, n,
           sizing, spy_w=SPY_W, vixy_w=VIXY_W):
    hw = [0.0] * n
    for i in range(1, n):
        if signal[i - 1] > 0:
            hw[i] = (vix[i - 1] / 100.0) if sizing else vixy_w
    eq = [100.0]
    pr = [0.0]
    sleq = [1.0]
    for i in range(1, n):
        r_spy = (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0.0
        was_in = hw[i - 1] > 0
        now_in = hw[i] > 0
        if now_in and not was_in:
            r_vixy = (vixy_close[i] / vixy_open[i] - 1) if vixy_open[i] > 0 else 0.0
            vixy_contrib = hw[i] * r_vixy
        elif was_in and not now_in:
            r_vixy = (vixy_open[i] / vixy_close[i - 1] - 1) if vixy_close[i - 1] > 0 else 0.0
            vixy_contrib = hw[i - 1] * r_vixy
        elif was_in and now_in:
            r_overnight = (vixy_open[i] / vixy_close[i - 1] - 1) if vixy_close[i - 1] > 0 else 0.0
            r_intraday = (vixy_close[i] / vixy_open[i] - 1) if vixy_open[i] > 0 else 0.0
            vixy_contrib = hw[i - 1] * r_overnight + hw[i] * r_intraday
        else:
            vixy_contrib = 0.0
        day_ret = spy_w * r_spy + vixy_contrib
        pr.append(day_ret)
        eq.append(eq[-1] * (1 + day_ret))
        sleq.append(sleq[-1] * (1 + vixy_contrib))
    return dict(
        equity=[round(v, 2) for v in eq],
        hedge_weight=[round(v, 4) for v in hw],
        vixy_sleeve_equity=[round(v, 6) for v in sleq],
        stats=_stats(eq, pr),
    )


def _strat_c2c(spy_close, vixy_close, signal, vix, n,
               sizing, spy_w=SPY_W, vixy_w=VIXY_W):
    hw = [0.0] * n
    for i in range(1, n):
        if signal[i - 1] > 0:
            hw[i] = (vix[i - 1] / 100.0) if sizing else vixy_w
    eq = [100.0]
    pr = [0.0]
    sleq = [1.0]
    for i in range(1, n):
        r_spy = (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0.0
        r_vixy = (vixy_close[i] / vixy_close[i - 1] - 1) if vixy_close[i - 1] > 0 else 0.0
        vixy_contrib = hw[i] * r_vixy
        day_ret = spy_w * r_spy + vixy_contrib
        pr.append(day_ret)
        eq.append(eq[-1] * (1 + day_ret))
        sleq.append(sleq[-1] * (1 + vixy_contrib))
    return dict(
        equity=[round(v, 2) for v in eq],
        hedge_weight=[round(v, 4) for v in hw],
        vixy_sleeve_equity=[round(v, 6) for v in sleq],
        stats=_stats(eq, pr),
    )


def _signals(vix, vix3m, gspc, n):
    gspc_r = _ret(gspc)
    rv5 = _rvol(gspc_r, RV_SHORT)
    rv10 = _rvol(gspc_r, RV_LONG)
    ma30 = _ma(vix, MA_WINDOW)
    evrp5 = [None if rv5[i] is None else vix[i] - rv5[i] for i in range(n)]
    evrp10 = [None if rv10[i] is None else vix[i] - rv10[i] for i in range(n)]
    sig_bm = [1.0 if vix[i] > vix3m[i] else 0.0 for i in range(n)]
    sig_e10 = [1.0 if evrp10[i] is not None and evrp10[i] <= 0 else 0.0 for i in range(n)]
    sig_e5_vix3m = [
        1.0 if (evrp5[i] is not None and evrp5[i] <= 0 and vix[i] > vix3m[i]) else 0.0
        for i in range(n)
    ]
    sig_e10_vix3m = [
        1.0 if (evrp10[i] is not None and evrp10[i] <= 0 and vix[i] > vix3m[i]) else 0.0
        for i in range(n)
    ]
    sig_e10_m30 = [
        1.0 if (evrp10[i] is not None and evrp10[i] <= 0
                and ma30[i] is not None and vix[i] > ma30[i]) else 0.0
        for i in range(n)
    ]
    sig_e5_m30 = [
        1.0 if (evrp5[i] is not None and evrp5[i] <= 0
                and ma30[i] is not None and vix[i] > ma30[i]) else 0.0
        for i in range(n)
    ]
    return dict(
        ma30=ma30, evrp5=evrp5, evrp10=evrp10,
        sig_bm=sig_bm, sig_e10=sig_e10,
        sig_e5_vix3m=sig_e5_vix3m, sig_e10_vix3m=sig_e10_vix3m,
        sig_e10_m30=sig_e10_m30, sig_e5_m30=sig_e5_m30,
    )


def _build(dates, spy_close, hedge_close, spy_open, hedge_open, vix, vix3m, gspc):
    n = len(dates)
    s = _signals(vix, vix3m, gspc, n)
    evrp5, evrp10 = s["evrp5"], s["evrp10"]
    strats = {}
    spy_eq = [100.0]
    spy_pr = [0.0]
    for i in range(1, n):
        r = (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0.0
        spy_pr.append(r)
        spy_eq.append(spy_eq[-1] * (1 + r))
    strats["100% SPY"] = dict(
        equity=[round(v, 2) for v in spy_eq],
        hedge_weight=[0.0] * n,
        vixy_sleeve_equity=[1.0] * n,
        stats=_stats(spy_eq, spy_pr),
    )

    def R(sig, sizing):
        return _strat(spy_close, hedge_close, spy_open, hedge_open, sig, vix, n, sizing)

    strats["Benchmark VIX>VIX3M"]   = R(s["sig_bm"], False)
    strats["Fixed eVRP(5D)+VIX3M"]  = R(s["sig_e5_vix3m"], False)
    strats["Sizing eVRP(5D)+VIX3M"] = R(s["sig_e5_vix3m"], True)
    strats["Fixed eVRP(10D)+VIX3M"] = R(s["sig_e10_vix3m"], False)
    strats["Sizing eVRP(10D)+VIX3M"]= R(s["sig_e10_vix3m"], True)
    strats["Fixed eVRP(5D)+MA30"]   = R(s["sig_e5_m30"], False)
    strats["Sizing eVRP(5D)+MA30"]  = R(s["sig_e5_m30"], True)
    strats["Fixed eVRP(10D)+MA30"]  = R(s["sig_e10_m30"], False)
    strats["Sizing eVRP(10D)+MA30"] = R(s["sig_e10_m30"], True)
    strats["Fixed eVRP(10D)"]       = R(s["sig_e10"], False)
    strats["Sizing eVRP(10D)"]      = R(s["sig_e10"], True)

    def C(sig, sizing):
        return _strat_c2c(spy_close, hedge_close, sig, vix, n, sizing)

    c2c = {}
    c2c["100% SPY"] = strats["100% SPY"]
    c2c["Benchmark VIX>VIX3M"]   = C(s["sig_bm"], False)
    c2c["Fixed eVRP(5D)+VIX3M"]  = C(s["sig_e5_vix3m"], False)
    c2c["Sizing eVRP(5D)+VIX3M"] = C(s["sig_e5_vix3m"], True)
    c2c["Fixed eVRP(10D)+VIX3M"] = C(s["sig_e10_vix3m"], False)
    c2c["Sizing eVRP(10D)+VIX3M"]= C(s["sig_e10_vix3m"], True)
    c2c["Fixed eVRP(5D)+MA30"]   = C(s["sig_e5_m30"], False)
    c2c["Sizing eVRP(5D)+MA30"]  = C(s["sig_e5_m30"], True)
    c2c["Fixed eVRP(10D)+MA30"]  = C(s["sig_e10_m30"], False)
    c2c["Sizing eVRP(10D)+MA30"] = C(s["sig_e10_m30"], True)
    c2c["Fixed eVRP(10D)"]       = C(s["sig_e10"], False)
    c2c["Sizing eVRP(10D)"]      = C(s["sig_e10"], True)

    rnd = lambda a, d=2: [round(v, d) if v is not None else None for v in a]
    return dict(
        dates=dates,
        spy_price=rnd(spy_close),
        hedge_price=rnd(hedge_close),
        spy_open=rnd(spy_open),
        hedge_open=rnd(hedge_open),
        vix=rnd(vix),
        vix3m=rnd(vix3m),
        evrp_10d=rnd(evrp10),
        evrp_5d=rnd(evrp5),
        strategies=strats,
        strategies_c2c=c2c,
    )


# ════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ════════════════════════════════════════════════════════════════════════════
def load_barchart_close(filename, log_fn=print):
    path = os.path.join(BARCHART, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Barchart CSV missing: {path}")
    df = pd.read_csv(path)
    tcol = df.columns[0]
    if "Last" not in df.columns:
        raise ValueError(f"{filename}: no 'Last' column (cols={list(df.columns)})")
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    vals = pd.to_numeric(df["Last"], errors="coerce")
    out = {}
    for t, v in zip(df[tcol], vals):
        if pd.notna(v) and v != 0:
            out[t.strftime("%Y-%m-%d")] = float(v)
    log_fn(f"    {filename}: {len(out)} rows (last {max(out) if out else 'NONE'})")
    return out


def _fetch_one(ticker, start, retries=4):
    import yfinance as yf
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True,
                             progress=False, threads=False)
            if df is None or df.empty:
                raise ValueError("empty frame")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Open", "Close"])
            dates = [d.strftime("%Y-%m-%d") for d in df.index]
            close = [round(float(x), 4) for x in df["Close"]]
            opn = [round(float(x), 4) for x in df["Open"]]
            return dates, close, opn
        except Exception as e:
            last_err = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"yfinance failed for {ticker}: {last_err}")


def _fetch_etfs(tickers, start, log_fn=print):
    out = {}
    for tk in tickers:
        log_fn(f"    yfinance {tk}...")
        out[tk] = _fetch_one(tk, start)
    return out


# ════════════════════════════════════════════════════════════════════════════
# COMPUTE
# ════════════════════════════════════════════════════════════════════════════
def compute(log_fn=print):
    log_fn("  Loading Barchart signal inputs (VIX / VIX3M / SPX)...")
    vix_d   = load_barchart_close(VIX_CSV, log_fn)
    vix3m_d = load_barchart_close(VIX3M_CSV, log_fn)
    spx_d   = load_barchart_close(SPX_CSV, log_fn)

    bc_axis = sorted(set(vix_d) & set(vix3m_d) & set(spx_d))
    if len(bc_axis) < MIN_DAYS:
        raise ValueError(f"Barchart axis too short: {len(bc_axis)} days")
    bc_axis = [d for d in bc_axis if d >= START]
    vix_b   = [vix_d[d] for d in bc_axis]
    vix3m_b = [vix3m_d[d] for d in bc_axis]
    spx_b   = [spx_d[d] for d in bc_axis]
    log_fn(f"  Barchart axis: {len(bc_axis)} days ({bc_axis[0]} -> {bc_axis[-1]})")

    log_fn("  Fetching ETF legs from yfinance...")
    all_tk = [SPY_TICKER] + list(HEDGE_TICKERS.values())
    raw = _fetch_etfs(all_tk, START, log_fn)
    spy_dates, spy_cl, spy_op = raw[SPY_TICKER]
    spy_cl_map = dict(zip(spy_dates, spy_cl))
    spy_op_map = dict(zip(spy_dates, spy_op))
    bc_close_map = dict(zip(bc_axis, spx_b))
    bc_vix_map   = dict(zip(bc_axis, vix_b))
    bc_vix3m_map = dict(zip(bc_axis, vix3m_b))
    bc_set = set(bc_axis)

    results = {}
    for key, tk in HEDGE_TICKERS.items():
        h_dates, h_cl, h_op = raw[tk]
        h_cl_map = dict(zip(h_dates, h_cl))
        h_op_map = dict(zip(h_dates, h_op))
        axis = sorted(set(spy_dates) & set(h_dates) & bc_set)
        if len(axis) < MIN_DAYS:
            raise ValueError(f"{tk} aligned axis too short: {len(axis)} days")
        results[key] = _build(
            axis,
            [spy_cl_map[d] for d in axis],
            [h_cl_map[d] for d in axis],
            [spy_op_map[d] for d in axis],
            [h_op_map[d] for d in axis],
            [bc_vix_map[d] for d in axis],
            [bc_vix3m_map[d] for d in axis],
            [bc_close_map[d] for d in axis],
        )
        log_fn(f"    {key}: {len(axis)} days ({axis[0]} -> {axis[-1]})")

    nb = len(bc_axis)
    s = _signals(vix_b, vix3m_b, spx_b, nb)
    li = nb - 1
    ma30, evrp5, evrp10 = s["ma30"], s["evrp5"], s["evrp10"]
    vixy_inst = results["vixy"]
    curr = dict(
        date=bc_axis[li],
        spy=vixy_inst["spy_price"][-1] if vixy_inst["spy_price"] else None,
        vixy=vixy_inst["hedge_price"][-1] if vixy_inst["hedge_price"] else None,
        vix=round(vix_b[li], 2),
        vix3m=round(vix3m_b[li], 2),
        evrp_5d=round(evrp5[li], 2) if evrp5[li] is not None else None,
        evrp_10d=round(evrp10[li], 2) if evrp10[li] is not None else None,
        vix_ma30=round(ma30[li], 2) if ma30[li] is not None else None,
        benchmark_on=vix_b[li] > vix3m_b[li],
        evrp10_on=evrp10[li] is not None and evrp10[li] <= 0,
        evrp10_ma30_on=(evrp10[li] is not None and evrp10[li] <= 0
                        and ma30[li] is not None and vix_b[li] > ma30[li]),
        sizing_e5_ma30_on=(evrp5[li] is not None and evrp5[li] <= 0
                           and ma30[li] is not None and vix_b[li] > ma30[li]),
        e5_vix3m_on=(evrp5[li] is not None and evrp5[li] <= 0
                     and vix_b[li] > vix3m_b[li]),
        e10_vix3m_on=(evrp10[li] is not None and evrp10[li] <= 0
                      and vix_b[li] > vix3m_b[li]),
        e5_ma30_on=(evrp5[li] is not None and evrp5[li] <= 0
                    and ma30[li] is not None and vix_b[li] > ma30[li]),
    )

    return dict(
        vixy=results["vixy"],
        sds=results["sds"],
        tza=results["tza"],
        current_signals=curr,
        computed_at=datetime.now(timezone.utc).isoformat(),
        signal_source="barchart",
        last_signal_date=bc_axis[li],
    )


def main():
    print("Computing VIXY model JSON...")
    payload = compute()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    cs = payload["current_signals"]
    print(f"  vixy.json saved ({os.path.getsize(OUT)/1024:.0f} KB)")
    print(f"  AS OF {cs['date']} | VIX {cs['vix']} VIX3M {cs['vix3m']} "
          f"eVRP5 {cs['evrp_5d']} eVRP10 {cs['evrp_10d']} MA30 {cs['vix_ma30']}")


if __name__ == "__main__":
    main()
