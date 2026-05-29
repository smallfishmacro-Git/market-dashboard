"""
equity_regime_updater.py
------------------------
Pulls S&P 500 earnings aggregates from Portfolio123 and writes equity_regime.json
for the SmallFishMacro Equity tab.

Series pulled:
  - #SPEPSCNY  : Blended CY/NY EPS (forward blend)
  - #SPEPSCY   : Pure current-year EPS
  - #SPEPSNY   : Pure next-year EPS
  - #SPEPSQ    : Blended quarterly EPS
  - #SPEPSTTM  : Trailing 12M EPS
  - #SPYieldBlend, #SPRPBlend : valuation overlays
  - DataSeries("S&P 500 - EPS Revisions Breadth") : custom aggregate

Output: equity_regime.json with:
  - levels : weekly levels for each series
  - yoy    : YoY % change for each series
  - composite : 4-factor regime score history
  - latest : current snapshot values for the KPI strip

Requires GitHub Actions secrets: P123_API_ID, P123_API_KEY
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import p123api
import pandas as pd


def _utcnow() -> datetime:
    """Timezone-aware UTC now (replaces deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Maximum history P123 allows on this subscription tier: 5 years.
# data_universe() returns: "You can only test historical data after <today - 5y>"
# So we cap the start at ~5 years back. Long-history merging (pre-2021) would
# require a separate CSV import (e.g. from Alfi's substack archive).
START_DATE = (_utcnow() - timedelta(days=5 * 365 + 10)).strftime("%Y-%m-%d")
END_DATE   = _utcnow().strftime("%Y-%m-%d")

# Output target — committed by the GitHub Action, consumed by smallfish-equity Vercel app
OUTPUT_PATH = Path(os.environ.get("EQUITY_REGIME_OUTPUT", "equity_regime.json"))

# Formulas to retrieve. P123 aggregate constants are global series — they return
# the same value for all stocks in the universe, so we collapse to one row per date.
# All entries below are derived/aggregate series that work WITHOUT a vendor data license.
#
# CRITICAL: Each aggregate must be wrapped in Close(0, <series>) to make it a
# point-in-time time-series lookup. Without Close(0, ...), P123 treats the #SP*
# constant as a literal that returns the current latest value for all asOfDts.
FORMULAS = [
    ("eps_cny",      "Close(0, #SPEPSCNY)"),     # Blended CY/NY EPS
    ("eps_cy",       "Close(0, #SPEPSCY)"),      # Pure current-year estimate
    ("eps_ny",       "Close(0, #SPEPSNY)"),      # Pure next-year estimate
    ("eps_q",        "Close(0, #SPEPSQ)"),       # Blended quarterly EPS
    ("eps_ttm",      "Close(0, #SPEPSTTM)"),     # Trailing 12M EPS
    ("yield_blend",  "Close(0, #SPYieldBlend)"), # Blended earnings yield
    ("rp_blend",     "Close(0, #SPRPBlend)"),    # Blended equity risk premium
    ("rev_breadth",  'Close(0, GetSeries("S&P 500 - EPS Revisions Breadth"))'),  # custom aggregate
]


# ----------------------------------------------------------------------------
# Historical CSV merge configuration
# ----------------------------------------------------------------------------
# Map P123 aggregate-series CSV exports to our column names. The CSVs are
# user-built aggregate series (e.g. MySPEPSCNY) that approximate the system
# constants (#SPEPSCNY) but typically differ in scale by a constant factor
# (because the user version omits the S&P 500 Close multiplication step).
#
# We resolve this with anchored rescaling: compute the ratio at the API join
# date and multiply all historical values by it. The drift is typically <2%
# over 5 years, well below chart-readable noise.
#
# Drop CSVs into HISTORICAL_DIR. Entries here are tried in order; missing
# files are skipped silently (so you can add files incrementally).
HISTORICAL_DIR = Path(os.environ.get("EQUITY_HISTORICAL_DIR", "historical"))

HISTORICAL_CSV_MAP = {
    # api_column     : list of CSV filenames to try (first match wins)
    "eps_cny":     ["MySPEPSCNY.csv",            "P123_Series_37_20260522.csv"],
    "eps_ttm":     ["SPEPSTTM.csv",              "P123_Series_16_20260522.csv"],
    "eps_q":       ["MySPEPSQ.csv"],
    "eps_cy":      ["MySPEPSCY.csv"],
    "eps_ny":      ["MySPEPSNY.csv"],
    # rev_breadth: P123 custom series, definition changed from 1-week to 1-month
    # revisions in May 2026. The CSV holds full 2003+ history of the NEW definition.
    # The API also returns the new definition going forward. No scale mismatch.
    # NOTE: this CSV is daily; merge code resamples to weekly Friday automatically.
    "rev_breadth": ["RevisionsBreadth1M.csv",    "P123_Series_19495_20260529.csv"],
    # yield_blend and rp_blend can be derived downstream from EPS + SPX price.
}


# ----------------------------------------------------------------------------
# Pull
# ----------------------------------------------------------------------------

def fetch_series(client: p123api.Client) -> pd.DataFrame:
    """
    Pull all aggregate series as weekly time series.

    Uses data_universe() instead of data() because:
      - data() requires a FactSet/Compustat license (limited to IBM/MSFT/INTC otherwise)
      - data_universe() works WITHOUT a license for derived/aggregate series
        (P123 aggregate constants #SP* and DataSeries() qualify as 'derived')

    Strategy:
      - Build a list of weekly Friday dates from START_DATE to END_DATE
      - Call data_universe() with universe='DJIA' (cheap, ~30 stocks)
      - The aggregate constants return the same value for all 30 stocks per date
      - We take the first row per date to collapse back to one row per week
    """
    names    = [n for n, _ in FORMULAS]
    formulas = [f for _, f in FORMULAS]

    # Build weekly Friday dates (P123 docs recommend weekend dates for asOfDts)
    start = pd.Timestamp(START_DATE)
    end   = pd.Timestamp(END_DATE)
    # W-FRI gives Friday-anchored weekly dates
    weekly_dates = pd.date_range(start=start, end=end, freq="W-FRI")
    as_of_dts = [d.strftime("%Y-%m-%d") for d in weekly_dates]
    print(f"[p123] requesting {len(formulas)} series across {len(as_of_dts)} weekly dates")
    print(f"[p123] range: {as_of_dts[0]} -> {as_of_dts[-1]}")

    # P123 has a limit on the number of asOfDts per call; chunk into batches.
    # Empirically ~250 dates per call works; we'll use 200 to be safe.
    CHUNK = 200
    all_frames = []

    for i in range(0, len(as_of_dts), CHUNK):
        batch = as_of_dts[i:i + CHUNK]
        payload = {
            "universe":  "DJIA",        # small, cheap carrier universe; aggregates ignore it
            "asOfDts":   batch,
            "formulas":  formulas,
            "names":     names,         # data_universe DOES accept 'names'
            "pitMethod": "Complete",
            "precision": 4,
            "type":      "stock",
        }
        print(f"[p123] batch {i // CHUNK + 1}: {batch[0]} -> {batch[-1]} ({len(batch)} dates)")
        df_batch = client.data_universe(payload, to_pandas=True)
        if df_batch is None or df_batch.empty:
            print(f"[warn] empty response for batch {i // CHUNK + 1}")
            continue
        all_frames.append(df_batch)

    if not all_frames:
        raise RuntimeError("P123 returned no data across all batches")

    raw = pd.concat(all_frames, ignore_index=True)
    print(f"[p123] raw shape: {raw.shape}, columns: {list(raw.columns)}")
    print(f"[p123] first row:\n{raw.head(1)}")

    # Identify the date column (P123 uses 'dates' lowercase plural)
    date_col = None
    for candidate in ("dates", "Date", "date", "asOfDt", "AsOfDt"):
        if candidate in raw.columns:
            date_col = candidate
            break
    if date_col is None:
        raise RuntimeError(f"Could not find date column in {list(raw.columns)}")

    raw[date_col] = pd.to_datetime(raw[date_col])

    # Drop P123 metadata columns we don't need
    for col in ("p123Uids", "tickers", "P123 UID", "Ticker", "Name", "ticker"):
        if col in raw.columns:
            raw = raw.drop(columns=col)

    # Aggregate constants return identical values across all tickers in the universe.
    # Take the first row per date to collapse back to one series.
    agg = raw.groupby(date_col).first().reset_index()

    # Keep only date + our named series columns
    keep_cols = [date_col] + [n for n in names if n in agg.columns]
    missing = [n for n in names if n not in agg.columns]
    if missing:
        print(f"[warn] columns not returned by P123: {missing}")
    agg = agg[keep_cols]

    # Set date as index
    agg = agg.set_index(date_col).sort_index()
    agg.index.name = "Date"

    # Coerce all to numeric
    for col in agg.columns:
        agg[col] = pd.to_numeric(agg[col], errors="coerce")

    # Sanity check on EPS scale. S&P 500 EPS should be roughly $50-300.
    # If P123 returns thousands, it's the aggregate (sum) not the index value.
    if "eps_cny" in agg.columns:
        sample = agg["eps_cny"].dropna().iloc[-1] if not agg["eps_cny"].dropna().empty else None
        print(f"[diag] eps_cny latest value: {sample}")
        if sample and sample > 1000:
            print(f"[diag] eps_cny is at index-aggregate scale (sum across constituents),")
            print(f"[diag] not per-share. Divisor ~190 typical for S&P 500 index value.")
            print(f"[diag] Will keep raw values; YoY% is scale-invariant for regime signals.")

    print(f"[p123] collapsed to {len(agg)} weekly rows; final columns: {list(agg.columns)}")
    return agg


# ----------------------------------------------------------------------------
# Historical CSV merge
# ----------------------------------------------------------------------------

def _read_p123_csv(path: Path) -> pd.DataFrame:
    """
    Read a P123 aggregate-series CSV export.

    Format:
      Aggregate Series,<name>
      Data Start,<yyyy-mm-dd>
      Data End,<yyyy-mm-dd>
      <blank>
      Date,Value
      <yyyy-mm-dd>,<float>
      ...

    Returns a DataFrame with Date index and a single 'value' column.

    If the source is daily (median gap == 1 calendar day), automatically
    resamples to weekly Friday snapshots using the last observation in each
    week — matches the weekly cadence of the rest of the pipeline.
    """
    # The first 3 lines are metadata; row 4 is blank; row 5 is the real header.
    df = pd.read_csv(path, skiprows=4)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Normalize to weekly-Friday cadence so all sources align on the same date grid.
    # P123 CSV conventions vary:
    #   - MySPEPSCNY, SPEPSTTM: weekly Saturday/Sunday snapshots
    #   - RevisionsBreadth1M: business-daily
    #   - API: weekly Friday
    # Without this normalization, the merge unions different date grids and the
    # output JSON balloons to ~2x rows with each series sparse.
    if len(df) >= 10:
        median_gap_days = df.index.to_series().diff().dt.days.dropna().median()
        if median_gap_days is not None and median_gap_days <= 3:
            # Daily source → resample to weekly Friday using last value in each week
            df = df.resample("W-FRI").last().dropna()
            print(f"[csv] {path.name}: resampled daily → weekly Friday ({len(df)} rows)")
        else:
            # Already weekly but possibly off-Friday: snap each date back to the
            # prior Friday so Sat/Sun snapshots align with Fri-anchored API.
            days_back = (df.index.weekday - 4) % 7   # Fri=0, Sat=1, Sun=2, Mon=3, ...
            df.index = df.index - pd.to_timedelta(days_back, unit="D")
            # Deduplicate in case two adjacent observations now share a Friday
            df = df[~df.index.duplicated(keep="last")]
            print(f"[csv] {path.name}: snapped weekly to Friday ({len(df)} rows)")

    return df[["value"]]


def _find_csv(filenames: list[str]) -> Path | None:
    """Search HISTORICAL_DIR for the first matching CSV; return None if absent."""
    if not HISTORICAL_DIR.exists():
        return None
    for fname in filenames:
        candidate = HISTORICAL_DIR / fname
        if candidate.exists():
            return candidate
    return None


def merge_with_historical(api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splice P123 CSV historical data onto the API frame.

    For each column with a configured CSV:
      1. Load the CSV (longer history, possibly different scale).
      2. Align CSV dates to API dates via nearest-match (P123 CSVs typically
         use Sat/Sun weekends while the API returns Fridays, so exact-equality
         intersection finds zero overlap).
      3. Compute the median ratio (API / CSV) in the overlap → scaling factor.
      4. Apply scaling to historical CSV values.
      5. Splice: rescaled CSV before the API's first date; API after.

    If no overlap exists, the CSV is skipped with a warning. Missing CSVs are
    silently skipped (the column keeps API-only history).
    """
    if not HISTORICAL_DIR.exists():
        print(f"[merge] no historical directory at {HISTORICAL_DIR}, skipping merge")
        return api_df

    merged = api_df.copy()
    api_first_date = api_df.index.min()

    for col, candidates in HISTORICAL_CSV_MAP.items():
        if col not in api_df.columns:
            continue

        csv_path = _find_csv(candidates)
        if csv_path is None:
            print(f"[merge] no CSV for '{col}' (tried: {candidates}); using API only")
            continue

        try:
            hist = _read_p123_csv(csv_path)
        except Exception as e:
            print(f"[merge] could not read {csv_path.name}: {e}; skipping")
            continue

        # Align CSV dates to API dates via nearest-match (tolerance: 4 days).
        # Both series are weekly; this handles Fri-vs-Sat/Sun snapshot differences.
        api_series  = api_df[col].dropna().rename("api").reset_index()
        hist_series = hist["value"].dropna().rename("hist").reset_index()
        api_series  = api_series.rename(columns={"index": "date", "Date": "date"})
        hist_series = hist_series.rename(columns={"index": "date", "Date": "date"})
        api_series["date"]  = pd.to_datetime(api_series["date"])
        hist_series["date"] = pd.to_datetime(hist_series["date"])
        api_series  = api_series.sort_values("date").reset_index(drop=True)
        hist_series = hist_series.sort_values("date").reset_index(drop=True)

        aligned = pd.merge_asof(
            api_series,
            hist_series,
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=4),
        ).dropna(subset=["api", "hist"])

        if len(aligned) < 5:
            print(f"[merge] insufficient overlap for '{col}' ({len(aligned)} aligned points); skipping")
            continue

        # Compute scaling factor (median for robustness)
        ratios = aligned["api"] / aligned["hist"]
        scale  = float(ratios.median())
        drift  = float(ratios.max() - ratios.min()) / scale * 100
        print(f"[merge] {col}: scale={scale:.4f} from {len(aligned)} aligned points "
              f"(drift {drift:.2f}%), CSV={csv_path.name}")

        # Build rescaled historical pre-API series
        hist_rescaled = hist["value"] * scale
        hist_pre      = hist_rescaled.loc[hist_rescaled.index < api_first_date]

        # Combine: historical (pre-API) + API (current)
        api_col  = api_df[col].dropna()
        combined = pd.concat([hist_pre, api_col]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

        # Reindex merged frame to the union of dates
        all_dates = merged.index.union(combined.index)
        merged    = merged.reindex(all_dates).sort_index()
        merged[col] = combined.reindex(all_dates)

    merged.index.name = "Date"
    print(f"[merge] final frame: {len(merged)} rows, "
          f"{merged.index.min().date()} → {merged.index.max().date()}")
    return merged


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

def add_yoy(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add YoY % change columns. 52-week lookback since data is weekly."""
    out = df.copy()
    for col in cols:
        if col in df.columns:
            out[f"{col}_yoy"] = (df[col] / df[col].shift(52) - 1.0) * 100.0
    return out


def composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    4-factor regime score (each component ∈ {-1, 0, +1}). Sum maps to:
        +3..+4  EARNINGS EXPANSION
        +1..+2  IMPROVING
         0      NEUTRAL
       -1..-2  WEAKENING
       -3..-4  CONTRACTION
    """
    s = pd.DataFrame(index=df.index)

    # 1. EPS growth: CY blend YoY positive
    s["f_growth"] = (df["eps_cny_yoy"] > 0).astype(int) - (df["eps_cny_yoy"] < 0).astype(int)

    # 2. Revisions breadth above its 52-week rolling median.
    # The custom series measures "% of stocks with 1-month positive revisions";
    # it's bounded roughly [14, 67] with long-term mean ~38. A static threshold
    # is meaningless on this scale; we use a dynamic baseline so the factor
    # captures regime shifts vs the recent past rather than absolute level.
    breadth_baseline = df["rev_breadth"].rolling(window=52, min_periods=12).median()
    s["f_breadth"] = (
        (df["rev_breadth"] > breadth_baseline).astype(int)
        - (df["rev_breadth"] < breadth_baseline).astype(int)
    )

    # 3. Acceleration: CY YoY rising vs 13 weeks ago
    accel = df["eps_cny_yoy"] - df["eps_cny_yoy"].shift(13)
    s["f_accel"] = (accel > 0).astype(int) - (accel < 0).astype(int)

    # 4. Quarterly EPS YoY positive (faster-moving confirmation)
    s["f_quarterly"] = (df["eps_q_yoy"] > 0).astype(int) - (df["eps_q_yoy"] < 0).astype(int)

    s["score"] = s[["f_growth", "f_breadth", "f_accel", "f_quarterly"]].sum(axis=1)

    def label(x: int) -> str:
        if x >= 3:  return "EARNINGS EXPANSION"
        if x >= 1:  return "IMPROVING"
        if x == 0:  return "NEUTRAL"
        if x >= -2: return "WEAKENING"
        return "CONTRACTION"

    s["regime"] = s["score"].apply(label)
    return s


# ----------------------------------------------------------------------------
# Serialize
# ----------------------------------------------------------------------------

def to_json_payload(df: pd.DataFrame, sig: pd.DataFrame) -> dict:
    """Build the equity_regime.json payload consumed by the Next.js tab."""
    # Reset index for serialization
    df_out  = df.reset_index().rename(columns={"Date": "date"})
    sig_out = sig.reset_index().rename(columns={"Date": "date"})
    df_out["date"]  = df_out["date"].dt.strftime("%Y-%m-%d")
    sig_out["date"] = sig_out["date"].dt.strftime("%Y-%m-%d")

    # CRITICAL: Replace NaN with None so json.dumps emits valid `null` instead of
    # the non-standard `NaN` literal (which Python tolerates but JSON.parse rejects).
    # The first ~52 weekly rows of every *_yoy column are NaN by design (no prior
    # year to compare against), so this affects ~52 rows × ~5 yoy cols = ~260 cells.
    df_out  = df_out.astype(object).where(pd.notna(df_out), None)
    sig_out = sig_out.astype(object).where(pd.notna(sig_out), None)

    # Latest snapshot for KPI strip
    last        = df.iloc[-1]
    last_sig    = sig.iloc[-1]
    prev_4w_sig = sig.iloc[-5] if len(sig) >= 5 else last_sig

    # Long-term breadth baseline for the KPI tile context line
    # "48.7 · +10pp above median"
    breadth_series   = df["rev_breadth"].dropna()
    breadth_median   = float(breadth_series.median()) if len(breadth_series) > 0 else None
    breadth_baseline_52w = float(breadth_series.rolling(52, min_periods=12).median().iloc[-1]) \
                           if len(breadth_series) >= 12 else None

    latest = {
        "asof":            df.index[-1].strftime("%Y-%m-%d"),
        "eps_cny":         _r(last.get("eps_cny")),
        "eps_cny_yoy":     _r(last.get("eps_cny_yoy")),
        "eps_cy":          _r(last.get("eps_cy")),
        "eps_ny":          _r(last.get("eps_ny")),
        "eps_q":           _r(last.get("eps_q")),
        "eps_q_yoy":       _r(last.get("eps_q_yoy")),
        "eps_ttm":         _r(last.get("eps_ttm")),
        "yield_blend":     _r(last.get("yield_blend")),
        "rp_blend":        _r(last.get("rp_blend")),
        "rev_breadth":     _r(last.get("rev_breadth")),
        "rev_breadth_prior":      _r(df["rev_breadth"].iloc[-5] if len(df) >= 5 else None),
        "rev_breadth_median_lt":  _r(breadth_median),     # long-term median, ~38
        "rev_breadth_baseline_52w": _r(breadth_baseline_52w),  # current 52w rolling median (trigger reference)
        "score":           int(last_sig["score"]),
        "regime":          last_sig["regime"],
        "components": {
            "growth":    int(last_sig["f_growth"]),
            "breadth":   int(last_sig["f_breadth"]),
            "accel":     int(last_sig["f_accel"]),
            "quarterly": int(last_sig["f_quarterly"]),
        },
    }

    return {
        "generated_at": _utcnow().isoformat(),
        "start_date":   df.index[0].strftime("%Y-%m-%d"),   # actual earliest row (post-merge)
        "end_date":     df.index[-1].strftime("%Y-%m-%d"),
        "api_start":    START_DATE,                          # for diagnostics: where the API window begins
        "frequency":    "weekly",
        "latest":       latest,
        "series":       df_out.to_dict(orient="records"),
        "composite":    sig_out.to_dict(orient="records"),
    }


def _r(v, ndigits: int = 2):
    """Round + handle NaN for JSON serialization."""
    if v is None or pd.isna(v):
        return None
    return round(float(v), ndigits)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    api_id  = os.environ.get("P123_API_ID")
    api_key = os.environ.get("P123_API_KEY")
    if not (api_id and api_key):
        print("ERROR: P123_API_ID and P123_API_KEY env vars required", file=sys.stderr)
        return 1

    with p123api.Client(api_id=api_id, api_key=api_key) as client:
        raw = fetch_series(client)

    # Splice in P123 CSV historical data (pre-API window) if present
    raw = merge_with_historical(raw)

    # Compute YoY for the level series
    yoy_cols = ["eps_cny", "eps_cy", "eps_ny", "eps_q", "eps_ttm"]
    enriched = add_yoy(raw, yoy_cols)

    # 4-factor composite
    sig = composite_signal(enriched)

    # Serialize. allow_nan=False makes json.dumps RAISE on any NaN that slipped
    # through (instead of silently emitting the non-standard `NaN` literal that
    # JSON.parse on the frontend would reject). We already convert NaN→None in
    # to_json_payload, so this should never trigger; if it does, it's a bug.
    payload = to_json_payload(enriched, sig)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"[ok] wrote {OUTPUT_PATH} — {len(payload['series'])} weekly rows, regime={payload['latest']['regime']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
