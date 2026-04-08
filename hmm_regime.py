"""
hmm_regime.py — Point-in-time HMM regime detection (no lookahead bias)
=======================================================================

Expanding-window quarterly refit with forward-only filtering.

For each historical date t, the regime label reflects ONLY information
available on or before t:

  1. Minimum training window: 5 years (~1260 trading days).
  2. Refit quarterly (first trading day of each calendar quarter).
     On each refit date t_r, fit GaussianHMM on returns[start : t_r].
  3. Between refits, use the most recent model and a manual forward
     algorithm to get P(state_t | x_1, ..., x_t) — filtered only.
     hmmlearn's score_samples uses forward-backward (smoothed), so
     we implement the forward pass ourselves.
  4. State-to-regime mapping: at each refit, derive bull_state =
     argmax(model.means_) using ONLY that model's parameters.

Used by:
  - data_updater.py  → compute_vix_hmm()
  - tab_market_risk.py → _compute_hmm()
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def _forward_filtered_states(model, X):
    """
    Run the forward algorithm and return the filtered state at each time step.

    Unlike hmmlearn's score_samples (which uses forward-backward and returns
    smoothed posteriors), this uses ONLY the forward pass, so the state at
    time t depends only on observations x_1, ..., x_t.

    Parameters
    ----------
    model : fitted GaussianHMM
        Must have startprob_, transmat_, means_, covars_ attributes.
    X : np.ndarray, shape (T, n_features)
        Observation sequence.

    Returns
    -------
    filtered_states : np.ndarray, shape (T,)
        Most likely state at each time step using only past + current obs.
    """
    n_states = model.n_components
    T = len(X)

    # Compute emission log-probabilities: log P(x_t | state=k)
    log_emit = np.zeros((T, n_states))
    for k in range(n_states):
        mean = model.means_[k]
        if model.covariance_type == "full":
            cov = model.covars_[k]
        elif model.covariance_type == "diag":
            cov = np.diag(model.covars_[k])
        elif model.covariance_type == "spherical":
            cov = model.covars_[k] * np.eye(X.shape[1])
        elif model.covariance_type == "tied":
            cov = model.covars_
        else:
            raise ValueError(f"Unknown covariance_type: {model.covariance_type}")
        log_emit[:, k] = multivariate_normal.logpdf(X, mean=mean, cov=cov)

    # Forward pass in log space
    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    # alpha[t, k] = log P(state_t=k, x_1..t)
    log_alpha = np.zeros((T, n_states))

    # Initialization
    log_alpha[0] = log_startprob + log_emit[0]

    # Recursion
    for t in range(1, T):
        for k in range(n_states):
            log_alpha[t, k] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, k])
                + log_emit[t, k]
            )

    # Filtered state = argmax P(state_t | x_1..t) = argmax alpha_t(k)
    filtered_states = np.argmax(log_alpha, axis=1)
    return filtered_states


def _forward_incremental(model, X_chunk, prev_log_alpha=None):
    """
    Run the forward algorithm on a chunk, optionally continuing from a
    previous forward pass. Returns filtered states AND the final log_alpha
    so the next chunk can continue from there.

    Parameters
    ----------
    model : fitted GaussianHMM
    X_chunk : np.ndarray, shape (T_chunk, n_features)
        New observations to process.
    prev_log_alpha : np.ndarray, shape (n_states,) or None
        The log_alpha from the last time step of the previous chunk.
        If None, starts fresh with model.startprob_.

    Returns
    -------
    filtered_states : np.ndarray, shape (T_chunk,)
    last_log_alpha : np.ndarray, shape (n_states,)
    """
    n_states = model.n_components
    T = len(X_chunk)

    # Compute emission log-probabilities for the chunk
    log_emit = np.zeros((T, n_states))
    for k in range(n_states):
        mean = model.means_[k]
        if model.covariance_type == "full":
            cov = model.covars_[k]
        elif model.covariance_type == "diag":
            cov = np.diag(model.covars_[k])
        elif model.covariance_type == "spherical":
            cov = model.covars_[k] * np.eye(X_chunk.shape[1])
        elif model.covariance_type == "tied":
            cov = model.covars_
        else:
            raise ValueError(f"Unknown covariance_type: {model.covariance_type}")
        log_emit[:, k] = multivariate_normal.logpdf(X_chunk, mean=mean, cov=cov)

    log_transmat = np.log(model.transmat_ + 1e-300)
    log_alpha = np.zeros((T, n_states))

    # First step of this chunk
    if prev_log_alpha is None:
        log_startprob = np.log(model.startprob_ + 1e-300)
        log_alpha[0] = log_startprob + log_emit[0]
    else:
        for k in range(n_states):
            log_alpha[0, k] = (
                np.logaddexp.reduce(prev_log_alpha + log_transmat[:, k])
                + log_emit[0, k]
            )

    # Recursion for rest of chunk
    for t in range(1, T):
        for k in range(n_states):
            log_alpha[t, k] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, k])
                + log_emit[t, k]
            )

    filtered_states = np.argmax(log_alpha, axis=1)
    return filtered_states, log_alpha[-1]


def fit_hmm_expanding(returns: pd.Series,
                      min_train_years: int = 5,
                      refit_freq: str = "QS",
                      n_components: int = 2,
                      random_state: int = 17,
                      n_iter: int = 300,
                      log_fn=None) -> pd.Series:
    """
    Expanding-window HMM with quarterly refit and forward-only filtering.

    Parameters
    ----------
    returns : pd.Series
        Daily returns with DatetimeIndex (NaN-free, sorted).
    min_train_years : int
        Minimum years of data before first regime label.
    refit_freq : str
        Pandas offset alias for refit schedule ("QS" = quarter start).
    n_components : int
        Number of HMM states (2 = bull/bear).
    random_state : int
        Seed for reproducibility.
    n_iter : int
        Max EM iterations per fit.
    log_fn : callable or None
        Logging function; receives progress messages.

    Returns
    -------
    pd.Series
        Integer regime labels (1 = bull, 0 = bear) indexed by date.
        Only dates with enough training history get a label.
    """
    from hmmlearn.hmm import GaussianHMM

    if log_fn is None:
        log_fn = lambda msg: None

    returns = returns.dropna().sort_index()
    if len(returns) < min_train_years * 252:
        log_fn("  HMM: not enough data for minimum training window")
        return pd.Series(dtype=float)

    # ── Determine refit dates ────────────────────────────────────────────────
    min_date = returns.index[0] + pd.DateOffset(years=min_train_years)
    # Generate quarter-start dates within the returns range
    all_refit_candidates = pd.date_range(
        start=returns.index[0], end=returns.index[-1], freq=refit_freq
    )
    # Keep only those on or after the min training window
    refit_dates = all_refit_candidates[all_refit_candidates >= min_date]
    # Snap each refit date to the nearest trading day on or after
    snapped_refits = []
    for rd in refit_dates:
        mask = returns.index >= rd
        if mask.any():
            snapped_refits.append(returns.index[mask][0])
    refit_dates = pd.DatetimeIndex(sorted(set(snapped_refits)))

    if len(refit_dates) == 0:
        log_fn("  HMM: no refit dates after minimum training window")
        return pd.Series(dtype=float)

    log_fn(f"  HMM: {len(refit_dates)} quarterly refits "
           f"({refit_dates[0].date()} → {refit_dates[-1].date()})")

    # ── Expanding-window refit loop ──────────────────────────────────────────
    regime_labels = pd.Series(np.nan, index=returns.index, dtype=float)

    # Build list of (refit_date, next_refit_date) intervals
    intervals = []
    for i, rd in enumerate(refit_dates):
        end = refit_dates[i + 1] if i + 1 < len(refit_dates) else returns.index[-1] + pd.Timedelta(days=1)
        intervals.append((rd, end))

    for i, (refit_date, next_refit) in enumerate(intervals):
        # Train on all data from start up to and including refit_date
        train_mask = returns.index <= refit_date
        train_data = returns[train_mask].values.reshape(-1, 1)

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(train_data)

        # Determine bull state from THIS model's parameters only
        bull_state = int(np.argmax(model.means_.ravel()))

        # Label dates in [refit_date, next_refit) using forward algorithm.
        # We warm up the forward pass on a trailing window before the chunk
        # to initialize the state distribution. A 252-day (~1yr) warm-up is
        # sufficient since the forward algorithm's memory of early states
        # decays exponentially. This makes the cost O(warmup + chunk) per
        # refit instead of O(full_history).
        chunk_mask = (returns.index >= refit_date) & (returns.index < next_refit)
        if not chunk_mask.any():
            continue

        WARMUP_DAYS = 252
        warmup_start_idx = max(0, train_data.shape[0] - WARMUP_DAYS)
        warmup_data = train_data[warmup_start_idx:]

        # Forward pass on warm-up window
        _, last_alpha = _forward_incremental(model, warmup_data, prev_log_alpha=None)

        # Forward pass on new chunk, continuing from warm-up's state
        chunk_data = returns[chunk_mask].values.reshape(-1, 1)
        chunk_states, _ = _forward_incremental(model, chunk_data, prev_log_alpha=last_alpha)
        chunk_dates = returns.index[chunk_mask]

        regime_labels.loc[chunk_dates] = (chunk_states == bull_state).astype(int)

        if (i + 1) % 20 == 0:
            log_fn(f"  HMM: {i + 1}/{len(intervals)} refits done...")

    # Drop NaN (dates before minimum training window)
    regime_labels = regime_labels.dropna().astype(int)
    regime_labels.name = "HMM_Regime"

    pct_bull = regime_labels.mean() * 100
    log_fn(f"  HMM: {len(regime_labels)} labels assigned, "
           f"bull = {pct_bull:.1f}%")

    return regime_labels
