"""
jm_regime.py — Point-in-time Statistical Jump Model regime detection (no lookahead)
====================================================================================

Drop-in companion to hmm_regime.py. Same architecture, same point-in-time
discipline (expanding-window quarterly refit + forward-only filtering), so the
JM Regime Indicator is directly comparable to the HMM Regime Indicator already
in the Trend Health Model.

Model
-----
Statistical Jump Model (Nystrup / Shu), 2 states, jump penalty lambda = 50.
Coordinate-descent + dynamic-programming solver. Feature set is the parsimonious
"Chapter 3" set from the Shu dissertation (validated in the jm_vs_hmm notebook):

    DD_10       EWM downside deviation, halflife 10
    Sortino_20  EWM mean return (hl 20) / EWM downside deviation (hl 20)
    Sortino_60  EWM mean return (hl 60) / EWM downside deviation (hl 60)

Point-in-time guarantee
-----------------------
For each historical date t the regime label uses ONLY information available on
or before t:

  1. Minimum training window: 5 years (~1260 trading days).
  2. Refit centroids YEARLY (first trading day of each calendar year) on
     an EXPANDING window: features from the start of history up to the refit
     date. Feature standardization (z-score) uses ONLY that training window.
  3. State-to-regime mapping is derived at each refit from the training period
     only (the centroid whose training days have the higher mean return = bull).
  4. Between refits, the regime at each day t is the FORWARD-FILTERED jump-model
     state: argmin_k V[t, k] of the forward DP recursion, which depends only on
     observations up to t. (This equals the last state of a Viterbi pass over
     data up to t, i.e. the quantity the notebook extracts in its point-in-time
     experiment, but computed forward-only so no future day can influence it.)

Returns 1 = bull (risk-on), 0 = bear (risk-off) — same convention as
fit_hmm_expanding, so _compute_jm can mirror _compute_hmm exactly.

No external dependencies beyond numpy / pandas (the JumpModel is implemented
here), so requirements.txt and the GitHub Actions install step are unchanged.

Used by:
  - tab_market_risk.py -> _compute_jm()
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Jump Model solver (coordinate descent + DP), lifted from the validated
#  jm_vs_hmm notebook. DP inner loop vectorized over states for speed.
# ─────────────────────────────────────────────────────────────────────────────
class JumpModel:
    def __init__(self, n_states=2, jump_penalty=50.0, max_iter=50,
                 n_init=10, tol=1e-6, random_state=42):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None

    def _loss_matrix(self, Y):
        # L[t, k] = 0.5 * ||y_t - c_k||^2
        # (Y - centroids[:,None,:]) -> (K, T, D); sum over D -> (K, T) -> transpose
        diff = Y[None, :, :] - self.centroids_[:, None, :]
        return 0.5 * np.sum(diff ** 2, axis=2).T

    def _dp(self, L):
        """Full Viterbi (forward + backtrack) — used during fitting.

        Fast scalar path for K=2 (the only case used here): avoids per-step
        numpy allocation/dispatch, which is the dominant cost over long
        training windows. Falls back to a vectorized path for K>2.
        """
        T, K = L.shape
        lam = self.jump_penalty
        if K == 2:
            L0 = L[:, 0]; L1 = L[:, 1]
            v0 = float(L0[0]); v1 = float(L1[0])
            bp0 = np.empty(T, dtype=np.int8); bp1 = np.empty(T, dtype=np.int8)
            for t in range(1, T):
                # state 0: stay (v0) vs jump from 1 (v1 + lam)
                if v0 <= v1 + lam:
                    nb0 = 0; c0 = v0
                else:
                    nb0 = 1; c0 = v1 + lam
                # state 1: stay (v1) vs jump from 0 (v0 + lam)
                if v1 <= v0 + lam:
                    nb1 = 1; c1 = v1
                else:
                    nb1 = 0; c1 = v0 + lam
                v0 = float(L0[t]) + c0
                v1 = float(L1[t]) + c1
                bp0[t] = nb0; bp1[t] = nb1
            states = np.empty(T, dtype=int)
            if v0 <= v1:
                states[T - 1] = 0; obj = v0
            else:
                states[T - 1] = 1; obj = v1
            for t in range(T - 2, -1, -1):
                states[t] = bp0[t + 1] if states[t + 1] == 0 else bp1[t + 1]
            return states, float(obj)
        # general fallback (K > 2)
        trans = lam * (~np.eye(K, dtype=bool)).astype(float)
        V = np.zeros((T, K))
        bp = np.zeros((T, K), dtype=int)
        V[0, :] = L[0, :]
        for t in range(1, T):
            costs = V[t - 1, :][:, None] + trans
            bp[t, :] = np.argmin(costs, axis=0)
            V[t, :] = L[t, :] + costs[bp[t, :], np.arange(K)]
        states = np.zeros(T, dtype=int)
        states[T - 1] = int(np.argmin(V[T - 1, :]))
        for t in range(T - 2, -1, -1):
            states[t] = bp[t + 1, states[t + 1]]
        return states, float(np.min(V[T - 1, :]))

    def _update_centroids(self, Y, states):
        K = self.n_states
        c = np.zeros((K, Y.shape[1]))
        for k in range(K):
            mask = states == k
            c[k] = Y[mask].mean(axis=0) if mask.sum() else Y[np.random.randint(len(Y))]
        return c

    def _kpp(self, Y, rng):
        T, D = Y.shape
        K = self.n_states
        c = np.zeros((K, D))
        c[0] = Y[rng.integers(0, T)]
        for k in range(1, K):
            d = np.min([np.sum((Y - c[j]) ** 2, axis=1) for j in range(k)], axis=0)
            s = d.sum()
            c[k] = Y[rng.choice(T, p=d / s)] if s > 0 else Y[rng.integers(0, T)]
        return c

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best_obj, best_states, best_c = np.inf, None, None
        for _ in range(self.n_init):
            self.centroids_ = self._kpp(Y, rng)
            prev = np.inf
            for _ in range(self.max_iter):
                L = self._loss_matrix(Y)
                states, obj = self._dp(L)
                self.centroids_ = self._update_centroids(Y, states)
                if abs(prev - obj) < self.tol:
                    break
                prev = obj
            L = self._loss_matrix(Y)
            states, obj = self._dp(L)
            if obj < best_obj:
                best_obj, best_states, best_c = obj, states.copy(), self.centroids_.copy()
        self.centroids_, self.labels_ = best_c, best_states
        return self


def build_jm_features(returns):
    """Chapter 3 feature set: DD_10, Sortino_20, Sortino_60 (raw, unstandardized)."""
    neg_sq = (returns.clip(upper=0)) ** 2
    dd_10 = neg_sq.ewm(halflife=10, min_periods=10).mean().apply(np.sqrt)
    dd_20 = neg_sq.ewm(halflife=20, min_periods=20).mean().apply(np.sqrt)
    dd_60 = neg_sq.ewm(halflife=60, min_periods=60).mean().apply(np.sqrt)
    avg_20 = returns.ewm(halflife=20, min_periods=20).mean()
    avg_60 = returns.ewm(halflife=60, min_periods=60).mean()
    feats = pd.DataFrame({
        "DD_10": dd_10,
        "Sortino_20": avg_20 / dd_20.replace(0, np.nan),
        "Sortino_60": avg_60 / dd_60.replace(0, np.nan),
    }).dropna()
    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  Forward-filtered (point-in-time) state assignment with frozen centroids.
#  V[t, k] = L[t, k] + min_j ( V[t-1, j] + lam * (j != k) )
#  filtered_state[t] = argmin_k V[t, k]   -> depends only on obs up to t.
#  V is carried across the whole inference span; when centroids change at a
#  refit, subsequent emission costs use the new centroids (natural online
#  behavior). Renormalizing V each step (subtract row min) prevents overflow
#  and does not change argmin or the relative state costs.
# ─────────────────────────────────────────────────────────────────────────────
def _forward_filtered_step(v_prev, loss_t, lam, K):
    if v_prev is None:
        v = loss_t.copy()
    else:
        trans = lam * (~np.eye(K, dtype=bool)).astype(float)
        costs = v_prev[:, None] + trans          # costs[j, k]
        v = loss_t + np.min(costs, axis=0)
    v = v - v.min()                              # renormalize (argmin invariant)
    return v


def fit_jm_expanding(returns: pd.Series,
                     n_states: int = 2,
                     jump_penalty: float = 50.0,
                     refit_freq: str = "YS",
                     min_train_years: int = 5,
                     n_init: int = 5,
                     max_iter: int = 20,
                     log_fn=print) -> pd.Series:
    """
    Point-in-time JM regime labels for a daily return series.

    Parameters
    ----------
    returns : pd.Series
        Daily (log) returns with a sorted, NaN-free DatetimeIndex.
    n_states : int
        Number of regimes (2 = bull/bear).
    jump_penalty : float
        Lambda. 50 matches the validated notebook.
    refit_freq : str
        Pandas offset alias for refit cadence ("QS" = quarter start, matching
        the HMM module).
    min_train_years : int
        Minimum training history before the first label is produced.
    n_init : int
        Random restarts per JM fit.

    Returns
    -------
    pd.Series
        Regime labels indexed by date: 1 = bull (risk-on), 0 = bear (risk-off).
        Forward-filtered only (no lookahead).
    """
    returns = returns.dropna().sort_index()
    if len(returns) < min_train_years * 252:
        log_fn(f"  JM: not enough history ({len(returns)} days) for "
               f"{min_train_years}y minimum window")
        return pd.Series(dtype=float)

    # Refit dates: first trading day on/after each quarter start, once the
    # minimum training window has elapsed.
    min_date = returns.index[0] + pd.DateOffset(years=min_train_years)
    cal = pd.date_range(start=returns.index[0], end=returns.index[-1], freq=refit_freq)
    refit_dates = []
    for rd in cal:
        if rd < min_date:
            continue
        mask = returns.index >= rd
        if mask.any():
            d = returns.index[mask][0]
            if not refit_dates or d != refit_dates[-1]:
                refit_dates.append(d)
    if not refit_dates:
        log_fn("  JM: no refit dates after minimum window")
        return pd.Series(dtype=float)

    labels = pd.Series(np.nan, index=returns.index, dtype=float)
    lam = float(jump_penalty)
    K = int(n_states)

    centroids = None
    feat_mean = feat_std = None
    bull_state = 0
    v_prev = None
    n_refits = 0

    for i, refit_date in enumerate(refit_dates):
        next_refit = (refit_dates[i + 1] if i + 1 < len(refit_dates)
                      else returns.index[-1] + pd.Timedelta(days=1))

        # ---- refit centroids on expanding window [start, refit_date] ----
        train_ret = returns[returns.index <= refit_date]
        feats_tr = build_jm_features(train_ret)
        if len(feats_tr) < min_train_years * 200:   # need enough post-warmup rows
            continue
        feat_mean = feats_tr.mean()
        feat_std = feats_tr.std().replace(0, 1.0)
        Y_tr = ((feats_tr - feat_mean) / feat_std).values

        jm = JumpModel(n_states=K, jump_penalty=lam, n_init=n_init,
                       max_iter=max_iter, random_state=42).fit(Y_tr)

        # state-to-regime mapping from TRAINING period only
        ret_tr_aligned = train_ret.reindex(feats_tr.index).values
        m = [ret_tr_aligned[jm.labels_ == k].mean() if (jm.labels_ == k).any()
             else -np.inf for k in range(K)]
        bull_state = int(np.argmax(m))
        centroids = jm.centroids_.copy()
        n_refits += 1
        log_fn(f"  JM refit #{n_refits} @ {refit_date.date()} "
               f"(train {feats_tr.index[0].date()}->{feats_tr.index[-1].date()}, "
               f"{len(feats_tr)} rows, bull_state={bull_state})")

        # reset the forward filter at each refit boundary so the new centroid
        # geometry isn't blended with the previous segment's accumulated costs
        v_prev = None

        # ---- forward-filtered inference for this segment ----
        seg_mask = (returns.index >= refit_date) & (returns.index < next_refit)
        seg_dates = returns.index[seg_mask]
        if len(seg_dates) == 0:
            continue

        # features for the segment, standardized with THIS refit's training stats
        feats_all = build_jm_features(returns[returns.index < next_refit])
        feats_seg = feats_all.reindex(seg_dates).dropna()

        infer = JumpModel(n_states=K, jump_penalty=lam)
        infer.centroids_ = centroids
        if len(feats_seg):
            Y_seg = ((feats_seg - feat_mean) / feat_std).values
            L_seg = infer._loss_matrix(Y_seg)   # (n_seg, K)
            for j, d in enumerate(feats_seg.index):
                v_prev = _forward_filtered_step(v_prev, L_seg[j], lam, K)
                state = int(np.argmin(v_prev))
                labels.loc[d] = 1.0 if state == bull_state else 0.0

    out = labels.dropna()
    if len(out):
        log_fn(f"  JM: produced {len(out)} labels "
               f"({out.index[0].date()}->{out.index[-1].date()}), "
               f"bear fraction {(out == 0).mean():.1%}, "
               f"switches {int((out.diff().abs() > 0).sum())}")
    return out
