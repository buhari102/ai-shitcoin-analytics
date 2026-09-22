"""Performance statistics with multiple-testing correction.

The central problem: if you test 500 strategies, the best one looks brilliant
by luck alone. A raw Sharpe ratio from a search over many candidates is not
evidence of anything. The Deflated Sharpe Ratio (Bailey and Lopez de Prado)
converts it into a probability that the true Sharpe exceeds zero, given how
many trials were actually run - which is exactly why the hypothesis ledger
tracks the trial count.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray | pd.Series) -> float:
    """Per-observation Sharpe. Not annualised: annualising trade-level
    returns invites nonsense when trade frequency varies."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd)


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected maximum Sharpe under the null that no strategy has edge.

    This is the bar a candidate must clear simply to be distinguishable from
    the best of N coin flips.
    """
    if n_trials <= 1 or trial_sharpe_std <= 0:
        return 0.0
    n = float(n_trials)
    z1 = sps.norm.ppf(1 - 1 / n)
    z2 = sps.norm.ppf(1 - 1 / (n * math.e))
    return float(trial_sharpe_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    n_trials: int,
    trial_sharpe_std: float | None = None,
) -> dict:
    """Probability that the true Sharpe is positive, after deflation.

    Args:
        returns: per-trade or per-period net returns.
        n_trials: how many strategy variants were evaluated to produce this one.
        trial_sharpe_std: dispersion of Sharpe across those trials. When
            unknown, 1/sqrt(n_obs) is the standard approximation.

    A DSR below ~0.95 means the result is not distinguishable from the best
    of N random strategies.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n_obs = len(r)
    if n_obs < 3:
        return {"sharpe": 0.0, "sharpe_threshold": 0.0, "dsr": 0.0, "n_obs": n_obs, "n_trials": n_trials}

    sr = sharpe_ratio(r)
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))

    if trial_sharpe_std is None:
        trial_sharpe_std = 1.0 / math.sqrt(n_obs)

    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std)

    # Variance of the Sharpe estimator under non-normal returns.
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom <= 0:
        return {
            "sharpe": sr,
            "sharpe_threshold": sr0,
            "dsr": 0.0,
            "skew": skew,
            "kurtosis": kurt,
            "n_obs": n_obs,
            "n_trials": n_trials,
        }

    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    dsr = float(sps.norm.cdf(z))

    return {
        "sharpe": sr,
        "sharpe_threshold": sr0,
        "dsr": dsr,
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": n_obs,
        "n_trials": n_trials,
    }


def probability_of_backtest_overfitting(returns_matrix: pd.DataFrame, n_splits: int = 8) -> dict:
    """Combinatorially Symmetric Cross-Validation (Bailey et al.).

    Takes a T x N matrix of per-period returns, one column per strategy
    variant. Splits time into S blocks, and for every way of choosing S/2
    blocks as in-sample, checks how the in-sample winner ranks out of sample.

    PBO is the fraction of splits where the in-sample winner lands in the
    bottom half out of sample. Above ~0.5 the selection process is worse than
    useless: you are reliably picking the strategy most overfit to noise.
    """
    df = returns_matrix.dropna(axis=1, how="all").dropna()
    n_periods, n_strategies = df.shape
    if n_strategies < 2 or n_periods < n_splits * 2:
        return {"pbo": float("nan"), "n_strategies": n_strategies, "n_periods": n_periods, "reason": "insufficient data"}

    if n_splits % 2:
        n_splits -= 1
    blocks = np.array_split(np.arange(n_periods), n_splits)

    logits = []
    for is_blocks in itertools.combinations(range(n_splits), n_splits // 2):
        oos_blocks = [b for b in range(n_splits) if b not in is_blocks]
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])

        is_perf = df.iloc[is_idx].apply(sharpe_ratio, axis=0)
        oos_perf = df.iloc[oos_idx].apply(sharpe_ratio, axis=0)

        best = is_perf.idxmax()
        # Relative rank of the in-sample winner within the OOS distribution.
        rank = oos_perf.rank(pct=True)[best]
        rank = min(max(float(rank), 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))

    logits_arr = np.array(logits)
    return {
        "pbo": float((logits_arr <= 0).mean()),
        "median_logit": float(np.median(logits_arr)),
        "n_strategies": n_strategies,
        "n_periods": n_periods,
        "n_combinations": len(logits),
    }


def trade_statistics(net_returns: np.ndarray | pd.Series) -> dict:
    """Descriptive stats the Risk and CIO agents reason over."""
    r = np.asarray(net_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"n_trades": 0}

    wins = r[r > 0]
    losses = r[r <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    # Compound in log space: over tens of thousands of trades a plain cumprod
    # overflows to inf (or underflows to zero) and the drawdown becomes NaN.
    safe = np.clip(r, -0.999999, None)
    log_equity = np.cumsum(np.log1p(safe))
    log_peak = np.maximum.accumulate(log_equity)
    drawdown = np.expm1(log_equity - log_peak)
    total_return = float(np.expm1(log_equity[-1])) if np.isfinite(log_equity[-1]) else -1.0

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0

    return {
        "n_trades": int(len(r)),
        "win_rate": float(len(wins) / len(r)),
        "mean_return": float(r.mean()),
        "median_return": float(np.median(r)),
        "total_return": total_return,
        "volatility": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "sharpe": sharpe_ratio(r),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": float(drawdown.min()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(avg_win / avg_loss) if avg_loss > 0 else float("inf"),
    }


def breakeven_win_rate(payoff_ratio: float) -> float:
    """Win rate required to break even at a given reward:risk."""
    if payoff_ratio <= 0:
        return 1.0
    return 1.0 / (1.0 + payoff_ratio)


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    """Full Kelly. Always scaled down by config.KELLY_FRACTION before use."""
    if payoff_ratio <= 0:
        return 0.0
    f = win_prob - (1 - win_prob) / payoff_ratio
    return max(0.0, float(f))
