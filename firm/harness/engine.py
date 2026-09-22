"""The referee itself.

A strategy is a pure function from a causal feature frame to a side series
(+1 long, -1 short, 0 flat). The engine labels the resulting trades, charges
adversarial costs, runs purged walk-forward folds, deflates the Sharpe by the
number of trials actually attempted, and returns a verdict.

Agents may propose strategies. They may not touch anything in this module,
and they do not get to argue with the scorecard.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import config
from firm.harness import costs, cv, features, stats

SignalFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class StrategySpec:
    """A candidate strategy. `signal_fn` must be causal and side-effect free."""

    name: str
    signal_fn: SignalFn
    symbols: list[str] = field(default_factory=lambda: list(config.WATCHLIST))
    interval: str = "1h"
    horizon: int = config.TRIPLE_BARRIER_HORIZON_BARS
    pt_atr: float = config.TRIPLE_BARRIER_PT_ATR
    sl_atr: float = config.TRIPLE_BARRIER_SL_ATR
    notional_usd: float = 1000.0
    description: str = ""

    def code_hash(self) -> str:
        try:
            source = inspect.getsource(self.signal_fn)
        except (OSError, TypeError):
            source = repr(self.signal_fn)
        payload = f"{source}|{self.interval}|{self.horizon}|{self.pt_atr}|{self.sl_atr}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def holdout_boundary() -> pd.Timestamp:
    """Start of the reserved tail slice. Research must stay strictly before it."""
    return pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=config.HOLDOUT_TAIL_DAYS)


def _adv_usd(df: pd.DataFrame) -> float:
    """Average daily dollar volume, used to scale market impact."""
    if "quote_volume" not in df.columns or df.empty:
        return 0.0
    bars_per_day = 24  # 1h bars
    recent = df["quote_volume"].tail(bars_per_day * 30)
    if recent.empty:
        return 0.0
    return float(recent.mean() * bars_per_day)


def build_events(
    spec: StrategySpec,
    segment: str = "research",
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate labeled, cost-adjusted trades across every symbol.

    `since`/`until` narrow the window further, which the decay monitor uses to
    score a strategy only on data that did not exist when it was discovered.
    """
    from firm.harness.labeling import triple_barrier_labels

    boundary = holdout_boundary()
    all_events = []

    for symbol in spec.symbols:
        feats = features.build_features(symbol, spec.interval, benchmark=config.BENCHMARK)
        if feats.empty:
            continue

        if segment == "research":
            feats = feats[feats["timestamp"] < boundary]
        elif segment == "holdout":
            feats = feats[feats["timestamp"] >= boundary]
        elif segment != "all":
            raise ValueError(f"unknown segment: {segment}")

        if since is not None:
            feats = feats[feats["timestamp"] >= since]
        if until is not None:
            feats = feats[feats["timestamp"] < until]

        if len(feats) < 200:
            continue

        feats = feats.reset_index(drop=True)

        try:
            side = spec.signal_fn(feats)
        except Exception as exc:  # noqa: BLE001 - a broken candidate scores zero, it does not crash the loop
            raise StrategyError(f"signal_fn failed on {symbol}: {exc}") from exc

        side = pd.Series(side, index=feats.index).fillna(0).clip(-1, 1)
        if (side == 0).all():
            continue

        events = triple_barrier_labels(
            feats, side, horizon=spec.horizon, pt_atr=spec.pt_atr, sl_atr=spec.sl_atr
        )
        if events.empty:
            continue

        events = costs.apply_costs(events, symbol, spec.notional_usd, _adv_usd(feats))
        events["symbol"] = symbol
        regimes = features.regime_label(feats)
        events["regime"] = regimes.iloc[events["entry_idx"]].to_numpy()
        all_events.append(events)

    if not all_events:
        return pd.DataFrame()

    combined = pd.concat(all_events, ignore_index=True)
    return combined.sort_values("entry_time").reset_index(drop=True)


class StrategyError(RuntimeError):
    """Raised when a candidate strategy cannot be evaluated."""


def evaluate_strategy(
    spec: StrategySpec,
    n_trials: int = 1,
    segment: str = "research",
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
) -> dict:
    """Score a strategy. `n_trials` is the honest count of variants attempted.

    The trial count comes from the hypothesis ledger, not from the agent.
    Understating it inflates the deflated Sharpe, which is precisely the
    failure mode the ledger exists to prevent.
    """
    events = build_events(spec, segment=segment, since=since, until=until)

    scorecard: dict = {
        "strategy": spec.name,
        "code_hash": spec.code_hash(),
        "harness_version": config.HARNESS_VERSION,
        "segment": segment,
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_trials_declared": n_trials,
        "symbols": spec.symbols,
        "interval": spec.interval,
    }

    if events.empty:
        scorecard.update({"verdict": "no_trades", "n_trades": 0, "passed": False})
        return scorecard

    net = events["net_return"].to_numpy()
    scorecard["overall"] = stats.trade_statistics(net)
    scorecard["gross"] = stats.trade_statistics(events["gross_return"].to_numpy())
    scorecard["mean_cost_bps"] = float(events["cost_bps"].mean())

    # How much of the gross edge is eaten by costs? Above 1.0 means the
    # strategy only exists in a frictionless fantasy.
    gross_mean = float(events["gross_return"].mean())
    cost_mean = float(events["cost_bps"].mean()) / 10_000
    scorecard["cost_drag_ratio"] = float(cost_mean / abs(gross_mean)) if gross_mean else float("inf")

    scorecard["deflated"] = stats.deflated_sharpe_ratio(net, n_trials=n_trials)

    folds = cv.purged_walk_forward_splits(events)
    fold_results = []
    fold_returns: dict[str, pd.Series] = {}

    for i, fold in enumerate(folds):
        test_returns = events.iloc[fold.test_idx]["net_return"]
        fold_stats = stats.trade_statistics(test_returns.to_numpy())
        fold_results.append(
            {
                "fold": i,
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "n_train": int(len(fold.train_idx)),
                "purged": fold.purged,
                "embargoed": fold.embargoed,
                **{k: fold_stats[k] for k in ("n_trades", "win_rate", "mean_return", "sharpe") if k in fold_stats},
            }
        )
        fold_returns[f"fold_{i}"] = test_returns.reset_index(drop=True)

    scorecard["folds"] = fold_results

    if fold_results:
        sharpes = [f["sharpe"] for f in fold_results if "sharpe" in f]
        scorecard["fold_consistency"] = {
            "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            "std_sharpe": float(np.std(sharpes)) if sharpes else 0.0,
            "positive_folds": int(sum(s > 0 for s in sharpes)),
            "total_folds": len(sharpes),
        }

    # Per-regime breakdown: an edge that only exists in one regime is a
    # regime bet, and should be sized and retired as one.
    if "regime" in events.columns:
        by_regime = {}
        for regime, group in events.groupby("regime"):
            by_regime[str(regime)] = stats.trade_statistics(group["net_return"].to_numpy())
        scorecard["by_regime"] = by_regime

    scorecard.update(_verdict(scorecard))
    return scorecard


def _verdict(scorecard: dict) -> dict:
    """Apply the promotion gates. Every failure reason is recorded."""
    overall = scorecard.get("overall", {})
    deflated = scorecard.get("deflated", {})
    consistency = scorecard.get("fold_consistency", {})

    reasons = []
    n_trades = overall.get("n_trades", 0)

    if n_trades < config.MIN_TRADES_FOR_SIGNIFICANCE:
        reasons.append(f"only {n_trades} trades, need {config.MIN_TRADES_FOR_SIGNIFICANCE}")
    if overall.get("mean_return", 0) <= 0:
        reasons.append("net mean return is not positive after costs")
    if deflated.get("dsr", 0) < 0.95:
        reasons.append(f"deflated Sharpe probability {deflated.get('dsr', 0):.3f} < 0.95")
    if consistency and consistency.get("positive_folds", 0) < consistency.get("total_folds", 1) * 0.6:
        reasons.append(
            f"only {consistency.get('positive_folds')}/{consistency.get('total_folds')} folds positive"
        )
    if scorecard.get("cost_drag_ratio", 0) > 0.8:
        reasons.append(f"costs consume {scorecard['cost_drag_ratio']:.0%} of gross edge")

    passed = not reasons
    return {
        "passed": passed,
        "verdict": "promote" if passed else "reject",
        "rejection_reasons": reasons,
    }
