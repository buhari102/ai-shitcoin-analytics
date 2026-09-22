"""Stage 7: alpha decay monitoring and retirement.

Every edge decays, and crypto edges decay fast. A strategy is re-scored only
on bars that did not exist when it was discovered, which is genuinely out of
sample without spending the holdout budget, and compared against its own
backtest distribution.

Retirement is automatic. Leaving a dead strategy in the book because it used
to work is how a research programme quietly becomes a museum.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import pandas as pd

import config
from firm.harness.engine import StrategySpec, evaluate_strategy
from firm.research import evolution
from firm.research.dsl import RuleSpec, compile_rule


def monitor_strategy(record: dict, symbols: list[str] | None = None) -> dict:
    """Compare post-discovery performance against the backtest distribution."""
    genome = record.get("genome")
    if not isinstance(genome, dict):
        return {"name": record.get("name"), "status": "unreadable genome"}

    rule = RuleSpec.from_dict(genome)
    discovered_at = pd.Timestamp(record["created_at"])
    backtest = record.get("scorecard", {}).get("overall", {})

    spec = StrategySpec(
        name=rule.name,
        signal_fn=compile_rule(rule),
        symbols=symbols or list(config.WATCHLIST),
        horizon=rule.horizon,
        pt_atr=rule.pt_atr,
        sl_atr=rule.sl_atr,
    )

    # 'all' rather than 'research': post-discovery data is the whole point,
    # and this is monitoring rather than selection.
    live = evaluate_strategy(spec, n_trials=1, segment="all", since=discovered_at)
    live_stats = live.get("overall", {})

    n_live = live_stats.get("n_trades", 0)
    if n_live < 5:
        return {
            "name": rule.name,
            "status": "insufficient post-discovery data",
            "live_trades": n_live,
            "discovered_at": str(discovered_at),
        }

    bt_mean = backtest.get("mean_return", 0.0)
    bt_vol = backtest.get("volatility", 0.0)
    live_mean = live_stats.get("mean_return", 0.0)

    # Is the live mean plausibly drawn from the backtest distribution?
    if bt_vol > 0 and n_live > 1:
        standard_error = bt_vol / math.sqrt(n_live)
        z = (live_mean - bt_mean) / standard_error
    else:
        z = 0.0

    decayed = z < config.DECAY_ZSCORE_RETIRE
    degraded = live_mean <= 0 < bt_mean

    return {
        "name": rule.name,
        "discovered_at": str(discovered_at),
        "backtest_mean": bt_mean,
        "backtest_sharpe": backtest.get("sharpe", 0.0),
        "live_trades": n_live,
        "live_mean": live_mean,
        "live_sharpe": live_stats.get("sharpe", 0.0),
        "live_win_rate": live_stats.get("win_rate", 0.0),
        "decay_zscore": float(z),
        "decayed": bool(decayed),
        "sign_flipped": bool(degraded),
        "recommendation": "retire" if (decayed or degraded) else "keep",
    }


def run_monitor(symbols: list[str] | None = None, auto_retire: bool = False) -> dict:
    """Check every active archived strategy."""
    active = [s for s in evolution.all_strategies(limit=500) if s.get("status") == "active"]
    reports = []
    retired = []

    for record in active:
        try:
            report = monitor_strategy(record, symbols)
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the sweep
            report = {"name": record.get("name"), "status": f"monitor failed: {exc}"}
        reports.append(report)

        if auto_retire and report.get("recommendation") == "retire":
            reason = "sign_flip" if report.get("sign_flipped") else "decay"
            evolution.retire(report["name"], reason)
            retired.append({"name": report["name"], "reason": reason})

    return {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "active_strategies": len(active),
        "reports": reports,
        "retired": retired,
    }


def attribution(symbols: list[str] | None = None) -> dict:
    """Which niches are actually carrying the book.

    Feeds the Post-Mortem agent, and through it the proposer prompts: knowing
    that funding-driven swing strategies keep dying is more useful than any
    individual scorecard.
    """
    strategies = evolution.all_strategies(limit=500)
    if not strategies:
        return {"niches": [], "note": "archive is empty"}

    rows = []
    for s in strategies:
        card = s.get("scorecard", {})
        overall = card.get("overall", {})
        rows.append(
            {
                "niche": s.get("niche"),
                "status": s.get("status"),
                "fitness": s.get("fitness", 0.0),
                "passed": bool(s.get("passed")),
                "sharpe": overall.get("sharpe", 0.0),
                "n_trades": overall.get("n_trades", 0),
                "cost_drag": card.get("cost_drag_ratio"),
            }
        )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("niche")
        .agg(
            candidates=("niche", "size"),
            promoted=("passed", "sum"),
            best_fitness=("fitness", "max"),
            mean_sharpe=("sharpe", "mean"),
            retired=("status", lambda s: int(sum(str(x).startswith("retired") for x in s))),
        )
        .reset_index()
        .sort_values("best_fitness", ascending=False)
    )

    return {
        "niches": grouped.to_dict(orient="records"),
        "total_candidates": len(df),
        "total_promoted": int(df["passed"].sum()),
        "promotion_rate": float(df["passed"].mean()),
    }


if __name__ == "__main__":
    print(json.dumps(run_monitor(), indent=2, default=str))
    print(json.dumps(attribution(), indent=2, default=str))
