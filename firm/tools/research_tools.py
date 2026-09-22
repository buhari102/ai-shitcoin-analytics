"""Tools that expose the referee, the archive, and the calibration record.

`test_hypothesis` is the only way an agent can make a claim about whether
something works. It pre-registers the hypothesis before evaluating it, so
the trial count used for deflation stays honest whether or not the agent
likes the answer.
"""

from __future__ import annotations

import json

from crewai.tools import tool

import config
from firm.research import calibration, decay, evolution, ledger
from firm.research.dsl import RuleSpec, feature_catalogue


@tool("test_hypothesis")
def test_hypothesis(genome_json: str, symbols: str = "", rationale: str = "") -> str:
    """Submit a strategy hypothesis to the immutable harness for scoring.

    The hypothesis is pre-registered before it runs, which increments the
    global trial count and makes the deflated Sharpe harder to clear. That is
    intentional: the more you search, the higher the bar.

    The genome is JSON of the form:
      {"name": "funding_crowded_short",
       "direction": "short",
       "conditions": [{"feature": "funding_z", "op": ">", "value": 2.0},
                      {"feature": "momentum_24", "op": ">", "value": 0.05}],
       "horizon": 24, "pt_atr": 2.0, "sl_atr": 1.0,
       "rationale": "Crowded longs paying extreme funding after a sharp rally
                     tend to mean-revert as leverage is flushed."}

    Available features:
    """
    try:
        data = json.loads(genome_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"genome is not valid JSON: {exc}", "features": list(_features())})

    if rationale and not data.get("rationale"):
        data["rationale"] = rationale

    try:
        spec = RuleSpec.from_dict(data)
        spec.validate()
    except (KeyError, ValueError, TypeError) as exc:
        return json.dumps({"error": f"invalid genome: {exc}", "available_features": list(_features())})

    if not (spec.rationale or "").strip():
        return json.dumps(
            {"error": "rationale is mandatory. State why this should work economically, "
                      "before you find out whether it did."}
        )

    universe = [s.strip().upper() for s in symbols.split(",") if s.strip()] or list(config.WATCHLIST)

    try:
        scorecard = evolution.evaluate_candidate(spec, universe, author="agent")
    except Exception as exc:  # noqa: BLE001 - report failures to the agent rather than crashing the crew
        return json.dumps({"error": f"evaluation failed: {exc}"})

    overall = scorecard.get("overall", {})
    deflated = scorecard.get("deflated", {})
    consistency = scorecard.get("fold_consistency", {})

    return json.dumps(
        {
            "strategy": spec.name,
            "rule": spec.describe(),
            "verdict": scorecard.get("verdict"),
            "passed": scorecard.get("passed"),
            "rejection_reasons": scorecard.get("rejection_reasons", []),
            "n_trades": overall.get("n_trades", 0),
            "win_rate": overall.get("win_rate"),
            "mean_net_return": overall.get("mean_return"),
            "sharpe": overall.get("sharpe"),
            "profit_factor": overall.get("profit_factor"),
            "max_drawdown": overall.get("max_drawdown"),
            "deflated_sharpe_probability": deflated.get("dsr"),
            "sharpe_threshold_from_luck": deflated.get("sharpe_threshold"),
            "trials_to_date": deflated.get("n_trials"),
            "fold_consistency": consistency,
            "mean_cost_bps": scorecard.get("mean_cost_bps"),
            "cost_drag_ratio": scorecard.get("cost_drag_ratio"),
            "by_regime": scorecard.get("by_regime"),
            "fitness": scorecard.get("fitness"),
            "note": "This scorecard is final. It cannot be re-run for a better number; "
                    "re-testing a variant increments the trial count and raises the bar further.",
        },
        default=str,
    )


def _features() -> dict:
    from firm.research.dsl import TRADEABLE_FEATURES

    return TRADEABLE_FEATURES


# Append the live feature catalogue to the tool description so proposers
# always see the exact vocabulary they are allowed to use.
test_hypothesis.description = (test_hypothesis.description or "") + "\n" + feature_catalogue()


@tool("get_archive_elites")
def get_archive_elites() -> str:
    """Best strategies found so far, one set per behavioural niche.

    Read this before proposing anything. Niches with no occupant are where
    the unexplored hypothesis space is.
    """
    elites = evolution.elites()
    summary = evolution.archive_summary()

    return json.dumps(
        {
            "archive": summary,
            "elites": [
                {
                    "name": e["name"],
                    "niche": e["niche"],
                    "fitness": e["fitness"],
                    "passed": bool(e["passed"]),
                    "genome": e.get("genome"),
                    "sharpe": e.get("scorecard", {}).get("overall", {}).get("sharpe"),
                    "n_trades": e.get("scorecard", {}).get("overall", {}).get("n_trades"),
                }
                for e in elites
            ],
            "attribution": decay.attribution(),
        },
        default=str,
    )


@tool("get_failure_log")
def get_failure_log(keyword: str = "") -> str:
    """Hypotheses that have already been tested and rejected, with reasons.

    Knowing why something failed is more informative than knowing something
    succeeded. Check here before proposing, so the crew stops re-testing the
    same idea and inflating the trial count for nothing.

    Args:
        keyword: Optional search term to filter past hypotheses.
    """
    return json.dumps(
        {
            "failures": ledger.failure_log(limit=25),
            "matching_hypotheses": ledger.search_hypotheses(keyword, limit=25) if keyword else [],
            "ledger": ledger.summary(),
        },
        default=str,
    )


@tool("get_research_state")
def get_research_state() -> str:
    """Overall state of the research programme: ledger, archive, calibration, decay.

    The CIO should read this before allocating, because it says how much of
    the book rests on validated work versus untested opinion.
    """
    return json.dumps(
        {
            "harness_version": config.HARNESS_VERSION,
            "ledger": ledger.summary(),
            "archive": evolution.archive_summary(),
            "forecasts": calibration.summary(),
            "holdout": {
                "tail_days_reserved": config.HOLDOUT_TAIL_DAYS,
                "budget_per_strategy": config.HOLDOUT_BUDGET_PER_STRATEGY,
                "spent": ledger.holdout_history(),
            },
            "risk_limits": {
                "max_position_pct": config.MAX_POSITION_PCT,
                "max_total_exposure_pct": config.MAX_TOTAL_EXPOSURE_PCT,
                "max_correlated_exposure_pct": config.MAX_CORRELATED_EXPOSURE_PCT,
                "min_r_multiple": config.MIN_R_MULTIPLE,
            },
        },
        default=str,
    )


@tool("get_agent_calibration")
def get_agent_calibration(agent_role: str = "") -> str:
    """How well an agent's stated probabilities have matched reality.

    An agent with a negative skill score is worse than useless at
    forecasting and its probabilities must be discounted, not trusted.

    Args:
        agent_role: Role name to score. Empty returns the full leaderboard.
    """
    if agent_role:
        return json.dumps(calibration.score_agent(agent_role), default=str)

    board = calibration.leaderboard()
    return json.dumps(
        {
            "leaderboard": board.to_dict(orient="records") if not board.empty else [],
            "summary": calibration.summary(),
            "note": "skill_score > 0 means better than always predicting the base rate. "
            "Agents below that lose their vote.",
        },
        default=str,
    )


@tool("record_agent_forecast")
def record_agent_forecast(
    agent_role: str,
    symbol: str,
    direction: str,
    probability: float,
    horizon_hours: int,
    entry_price: float,
    target_price: float,
    invalidation: float,
    thesis: str = "",
) -> str:
    """Record a falsifiable forecast so it can be scored later.

    Every trade idea in the final report must be recorded here. A call that
    cannot be graded afterwards is not research.

    Args:
        agent_role: Which agent is making the call.
        symbol: Binance spot symbol.
        direction: 'long' or 'short'.
        probability: Probability the target is hit before invalidation, 0 to 1.
        horizon_hours: How long the thesis has to play out.
        entry_price: Intended entry.
        target_price: Profit target.
        invalidation: Hard stop.
        thesis: One sentence on why.
    """
    try:
        forecast_id = calibration.record_forecast(
            agent_role=agent_role,
            symbol=symbol,
            direction=direction,
            probability=float(probability),
            horizon_hours=int(horizon_hours),
            entry_price=float(entry_price),
            target_price=float(target_price),
            invalidation=float(invalidation),
            thesis=thesis,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    calibrated = calibration.apply_calibration(agent_role, float(probability))

    return json.dumps(
        {
            "forecast_id": forecast_id,
            "recorded": True,
            "stated_probability": probability,
            "calibrated_probability": calibrated,
            "note": "The calibrated probability is what sizing should use. It reflects this "
            "agent's historical accuracy, not its confidence.",
        },
        default=str,
    )
