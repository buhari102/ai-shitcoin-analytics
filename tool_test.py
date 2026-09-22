"""Invoke every agent tool directly, with no LLM involved.

This is the audit the plan calls for: confirm each tool returns real,
traceable numbers before trusting anything an agent says about them. Costs
nothing and needs no API key.

    venv\\Scripts\\python.exe tool_test.py
"""

from __future__ import annotations

import json
import sys

from firm import tools

SYMBOL = "DOGEUSDT"


def show(name: str, raw: str, keys: list[str] | None = None) -> bool:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  {name}: NOT VALID JSON -> {raw[:120]}")
        return False

    if isinstance(data, dict) and "error" in data:
        print(f"  {name}: error -> {data['error']}")
        return False

    print(f"  {name}: ok")
    for k in keys or []:
        if isinstance(data, dict) and k in data:
            v = data[k]
            text = json.dumps(v, default=str)
            print(f"      {k} = {text[:150]}")
    return True


def main() -> int:
    results = []

    print("=" * 70)
    print("MARKET TOOLS")
    print("=" * 70)
    results.append(
        show(
            "screen_memecoin_universe",
            tools.screen_memecoin_universe.run(min_quote_volume_usd=5_000_000),
            ["liquid_count"],
        )
    )
    results.append(
        show(
            "get_price_history_stats",
            tools.get_price_history_stats.run(symbol=SYMBOL),
            ["last_price", "returns", "realised_volatility_per_bar"],
        )
    )
    results.append(
        show(
            "get_order_book_depth",
            tools.get_order_book_depth.run(symbol=SYMBOL, notional_usd=1000.0),
            ["spread_bps", "modelled_round_trip_bps", "estimated_slippage_bps"],
        )
    )

    print("\n" + "=" * 70)
    print("DERIVATIVES TOOLS")
    print("=" * 70)
    results.append(
        show(
            "get_funding_profile",
            tools.get_funding_profile.run(symbol=SYMBOL),
            ["current_funding_rate", "current_annualised_pct", "percentile_vs_own_history"],
        )
    )
    results.append(
        show(
            "get_live_positioning",
            tools.get_live_positioning.run(symbol=SYMBOL),
            ["long_short_account_ratio", "backtestable"],
        )
    )
    results.append(
        show("get_flow_coverage", tools.get_flow_coverage.run(), ["max_span_days", "assessment"])
    )

    print("\n" + "=" * 70)
    print("QUANT TOOLS")
    print("=" * 70)
    results.append(
        show(
            "analyze_symbol",
            tools.analyze_symbol.run(symbol=SYMBOL),
            ["volatility", "btc_relationship", "risk"],
        )
    )

    price = json.loads(tools.get_price_history_stats.run(symbol=SYMBOL)).get("last_price", 0.1)
    results.append(
        show(
            "evaluate_trade_ev",
            tools.evaluate_trade_ev.run(
                symbol=SYMBOL,
                direction="long",
                entry=price,
                stop=price * 0.97,
                target=price * 1.09,
                win_probability=0.4,
            ),
            ["r_multiple", "breakeven_win_rate", "net_ev_pct", "recommended_position_pct", "verdict"],
        )
    )
    results.append(
        show(
            "get_correlation_matrix",
            tools.get_correlation_matrix.run(symbols="DOGEUSDT,SHIBUSDT,PEPEUSDT"),
            ["correlation_to_btc", "highly_correlated_pairs"],
        )
    )

    print("\n" + "=" * 70)
    print("RESEARCH TOOLS")
    print("=" * 70)
    results.append(
        show("get_research_state", tools.get_research_state.run(), ["ledger", "archive"])
    )
    results.append(show("get_archive_elites", tools.get_archive_elites.run(), ["archive"]))
    results.append(show("get_failure_log", tools.get_failure_log.run(), ["ledger"]))
    results.append(
        show("get_agent_calibration", tools.get_agent_calibration.run(), ["leaderboard"])
    )

    # The referee, exercised with a deliberately weak hypothesis.
    genome = json.dumps(
        {
            "name": "tooltest_probe",
            "direction": "long",
            "conditions": [{"feature": "return_zscore", "op": "<", "value": -2.0}],
            "horizon": 24,
            "pt_atr": 2.0,
            "sl_atr": 1.0,
            "rationale": "Tool-layer smoke test of the referee path.",
        }
    )
    results.append(
        show(
            "test_hypothesis",
            tools.test_hypothesis.run(genome_json=genome, symbols="DOGEUSDT,SHIBUSDT"),
            ["verdict", "n_trades", "sharpe", "deflated_sharpe_probability", "rejection_reasons"],
        )
    )

    print("\n" + "=" * 70)
    passed = sum(results)
    print(f"  {passed}/{len(results)} tools returned usable data")
    print("=" * 70)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
