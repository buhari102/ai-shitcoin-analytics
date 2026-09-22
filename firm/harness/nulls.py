"""Null validation: prove the harness can reject noise.

Before any positive result is believed, the referee must demonstrate that it
correctly scores strategies known to have no edge. If random signals pass the
gates, every downstream result is noise with good formatting.

Run:
    venv\\Scripts\\python.exe -m firm.harness.nulls
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config
from firm.harness.engine import StrategySpec, evaluate_strategy


def random_signal(seed: int, density: float = 0.05):
    """Pure coin flips. Any edge found here is a harness bug."""

    def fn(df: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(seed + len(df))
        draws = rng.random(len(df))
        side = np.zeros(len(df))
        side[draws < density / 2] = 1
        side[(draws >= density / 2) & (draws < density)] = -1
        return pd.Series(side, index=df.index)

    return fn


def always_long(df: pd.DataFrame) -> pd.Series:
    """Buy and hold, sampled periodically. Not an edge, just beta."""
    side = np.zeros(len(df))
    side[::24] = 1
    return pd.Series(side, index=df.index)


def alternating(df: pd.DataFrame) -> pd.Series:
    """Deterministic but information-free."""
    side = np.zeros(len(df))
    side[::12] = 1
    side[6::12] = -1
    return pd.Series(side, index=df.index)


def lookahead_cheat(df: pd.DataFrame) -> pd.Series:
    """Deliberate look-ahead: trades on the NEXT bar's return.

    This is a positive control. It must score spectacularly. If it does not,
    the labeling or cost code is broken in a way that destroys real edges too.
    """
    future = df["close"].pct_change().shift(-1)
    return pd.Series(np.sign(future).fillna(0), index=df.index)


def run_null_suite(symbols: list[str], interval: str = "1h", n_random: int = 5) -> dict:
    """Score the null battery and report whether the harness behaved."""
    results = []

    for i in range(n_random):
        spec = StrategySpec(
            name=f"null_random_{i}",
            signal_fn=random_signal(seed=1000 + i),
            symbols=symbols,
            interval=interval,
            description="Coin-flip entries, no information.",
        )
        # Declare the trial count honestly: this IS a search over n_random variants.
        results.append(evaluate_strategy(spec, n_trials=n_random))

    for name, fn in (("null_always_long", always_long), ("null_alternating", alternating)):
        spec = StrategySpec(name=name, signal_fn=fn, symbols=symbols, interval=interval)
        results.append(evaluate_strategy(spec, n_trials=n_random))

    control = StrategySpec(
        name="positive_control_lookahead",
        signal_fn=lookahead_cheat,
        symbols=symbols,
        interval=interval,
        description="Look-ahead cheat. Must pass, or the harness is broken.",
    )
    control_result = evaluate_strategy(control, n_trials=1)

    nulls_passed = [r["strategy"] for r in results if r.get("passed")]
    control_ok = control_result.get("passed", False)

    # PBO across the random variants: selection among pure noise should look
    # exactly as bad as it is.
    pbo = _pbo_across_nulls(symbols, interval, n_random)

    verdict = {
        "harness_version": config.HARNESS_VERSION,
        "nulls_evaluated": len(results),
        "nulls_incorrectly_passed": nulls_passed,
        "positive_control_passed": control_ok,
        "pbo_among_nulls": pbo,
        "harness_trustworthy": (not nulls_passed) and control_ok,
    }

    return {"verdict": verdict, "nulls": results, "positive_control": control_result}


def _pbo_across_nulls(symbols: list[str], interval: str, n_random: int) -> dict:
    """Build a returns matrix from the null variants and compute PBO."""
    from firm.harness.engine import build_events
    from firm.harness.stats import probability_of_backtest_overfitting

    series = {}
    for i in range(n_random):
        spec = StrategySpec(
            name=f"null_random_{i}",
            signal_fn=random_signal(seed=1000 + i),
            symbols=symbols,
            interval=interval,
        )
        events = build_events(spec)
        if events.empty:
            continue
        s = events.set_index("entry_time")["net_return"]
        # Aggregate to daily so variants share a common time index.
        series[f"null_{i}"] = s.resample("1D").mean()

    if len(series) < 2:
        return {"pbo": None, "reason": "not enough null variants produced trades"}

    matrix = pd.DataFrame(series)
    return probability_of_backtest_overfitting(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the harness against known-null strategies.")
    parser.add_argument("--symbols", nargs="*", default=["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out = run_null_suite(args.symbols, args.interval, args.n_random)
    verdict = out["verdict"]

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return

    print("=" * 66)
    print("HARNESS NULL VALIDATION")
    print("=" * 66)
    for r in out["nulls"]:
        stats_ = r.get("overall", {})
        d = r.get("deflated", {})
        print(
            f"  {r['strategy']:<26} trades={stats_.get('n_trades', 0):>5} "
            f"mean={stats_.get('mean_return', 0):>+8.5f} "
            f"sharpe={stats_.get('sharpe', 0):>+6.3f} "
            f"DSR={d.get('dsr', 0):.3f} -> {r.get('verdict')}"
        )

    c = out["positive_control"]
    cs = c.get("overall", {})
    print(
        f"\n  {c['strategy']:<26} trades={cs.get('n_trades', 0):>5} "
        f"mean={cs.get('mean_return', 0):>+8.5f} "
        f"sharpe={cs.get('sharpe', 0):>+6.3f} -> {c.get('verdict')}"
    )

    print("\n" + "-" * 66)
    print(f"  nulls incorrectly passed : {verdict['nulls_incorrectly_passed'] or 'none'}")
    print(f"  positive control passed  : {verdict['positive_control_passed']}")
    print(f"  PBO among nulls          : {verdict['pbo_among_nulls'].get('pbo')}")
    print(f"\n  HARNESS TRUSTWORTHY      : {verdict['harness_trustworthy']}")
    print("=" * 66)

    if not verdict["harness_trustworthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
