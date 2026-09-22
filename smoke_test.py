"""Standalone sanity checks for the data and harness layers.

Run before trusting any agent output:
    venv\\Scripts\\python.exe smoke_test.py
"""

from __future__ import annotations

import sys

import pandas as pd

import config
from firm.data import storage
from firm.harness import features
from firm.harness.engine import StrategySpec, evaluate_strategy
from firm.research.dsl import Condition, RuleSpec, compile_rule


def check_data() -> list[str]:
    print("=" * 70)
    print("DATA INVENTORY")
    print("=" * 70)
    inv = storage.inventory()

    ready = []
    for name, meta in inv["bars"].items():
        print(f"  {name:<22} {meta['rows']:>7} bars  {meta['first'][:10]} -> {meta['last'][:10]}")
        if name.endswith("_1h") and meta["rows"] > 500:
            ready.append(name.replace("_1h", ""))

    print(f"\n  funding series: {len(inv['funding'])}")
    for name, meta in list(inv["funding"].items())[:20]:
        print(f"  {name:<22} {meta['rows']:>7} points {meta['first'][:10]} -> {meta['last'][:10]}")

    print(f"\n  flow coverage (self-collected, cannot be backfilled):")
    if inv["flow"]:
        for row in inv["flow"]:
            print(f"    {row['symbol']:<14} {row['snapshots']:>6} snapshots, span {row['span_days']:.2f} days")
    else:
        print("    none yet - run the collector")

    return ready


def check_features(symbol: str) -> bool:
    print("\n" + "=" * 70)
    print(f"FEATURES: {symbol}")
    print("=" * 70)
    f = features.build_features(symbol, "1h", benchmark=config.BENCHMARK)
    if f.empty:
        print("  EMPTY - not enough bars")
        return False

    print(f"  shape: {f.shape}")
    for col in ("funding_rate", "funding_z", "beta_168", "residual_return",
                "taker_buy_share_z", "return_zscore", "atr_pct"):
        if col in f.columns:
            print(f"    {col:<20} non-null {f[col].notna().sum():>7} / {len(f)}")
        else:
            print(f"    {col:<20} MISSING")

    # Look-ahead guard: a feature must not correlate suspiciously with the
    # very next return. Anything above ~0.2 means a shift is wrong.
    fwd = f["close"].pct_change().shift(-1)
    print("\n  correlation with NEXT bar return (must be near zero):")
    worst = 0.0
    for col in ("momentum_24", "return_zscore", "funding_z", "taker_buy_share_z", "range_position"):
        if col in f.columns:
            c = f[col].corr(fwd)
            worst = max(worst, abs(c) if pd.notna(c) else 0)
            flag = "  <-- SUSPICIOUS" if pd.notna(c) and abs(c) > 0.2 else ""
            print(f"    {col:<20} {c:+.4f}{flag}")
    return worst < 0.2


def check_strategy(symbols: list[str]) -> None:
    print("\n" + "=" * 70)
    print("HARNESS END-TO-END")
    print("=" * 70)

    rule = RuleSpec(
        name="smoke_meanrev",
        direction="long",
        conditions=[Condition("return_zscore", "<", -1.5)],
        horizon=24,
        pt_atr=2.0,
        sl_atr=1.0,
        rationale="Short-term oversold bounce; a plain mean-reversion baseline.",
    )
    print(f"  {rule.describe()}")

    spec = StrategySpec(
        name=rule.name,
        signal_fn=compile_rule(rule),
        symbols=symbols,
        interval="1h",
        horizon=rule.horizon,
        pt_atr=rule.pt_atr,
        sl_atr=rule.sl_atr,
    )
    card = evaluate_strategy(spec, n_trials=1)

    overall = card.get("overall", {})
    print(f"\n  trades          : {overall.get('n_trades', 0)}")
    print(f"  win rate        : {overall.get('win_rate', 0):.1%}")
    print(f"  mean net return : {overall.get('mean_return', 0):+.5f}")
    print(f"  sharpe          : {overall.get('sharpe', 0):+.4f}")
    print(f"  profit factor   : {overall.get('profit_factor', 0):.3f}")
    print(f"  max drawdown    : {overall.get('max_drawdown', 0):.2%}")
    print(f"  mean cost (bps) : {card.get('mean_cost_bps', 0):.2f}")
    print(f"  cost drag ratio : {card.get('cost_drag_ratio', 0):.3f}")
    print(f"  deflated Sharpe : {card.get('deflated', {}).get('dsr', 0):.4f}")
    print(f"  folds           : {len(card.get('folds', []))}")

    fc = card.get("fold_consistency", {})
    if fc:
        print(f"  positive folds  : {fc.get('positive_folds')}/{fc.get('total_folds')}")

    print(f"\n  VERDICT: {card.get('verdict')}")
    for reason in card.get("rejection_reasons", []):
        print(f"    - {reason}")


def main() -> int:
    ready = check_data()
    if not ready:
        print("\nNo symbols have enough 1h bars yet. Run the backfill first.")
        return 1

    symbol = "DOGEUSDT" if "DOGEUSDT" in ready else ready[0]
    clean = check_features(symbol)
    if not clean:
        print("\nWARNING: possible look-ahead in features. Do not trust results.")

    check_strategy([s for s in ready if s != config.BENCHMARK][:4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
