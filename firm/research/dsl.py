"""A declarative strategy DSL.

Strategies are JSON rule specs, not Python source. This matters for two
reasons: agents can propose candidates without the system ever executing
model-generated code, and a structured genome makes mutation and
recombination well defined rather than a matter of hoping the LLM edits
code correctly.

Expressiveness is deliberately bounded. A conjunction of threshold
conditions over causal features covers most of the hypothesis space that is
testable with this much data, and the narrow grammar keeps the search from
wandering into strategies too complex to survive deflation anyway.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# Features the DSL may reference. Anything not listed here does not exist as
# far as a proposer is concerned.
TRADEABLE_FEATURES: dict[str, str] = {
    "momentum_6": "6-bar price momentum",
    "momentum_24": "24-bar price momentum",
    "momentum_72": "72-bar price momentum",
    "momentum_168": "weekly price momentum",
    "volatility_24": "24-bar realised volatility",
    "volatility_168": "weekly realised volatility",
    "return_zscore": "latest return vs its 30-day distribution",
    "atr_pct": "ATR as a fraction of price",
    "quote_volume_z": "dollar volume vs its weekly norm",
    "taker_buy_share": "share of volume lifting the offer",
    "taker_buy_share_z": "taker buy share vs its weekly norm",
    "range_position": "position within the 24-bar range, 0 to 1",
    "funding_rate": "current perp funding rate",
    "funding_z": "funding vs its weekly norm",
    "funding_cum_7d": "funding accumulated over 7 days",
    "beta_168": "rolling beta to BTC",
    "residual_return": "return net of BTC beta",
}

OPERATORS = (">", "<", ">=", "<=")


@dataclass
class Condition:
    feature: str
    op: Literal[">", "<", ">=", "<="]
    value: float

    def describe(self) -> str:
        return f"{self.feature} {self.op} {self.value:.4g}"


@dataclass
class RuleSpec:
    """A complete candidate strategy genome."""

    name: str
    direction: Literal["long", "short"]
    conditions: list[Condition] = field(default_factory=list)
    horizon: int = 24
    pt_atr: float = 2.0
    sl_atr: float = 1.0
    rationale: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_dict(data: dict) -> "RuleSpec":
        conds = [Condition(**c) if isinstance(c, dict) else c for c in data.get("conditions", [])]
        return RuleSpec(
            name=data["name"],
            direction=data["direction"],
            conditions=conds,
            horizon=int(data.get("horizon", 24)),
            pt_atr=float(data.get("pt_atr", 2.0)),
            sl_atr=float(data.get("sl_atr", 1.0)),
            rationale=data.get("rationale", ""),
        )

    def describe(self) -> str:
        clauses = " AND ".join(c.describe() for c in self.conditions) or "always"
        return (
            f"{self.direction.upper()} when {clauses} "
            f"[horizon={self.horizon}b, target={self.pt_atr}ATR, stop={self.sl_atr}ATR]"
        )

    def validate(self) -> None:
        if self.direction not in ("long", "short"):
            raise ValueError(f"bad direction: {self.direction}")
        if not self.conditions:
            raise ValueError("a strategy with no conditions is just buy-and-hold")
        if len(self.conditions) > 4:
            raise ValueError("more than 4 conditions will not survive deflation on this much data")
        for c in self.conditions:
            if c.feature not in TRADEABLE_FEATURES:
                raise ValueError(f"unknown feature '{c.feature}'; allowed: {sorted(TRADEABLE_FEATURES)}")
            if c.op not in OPERATORS:
                raise ValueError(f"bad operator: {c.op}")
        if not 1 <= self.horizon <= 336:
            raise ValueError("horizon must be between 1 and 336 bars")
        if self.pt_atr <= 0 or self.sl_atr <= 0:
            raise ValueError("barrier distances must be positive")


def compile_rule(spec: RuleSpec):
    """Turn a rule spec into a causal signal function for the harness."""
    spec.validate()
    side_value = 1.0 if spec.direction == "long" else -1.0

    def signal_fn(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for cond in spec.conditions:
            if cond.feature not in df.columns:
                return pd.Series(0.0, index=df.index)
            col = df[cond.feature]
            if cond.op == ">":
                mask &= col > cond.value
            elif cond.op == "<":
                mask &= col < cond.value
            elif cond.op == ">=":
                mask &= col >= cond.value
            else:
                mask &= col <= cond.value
        return pd.Series(np.where(mask.fillna(False), side_value, 0.0), index=df.index)

    return signal_fn


# --------------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------------


def mutate(spec: RuleSpec, rng: random.Random | None = None) -> RuleSpec:
    """Produce a neighbouring genome. Used when no LLM proposer is available."""
    rng = rng or random.Random()
    child = RuleSpec.from_dict(json.loads(spec.to_json()))
    child.name = f"{spec.name}_m{rng.randint(1000, 9999)}"

    choice = rng.choice(["threshold", "operator", "add", "drop", "barrier", "horizon"])

    if choice == "threshold" and child.conditions:
        c = rng.choice(child.conditions)
        c.value = float(c.value * rng.uniform(0.6, 1.6) + rng.gauss(0, 0.1))
    elif choice == "operator" and child.conditions:
        c = rng.choice(child.conditions)
        c.op = ">" if c.op in ("<", "<=") else "<"
    elif choice == "add" and len(child.conditions) < 4:
        child.conditions.append(_random_condition(rng))
    elif choice == "drop" and len(child.conditions) > 1:
        child.conditions.remove(rng.choice(child.conditions))
    elif choice == "barrier":
        child.pt_atr = round(max(0.5, child.pt_atr * rng.uniform(0.7, 1.4)), 2)
        child.sl_atr = round(max(0.3, child.sl_atr * rng.uniform(0.7, 1.4)), 2)
    else:
        child.horizon = int(max(2, min(336, child.horizon * rng.uniform(0.5, 2.0))))

    return child


def crossover(a: RuleSpec, b: RuleSpec, rng: random.Random | None = None) -> RuleSpec:
    """Recombine two parents' conditions."""
    rng = rng or random.Random()
    pool = a.conditions + b.conditions
    rng.shuffle(pool)
    n = min(len(pool), rng.randint(1, 3))
    return RuleSpec(
        name=f"x_{a.name[:8]}_{b.name[:8]}_{rng.randint(100, 999)}",
        direction=rng.choice([a.direction, b.direction]),
        conditions=pool[:n],
        horizon=rng.choice([a.horizon, b.horizon]),
        pt_atr=rng.choice([a.pt_atr, b.pt_atr]),
        sl_atr=rng.choice([a.sl_atr, b.sl_atr]),
        rationale=f"crossover of '{a.name}' and '{b.name}'",
    )


def _random_condition(rng: random.Random) -> Condition:
    feature = rng.choice(list(TRADEABLE_FEATURES))
    op = rng.choice(OPERATORS)
    # Most features here are z-scores or small ratios, so this range covers
    # the meaningful part of their distributions.
    value = round(rng.uniform(-2.5, 2.5), 3)
    return Condition(feature=feature, op=op, value=value)


def random_rule(rng: random.Random | None = None, name: str | None = None) -> RuleSpec:
    rng = rng or random.Random()
    n = rng.randint(1, 3)
    return RuleSpec(
        name=name or f"rand_{rng.randint(10000, 99999)}",
        direction=rng.choice(["long", "short"]),
        conditions=[_random_condition(rng) for _ in range(n)],
        horizon=rng.choice([6, 12, 24, 48, 72, 168]),
        pt_atr=rng.choice([1.5, 2.0, 2.5, 3.0]),
        sl_atr=rng.choice([0.75, 1.0, 1.5]),
        rationale="randomly generated",
    )


def feature_catalogue() -> str:
    """Human-readable feature list for proposer prompts."""
    return "\n".join(f"  {k:<22} {v}" for k, v in TRADEABLE_FEATURES.items())
