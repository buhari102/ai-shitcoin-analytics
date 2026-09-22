"""Stage 6: meta-labeling for sizing.

The primary signal decides direction. A separate model, trained on how that
signal has actually performed, decides whether to act and how large. This is
the division of labour that makes an agent firm workable: the creative side
supplies recall, the quantitative side supplies precision, and nobody has to
pretend an LLM's stated confidence is a probability.

Training uses the same purged walk-forward folds as the harness, so the
reported AUC is not contaminated by overlapping labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from firm.harness import cv, engine, features, stats

MODEL_DIR = config.DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Features the meta-model sees. Deliberately about *context*, not direction:
# the primary signal already decided direction.
META_FEATURES = [
    "volatility_24",
    "volatility_168",
    "atr_pct",
    "return_zscore",
    "quote_volume_z",
    "taker_buy_share_z",
    "range_position",
    "funding_z",
    "funding_cum_7d",
    "beta_168",
    "momentum_24",
    "momentum_168",
]


def build_training_set(spec: engine.StrategySpec) -> pd.DataFrame:
    """Join each primary signal to the context it fired in, plus its outcome."""
    rows = []

    for symbol in spec.symbols:
        feats = features.build_features(symbol, spec.interval, benchmark=config.BENCHMARK)
        if feats.empty:
            continue
        feats = feats[feats["timestamp"] < engine.holdout_boundary()].reset_index(drop=True)
        if len(feats) < 200:
            continue

        single = engine.StrategySpec(
            name=spec.name, signal_fn=spec.signal_fn, symbols=[symbol],
            interval=spec.interval, horizon=spec.horizon,
            pt_atr=spec.pt_atr, sl_atr=spec.sl_atr, notional_usd=spec.notional_usd,
        )
        events = engine.build_events(single, segment="research")
        if events.empty:
            continue

        context = feats.iloc[events["entry_idx"].to_numpy()][
            [c for c in META_FEATURES if c in feats.columns]
        ].reset_index(drop=True)
        context["side"] = events["side"].to_numpy()
        context["symbol"] = symbol
        context["entry_time"] = events["entry_time"].to_numpy()
        context["exit_time"] = events["exit_time"].to_numpy()
        context["net_return"] = events["net_return"].to_numpy()
        context["meta_label"] = (events["net_return"] > 0).astype(int).to_numpy()
        rows.append(context)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def train(spec: engine.StrategySpec, min_samples: int = 100) -> dict:
    """Fit and purged-CV-validate a meta-model for one primary strategy."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    data = build_training_set(spec)
    if data.empty or len(data) < min_samples:
        return {
            "strategy": spec.name,
            "trained": False,
            "reason": f"only {len(data)} samples, need {min_samples}",
        }

    feature_cols = [c for c in META_FEATURES if c in data.columns]
    X = data[feature_cols].to_numpy(dtype=float)
    y = data["meta_label"].to_numpy(dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(np.unique(y)) < 2:
        return {"strategy": spec.name, "trained": False, "reason": "only one outcome class present"}

    folds = cv.purged_walk_forward_splits(data[["entry_time", "exit_time"]])
    fold_scores = []

    for fold in folds:
        if len(np.unique(y[fold.train_idx])) < 2 or len(np.unique(y[fold.test_idx])) < 2:
            continue
        model = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
        model.fit(X[fold.train_idx], y[fold.train_idx])
        proba = model.predict_proba(X[fold.test_idx])[:, 1]
        fold_scores.append(
            {
                "auc": float(roc_auc_score(y[fold.test_idx], proba)),
                "brier": float(np.mean((proba - y[fold.test_idx]) ** 2)),
                "n_test": int(len(fold.test_idx)),
            }
        )

    final = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
    final.fit(X, y)

    import joblib

    path = MODEL_DIR / f"meta_{spec.name}.joblib"
    joblib.dump({"model": final, "features": feature_cols}, path)

    mean_auc = float(np.mean([f["auc"] for f in fold_scores])) if fold_scores else float("nan")

    return {
        "strategy": spec.name,
        "trained": True,
        "model_path": str(path),
        "n_samples": int(len(data)),
        "base_rate": float(y.mean()),
        "oos_auc_mean": mean_auc,
        "oos_folds": fold_scores,
        # Below 0.55 the meta-model adds nothing and sizing should fall back
        # to a flat fraction rather than pretending to be informed.
        "usable": bool(np.isfinite(mean_auc) and mean_auc > 0.55),
        "feature_importance": dict(
            sorted(
                zip(feature_cols, (float(v) for v in final.feature_importances_)),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }


def load(strategy_name: str):
    import joblib

    path = MODEL_DIR / f"meta_{strategy_name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict_success_probability(strategy_name: str, context: dict) -> float | None:
    """P(this particular signal works), given current market context."""
    bundle = load(strategy_name)
    if bundle is None:
        return None
    model, cols = bundle["model"], bundle["features"]
    x = np.array([[float(context.get(c, 0.0) or 0.0) for c in cols]])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return float(model.predict_proba(x)[0, 1])


def size_position(
    win_probability: float,
    payoff_ratio: float,
    max_position_pct: float = config.MAX_POSITION_PCT,
) -> dict:
    """Fractional-Kelly sizing with hard caps.

    Full Kelly on a probability this uncertain is ruin-seeking, so the result
    is scaled by config.KELLY_FRACTION and then clipped.
    """
    full_kelly = stats.kelly_fraction(win_probability, payoff_ratio)
    scaled = full_kelly * config.KELLY_FRACTION
    capped = float(min(scaled * 100, max_position_pct))
    breakeven = stats.breakeven_win_rate(payoff_ratio)

    return {
        "win_probability": float(win_probability),
        "payoff_ratio": float(payoff_ratio),
        "breakeven_win_rate": breakeven,
        "edge": float(win_probability - breakeven),
        "full_kelly_pct": float(full_kelly * 100),
        "fractional_kelly_pct": float(scaled * 100),
        "position_pct": max(0.0, capped),
        "tradeable": win_probability > breakeven and capped > 0,
    }


if __name__ == "__main__":
    import sys

    from firm.research.dsl import RuleSpec, compile_rule

    genome = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    if not genome:
        print("usage: python -m firm.research.metalabel '<genome json>'")
        raise SystemExit(1)

    rule = RuleSpec.from_dict(genome)
    spec = engine.StrategySpec(name=rule.name, signal_fn=compile_rule(rule), horizon=rule.horizon)
    print(json.dumps(train(spec), indent=2, default=str))
