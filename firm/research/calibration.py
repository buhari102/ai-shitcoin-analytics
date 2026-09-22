"""Stage 4: forecast scoring and per-agent calibration.

Every agent claim is stored as a falsifiable forecast - symbol, direction,
probability, horizon, target, invalidation - and resolved against what
actually happened. Brier score and log loss then measure whether that agent's
stated probabilities mean anything.

This is what converts agent output from opinion into a measurable skill
signal. An agent that says "75% confident" and is right 40% of the time gets
its probabilities remapped before they reach position sizing, and an agent
that never beats the base rate loses its vote entirely.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd

import config
from firm.data import storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    agent_role    TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    direction     TEXT NOT NULL,
    probability   REAL NOT NULL,
    horizon_hours INTEGER NOT NULL,
    entry_price   REAL NOT NULL,
    target_price  REAL NOT NULL,
    invalidation  REAL NOT NULL,
    regime        TEXT,
    thesis        TEXT,
    resolved_at   TEXT,
    outcome       INTEGER,
    resolution    TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecast_agent ON forecasts(agent_role);
CREATE INDEX IF NOT EXISTS idx_forecast_open ON forecasts(outcome);
"""


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.FORECAST_DB)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def record_forecast(
    agent_role: str,
    symbol: str,
    direction: str,
    probability: float,
    horizon_hours: int,
    entry_price: float,
    target_price: float,
    invalidation: float,
    regime: str | None = None,
    thesis: str | None = None,
) -> int:
    """Store a falsifiable forecast. Probability must be a real number in (0,1)."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    if direction not in ("long", "short"):
        raise ValueError("direction must be long or short")

    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO forecasts
               (created_at, agent_role, symbol, direction, probability, horizon_hours,
                entry_price, target_price, invalidation, regime, thesis)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(), agent_role, symbol, direction, float(probability),
                int(horizon_hours), float(entry_price), float(target_price),
                float(invalidation), regime, thesis,
            ),
        )
        return int(cur.lastrowid)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def resolve_forecasts() -> dict:
    """Resolve every open forecast whose horizon has elapsed.

    Uses the same first-touch logic as the harness: target before
    invalidation is a win, invalidation first is a loss, and if neither is
    touched the position is marked to market at the horizon.
    """
    with _db() as conn:
        rows = conn.execute("SELECT * FROM forecasts WHERE outcome IS NULL").fetchall()
        open_forecasts = [dict(r) for r in rows]

    if not open_forecasts:
        return {"checked": 0, "resolved": 0, "still_open": 0}

    resolved = 0
    still_open = 0

    for fc in open_forecasts:
        created = pd.Timestamp(fc["created_at"])
        deadline = created + pd.Timedelta(hours=fc["horizon_hours"])
        if pd.Timestamp.now(tz="UTC") < deadline:
            still_open += 1
            continue

        bars = storage.load_bars(fc["symbol"], "1h")
        if bars.empty:
            still_open += 1
            continue

        window = bars[(bars["timestamp"] > created) & (bars["timestamp"] <= deadline)]
        if window.empty:
            still_open += 1
            continue

        outcome, note = _first_touch(fc, window)
        with _db() as conn:
            conn.execute(
                "UPDATE forecasts SET outcome = ?, resolved_at = ?, resolution = ? WHERE id = ?",
                (outcome, _now(), note, fc["id"]),
            )
        resolved += 1

    return {"checked": len(open_forecasts), "resolved": resolved, "still_open": still_open}


def _first_touch(fc: dict, window: pd.DataFrame) -> tuple[int, str]:
    long = fc["direction"] == "long"
    target, stop = fc["target_price"], fc["invalidation"]

    for row in window.itertuples():
        if long:
            if row.low <= stop:
                return 0, "invalidation hit first"
            if row.high >= target:
                return 1, "target hit first"
        else:
            if row.high >= stop:
                return 0, "invalidation hit first"
            if row.low <= target:
                return 1, "target hit first"

    final = float(window["close"].iloc[-1])
    entry = fc["entry_price"]
    profit = (final - entry) if long else (entry - final)
    return (1 if profit > 0 else 0), f"horizon expiry, marked at {final:.8g}"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_agent(agent_role: str, regime: str | None = None) -> dict:
    """Brier score, log loss, and skill versus the base rate."""
    with _db() as conn:
        query = "SELECT probability, outcome, regime FROM forecasts WHERE agent_role = ? AND outcome IS NOT NULL"
        params: list = [agent_role]
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return {"agent": agent_role, "n": 0, "status": "no resolved forecasts"}

    p = np.array([r["probability"] for r in rows], dtype=float)
    y = np.array([r["outcome"] for r in rows], dtype=float)
    n = len(y)

    base_rate = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    # Brier score of always predicting the base rate. Skill above this is the
    # only thing that counts as forecasting ability.
    brier_base = float(np.mean((base_rate - y) ** 2))
    skill = float(1 - brier / brier_base) if brier_base > 0 else 0.0

    eps = 1e-12
    log_loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    # Reliability: mean predicted probability versus realised frequency.
    overconfidence = float(p.mean() - base_rate)

    trusted = n >= config.MIN_FORECASTS_FOR_CALIBRATION and skill > config.DEMOTION_SKILL_THRESHOLD

    return {
        "agent": agent_role,
        "regime": regime or "all",
        "n": n,
        "base_rate": base_rate,
        "mean_probability": float(p.mean()),
        "overconfidence": overconfidence,
        "brier": brier,
        "brier_base_rate": brier_base,
        "skill_score": skill,
        "log_loss": log_loss,
        "status": "voting" if trusted else ("demoted" if n >= config.MIN_FORECASTS_FOR_CALIBRATION else "provisional"),
        "reliability_curve": _reliability_curve(p, y),
    }


def _reliability_curve(p: np.ndarray, y: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if not mask.any():
            continue
        out.append(
            {
                "bucket": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "actual": float(y[mask].mean()),
            }
        )
    return out


# --------------------------------------------------------------------------
# Calibration mapping
# --------------------------------------------------------------------------


def calibration_map(agent_role: str):
    """Fit a monotone map from stated probability to realised frequency.

    Returns a callable. With too little history it returns a shrink-to-base-rate
    function rather than the identity, because an uncalibrated agent's raw
    confidence should never flow straight into position sizing.
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT probability, outcome FROM forecasts WHERE agent_role = ? AND outcome IS NOT NULL",
            (agent_role,),
        ).fetchall()

    if len(rows) < config.MIN_FORECASTS_FOR_CALIBRATION:
        def shrink(prob: float) -> float:
            # Pull halfway to 0.5 until the agent has earned credibility.
            return 0.5 + (float(prob) - 0.5) * 0.5
        return shrink

    p = np.array([r["probability"] for r in rows], dtype=float)
    y = np.array([r["outcome"] for r in rows], dtype=float)

    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(p, y)

    def mapped(prob: float) -> float:
        return float(np.clip(iso.predict([float(prob)])[0], 0.01, 0.99))

    return mapped


def apply_calibration(agent_role: str, probability: float) -> float:
    """Transform a stated probability into a calibrated one."""
    return calibration_map(agent_role)(probability)


def leaderboard() -> pd.DataFrame:
    """All agents ranked by skill score. Drives who keeps a vote."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT agent_role FROM forecasts WHERE outcome IS NOT NULL"
        ).fetchall()

    scores = [score_agent(r["agent_role"]) for r in rows]
    if not scores:
        return pd.DataFrame(columns=["agent", "n", "brier", "skill_score", "status"])

    df = pd.DataFrame(
        [{k: s.get(k) for k in ("agent", "n", "base_rate", "brier", "skill_score", "overconfidence", "status")} for s in scores]
    )
    return df.sort_values("skill_score", ascending=False).reset_index(drop=True)


def open_forecasts() -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, agent_role, symbol, direction, probability, horizon_hours FROM forecasts WHERE outcome IS NULL ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def summary() -> dict:
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM forecasts").fetchone()["n"]
        resolved = conn.execute("SELECT COUNT(*) AS n FROM forecasts WHERE outcome IS NOT NULL").fetchone()["n"]
    board = leaderboard()
    return {
        "forecasts_recorded": total,
        "resolved": resolved,
        "open": total - resolved,
        "leaderboard": board.to_dict(orient="records") if not board.empty else [],
    }


if __name__ == "__main__":
    print(json.dumps(resolve_forecasts(), indent=2))
    print(json.dumps(summary(), indent=2, default=str))
