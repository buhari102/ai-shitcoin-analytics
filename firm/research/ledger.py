"""Stage 3: pre-registered hypothesis ledger and the holdout vault.

Two jobs, both about honesty rather than bookkeeping.

1. Pre-registration. A hypothesis must be written down - including why it
   should work economically and which direction it predicts - before the
   harness runs. This makes the trial count real, which is what the deflated
   Sharpe needs, and it stops an agent retrofitting a story onto a lucky
   backtest.

2. The holdout vault. The most recent slice of data is never touched by the
   research loop. A strategy may query it a fixed number of times, and a
   failure there is terminal. Unbudgeted holdout access is the quiet way
   serious teams destroy their own validation.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    author             TEXT NOT NULL,
    title              TEXT NOT NULL,
    statement          TEXT NOT NULL,
    economic_rationale TEXT NOT NULL,
    predicted_direction TEXT NOT NULL,
    feature_definition TEXT NOT NULL,
    horizon_bars       INTEGER NOT NULL,
    symbols            TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'registered',
    code_hash          TEXT
);

CREATE TABLE IF NOT EXISTS tests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id   INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    harness_version TEXT NOT NULL,
    segment         TEXT NOT NULL,
    n_trials        INTEGER NOT NULL,
    passed          INTEGER NOT NULL,
    scorecard       TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS holdout_access (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    accessed_at   TEXT NOT NULL,
    passed        INTEGER NOT NULL,
    scorecard     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tests_hypothesis ON tests(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_holdout_strategy ON holdout_access(strategy_name);
"""


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.LEDGER_DB)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class HoldoutExhausted(RuntimeError):
    """Raised when a strategy has spent its holdout budget."""


class NotPreRegistered(RuntimeError):
    """Raised when a test is attempted without a registered hypothesis."""


# --------------------------------------------------------------------------
# Pre-registration
# --------------------------------------------------------------------------


def register_hypothesis(
    author: str,
    title: str,
    statement: str,
    economic_rationale: str,
    predicted_direction: str,
    feature_definition: str,
    horizon_bars: int,
    symbols: list[str],
    code_hash: str | None = None,
) -> int:
    """Record a hypothesis before it is tested. Returns the ledger id."""
    if predicted_direction not in ("long", "short", "both"):
        raise ValueError("predicted_direction must be long, short, or both")
    if not economic_rationale.strip():
        raise ValueError("economic_rationale is mandatory: why should this work?")

    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO hypotheses
               (created_at, author, title, statement, economic_rationale,
                predicted_direction, feature_definition, horizon_bars, symbols, code_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(), author, title, statement, economic_rationale,
                predicted_direction, feature_definition, horizon_bars,
                json.dumps(symbols), code_hash,
            ),
        )
        return int(cur.lastrowid)


def trial_count() -> int:
    """Total hypotheses ever registered.

    This is the N that deflation uses. It only ever grows, which correctly
    makes it harder to claim an edge the longer you have been searching.
    """
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM hypotheses").fetchone()
        return int(row["n"])


def test_count() -> int:
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM tests").fetchone()
        return int(row["n"])


def record_test(hypothesis_id: int, scorecard: dict, n_trials: int) -> int:
    with _db() as conn:
        exists = conn.execute(
            "SELECT id FROM hypotheses WHERE id = ?", (hypothesis_id,)
        ).fetchone()
        if not exists:
            raise NotPreRegistered(f"hypothesis {hypothesis_id} was never registered")

        cur = conn.execute(
            """INSERT INTO tests
               (hypothesis_id, created_at, harness_version, segment, n_trials, passed, scorecard)
               VALUES (?,?,?,?,?,?,?)""",
            (
                hypothesis_id,
                _now(),
                scorecard.get("harness_version", config.HARNESS_VERSION),
                scorecard.get("segment", "research"),
                n_trials,
                1 if scorecard.get("passed") else 0,
                json.dumps(scorecard, default=str),
            ),
        )
        conn.execute(
            "UPDATE hypotheses SET status = ? WHERE id = ?",
            ("promoted" if scorecard.get("passed") else "rejected", hypothesis_id),
        )
        return int(cur.lastrowid)


# --------------------------------------------------------------------------
# Research memory
# --------------------------------------------------------------------------


def search_hypotheses(keyword: str = "", limit: int = 50) -> list[dict]:
    """Searchable memory so the crew stops re-testing the same idea."""
    with _db() as conn:
        if keyword:
            pattern = f"%{keyword}%"
            rows = conn.execute(
                """SELECT * FROM hypotheses
                   WHERE title LIKE ? OR statement LIKE ? OR feature_definition LIKE ?
                   ORDER BY id DESC LIMIT ?""",
                (pattern, pattern, pattern, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM hypotheses ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def failure_log(limit: int = 30) -> list[dict]:
    """Rejected hypotheses with their reasons.

    Fed back into the proposer prompts: knowing why something failed is more
    informative than knowing that something succeeded.
    """
    with _db() as conn:
        rows = conn.execute(
            """SELECT h.title, h.statement, h.economic_rationale, t.scorecard, t.created_at
               FROM tests t JOIN hypotheses h ON h.id = t.hypothesis_id
               WHERE t.passed = 0 ORDER BY t.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    out = []
    for r in rows:
        try:
            card = json.loads(r["scorecard"])
        except json.JSONDecodeError:
            card = {}
        out.append(
            {
                "title": r["title"],
                "statement": r["statement"],
                "rationale": r["economic_rationale"],
                "reasons": card.get("rejection_reasons", []),
                "n_trades": card.get("overall", {}).get("n_trades"),
                "tested_at": r["created_at"],
            }
        )
    return out


# --------------------------------------------------------------------------
# Holdout vault
# --------------------------------------------------------------------------


def holdout_uses(strategy_name: str) -> int:
    with _db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM holdout_access WHERE strategy_name = ?",
            (strategy_name,),
        ).fetchone()
        return int(row["n"])


def can_access_holdout(strategy_name: str) -> bool:
    return holdout_uses(strategy_name) < config.HOLDOUT_BUDGET_PER_STRATEGY


def spend_holdout(strategy_name: str, scorecard: dict) -> None:
    """Record a holdout evaluation. Raises if the budget is already spent."""
    if not can_access_holdout(strategy_name):
        raise HoldoutExhausted(
            f"{strategy_name} has already used its {config.HOLDOUT_BUDGET_PER_STRATEGY} "
            "holdout evaluation. The result stands; it cannot be re-rolled."
        )
    with _db() as conn:
        conn.execute(
            "INSERT INTO holdout_access (strategy_name, accessed_at, passed, scorecard) VALUES (?,?,?,?)",
            (strategy_name, _now(), 1 if scorecard.get("passed") else 0, json.dumps(scorecard, default=str)),
        )


def holdout_history() -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT strategy_name, accessed_at, passed FROM holdout_access ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def summary() -> dict:
    """One-glance state of the research programme."""
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM hypotheses").fetchone()["n"]
        promoted = conn.execute(
            "SELECT COUNT(*) AS n FROM hypotheses WHERE status = 'promoted'"
        ).fetchone()["n"]
        rejected = conn.execute(
            "SELECT COUNT(*) AS n FROM hypotheses WHERE status = 'rejected'"
        ).fetchone()["n"]
        tests = conn.execute("SELECT COUNT(*) AS n FROM tests").fetchone()["n"]
        holdouts = conn.execute("SELECT COUNT(*) AS n FROM holdout_access").fetchone()["n"]

    return {
        "hypotheses_registered": total,
        "promoted": promoted,
        "rejected": rejected,
        "untested": total - promoted - rejected,
        "tests_run": tests,
        "holdout_evaluations_spent": holdouts,
        "deflation_trial_count": total,
    }
