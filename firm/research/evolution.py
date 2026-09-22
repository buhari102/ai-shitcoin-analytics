"""Stage 5: the evolutionary loop.

A genetic algorithm in which the proposer (an LLM agent, or the built-in
mutation operators as a fallback) generates candidate genomes and the
immutable harness assigns fitness. The proposer never sees the fitness
function and never scores its own work.

Diversity pressure is the part people skip. Without niching, the population
collapses onto one overfit family that looks excellent and then dies all at
once, because every survivor is the same bet wearing different thresholds.
Elites are therefore kept per niche (holding period x driver x direction),
in the style of MAP-Elites.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import config
from firm.harness.engine import StrategySpec, evaluate_strategy
from firm.research import ledger
from firm.research.dsl import RuleSpec, compile_rule, crossover, mutate, random_rule

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    name         TEXT NOT NULL UNIQUE,
    generation   INTEGER NOT NULL DEFAULT 0,
    parent       TEXT,
    genome       TEXT NOT NULL,
    niche        TEXT NOT NULL,
    fitness      REAL NOT NULL,
    passed       INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    scorecard    TEXT NOT NULL,
    hypothesis_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_strategies_niche ON strategies(niche);
CREATE INDEX IF NOT EXISTS idx_strategies_fitness ON strategies(fitness DESC);
"""


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.ARCHIVE_DB)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Fitness and niching
# --------------------------------------------------------------------------


def fitness(scorecard: dict) -> float:
    """Single scalar the archive sorts on.

    Deliberately built from the deflated Sharpe rather than the raw one, so
    that fitness already accounts for how much searching produced it. A
    strategy that fails the trade-count floor scores zero rather than
    negative, because "we do not know" is not the same as "it is bad".
    """
    overall = scorecard.get("overall", {})
    if overall.get("n_trades", 0) < config.MIN_TRADES_FOR_SIGNIFICANCE:
        return 0.0

    dsr = scorecard.get("deflated", {}).get("dsr", 0.0)
    sharpe = overall.get("sharpe", 0.0)
    consistency = scorecard.get("fold_consistency", {})
    total_folds = consistency.get("total_folds", 0) or 1
    fold_ratio = consistency.get("positive_folds", 0) / total_folds

    # DSR gates the result, Sharpe scales it, fold consistency discounts
    # strategies that only worked in one stretch of history.
    return float(dsr * max(sharpe, 0.0) * (0.5 + 0.5 * fold_ratio))


def niche_of(spec: RuleSpec, scorecard: dict) -> str:
    """Behavioural descriptor used to keep the population diverse."""
    horizon = spec.horizon
    if horizon <= 12:
        holding = "intraday"
    elif horizon <= 48:
        holding = "swing"
    else:
        holding = "position"

    drivers = {c.feature.split("_")[0] for c in spec.conditions}
    if drivers & {"funding"}:
        driver = "funding"
    elif drivers & {"taker", "quote"}:
        driver = "flow"
    elif drivers & {"beta", "residual"}:
        driver = "relative"
    elif drivers & {"volatility", "atr"}:
        driver = "volatility"
    else:
        driver = "price"

    return f"{holding}|{driver}|{spec.direction}"


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------


def add_to_archive(
    spec: RuleSpec,
    scorecard: dict,
    generation: int = 0,
    parent: str | None = None,
    hypothesis_id: int | None = None,
) -> int:
    with _db() as conn:
        cur = conn.execute(
            """INSERT OR REPLACE INTO strategies
               (created_at, name, generation, parent, genome, niche, fitness, passed, scorecard, hypothesis_id)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                dt.datetime.now(dt.timezone.utc).isoformat(),
                spec.name,
                generation,
                parent,
                spec.to_json(),
                niche_of(spec, scorecard),
                fitness(scorecard),
                1 if scorecard.get("passed") else 0,
                json.dumps(scorecard, default=str),
                hypothesis_id,
            ),
        )
        return int(cur.lastrowid)


def elites(per_niche: int = config.ARCHIVE_ELITES_PER_NICHE) -> list[dict]:
    """Best performers in each behavioural niche."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT * FROM (
                   SELECT *, ROW_NUMBER() OVER (PARTITION BY niche ORDER BY fitness DESC) AS rank
                   FROM strategies WHERE status = 'active'
               ) WHERE rank <= ? ORDER BY fitness DESC""",
            (per_niche,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def all_strategies(limit: int = 200) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY fitness DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def retire(name: str, reason: str) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE strategies SET status = ? WHERE name = ?",
            (f"retired:{reason}", name),
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["genome"] = json.loads(d["genome"])
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        d["scorecard"] = json.loads(d["scorecard"])
    except (json.JSONDecodeError, TypeError):
        pass
    return d


def archive_summary() -> dict:
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM strategies").fetchone()["n"]
        active = conn.execute("SELECT COUNT(*) AS n FROM strategies WHERE status='active'").fetchone()["n"]
        promoted = conn.execute("SELECT COUNT(*) AS n FROM strategies WHERE passed=1").fetchone()["n"]
        niches = conn.execute(
            "SELECT niche, COUNT(*) AS n, MAX(fitness) AS best FROM strategies GROUP BY niche ORDER BY best DESC"
        ).fetchall()

    return {
        "total": total,
        "active": active,
        "promoted": promoted,
        "niches_occupied": len(niches),
        "niches": [dict(n) for n in niches],
    }


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def evaluate_candidate(
    spec: RuleSpec,
    symbols: list[str],
    author: str = "evolution",
    interval: str = "1h",
) -> dict:
    """Pre-register, evaluate, and archive one candidate.

    Registration happens first and unconditionally, so the trial count that
    feeds deflation cannot be gamed by quietly discarding failures.
    """
    spec.validate()

    hypothesis_id = ledger.register_hypothesis(
        author=author,
        title=spec.name,
        statement=spec.describe(),
        economic_rationale=spec.rationale or "not stated",
        predicted_direction=spec.direction,
        feature_definition=spec.to_json(),
        horizon_bars=spec.horizon,
        symbols=symbols,
    )

    n_trials = ledger.trial_count()

    strategy = StrategySpec(
        name=spec.name,
        signal_fn=compile_rule(spec),
        symbols=symbols,
        interval=interval,
        horizon=spec.horizon,
        pt_atr=spec.pt_atr,
        sl_atr=spec.sl_atr,
        description=spec.describe(),
    )

    scorecard = evaluate_strategy(strategy, n_trials=n_trials)
    scorecard["genome"] = json.loads(spec.to_json())
    scorecard["fitness"] = fitness(scorecard)

    ledger.record_test(hypothesis_id, scorecard, n_trials)
    add_to_archive(spec, scorecard, hypothesis_id=hypothesis_id)

    return scorecard


def evolve(
    symbols: list[str],
    generations: int = 3,
    population: int = config.EVOLUTION_POPULATION,
    seed: int | None = None,
    seeds: list[RuleSpec] | None = None,
    verbose: bool = True,
) -> dict:
    """Run the search. Returns the archive state afterwards."""
    rng = random.Random(seed)
    current = list(seeds) if seeds else [random_rule(rng) for _ in range(population)]
    history = []

    for gen in range(generations):
        results = []
        for spec in current:
            try:
                card = evaluate_candidate(spec, symbols, author=f"evolution_gen{gen}")
            except Exception as exc:  # noqa: BLE001 - a bad genome must not halt the search
                if verbose:
                    print(f"  {spec.name}: invalid ({exc})")
                continue
            results.append((spec, card))
            if verbose:
                o = card.get("overall", {})
                print(
                    f"  gen{gen} {spec.name:<28} trades={o.get('n_trades', 0):>4} "
                    f"sharpe={o.get('sharpe', 0):>+6.3f} fit={card.get('fitness', 0):.4f} "
                    f"{card.get('verdict')}"
                )

        if not results:
            break

        results.sort(key=lambda r: r[1].get("fitness", 0), reverse=True)
        history.append(
            {
                "generation": gen,
                "evaluated": len(results),
                "best_fitness": results[0][1].get("fitness", 0),
                "best_name": results[0][0].name,
            }
        )

        # Next generation: elites from the archive (preserving diversity)
        # plus mutations and crossovers of this round's best.
        survivors = [spec for spec, _ in results[: max(2, population // 2)]]
        children: list[RuleSpec] = []
        while len(children) < population:
            if len(survivors) >= 2 and rng.random() < 0.3:
                children.append(crossover(rng.choice(survivors), rng.choice(survivors), rng))
            else:
                children.append(mutate(rng.choice(survivors), rng))
        current = children

    return {
        "history": history,
        "archive": archive_summary(),
        "elites": [
            {"name": e["name"], "niche": e["niche"], "fitness": e["fitness"], "passed": bool(e["passed"])}
            for e in elites()
        ],
    }
