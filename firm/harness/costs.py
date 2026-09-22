"""Adversarial transaction cost model.

Costs are modelled pessimistically on purpose. In memecoin markets most
"discovered edges" are entirely explained by unmodelled spread and impact,
so the harness is built to err toward killing marginal strategies rather
than promoting them.

Four components per round trip:
  fees      taker in and taker out, no BNB discount assumed
  spread    observed half-spread, crossed twice, times a safety multiplier
  impact    square-root law against realised dollar volume
  funding   paid for the duration of the hold, signed by side
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import config
from firm.data import storage

# Used when no order book snapshot exists yet for a symbol. Deliberately wide.
FALLBACK_SPREAD_BPS = 8.0


# Backtests evaluate thousands of trades per symbol; without caching, every
# one of them would re-read the same Parquet file from disk.
@lru_cache(maxsize=64)
def _funding_series(symbol: str) -> pd.DataFrame:
    return storage.load_funding(symbol)


@lru_cache(maxsize=64)
def observed_spread_bps(symbol: str) -> tuple[float, str]:
    """Median spread from collected book snapshots, or a pessimistic default."""
    book = storage.load_book(symbol)
    if book.empty or "spread_bps" not in book.columns:
        return FALLBACK_SPREAD_BPS, "fallback"
    median = float(book["spread_bps"].median())
    if not np.isfinite(median) or median <= 0:
        return FALLBACK_SPREAD_BPS, "fallback"
    # Use the 75th percentile, not the median: you do not get to trade at the
    # average spread when you most want to.
    p75 = float(book["spread_bps"].quantile(0.75))
    return max(median, p75), f"observed:{len(book)}"


def clear_cache() -> None:
    """Call after the collector writes new data mid-session."""
    _funding_series.cache_clear()
    observed_spread_bps.cache_clear()


def impact_bps(notional_usd: float, adv_usd: float) -> float:
    """Square-root market impact. Zero ADV means untradeable, not free."""
    if adv_usd <= 0 or notional_usd <= 0:
        return config.IMPACT_COEFFICIENT_BPS * 10
    participation = notional_usd / adv_usd
    return config.IMPACT_COEFFICIENT_BPS * float(np.sqrt(participation))


def funding_cost_bps(
    symbol: str,
    side: int,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
) -> float:
    """Funding actually paid over the holding window, in bps.

    A long pays positive funding; a short receives it. This is a real cost
    that dominates the economics of multi-day memecoin perp positions.
    """
    funding = _funding_series(symbol)
    if funding.empty:
        return 0.0
    window = funding[(funding["timestamp"] > entry_time) & (funding["timestamp"] <= exit_time)]
    if window.empty:
        return 0.0
    return float(window["funding_rate"].sum()) * side * 10_000


def round_trip_cost_bps(
    symbol: str,
    notional_usd: float,
    adv_usd: float,
    side: int = 1,
    entry_time: pd.Timestamp | None = None,
    exit_time: pd.Timestamp | None = None,
    include_funding: bool = True,
) -> dict:
    """Total round-trip cost in basis points, itemised for auditability."""
    spread, spread_source = observed_spread_bps(symbol)

    fees = 2 * config.TAKER_FEE_BPS
    # Cross the half-spread on entry and on exit.
    spread_cost = spread * config.SPREAD_SAFETY_MULTIPLIER
    impact = 2 * impact_bps(notional_usd, adv_usd)

    funding = 0.0
    if include_funding and entry_time is not None and exit_time is not None:
        funding = funding_cost_bps(symbol, side, entry_time, exit_time)

    total = fees + spread_cost + impact + max(funding, 0.0)
    # Funding received (negative cost) is allowed to help, but never below the floor.
    if funding < 0:
        total = fees + spread_cost + impact + funding

    total = max(total, config.MIN_ROUND_TRIP_BPS)

    return {
        "fees_bps": fees,
        "spread_bps": spread_cost,
        "impact_bps": impact,
        "funding_bps": funding,
        "total_bps": total,
        "spread_source": spread_source,
    }


def _vectorised_funding_bps(
    symbol: str, entry_times: pd.Series, exit_times: pd.Series, sides: pd.Series
) -> np.ndarray:
    """Funding paid per trade, via a cumulative sum and two binary searches.

    Slicing the funding frame per trade is quadratic and dominates runtime on
    a multi-thousand-trade backtest.
    """
    funding = _funding_series(symbol)
    if funding.empty:
        return np.zeros(len(entry_times))

    times = funding["timestamp"].to_numpy()
    cumulative = np.concatenate([[0.0], funding["funding_rate"].to_numpy().cumsum()])

    # Funding strictly after entry and up to and including exit.
    start = np.searchsorted(times, entry_times.to_numpy(), side="right")
    end = np.searchsorted(times, exit_times.to_numpy(), side="right")
    paid = cumulative[end] - cumulative[start]
    return paid * sides.to_numpy() * 10_000


def apply_costs(
    events: pd.DataFrame,
    symbol: str,
    notional_usd: float,
    adv_usd: float,
) -> pd.DataFrame:
    """Attach per-trade costs and net returns to a labeled event frame."""
    if events.empty:
        return events.assign(cost_bps=[], net_return=[])

    spread, _ = observed_spread_bps(symbol)
    fees = 2 * config.TAKER_FEE_BPS
    spread_cost = spread * config.SPREAD_SAFETY_MULTIPLIER
    impact = 2 * impact_bps(notional_usd, adv_usd)

    funding = _vectorised_funding_bps(
        symbol, events["entry_time"], events["exit_time"], events["side"]
    )

    total = fees + spread_cost + impact + funding
    total = np.maximum(total, config.MIN_ROUND_TRIP_BPS)

    out = events.copy()
    out["funding_bps"] = funding
    out["cost_bps"] = total
    out["net_return"] = out["gross_return"] - total / 10_000
    return out
