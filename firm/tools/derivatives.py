"""Derivatives and positioning tools.

These answer "what is actually moving this market" with measured data rather
than narrative. Funding has six years of history and is fully backtestable;
open interest and long/short positioning do not, which the coverage tool
makes explicit so no agent can quietly treat three weeks of data as a
validated signal.
"""

from __future__ import annotations

import json

import pandas as pd
from crewai.tools import tool

from firm.data import binance, storage
from firm.data.http import BinanceUnavailable


@tool("get_funding_profile")
def get_funding_profile(symbol: str) -> str:
    """Funding rate history and the current carry regime for a symbol.

    Positive funding means longs pay shorts: the crowd is positioned long and
    paying for it. Extreme funding is both a crowding signal and a direct
    cost of carry, which is why it appears in the cost model too.

    Args:
        symbol: Binance spot symbol, e.g. DOGEUSDT.
    """
    funding = storage.load_funding(symbol)
    if funding.empty:
        return json.dumps({"error": f"no funding history for {symbol}; run the backfill"})

    rates = funding["funding_rate"]
    recent = rates.tail(21)  # ~7 days at 8h intervals
    current = float(rates.iloc[-1])

    percentile = float((rates < current).mean() * 100)
    annualised = current * 3 * 365 * 100  # 3 payments per day

    live = {}
    try:
        perp = binance.perp_symbol(symbol)
        if perp:
            idx = binance.premium_index(perp)
            live = {
                "perp_symbol": perp,
                "mark_price": float(idx["markPrice"]),
                "index_price": float(idx["indexPrice"]),
                "next_funding_rate": float(idx["lastFundingRate"]),
                "basis_pct": (float(idx["markPrice"]) / float(idx["indexPrice"]) - 1) * 100,
            }
    except (BinanceUnavailable, KeyError, ValueError) as exc:
        live = {"error": f"live premium index unavailable: {exc}"}

    return json.dumps(
        {
            "symbol": symbol,
            "history_points": int(len(funding)),
            "history_since": str(funding["timestamp"].min())[:10],
            "current_funding_rate": current,
            "current_annualised_pct": annualised,
            "percentile_vs_own_history": percentile,
            "mean_7d": float(recent.mean()),
            "cumulative_7d": float(recent.sum()),
            "mean_all_time": float(rates.mean()),
            "std_all_time": float(rates.std()),
            "zscore_vs_7d_norm": float((current - recent.mean()) / recent.std())
            if recent.std() > 0
            else None,
            "live": live,
            "backtestable": True,
            "interpretation": "Funding has multi-year history, so a funding-based "
            "hypothesis CAN be tested with test_hypothesis.",
        },
        default=str,
    )


@tool("get_live_positioning")
def get_live_positioning(symbol: str) -> str:
    """Current open interest, long/short account ratio, and taker flow.

    Important limitation: Binance retains only about three weeks of this
    data, so it describes the present but cannot be backtested until the
    local collector has accumulated enough history. Check get_flow_coverage
    before treating any of this as a validated edge.

    Args:
        symbol: Binance spot symbol, e.g. DOGEUSDT.
    """
    perp = None
    try:
        perp = binance.perp_symbol(symbol)
        if perp is None:
            return json.dumps({"error": f"{symbol} has no Binance perp market"})

        oi = binance.open_interest_hist(perp, "1h", 168)
        lsr = binance.long_short_account_ratio(perp, "1h", 168)
        taker = binance.taker_long_short_ratio(perp, "1h", 168)
    except BinanceUnavailable as exc:
        return json.dumps({"error": f"Binance unreachable: {exc}", "perp_symbol": perp})

    def series(rows: list[dict], key: str) -> pd.Series:
        return pd.Series([float(r[key]) for r in rows]) if rows else pd.Series(dtype=float)

    oi_usd = series(oi, "sumOpenInterestValue")
    ratio = series(lsr, "longShortRatio")
    buy_sell = series(taker, "buySellRatio")

    def summarise(s: pd.Series, name: str) -> dict:
        if s.empty:
            return {"error": f"no {name} data"}
        current = float(s.iloc[-1])
        return {
            "current": current,
            "mean_7d": float(s.mean()),
            "change_24h_pct": float(s.iloc[-1] / s.iloc[-25] - 1) * 100 if len(s) > 25 else None,
            "zscore": float((current - s.mean()) / s.std()) if s.std() > 0 else None,
        }

    return json.dumps(
        {
            "symbol": symbol,
            "perp_symbol": perp,
            "window": "last 7 days, 1h resolution",
            "open_interest_usd": summarise(oi_usd, "open interest"),
            "long_short_account_ratio": summarise(ratio, "long/short ratio"),
            "taker_buy_sell_ratio": summarise(buy_sell, "taker flow"),
            "backtestable": False,
            "caveat": "Binance retains only ~21 days of this data. It is DESCRIPTIVE ONLY "
            "until the local collector has months of history. Do not claim a positioning "
            "edge is validated.",
        },
        default=str,
    )


@tool("get_flow_coverage")
def get_flow_coverage() -> str:
    """How much self-collected flow history exists, per symbol.

    This is the honest answer to whether a positioning or flow hypothesis can
    be tested yet. Until the span covers multiple market regimes, it cannot.
    """
    coverage = storage.flow_coverage()
    book = storage.load_book()

    if coverage.empty:
        return json.dumps(
            {
                "symbols_covered": 0,
                "status": "no flow data collected yet",
                "action": "run 'python -m firm.data.collector --loop' or schedule it every 15 minutes",
                "why_it_matters": "Binance does not retain this data. Every day the collector "
                "is not running is a day of history that can never be recovered.",
            }
        )

    records = coverage.to_dict(orient="records")
    max_span = max(r["span_days"] for r in records)

    return json.dumps(
        {
            "symbols_covered": len(records),
            "max_span_days": max_span,
            "book_snapshots": int(len(book)),
            "coverage": records,
            "testable": max_span >= 90,
            "assessment": (
                "Enough history to begin testing flow hypotheses."
                if max_span >= 90
                else f"Only {max_span:.1f} days collected. Flow signals are NOT yet testable; "
                "treat any flow observation as descriptive context, not evidence."
            ),
        },
        default=str,
    )
