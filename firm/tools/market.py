"""Market data tools.

These read local Parquet storage wherever possible so the agents cannot
stall on a slow network mid-run, and so two agents asking the same question
get the same answer.
"""

from __future__ import annotations

import json

import pandas as pd
from crewai.tools import tool

import config
from firm.data import binance, storage
from firm.data.http import BinanceUnavailable


@tool("screen_memecoin_universe")
def screen_memecoin_universe(min_quote_volume_usd: float = 5_000_000) -> str:
    """Screen Binance memecoin pairs by liquidity and recent activity.

    Returns live 24h price change, dollar volume, and a local-history check
    for each watchlist symbol, sorted by dollar volume. Use this first to cut
    the universe down before doing any deeper work.

    Args:
        min_quote_volume_usd: Minimum 24h quote volume to be considered liquid.
    """
    try:
        ticker = binance.ticker_24hr(config.WATCHLIST)
    except BinanceUnavailable as exc:
        return json.dumps({"error": f"Binance unreachable: {exc}", "symbols": []})

    rows = []
    for r in ticker.itertuples():
        bars = storage.load_bars(r.symbol, "1h")
        rows.append(
            {
                "symbol": r.symbol,
                "last_price": float(r.lastPrice),
                "change_24h_pct": float(r.priceChangePercent),
                "quote_volume_24h_usd": float(r.quoteVolume),
                "trades_24h": int(r.count),
                "liquid": bool(r.quoteVolume >= min_quote_volume_usd),
                "local_bars": int(len(bars)),
                "history_since": str(bars["timestamp"].min())[:10] if not bars.empty else None,
            }
        )

    rows.sort(key=lambda x: x["quote_volume_24h_usd"], reverse=True)
    liquid = [r for r in rows if r["liquid"]]

    return json.dumps(
        {
            "screened_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "liquidity_threshold_usd": min_quote_volume_usd,
            "liquid_count": len(liquid),
            "symbols": rows,
            "note": "quote_volume is live from Binance; local_bars is what is actually backtestable.",
        },
        default=str,
    )


@tool("get_price_history_stats")
def get_price_history_stats(symbol: str, interval: str = "1h", lookback_bars: int = 720) -> str:
    """Summarise recent price action for one symbol from local storage.

    Returns returns over several windows, realised volatility, ATR, the
    current position within the recent range, and volume behaviour.

    Args:
        symbol: Binance spot symbol, e.g. DOGEUSDT.
        interval: Bar interval, 1h or 1d.
        lookback_bars: How many recent bars to summarise.
    """
    df = storage.load_bars(symbol, interval)
    if df.empty:
        return json.dumps({"error": f"no local bars for {symbol} {interval}; run the backfill"})

    recent = df.tail(lookback_bars)
    close = recent["close"]
    returns = close.pct_change().dropna()

    def pct_change_over(bars: int) -> float | None:
        if len(close) <= bars:
            return None
        return float(close.iloc[-1] / close.iloc[-1 - bars] - 1)

    high = float(recent["high"].max())
    low = float(recent["low"].min())
    last = float(close.iloc[-1])

    return json.dumps(
        {
            "symbol": symbol,
            "interval": interval,
            "bars_available": int(len(df)),
            "history_since": str(df["timestamp"].min())[:10],
            "last_price": last,
            "returns": {
                "6b": pct_change_over(6),
                "24b": pct_change_over(24),
                "72b": pct_change_over(72),
                "168b": pct_change_over(168),
            },
            "realised_volatility_per_bar": float(returns.std()) if len(returns) > 1 else None,
            "range": {
                "high": high,
                "low": low,
                "position_in_range": float((last - low) / (high - low)) if high > low else None,
            },
            "volume": {
                "mean_quote_volume": float(recent["quote_volume"].mean()),
                "latest_vs_mean": float(
                    recent["quote_volume"].iloc[-1] / recent["quote_volume"].mean()
                )
                if recent["quote_volume"].mean() > 0
                else None,
            },
        },
        default=str,
    )


@tool("get_order_book_depth")
def get_order_book_depth(symbol: str, notional_usd: float = 1000.0) -> str:
    """Live order book depth, spread, and an estimate of round-trip cost.

    This is the reality check on whether an idea survives execution. A
    strategy with a 1% edge and a 1.2% round trip is not a strategy.

    Args:
        symbol: Binance spot symbol.
        notional_usd: Trade size to estimate slippage for.
    """
    try:
        book = binance.order_book(symbol, config.ORDER_BOOK_DEPTH_LIMIT)
    except BinanceUnavailable as exc:
        return json.dumps({"error": f"Binance unreachable: {exc}"})

    bids = [(float(p), float(q)) for p, q in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])]
    if not bids or not asks:
        return json.dumps({"error": f"empty order book for {symbol}"})

    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2

    # Walk the book to find the fill price for the requested size.
    filled = 0.0
    cost = 0.0
    for price, qty in asks:
        level_usd = price * qty
        take = min(level_usd, notional_usd - filled)
        cost += take * price
        filled += take
        if filled >= notional_usd:
            break

    avg_fill = cost / filled if filled > 0 else float("nan")
    slippage_bps = (avg_fill - best_ask) / mid * 10_000 if filled > 0 else None

    bid_usd = sum(p * q for p, q in bids)
    ask_usd = sum(p * q for p, q in asks)

    from firm.harness import costs

    modelled = costs.round_trip_cost_bps(
        symbol,
        notional_usd=notional_usd,
        adv_usd=float(storage.load_bars(symbol, "1h")["quote_volume"].tail(720).mean() * 24)
        if not storage.load_bars(symbol, "1h").empty
        else 0.0,
        include_funding=False,
    )

    return json.dumps(
        {
            "symbol": symbol,
            "mid": mid,
            "spread_bps": (best_ask - best_bid) / mid * 10_000,
            "bid_depth_usd": bid_usd,
            "ask_depth_usd": ask_usd,
            "depth_imbalance": (bid_usd - ask_usd) / (bid_usd + ask_usd),
            "requested_notional_usd": notional_usd,
            "fillable": filled >= notional_usd * 0.99,
            "estimated_slippage_bps": slippage_bps,
            "modelled_round_trip_bps": modelled["total_bps"],
            "cost_breakdown": {k: v for k, v in modelled.items() if k != "total_bps"},
            "interpretation": "modelled_round_trip_bps is what the harness charges; "
            "an idea must clear it to be worth taking.",
        },
        default=str,
    )
