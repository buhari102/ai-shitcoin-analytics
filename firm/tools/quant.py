"""Quantitative tools.

Every number here is computed in Python. The agents interpret; they never
calculate. This is the difference between a research firm and a chatbot
producing confident arithmetic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from crewai.tools import tool

import config
from firm.data import storage
from firm.harness import costs, features, stats
from firm.research import metalabel


@tool("analyze_symbol")
def analyze_symbol(symbol: str, interval: str = "1h") -> str:
    """Full quantitative profile of one symbol: volatility, beta, extension, costs.

    Returns realised volatility across windows, ATR, how extended the current
    move is versus its own distribution, beta and correlation to BTC, maximum
    drawdown, and the modelled round-trip cost. Use this before assigning any
    probability to a trade.

    Args:
        symbol: Binance spot symbol, e.g. DOGEUSDT.
        interval: Bar interval, default 1h.
    """
    f = features.build_features(symbol, interval, benchmark=config.BENCHMARK)
    if f.empty:
        return json.dumps({"error": f"insufficient local history for {symbol}"})

    last = f.iloc[-1]
    returns = f["return_1"].dropna()

    log_eq = np.cumsum(np.log1p(np.clip(returns.to_numpy(), -0.999999, None)))
    drawdown = float(np.expm1(log_eq - np.maximum.accumulate(log_eq)).min())

    adv = float(f["quote_volume"].tail(720).mean() * 24)
    cost = costs.round_trip_cost_bps(symbol, 1000.0, adv, include_funding=False)

    def val(key: str):
        v = last.get(key)
        return float(v) if v is not None and pd.notna(v) else None

    vol_24 = val("volatility_24")

    return json.dumps(
        {
            "symbol": symbol,
            "bars": int(len(f)),
            "history_since": str(f["timestamp"].min())[:10],
            "last_price": float(last["close"]),
            "volatility": {
                "per_bar_24": vol_24,
                "per_bar_168": val("volatility_168"),
                "atr_pct": val("atr_pct"),
                "annualised_pct": vol_24 * np.sqrt(24 * 365) * 100 if vol_24 else None,
            },
            "momentum": {
                "6b": val("momentum_6"),
                "24b": val("momentum_24"),
                "72b": val("momentum_72"),
                "168b": val("momentum_168"),
            },
            "extension": {
                "return_zscore": val("return_zscore"),
                "range_position_24b": val("range_position"),
                "reading": "already extended" if (val("return_zscore") or 0) > 2 else "not extreme",
            },
            "btc_relationship": {
                "beta_168": val("beta_168"),
                "correlation": float(f["return_1"].tail(720).corr(f["benchmark_return"].tail(720)))
                if "benchmark_return" in f.columns
                else None,
                "residual_return_24b": val("residual_return"),
                "reading": "mostly a leveraged BTC bet"
                if (val("beta_168") or 0) > 1.3
                else "meaningful idiosyncratic component",
            },
            "funding": {
                "current": val("funding_rate"),
                "zscore_7d": val("funding_z"),
                "cumulative_7d": val("funding_cum_7d"),
            },
            "flow": {
                "taker_buy_share": val("taker_buy_share"),
                "taker_buy_share_z": val("taker_buy_share_z"),
                "volume_zscore": val("quote_volume_z"),
            },
            "risk": {
                "max_drawdown_all_time": drawdown,
                "adv_usd": adv,
                "round_trip_cost_bps": cost["total_bps"],
                "spread_source": cost["spread_source"],
            },
            "note": "All values computed from local data. round_trip_cost_bps is the hurdle "
            "any trade must clear before it has an edge at all.",
        },
        default=str,
    )


@tool("evaluate_trade_ev")
def evaluate_trade_ev(
    symbol: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    win_probability: float,
) -> str:
    """Compute R-multiple, breakeven win rate, expected value and Kelly size.

    This is a pure function: it reports the arithmetic of the trade as
    specified, including the cost hurdle. If the stated win probability is
    below the breakeven rate, the trade is negative expectancy no matter how
    good the story is.

    Args:
        symbol: Binance spot symbol.
        direction: 'long' or 'short'.
        entry: Intended entry price.
        stop: Invalidation price.
        target: Primary profit target.
        win_probability: Probability the target is hit before the stop, 0 to 1.
    """
    if direction not in ("long", "short"):
        return json.dumps({"error": "direction must be 'long' or 'short'"})
    if not 0 < win_probability < 1:
        return json.dumps({"error": "win_probability must be strictly between 0 and 1"})

    if direction == "long":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target

    if risk <= 0:
        return json.dumps({"error": f"stop is on the wrong side of entry for a {direction}"})
    if reward <= 0:
        return json.dumps({"error": f"target is on the wrong side of entry for a {direction}"})

    risk_pct = risk / entry
    reward_pct = reward / entry
    payoff = reward / risk

    f = features.build_features(symbol, "1h", benchmark=config.BENCHMARK)
    adv = float(f["quote_volume"].tail(720).mean() * 24) if not f.empty else 0.0
    cost = costs.round_trip_cost_bps(symbol, 1000.0, adv, include_funding=False)
    cost_pct = cost["total_bps"] / 10_000

    gross_ev = win_probability * reward_pct - (1 - win_probability) * risk_pct
    net_ev = gross_ev - cost_pct

    sizing = metalabel.size_position(win_probability, payoff)

    return json.dumps(
        {
            "symbol": symbol,
            "direction": direction,
            "risk_pct": risk_pct * 100,
            "reward_pct": reward_pct * 100,
            "r_multiple": payoff,
            "breakeven_win_rate": sizing["breakeven_win_rate"],
            "stated_win_probability": win_probability,
            "edge_over_breakeven": sizing["edge"],
            "gross_ev_pct": gross_ev * 100,
            "round_trip_cost_pct": cost_pct * 100,
            "net_ev_pct": net_ev * 100,
            "full_kelly_pct": sizing["full_kelly_pct"],
            "recommended_position_pct": sizing["position_pct"],
            "meets_min_r_multiple": payoff >= config.MIN_R_MULTIPLE,
            "tradeable": net_ev > 0 and payoff >= config.MIN_R_MULTIPLE,
            "verdict": "positive expectancy after costs"
            if net_ev > 0
            else "NEGATIVE expectancy after costs - do not take this trade",
        },
        default=str,
    )


@tool("get_correlation_matrix")
def get_correlation_matrix(symbols: str = "", lookback_bars: int = 720) -> str:
    """Pairwise return correlations across symbols, plus correlation to BTC.

    Use this before allocating. Five memecoin longs with 0.9 pairwise
    correlation are one position at five times the intended size, which is
    the most common way a book blows up.

    Args:
        symbols: Comma-separated symbols. Empty means the full watchlist.
        lookback_bars: Bars of history to use.
    """
    requested = [s.strip().upper() for s in symbols.split(",") if s.strip()] or list(config.WATCHLIST)

    series = {}
    for sym in requested + [config.BENCHMARK]:
        bars = storage.load_bars(sym, "1h")
        if bars.empty:
            continue
        s = bars.set_index("timestamp")["close"].pct_change().tail(lookback_bars)
        if len(s) > 50:
            series[sym] = s

    if len(series) < 2:
        return json.dumps({"error": "need at least two symbols with local history"})

    df = pd.DataFrame(series).dropna()
    corr = df.corr()

    # Flag clusters that are effectively the same bet.
    clusters = []
    cols = [c for c in corr.columns if c != config.BENCHMARK]
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = float(corr.loc[a, b])
            if c > config.CORRELATION_CLUSTER_THRESHOLD:
                clusters.append({"pair": [a, b], "correlation": c})

    btc_corr = (
        {c: float(corr.loc[c, config.BENCHMARK]) for c in cols}
        if config.BENCHMARK in corr.columns
        else {}
    )

    return json.dumps(
        {
            "symbols": list(df.columns),
            "observations": int(len(df)),
            "correlation_to_btc": btc_corr,
            "mean_pairwise_correlation": float(
                corr.loc[cols, cols].where(~np.eye(len(cols), dtype=bool)).stack().mean()
            )
            if len(cols) > 1
            else None,
            "highly_correlated_pairs": sorted(clusters, key=lambda x: -x["correlation"]),
            "threshold": config.CORRELATION_CLUSTER_THRESHOLD,
            "risk_note": f"Pairs above {config.CORRELATION_CLUSTER_THRESHOLD} share a "
            f"{config.MAX_CORRELATED_EXPOSURE_PCT}% combined exposure cap.",
        },
        default=str,
    )
