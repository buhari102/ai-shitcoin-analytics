"""Frozen feature builders.

Every feature here is causal: it uses only information available at the close
of the bar it is attached to. Shifts are explicit and deliberate. A single
accidental look-ahead here would invalidate the entire research programme,
which is why features live inside the immutable harness rather than in
agent-editable code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm.data import storage
from firm.harness.labeling import average_true_range


def build_features(symbol: str, interval: str = "1h", benchmark: str | None = None) -> pd.DataFrame:
    """Assemble the standard feature frame for one symbol."""
    df = storage.load_bars(symbol, interval)
    if df.empty or len(df) < 100:
        return pd.DataFrame()

    df = df.copy()
    close = df["close"]

    df["return_1"] = close.pct_change()
    df["log_return"] = np.log(close).diff()

    for window in (6, 24, 72, 168):
        df[f"momentum_{window}"] = close.pct_change(window)
        df[f"volatility_{window}"] = df["log_return"].rolling(window).std()

    # Where does the latest move sit in its own recent distribution?
    roll_mean = df["return_1"].rolling(720).mean()
    roll_std = df["return_1"].rolling(720).std()
    df["return_zscore"] = (df["return_1"] - roll_mean) / roll_std

    df["atr"] = average_true_range(df)
    df["atr_pct"] = df["atr"] / close

    # Volume pressure relative to its own history.
    df["quote_volume_z"] = (
        df["quote_volume"] - df["quote_volume"].rolling(168).mean()
    ) / df["quote_volume"].rolling(168).std()

    # Taker imbalance straight from the kline payload: the only flow metric
    # with deep history, since the futures/data endpoints retain ~21 days.
    if "taker_buy_quote" in df.columns:
        total = df["quote_volume"].replace(0, np.nan)
        df["taker_buy_share"] = df["taker_buy_quote"] / total
        df["taker_buy_share_z"] = (
            df["taker_buy_share"] - df["taker_buy_share"].rolling(168).mean()
        ) / df["taker_buy_share"].rolling(168).std()

    df["range_position"] = (close - df["low"].rolling(24).min()) / (
        df["high"].rolling(24).max() - df["low"].rolling(24).min()
    )

    df = _attach_funding(df, symbol)
    if benchmark and benchmark != symbol:
        df = _attach_benchmark(df, benchmark, interval)

    # Guard against any feature accidentally referencing the current bar's
    # future: shift every derived column by one bar so a signal computed at
    # bar t can only be acted on at t's close.
    feature_cols = [c for c in df.columns if c not in
                    ("open_time", "open", "high", "low", "close", "volume",
                     "quote_volume", "trades", "taker_buy_quote", "timestamp")]
    df[feature_cols] = df[feature_cols].shift(1)

    return df.reset_index(drop=True)


def _attach_funding(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Merge funding history as-of each bar, plus its own z-score."""
    funding = storage.load_funding(symbol)
    if funding.empty:
        df["funding_rate"] = np.nan
        df["funding_z"] = np.nan
        df["funding_cum_7d"] = np.nan
        return df

    funding = funding[["timestamp", "funding_rate"]].sort_values("timestamp")
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        funding,
        on="timestamp",
        direction="backward",
    )
    merged["funding_z"] = (
        merged["funding_rate"] - merged["funding_rate"].rolling(168).mean()
    ) / merged["funding_rate"].rolling(168).std()
    # Carry accumulated over the last week: the core of the funding-carry trade.
    merged["funding_cum_7d"] = merged["funding_rate"].rolling(21).sum()
    return merged


def _attach_benchmark(df: pd.DataFrame, benchmark: str, interval: str) -> pd.DataFrame:
    """Beta and residual return vs BTC.

    Separates 'this is an idea' from 'this is leverage on beta', which is the
    single most common way a memecoin thesis turns out to be a BTC thesis.
    """
    bench = storage.load_bars(benchmark, interval)
    if bench.empty:
        df["benchmark_return"] = np.nan
        df["beta_168"] = np.nan
        df["residual_return"] = np.nan
        return df

    bench = bench[["timestamp", "close"]].rename(columns={"close": "benchmark_close"})
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        bench.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged["benchmark_return"] = merged["benchmark_close"].pct_change()

    cov = merged["return_1"].rolling(168).cov(merged["benchmark_return"])
    var = merged["benchmark_return"].rolling(168).var()
    merged["beta_168"] = cov / var
    merged["residual_return"] = merged["return_1"] - merged["beta_168"] * merged["benchmark_return"]
    return merged


def regime_label(df: pd.DataFrame) -> pd.Series:
    """Coarse regime tag used for diversity niches and calibration slicing."""
    vol = df.get("volatility_168")
    trend = df.get("momentum_168")
    if vol is None or trend is None:
        return pd.Series(["unknown"] * len(df), index=df.index)

    vol_high = vol > vol.rolling(720, min_periods=100).median()
    trend_up = trend > 0

    return pd.Series(
        np.select(
            [vol_high & trend_up, vol_high & ~trend_up, ~vol_high & trend_up],
            ["volatile_up", "volatile_down", "calm_up"],
            default="calm_down",
        ),
        index=df.index,
    )
