"""Triple-barrier labeling (Lopez de Prado, Advances in Financial ML).

Labeling by raw forward return is the wrong target for a trading system: it
ignores the fact that a position gets stopped out on the way to being right.
The triple barrier asks the question a trader actually faces - which happens
first, the profit target, the stop, or the clock?

Barriers are set in ATR units so they adapt to each symbol's volatility.
A 2% target means something completely different for DOGE than for a fresh
low-float memecoin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's ATR."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def triple_barrier_labels(
    df: pd.DataFrame,
    side: pd.Series,
    horizon: int = config.TRIPLE_BARRIER_HORIZON_BARS,
    pt_atr: float = config.TRIPLE_BARRIER_PT_ATR,
    sl_atr: float = config.TRIPLE_BARRIER_SL_ATR,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Label each bar where `side` is non-zero.

    Args:
        df: OHLC frame with a `timestamp` column, ascending.
        side: +1 long, -1 short, 0 no position, aligned to df's index.
        horizon: vertical barrier, in bars.
        pt_atr / sl_atr: barrier distances in ATR units.

    Returns a frame with one row per signal, carrying the realised gross
    return, which barrier was touched, and how many bars the trade was held.
    The holding period matters downstream: it drives both funding cost and
    the purge width in cross-validation.
    """
    atr = average_true_range(df, atr_window)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_arr = atr.to_numpy(dtype=float)
    side_arr = side.to_numpy(dtype=float)
    n = len(df)

    records = []
    for i in range(n):
        s = side_arr[i]
        if s == 0 or not np.isfinite(s):
            continue
        if not np.isfinite(atr_arr[i]) or atr_arr[i] <= 0:
            continue

        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        pt_dist = pt_atr * atr_arr[i]
        sl_dist = sl_atr * atr_arr[i]
        if s > 0:
            pt_price, sl_price = entry + pt_dist, entry - sl_dist
        else:
            pt_price, sl_price = entry - pt_dist, entry + sl_dist

        end = min(i + horizon, n - 1)
        touched = "vertical"
        exit_idx = end
        exit_price = close[end]

        for j in range(i + 1, end + 1):
            if s > 0:
                hit_sl = low[j] <= sl_price
                hit_pt = high[j] >= pt_price
            else:
                hit_sl = high[j] >= sl_price
                hit_pt = low[j] <= pt_price

            # When both barriers fall inside the same bar we cannot know the
            # intrabar path, so we assume the stop hit first. Optimism here is
            # the single easiest way to manufacture a fake edge.
            if hit_sl:
                touched, exit_idx, exit_price = "stop", j, sl_price
                break
            if hit_pt:
                touched, exit_idx, exit_price = "target", j, pt_price
                break

        gross_return = s * (exit_price - entry) / entry
        records.append(
            {
                "entry_idx": i,
                "exit_idx": exit_idx,
                "entry_time": df["timestamp"].iloc[i],
                "exit_time": df["timestamp"].iloc[exit_idx],
                "side": int(s),
                "entry_price": entry,
                "exit_price": exit_price,
                "atr": atr_arr[i],
                "barrier": touched,
                "bars_held": exit_idx - i,
                "gross_return": gross_return,
                "label": 1 if gross_return > 0 else 0,
            }
        )

    return pd.DataFrame(records)
