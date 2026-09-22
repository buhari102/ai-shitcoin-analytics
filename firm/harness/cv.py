"""Purged walk-forward cross-validation with an embargo.

Plain k-fold leaks badly on financial data. A label at time t depends on
prices up to t + horizon, so a training sample adjacent to the test fold
partially contains the test fold's future. Two corrections:

  purge    drop training samples whose label window overlaps the test window
  embargo  additionally drop training samples immediately after the test
           window, because serial correlation leaks backwards too

Walk-forward (expanding window, train always before test) is used rather than
shuffled folds, because in production you only ever have the past.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged: int
    embargoed: int


def purged_walk_forward_splits(
    events: pd.DataFrame,
    n_splits: int = config.CV_N_SPLITS,
    embargo_pct: float = config.CV_EMBARGO_PCT,
) -> list[Fold]:
    """Generate expanding-window folds over a labeled event frame.

    Args:
        events: must contain `entry_time` and `exit_time`, sorted ascending.
        n_splits: number of sequential test folds.
        embargo_pct: fraction of total samples embargoed after each test fold.
    """
    if events.empty:
        return []

    events = events.sort_values("entry_time").reset_index(drop=True)
    n = len(events)
    if n < n_splits * 2:
        return []

    entry = events["entry_time"].to_numpy()
    exit_ = events["exit_time"].to_numpy()
    embargo_n = int(n * embargo_pct)

    # Expanding window: fold k trains on everything before its test block.
    block = n // (n_splits + 1)
    folds: list[Fold] = []

    for k in range(1, n_splits + 1):
        test_start_i = k * block
        test_end_i = (k + 1) * block if k < n_splits else n
        test_idx = np.arange(test_start_i, test_end_i)
        if len(test_idx) == 0:
            continue

        test_window_start = entry[test_idx].min()
        test_window_end = exit_[test_idx].max()

        candidate_train = np.arange(0, test_start_i)
        if len(candidate_train) == 0:
            continue

        # Purge: a training label whose life extends into the test window has
        # seen test-period prices.
        overlaps = exit_[candidate_train] >= test_window_start
        purged_count = int(overlaps.sum())
        train_idx = candidate_train[~overlaps]

        # Embargo: drop the tail of training immediately preceding the test
        # window to break residual serial correlation.
        embargoed_count = 0
        if embargo_n > 0 and len(train_idx) > embargo_n:
            embargoed_count = embargo_n
            train_idx = train_idx[:-embargo_n]

        if len(train_idx) == 0:
            continue

        folds.append(
            Fold(
                train_idx=train_idx,
                test_idx=test_idx,
                test_start=pd.Timestamp(test_window_start),
                test_end=pd.Timestamp(test_window_end),
                purged=purged_count,
                embargoed=embargoed_count,
            )
        )

    return folds
