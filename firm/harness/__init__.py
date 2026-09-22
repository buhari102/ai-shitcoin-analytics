"""The immutable referee.

Nothing in this package may be modified by an agent. Changing the scoring
rules invalidates every stored result, so any edit requires bumping
config.HARNESS_VERSION, which in turn marks prior scorecards as stale.
"""

from firm.harness.engine import evaluate_strategy
from firm.harness.stats import deflated_sharpe_ratio, probability_of_backtest_overfitting

__all__ = [
    "evaluate_strategy",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
]
