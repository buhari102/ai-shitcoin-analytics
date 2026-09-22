"""CrewAI tools. Every factual claim an agent makes must trace to one of these."""

from firm.tools.market import (
    get_order_book_depth,
    get_price_history_stats,
    screen_memecoin_universe,
)
from firm.tools.derivatives import (
    get_flow_coverage,
    get_funding_profile,
    get_live_positioning,
)
from firm.tools.quant import analyze_symbol, evaluate_trade_ev, get_correlation_matrix
from firm.tools.research_tools import (
    get_agent_calibration,
    get_archive_elites,
    get_failure_log,
    get_research_state,
    record_agent_forecast,
    test_hypothesis,
)

MARKET_TOOLS = [screen_memecoin_universe, get_price_history_stats, get_order_book_depth]
DERIVATIVE_TOOLS = [get_funding_profile, get_live_positioning, get_flow_coverage]
QUANT_TOOLS = [analyze_symbol, evaluate_trade_ev, get_correlation_matrix]
RESEARCH_TOOLS = [test_hypothesis, get_archive_elites, get_failure_log, get_research_state]
CALIBRATION_TOOLS = [get_agent_calibration, record_agent_forecast]

__all__ = [
    "MARKET_TOOLS",
    "DERIVATIVE_TOOLS",
    "QUANT_TOOLS",
    "RESEARCH_TOOLS",
    "CALIBRATION_TOOLS",
]
