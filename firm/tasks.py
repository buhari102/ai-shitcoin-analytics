"""The pipeline.

Sequential with explicit context chaining rather than a hierarchical manager:
the workflow is fixed, so a manager agent would add cost and nondeterminism
without adding judgement. Research and Quant fan out from the same shortlist
and rejoin at the timing task.
"""

from __future__ import annotations

import datetime as dt

from crewai import Agent, Task

import config
from firm.schemas import Portfolio


def build_tasks(agents: dict[str, Agent], report_path: str) -> list[Task]:
    today = dt.date.today().isoformat()

    screen = Task(
        description=(
            "Screen the Binance memecoin watchlist for today. Call screen_memecoin_universe. "
            "Then shortlist the 4 to 6 symbols that combine real liquidity with unusual "
            "current activity.\n\n"
            "For each shortlisted symbol state: last price, 24h change, 24h dollar volume, "
            "how many bars of local history exist, and the date that history starts. "
            "Explicitly exclude anything below the liquidity threshold and say why.\n\n"
            "If a symbol has less than a year of local history, flag it: conclusions about "
            "how it behaves across regimes cannot be supported by that sample."
        ),
        expected_output=(
            "A shortlist of 4-6 symbols. For each: price, 24h change, dollar volume, bars of "
            "local history, history start date, and a one-line reason for inclusion. Plus a "
            "short list of what was excluded and why."
        ),
        agent=agents["scout"],
    )

    research = Task(
        description=(
            "For each shortlisted symbol, identify the mechanism that would actually move it.\n\n"
            "Call get_funding_profile and get_live_positioning for each. Call get_flow_coverage "
            "ONCE at the start and obey what it says.\n\n"
            "Funding has multi-year history and can support a testable claim. Open interest and "
            "long/short ratio are retained by Binance for only about three weeks, so they are "
            "DESCRIPTIVE CONTEXT ONLY until the local collector has months of data. Do not imply "
            "otherwise.\n\n"
            "For each symbol give: the current funding regime and where it sits in its own "
            "history, what positioning currently looks like, whether the move is idiosyncratic "
            "or just BTC beta, and the single most likely driver over the next few days. Label "
            "every claim as MEASURED or PRIOR KNOWLEDGE."
        ),
        expected_output=(
            "Per symbol: funding regime with percentile, current positioning, the dominant "
            "driver, and a MEASURED/PRIOR KNOWLEDGE label on every claim. Plus one paragraph "
            "on the overall market regime and an explicit statement of the flow data limitation."
        ),
        agent=agents["research"],
        context=[screen],
        async_execution=True,
    )

    quant = Task(
        description=(
            "Quantify the shortlist.\n\n"
            "1. Call analyze_symbol for each shortlisted symbol.\n"
            "2. Call get_correlation_matrix once across the shortlist.\n"
            "3. Call get_failure_log to see what has already been tried and rejected.\n"
            "4. Propose at least one NEW falsifiable hypothesis and submit it via "
            "test_hypothesis. Use only the listed features. State the economic rationale "
            "before you see the result.\n\n"
            "Report the harness verdict exactly as returned, including the rejection reasons "
            "if it failed. Do not reinterpret a rejection as a partial success.\n\n"
            "For each symbol give realised volatility, beta to BTC, how extended the current "
            "move is, the round-trip cost hurdle, and a preliminary win probability WITH the "
            "breakeven win rate it must beat."
        ),
        expected_output=(
            "Per symbol: volatility, BTC beta and correlation, extension z-score, round-trip "
            "cost in bps, preliminary win probability and the breakeven rate. Plus the full "
            "verdict on at least one newly tested hypothesis, quoted from the harness."
        ),
        agent=agents["quant"],
        context=[screen],
        async_execution=True,
    )

    timing = Task(
        description=(
            "Turn the analysis into executable levels.\n\n"
            "For each surviving candidate call get_order_book_depth and evaluate_trade_ev.\n\n"
            "Specify: an entry zone as two prices, a hard invalidation price, at least one "
            "target, and the resulting R-multiple. Then check the idea against the modelled "
            "round-trip cost and state whether the expected move is large enough to clear it "
            "at a realistic size.\n\n"
            "Drop any idea whose R-multiple is below "
            f"{config.MIN_R_MULTIPLE} or whose net expected value is negative. Say which ones "
            "you dropped and why."
        ),
        expected_output=(
            "Per surviving candidate: entry zone, invalidation, targets, R-multiple, net EV "
            "after costs, and depth/slippage assessment. Plus an explicit list of candidates "
            "dropped at this stage with reasons."
        ),
        agent=agents["microstructure"],
        context=[screen, research, quant],
    )

    red_team = Task(
        description=(
            "Attack every surviving idea. Your job is to kill them, not to balance them.\n\n"
            "For each, work through the checklist explicitly:\n"
            "  1. Is this just leveraged BTC exposure? Check the beta.\n"
            "  2. Does the edge survive the modelled round-trip cost?\n"
            "  3. Is the sample large enough to distinguish from noise?\n"
            "  4. Did it only work in one regime? Check by_regime if a hypothesis was tested.\n"
            "  5. Has the firm already tested and rejected this? Call get_failure_log.\n"
            "  6. Is any claim resting on flow data that is too short to be validated?\n\n"
            "Call get_research_state to see how many trials the firm has run. A strategy that "
            "passed on a high trial count deserves more suspicion, not less.\n\n"
            "For each idea deliver a verdict: KILLED with the reason, or SURVIVED with the "
            "strongest remaining objection and what evidence would change your mind."
        ),
        expected_output=(
            "Per idea: the six-point checklist worked through, and a verdict of KILLED (with "
            "reason) or SURVIVED (with the strongest remaining objection). Plus an overall "
            "statement on how much of today's work rests on validated evidence."
        ),
        agent=agents["red_team"],
        context=[research, quant, timing],
    )

    risk = Task(
        description=(
            "Size the survivors and enforce the limits.\n\n"
            "1. Call get_correlation_matrix and identify clusters that are really one bet.\n"
            "2. Call get_agent_calibration to see whether this firm's stated probabilities have "
            "historically meant anything. If there is not enough forecast history yet, say so "
            "and discount confidence accordingly.\n"
            "3. For each surviving idea call evaluate_trade_ev and use the recommended position "
            "size as a ceiling, not a target.\n"
            "4. Call record_agent_forecast for EVERY idea you approve, so it can be graded later.\n\n"
            f"Hard limits: {config.MAX_POSITION_PCT}% per idea, "
            f"{config.MAX_TOTAL_EXPOSURE_PCT}% total, "
            f"{config.MAX_CORRELATED_EXPOSURE_PCT}% per correlated cluster, minimum R-multiple "
            f"{config.MIN_R_MULTIPLE}. Reject anything that breaches them.\n\n"
            "Approving zero ideas is a valid and often correct outcome."
        ),
        expected_output=(
            "Per approved idea: final position size as a percent of capital, the calibrated "
            "probability used, and confirmation the forecast was recorded. Plus total exposure, "
            "cash percentage, any correlation cluster warning, and a list of ideas rejected on "
            "risk grounds with the specific limit breached."
        ),
        agent=agents["risk"],
        context=[timing, red_team],
    )

    final = Task(
        description=(
            "Produce the firm's book for " + today + ".\n\n"
            "Call get_research_state first and report the state of the programme honestly: "
            "how many hypotheses have been registered, how many survived, whether the harness "
            "has been validated, and how much flow history exists.\n\n"
            "Rank the approved ideas by expected value after costs. For each, state plainly "
            "what would make it wrong.\n\n"
            "In data_caveats, list what the crew could NOT verify today. At minimum this must "
            "mention that open interest and long/short positioning data has only about three "
            "weeks of history and cannot yet be backtested.\n\n"
            "If nothing cleared the hurdle, return an empty ideas list and say so directly. "
            "That is a real result, not a failure to produce one.\n\n"
            "Every numeric field must trace to a tool call. Populate the evidence list for "
            "each idea with the tool that produced each figure."
        ),
        expected_output=(
            "A Portfolio object: regime note, ranked trade ideas with entry zones, "
            "invalidation, targets, calibrated win probabilities, R-multiples, expected values "
            "and position sizes, each with evidence and the red team's rebuttal; plus total "
            "exposure, cash, correlation warnings, data caveats, and the harness state."
        ),
        agent=agents["cio"],
        context=[screen, research, quant, timing, red_team, risk],
        output_pydantic=Portfolio,
        output_file=report_path,
    )

    return [screen, research, quant, timing, red_team, risk, final]


def build_postmortem_task(agents: dict[str, Agent]) -> Task:
    """Weekly review. Run separately from the daily book."""
    return Task(
        description=(
            "Grade the firm.\n\n"
            "1. Call get_agent_calibration with no argument for the full leaderboard.\n"
            "2. Call get_research_state for the ledger and archive.\n"
            "3. Call get_archive_elites for niche attribution.\n\n"
            "Report: which agents are calibrated and which are systematically overconfident, "
            "which niches produce edge and which keep dying, whether promoted strategies are "
            "decaying faster than their backtests implied, and the promotion rate against the "
            "total trial count.\n\n"
            "Finish with three specific, actionable changes to how the firm proposes "
            "hypotheses. Be blunt. Vague encouragement is worthless here."
        ),
        expected_output=(
            "A review covering agent calibration, niche attribution, decay, and the promotion "
            "rate versus trial count, ending with three specific changes to hypothesis "
            "generation."
        ),
        agent=agents["postmortem"],
    )
