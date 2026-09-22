"""Central configuration for the memecoin research firm.

Everything tunable lives here so the agents never hardcode assumptions and the
harness can record exactly which settings produced a result.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BARS_DIR = DATA_DIR / "bars"
FUNDING_DIR = DATA_DIR / "funding"
FLOW_DIR = DATA_DIR / "flow"
BOOK_DIR = DATA_DIR / "book"
REPORTS_DIR = ROOT / "reports"
LEDGER_DB = DATA_DIR / "ledger.sqlite"
ARCHIVE_DB = DATA_DIR / "archive.sqlite"
FORECAST_DB = DATA_DIR / "forecasts.sqlite"
COLLECTOR_LOG = DATA_DIR / "collector.log"

for _d in (DATA_DIR, BARS_DIR, FUNDING_DIR, FLOW_DIR, BOOK_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

# Binance memecoin pairs available on SPOT, which is where bar history comes
# from. The screener intersects this with exchangeInfo, so a delisting drops
# out automatically rather than silently poisoning a backtest.
#
# Funding and flow are fetched from the USD-M perp, which for low-priced
# tokens is quoted in units of 1000 (SHIBUSDT spot -> 1000SHIBUSDT perp).
# firm.data.binance.perp_symbol resolves that mapping.
WATCHLIST: list[str] = [
    "DOGEUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "BONKUSDT",
    "FLOKIUSDT",
    "PENGUUSDT",
    "BOMEUSDT",
    "NEIROUSDT",
    "TURBOUSDT",
    "PNUTUSDT",
    "ACTUSDT",
    "TRUMPUSDT",
]

# Verified as futures-only (no Binance spot market), so they have no kline
# history in this pipeline. Add them back if spot listings appear.
FUTURES_ONLY: list[str] = ["SPXUSDT", "POPCATUSDT", "MEWUSDT", "FARTCOINUSDT"]

# Benchmark used for beta / regime conditioning.
BENCHMARK = "BTCUSDT"

# Backfill horizons. Verified available: klines from 2019-07, funding from 2020-07.
BACKFILL_START_MS = 1_500_000_000_000  # 2017-07; API clamps to first listing
BAR_INTERVALS: list[str] = ["1h", "4h", "1d"]

# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------

# Flow endpoints only retain ~21-30 days, so this must run continuously to
# accumulate history that cannot be bought or backfilled later.
COLLECTOR_INTERVAL_MINUTES = 15
COLLECTOR_PERIOD = "5m"  # granularity requested from the futures/data endpoints
ORDER_BOOK_DEPTH_LIMIT = 100

# --------------------------------------------------------------------------
# Adversarial cost model
# --------------------------------------------------------------------------

TAKER_FEE_BPS = 5.0  # 0.05% Binance taker, no BNB discount assumed
# Extra spread charged on top of observed half-spread, as a safety margin.
SPREAD_SAFETY_MULTIPLIER = 1.5
# Square-root market impact coefficient; impact_bps = COEF * sqrt(notional / adv)
IMPACT_COEFFICIENT_BPS = 40.0
# Minimum cost floor per round trip regardless of what the data says.
MIN_ROUND_TRIP_BPS = 15.0

# --------------------------------------------------------------------------
# Risk limits (hard caps the Risk Controller cannot exceed)
# --------------------------------------------------------------------------

MAX_POSITION_PCT = 2.0
MAX_TOTAL_EXPOSURE_PCT = 10.0
MAX_CORRELATED_EXPOSURE_PCT = 5.0  # across ideas with pairwise corr > 0.7
CORRELATION_CLUSTER_THRESHOLD = 0.7
MIN_R_MULTIPLE = 2.5
KELLY_FRACTION = 0.25  # quarter-Kelly; full Kelly is ruin-seeking here

# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

HARNESS_VERSION = "1.0.0"
CV_N_SPLITS = 6
CV_EMBARGO_PCT = 0.01  # fraction of total samples embargoed after each test fold
TRIPLE_BARRIER_HORIZON_BARS = 24
TRIPLE_BARRIER_PT_ATR = 2.0  # profit target in ATR units
TRIPLE_BARRIER_SL_ATR = 1.0  # stop loss in ATR units
MIN_TRADES_FOR_SIGNIFICANCE = 30

# Holdout vault: reserved slices the research loop must never read.
HOLDOUT_TAIL_DAYS = 90  # most recent N days
HOLDOUT_BUDGET_PER_STRATEGY = 1

# --------------------------------------------------------------------------
# Evolution
# --------------------------------------------------------------------------

ARCHIVE_ELITES_PER_NICHE = 3
EVOLUTION_POPULATION = 8

# --------------------------------------------------------------------------
# Decay monitoring
# --------------------------------------------------------------------------

DECAY_WINDOW_TRADES = 40
DECAY_ZSCORE_RETIRE = -2.0

# --------------------------------------------------------------------------
# Forecast calibration
# --------------------------------------------------------------------------

# Minimum resolved forecasts before an agent's probabilities are trusted at all.
MIN_FORECASTS_FOR_CALIBRATION = 25
# Below this skill score (vs the base rate) an agent loses its vote.
DEMOTION_SKILL_THRESHOLD = 0.0

# --------------------------------------------------------------------------
# LLM assignment per role (cheap models for grunt work, strong for judgement)
# --------------------------------------------------------------------------

CHEAP_MODEL = os.getenv("FIRM_CHEAP_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.getenv("FIRM_STRONG_MODEL", "gpt-4o")

ROLE_MODELS: dict[str, str] = {
    "scout": CHEAP_MODEL,
    "research": STRONG_MODEL,
    "quant": STRONG_MODEL,
    "microstructure": CHEAP_MODEL,
    "red_team": STRONG_MODEL,
    "risk": STRONG_MODEL,
    "cio": STRONG_MODEL,
    "postmortem": CHEAP_MODEL,
}

CREW_MAX_RPM = 20
AGENT_MAX_ITER = 12
