"""Economically motivated starting hypotheses.

Random search wastes trials, and every wasted trial permanently raises the
deflation bar for everything tested afterwards. These seeds encode the few
effects in crypto perps that have some documented basis, so the search
starts from plausible territory rather than noise.

None of these are expected to pass as written. They are starting points for
the evolutionary loop, and several of them exist specifically so the firm
can establish that they do NOT work on this universe after costs.
"""

from __future__ import annotations

from firm.research.dsl import Condition, RuleSpec


def seed_hypotheses() -> list[RuleSpec]:
    return [
        RuleSpec(
            name="funding_carry_crowded_short",
            direction="short",
            conditions=[
                Condition("funding_z", ">", 1.5),
                Condition("momentum_24", ">", 0.03),
            ],
            horizon=24,
            pt_atr=2.0,
            sl_atr=1.0,
            rationale=(
                "When funding is far above its own norm after a sharp rally, longs are crowded "
                "and paying to stay in. Leverage flushes tend to resolve downward, and the "
                "short additionally receives carry while waiting."
            ),
        ),
        RuleSpec(
            name="funding_carry_capitulation_long",
            direction="long",
            conditions=[
                Condition("funding_z", "<", -1.5),
                Condition("momentum_24", "<", -0.03),
            ],
            horizon=24,
            pt_atr=2.0,
            sl_atr=1.0,
            rationale=(
                "The mirror image: deeply negative funding after a sharp drop means shorts are "
                "crowded and paying. Short squeezes are the most reliable upside catalyst in "
                "low-float memecoins."
            ),
        ),
        RuleSpec(
            name="timeseries_momentum_weekly",
            direction="long",
            conditions=[
                Condition("momentum_168", ">", 0.10),
                Condition("volatility_168", "<", 0.05),
            ],
            horizon=72,
            pt_atr=2.5,
            sl_atr=1.0,
            rationale=(
                "Time-series momentum is among the most robust cross-asset anomalies. The "
                "volatility filter avoids buying parabolic moves where the ATR-scaled stop is "
                "already enormous."
            ),
        ),
        RuleSpec(
            name="volume_breakout_confirmed",
            direction="long",
            conditions=[
                Condition("quote_volume_z", ">", 2.0),
                Condition("range_position", ">", 0.8),
                Condition("taker_buy_share_z", ">", 0.5),
            ],
            horizon=24,
            pt_atr=2.5,
            sl_atr=1.0,
            rationale=(
                "A breakout to the top of the recent range on abnormal volume AND abnormal "
                "aggressive buying is the classic memecoin ignition pattern. Taker share is "
                "the only flow metric with deep history, since the futures/data endpoints "
                "retain only three weeks."
            ),
        ),
        RuleSpec(
            name="idiosyncratic_reversal",
            direction="long",
            conditions=[
                Condition("return_zscore", "<", -2.0),
                Condition("beta_168", "<", 1.0),
            ],
            horizon=24,
            pt_atr=2.0,
            sl_atr=1.0,
            rationale=(
                "An extreme down move in a name with low BTC beta is more likely to be "
                "idiosyncratic liquidation than a market-wide repricing, and liquidation "
                "cascades overshoot."
            ),
        ),
        RuleSpec(
            name="exhaustion_after_parabolic",
            direction="short",
            conditions=[
                Condition("return_zscore", ">", 2.5),
                Condition("quote_volume_z", ">", 1.5),
                Condition("funding_z", ">", 1.0),
            ],
            horizon=12,
            pt_atr=2.0,
            sl_atr=1.0,
            rationale=(
                "A parabolic move on heavy volume with funding already elevated is late-stage "
                "distribution. The short horizon reflects how fast these unwind, and the ATR "
                "stop keeps the risk bounded when they do not."
            ),
        ),
        RuleSpec(
            name="low_vol_compression_long",
            direction="long",
            conditions=[
                Condition("volatility_24", "<", 0.01),
                Condition("quote_volume_z", ">", 1.0),
            ],
            horizon=48,
            pt_atr=3.0,
            sl_atr=1.0,
            rationale=(
                "Volatility clusters. Compression with rising volume often precedes expansion, "
                "and buying the quiet period gives a favourable ATR-scaled reward to risk "
                "because the stop is tight while the range is narrow."
            ),
        ),
        RuleSpec(
            name="btc_beta_laggard",
            direction="long",
            conditions=[
                Condition("residual_return", "<", -0.02),
                Condition("beta_168", ">", 1.0),
            ],
            horizon=24,
            pt_atr=2.0,
            sl_atr=1.0,
            rationale=(
                "A high-beta name that underperformed its BTC-implied return has a catch-up "
                "tendency. This explicitly separates the idea from simple BTC exposure, which "
                "is the failure mode the Red Team checks first."
            ),
        ),
    ]
