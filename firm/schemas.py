"""Structured outputs.

The final deliverable is machine-readable so that tomorrow's post-mortem can
score today's calls. A report you cannot grade later is entertainment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A single factual claim traced to the tool that produced it."""

    claim: str = Field(description="The factual statement being made.")
    source: str = Field(description="Tool name that produced it, or 'prior_knowledge'.")
    value: str | None = Field(default=None, description="The specific number or value observed.")


class TradeIdea(BaseModel):
    symbol: str
    direction: Literal["long", "short", "avoid"]
    conviction: int = Field(ge=1, le=5, description="1 = marginal, 5 = highest")

    entry_low: float = Field(description="Bottom of the acceptable entry zone.")
    entry_high: float = Field(description="Top of the acceptable entry zone.")
    invalidation: float = Field(description="Hard stop. The thesis is wrong below/above this.")
    targets: list[float] = Field(description="Profit targets in order.")

    win_probability: float = Field(ge=0.0, le=1.0, description="Calibrated, not gut feel.")
    r_multiple: float = Field(description="Reward divided by risk at the primary target.")
    expected_value_pct: float = Field(description="EV per unit risked, after costs.")
    position_size_pct: float = Field(description="Percent of capital, after Risk Controller caps.")

    horizon_hours: int
    primary_driver: str = Field(description="The one thing that has to happen for this to work.")
    evidence: list[Evidence] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    red_team_rebuttal: str = Field(
        default="", description="The strongest argument against this idea, and why it was overruled."
    )


class Portfolio(BaseModel):
    generated_at: str
    regime_note: str = Field(description="What kind of market this is right now, per measured data.")
    ideas: list[TradeIdea] = Field(default_factory=list)

    total_exposure_pct: float
    cash_pct: float
    correlation_warning: str = Field(
        default="", description="Any cluster of ideas that is really the same bet."
    )

    data_caveats: list[str] = Field(
        default_factory=list,
        description="What the crew could NOT verify, e.g. flow history too short to test.",
    )
    harness_state: str = Field(
        default="", description="Whether the referee is validated and what the archive holds."
    )
