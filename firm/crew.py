"""Crew assembly and the report writer."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from crewai import Crew, Process

import config
from firm import events
from firm.agents import build_agents
from firm.schemas import Portfolio
from firm.tasks import build_postmortem_task, build_tasks


def report_paths(stamp: str | None = None) -> tuple[Path, Path]:
    stamp = stamp or dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    return (
        config.REPORTS_DIR / f"book_{stamp}.json",
        config.REPORTS_DIR / f"book_{stamp}.md",
    )


def build_crew(json_path: Path) -> Crew:
    agents = build_agents()
    tasks = build_tasks(agents, str(json_path))

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        max_rpm=config.CREW_MAX_RPM,
        step_callback=events.make_step_callback(),
        task_callback=events.make_task_callback(),
        verbose=True,
    )


def build_postmortem_crew() -> Crew:
    agents = build_agents()
    return Crew(
        agents=[agents["postmortem"]],
        tasks=[build_postmortem_task(agents)],
        process=Process.sequential,
        max_rpm=config.CREW_MAX_RPM,
        step_callback=events.make_step_callback(),
        task_callback=events.make_task_callback(),
        verbose=True,
    )


def render_markdown(portfolio: Portfolio) -> str:
    """Human-readable version of the book."""
    lines = [
        "# Memecoin Research Firm - Daily Book",
        "",
        f"Generated: {portfolio.generated_at}",
        f"Harness: {config.HARNESS_VERSION}",
        "",
        "## Regime",
        "",
        portfolio.regime_note or "_not stated_",
        "",
        "## Positioning",
        "",
        f"- Total exposure: {portfolio.total_exposure_pct:.2f}%",
        f"- Cash: {portfolio.cash_pct:.2f}%",
    ]

    if portfolio.correlation_warning:
        lines += ["", f"> Correlation warning: {portfolio.correlation_warning}"]

    lines += ["", "## Ideas", ""]

    if not portfolio.ideas:
        lines += [
            "**No ideas cleared the hurdle today.**",
            "",
            "This is a result, not a failure. On most days nothing passes.",
            "",
        ]
    else:
        ranked = sorted(portfolio.ideas, key=lambda i: i.expected_value_pct, reverse=True)
        for i, idea in enumerate(ranked, 1):
            lines += [
                f"### {i}. {idea.symbol} - {idea.direction.upper()} "
                f"(conviction {idea.conviction}/5)",
                "",
                f"- Entry zone: {idea.entry_low:.8g} to {idea.entry_high:.8g}",
                f"- Invalidation: {idea.invalidation:.8g}",
                f"- Targets: {', '.join(f'{t:.8g}' for t in idea.targets)}",
                f"- Win probability (calibrated): {idea.win_probability:.1%}",
                f"- R-multiple: {idea.r_multiple:.2f}",
                f"- Expected value: {idea.expected_value_pct:+.2f}%",
                f"- Position size: {idea.position_size_pct:.2f}%",
                f"- Horizon: {idea.horizon_hours}h",
                "",
                f"**Driver:** {idea.primary_driver}",
                "",
            ]
            if idea.evidence:
                lines.append("**Evidence:**")
                lines += [
                    f"- {e.claim} (`{e.source}`{f': {e.value}' if e.value else ''})"
                    for e in idea.evidence
                ]
                lines.append("")
            if idea.key_risks:
                lines.append("**Risks:**")
                lines += [f"- {r}" for r in idea.key_risks]
                lines.append("")
            if idea.red_team_rebuttal:
                lines += [f"**Red team:** {idea.red_team_rebuttal}", ""]

    if portfolio.data_caveats:
        lines += ["## What could not be verified", ""]
        lines += [f"- {c}" for c in portfolio.data_caveats]
        lines.append("")

    if portfolio.harness_state:
        lines += ["## Research programme state", "", portfolio.harness_state, ""]

    lines += [
        "---",
        "",
        "_Research output only. No orders are placed by this system. Win probabilities are "
        "assumptions to be tested, not forecasts._",
    ]
    return "\n".join(lines)


def run_daily(stamp: str | None = None) -> dict:
    """Run the full pipeline and write both report formats."""
    json_path, md_path = report_paths(stamp)
    crew = build_crew(json_path)

    events.start_run(label=json_path.stem)
    try:
        result = crew.kickoff()
    except Exception as exc:
        events.emit("run_failed", error=str(exc)[:500])
        raise

    portfolio = getattr(result, "pydantic", None)
    if isinstance(portfolio, Portfolio):
        md_path.write_text(render_markdown(portfolio), encoding="utf-8")
        json_path.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")
    else:
        # The model failed to produce the structured shape; keep the raw text
        # rather than losing the run.
        md_path.write_text(str(result), encoding="utf-8")

    events.emit(
        "run_completed",
        report=md_path.name,
        structured=isinstance(portfolio, Portfolio),
        ideas=len(portfolio.ideas) if isinstance(portfolio, Portfolio) else None,
    )

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "structured": isinstance(portfolio, Portfolio),
        "token_usage": getattr(crew, "usage_metrics", None),
    }
