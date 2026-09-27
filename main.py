"""Memecoin research firm - command line entrypoint.

    python main.py dashboard              live web UI at http://127.0.0.1:8787
    python main.py status                 what data and research state exists
    python main.py collect [--loop]       run the flow collector
    python main.py backfill               top up klines and funding history
    python main.py validate               prove the harness rejects noise
    python main.py evolve --generations 3 run the evolutionary search
    python main.py monitor                check promoted strategies for decay
    python main.py resolve                grade past forecasts
    python main.py book                   run the full agent crew (costs tokens)
    python main.py postmortem             weekly review of the firm itself

Research output only. This system never places an order.
"""

from __future__ import annotations

import argparse
import json
import sys

# CrewAI's verbose output contains emoji. The default Windows console codepage
# is cp1252, which cannot encode them, and the resulting UnicodeEncodeError
# fires on every event the bus emits. Force UTF-8 before anything prints.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import config  # noqa: E402


def cmd_status(args: argparse.Namespace) -> int:
    from firm.data import storage
    from firm.research import calibration, evolution, ledger

    inv = storage.inventory()
    bars_1h = {k: v for k, v in inv["bars"].items() if k.endswith("_1h")}

    print("=" * 68)
    print("DATA")
    print("=" * 68)
    print(f"  symbols with 1h bars : {len(bars_1h)}")
    print(f"  funding series       : {len(inv['funding'])}")
    total_bars = sum(v["rows"] for v in inv["bars"].values())
    print(f"  total bars stored    : {total_bars:,}")

    if inv["flow"]:
        span = max(r["span_days"] for r in inv["flow"])
        print(f"  flow symbols         : {len(inv['flow'])}")
        print(f"  flow span            : {span:.2f} days")
        if span < 90:
            print(f"    -> NOT yet testable. Need ~90 days; keep the collector running.")
    else:
        print("  flow                 : NONE COLLECTED")
        print("    -> Binance does not retain this. Start the collector today.")

    print()
    print("=" * 68)
    print("RESEARCH")
    print("=" * 68)
    for key, value in ledger.summary().items():
        print(f"  {key:<28} {value}")

    arch = evolution.archive_summary()
    print(f"  archive strategies           {arch['total']} ({arch['promoted']} promoted)")
    print(f"  niches occupied              {arch['niches_occupied']}")

    fc = calibration.summary()
    print(f"  forecasts recorded           {fc['forecasts_recorded']} ({fc['resolved']} resolved)")
    if fc["leaderboard"]:
        print("\n  agent calibration:")
        for row in fc["leaderboard"]:
            print(
                f"    {row['agent']:<22} n={row['n']:<5} brier={row['brier']:.4f} "
                f"skill={row['skill_score']:+.4f} {row['status']}"
            )
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from firm.data import collector

    sys.argv = ["collector"] + (["--loop"] if args.loop else [])
    collector.main()
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    from firm.data import backfill

    sys.argv = ["backfill"]
    backfill.main()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from firm.harness.nulls import run_null_suite

    out = run_null_suite(args.symbols, "1h", args.n_random)
    verdict = out["verdict"]

    print("=" * 68)
    print("HARNESS NULL VALIDATION")
    print("=" * 68)
    for r in out["nulls"]:
        o = r.get("overall", {})
        print(
            f"  {r['strategy']:<26} trades={o.get('n_trades', 0):>6} "
            f"sharpe={o.get('sharpe', 0):>+7.4f} "
            f"DSR={r.get('deflated', {}).get('dsr', 0):.4f} -> {r.get('verdict')}"
        )
    c = out["positive_control"]
    print(f"\n  {c['strategy']:<26} -> {c.get('verdict')} (must be 'promote')")
    print(f"\n  nulls incorrectly passed : {verdict['nulls_incorrectly_passed'] or 'none'}")
    print(f"  PBO among nulls          : {verdict['pbo_among_nulls'].get('pbo')}")
    print(f"  HARNESS TRUSTWORTHY      : {verdict['harness_trustworthy']}")
    return 0 if verdict["harness_trustworthy"] else 1


def cmd_evolve(args: argparse.Namespace) -> int:
    from firm.research.evolution import evolve
    from firm.research.seeds import seed_hypotheses

    symbols = args.symbols or [s for s in config.WATCHLIST]
    seeds = seed_hypotheses() if args.seeded else None

    print(f"Evolving over {len(symbols)} symbols, {args.generations} generations.")
    print("Each candidate is pre-registered, which raises the deflation bar for all future work.\n")

    out = evolve(
        symbols=symbols,
        generations=args.generations,
        population=args.population,
        seed=args.seed,
        seeds=seeds,
    )

    print("\n" + "=" * 68)
    print("ARCHIVE")
    print("=" * 68)
    for e in out["elites"]:
        flag = "PASS" if e["passed"] else "    "
        print(f"  [{flag}] {e['niche']:<28} {e['name']:<30} fitness={e['fitness']:.4f}")

    print(f"\n  niches occupied: {out['archive']['niches_occupied']}")
    print(f"  promoted       : {out['archive']['promoted']} / {out['archive']['total']}")
    if out["archive"]["promoted"] == 0:
        print("\n  Nothing passed. That is the expected outcome of an honest search.")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from firm.research.decay import attribution, run_monitor

    out = run_monitor(auto_retire=args.auto_retire)
    print(json.dumps(out, indent=2, default=str))
    print("\nATTRIBUTION")
    print(json.dumps(attribution(), indent=2, default=str))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    from firm.research import calibration

    print(json.dumps(calibration.resolve_forecasts(), indent=2))
    board = calibration.leaderboard()
    if board.empty:
        print("\nNo resolved forecasts yet.")
    else:
        print("\n" + board.to_string(index=False))
    return 0


def cmd_book(args: argparse.Namespace) -> int:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Put it in .env at the project root.")
        return 1

    from firm.crew import run_daily

    print("Running the crew. This makes real LLM calls and costs tokens.\n")
    try:
        out = run_daily()
    except Exception as exc:  # noqa: BLE001 - surface API problems readably
        message = str(exc)
        if "insufficient_quota" in message or "credit_balance_exhausted" in message:
            print(
                "\nOpenAI rejected the request: the account has no credits remaining.\n"
                "Add credits at https://platform.openai.com/settings/organization/billing/\n\n"
                "Everything except the agent layer runs without an API key:\n"
                "  python main.py status     data and research state\n"
                "  python main.py validate   prove the harness rejects noise\n"
                "  python main.py evolve     search for strategies\n"
                "  python main.py collect    keep accumulating flow data\n"
            )
            return 2
        if "AuthenticationError" in type(exc).__name__ or "invalid_api_key" in message:
            print("\nOpenAI rejected the API key. Check OPENAI_API_KEY in .env.")
            return 2
        raise

    print(f"\nReport written:\n  {out['markdown']}\n  {out['json']}")
    if not out["structured"]:
        print("\nWARNING: the model did not return the structured Portfolio shape; "
              "raw text was saved instead.")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from firm.dashboard.server import serve

    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_postmortem(args: argparse.Namespace) -> int:
    from firm.crew import build_postmortem_crew

    result = build_postmortem_crew().kickoff()
    print(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show data and research state").set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="run the flow collector")
    p.add_argument("--loop", action="store_true")
    p.set_defaults(func=cmd_collect)

    sub.add_parser("backfill", help="top up klines and funding").set_defaults(func=cmd_backfill)

    p = sub.add_parser("validate", help="prove the harness rejects noise")
    p.add_argument("--symbols", nargs="*", default=["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"])
    p.add_argument("--n-random", type=int, default=5)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("evolve", help="run the evolutionary search")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--population", type=int, default=config.EVOLUTION_POPULATION)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seeded", action="store_true", help="start from the economic seed hypotheses")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("monitor", help="check promoted strategies for decay")
    p.add_argument("--auto-retire", action="store_true")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("dashboard", help="open the live web dashboard")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_dashboard)

    sub.add_parser("resolve", help="grade past forecasts").set_defaults(func=cmd_resolve)
    sub.add_parser("book", help="run the full agent crew").set_defaults(func=cmd_book)
    sub.add_parser("postmortem", help="weekly review of the firm").set_defaults(func=cmd_postmortem)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
