"""Command line: python -m marketscout "oat milk" --country FI"""
from __future__ import annotations

import argparse
import logging
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from .graph import run_agent, run_chain
from .llm import get_llm


def save_run(state: dict, out_dir: Path) -> Path:
    name = f"{state['category']}-vs-{state['versus']}" if state.get("versus") else state["category"]
    slug = re.sub(r"[^a-z0-9]+", "-", f"{name}-{state['country']}".lower()).strip("-")
    folder = out_dir / f"{slug}-{state['metrics']['mode']}"
    folder.mkdir(parents=True, exist_ok=True)
    sources = "\n".join(f"- **{e['id']}** ({e['tool']}): {e['summary']} [source]({e['source']})"
                        for e in state["evidence"])
    (folder / "brief.md").write_text(
        state["brief"] + ("\n\n## Evidence\n" + sources if sources else "") + "\n", encoding="utf-8")
    keep = {k: state.get(k) for k in ("category", "versus", "country", "questions", "brief", "evidence", "verdicts", "tool_log", "metrics")}
    (folder / "run.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False), encoding="utf-8")
    return folder


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="marketscout", description=__doc__)
    p.add_argument("category", help="product category, e.g. 'oat milk'")
    p.add_argument("--country", default="FI", help="ISO country code (default FI)")
    p.add_argument("--mode", choices=["agent", "chain"], default="agent",
                   help="agent = tool-using graph; chain = single-prompt v1 baseline")
    p.add_argument("--versus", help="a second concept to compare head to head, e.g. 'dark luxury streetwear'")
    p.add_argument("--offline", action="store_true", help="scripted LLM, no API key needed")
    p.add_argument("--out", default="runs", type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="  %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    llm = get_llm(args.offline)
    if llm.name == "offline":
        print("[offline demo mode: scripted LLM, real data tools]", file=sys.stderr)
    if args.mode == "agent":
        state = run_agent(llm, args.category, args.country, versus=args.versus)
    else:
        state = run_chain(llm, args.category, args.country)
    folder = save_run(state, args.out)
    sys.stdout.reconfigure(encoding="utf-8")
    print(state["brief"])
    print("\n" + json.dumps(state["metrics"], indent=2))
    print(f"\nSaved to {folder}")
