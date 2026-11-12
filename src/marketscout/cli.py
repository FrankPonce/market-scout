"""Command line: python -m marketscout "oat milk" --country FI"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from .graph import run_agent, run_chain
from .llm import get_llm


def save_run(state: dict, out_dir: Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{state['category']}-{state['country']}".lower())
    folder = out_dir / f"{slug}-{state['metrics']['mode']}"
    folder.mkdir(parents=True, exist_ok=True)
    sources = "\n".join(f"- **{e['id']}** ({e['tool']}): {e['summary']} [source]({e['source']})"
                        for e in state["evidence"])
    (folder / "brief.md").write_text(
        state["brief"] + ("\n\n## Evidence\n" + sources if sources else "") + "\n", encoding="utf-8")
    keep = {k: state.get(k) for k in ("category", "country", "questions", "brief", "evidence", "verdicts", "tool_log", "metrics")}
    (folder / "run.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False), encoding="utf-8")
    return folder


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="marketscout", description=__doc__)
    p.add_argument("category", help="product category, e.g. 'oat milk'")
    p.add_argument("--country", default="FI", help="ISO country code (default FI)")
    p.add_argument("--mode", choices=["agent", "chain"], default="agent",
                   help="agent = tool-using graph; chain = single-prompt v1 baseline")
    p.add_argument("--offline", action="store_true", help="scripted LLM, no API key needed")
    p.add_argument("--out", default="runs", type=Path)
    args = p.parse_args()

    llm = get_llm(args.offline)
    if llm.name == "offline":
        print("[offline demo mode: scripted LLM, real data tools]", file=sys.stderr)
    runner = run_agent if args.mode == "agent" else run_chain
    state = runner(llm, args.category, args.country)
    folder = save_run(state, args.out)
    sys.stdout.reconfigure(encoding="utf-8")
    print(state["brief"])
    print("\n" + json.dumps(state["metrics"], indent=2))
    print(f"\nSaved to {folder}")
