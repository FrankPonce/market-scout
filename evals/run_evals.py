"""Run every case in cases.json through both modes and write evals/scorecard.md.

    python evals/run_evals.py            # uses your API key if set
    python evals/run_evals.py --offline  # scripted LLM

The point is the comparison: the same categories through the v1 single-prompt chain
and the v2 tool-using agent, scored the same way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from marketscout.graph import run_agent, run_chain
from marketscout.llm import get_llm

HERE = Path(__file__).parent
COLUMNS = ["mode", "claims", "citation_coverage", "supported", "partial", "not_supported",
           "faithfulness", "evidence", "tool_errors", "revisions", "tokens", "seconds"]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    rows, llm_name = [], ""
    for case in cases:
        for runner in (run_chain, run_agent):
            llm = get_llm(args.offline)      # fresh client so usage is per run
            llm_name = llm.name
            metrics = runner(llm, case["category"], case["country"])["metrics"]
            rows.append((f"{case['category']} / {case['country']}", metrics))
            print(rows[-1])

    lines = [f"# Scorecard ({llm_name} LLM, {len(cases)} cases)", "",
             "| case | " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * (len(COLUMNS) + 1)]
    lines += [f"| {name} | " + " | ".join(str(m[c]) for c in COLUMNS) + " |" for name, m in rows]
    (HERE / "scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {HERE / 'scorecard.md'}")


if __name__ == "__main__":
    main()
