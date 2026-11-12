"""The faithfulness check: does the cited evidence actually back each claim?

The citation check in graph.py only proves a claim *points at* some evidence. This
module asks a model to read each claim next to the evidence it cites and label it:

    supported    the evidence states it
    partial      the evidence backs part of it, or the claim adds an inference
    unsupported  the evidence does not contain it, or contradicts it

Claims with no valid citation are unsupported by definition and cost no model call.
When more than one model provider has a key set, the judge runs on a different one
from the writer (see llm.py), so the writer does not mark its own homework.
"""
from __future__ import annotations

import json
import re

CITATION = re.compile(r"[\[【](E\d+)[\]】]")   # [E1] or the full-width 【E1】 some models emit
VERDICTS = ("supported", "partial", "unsupported")
EVIDENCE_CHARS = 1500

SYSTEM = (
    "You audit market research. For each numbered claim, read ONLY the evidence it cites "
    "and decide:\n"
    "- supported: the evidence directly states the facts in the claim.\n"
    "- partial: the evidence backs some of it, or the claim adds an inference, cause, or "
    "generalisation the evidence does not state.\n"
    "- unsupported: the facts are not in the cited evidence, or it contradicts them.\n"
    "For a recommendation or risk, judge the facts it rests on, not whether the advice is wise. "
    "Be strict about numbers. Reply with JSON: "
    '{"verdicts": [{"n": 1, "verdict": "supported|partial|unsupported", "reason": "at most 15 words"}]}'
)


def claim_lines(brief: str) -> list[str]:
    """Every '- ' bullet outside 'Open questions' is a claim."""
    lines, exempt = [], False
    for line in brief.splitlines():
        line = line.strip()
        if line.startswith("#"):
            exempt = "open questions" in line.lower()
        elif line.startswith("- ") and not exempt:
            lines.append(line)
    return lines


def judge_claims(llm, brief: str, evidence: list[dict]) -> list[dict]:
    """One verdict per claim, in the order the claims appear in the brief."""
    by_id = {e["id"]: e for e in evidence}
    results, to_judge = [], []
    for line in claim_lines(brief):
        cites = list(dict.fromkeys(CITATION.findall(line)))
        text = CITATION.sub("", line[2:]).replace("**", "").strip()
        item = {"claim": text, "cites": cites, "verdict": "unchecked", "reason": ""}
        if not cites or any(c not in by_id for c in cites):
            item.update(verdict="unsupported", reason="No valid evidence cited.")
        else:
            to_judge.append(item)
        results.append(item)
    if not to_judge:
        return results

    used = sorted({c for item in to_judge for c in item["cites"]}, key=lambda c: int(c[1:]))
    evidence_text = "\n\n".join(f"[{c}] {by_id[c]['summary'][:EVIDENCE_CHARS]}" for c in used)
    claims_text = "\n".join(f"{n}. {item['claim']} (cites {', '.join(item['cites'])})"
                            for n, item in enumerate(to_judge, 1))
    reply = llm.chat("judge", [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Evidence:\n{evidence_text}\n\nClaims:\n{claims_text}"},
    ], json_mode=True, context={"claims": len(to_judge)})
    try:
        for v in json.loads(reply["content"])["verdicts"]:
            n = int(v["n"])
            verdict = str(v.get("verdict", "")).lower().strip()
            if 1 <= n <= len(to_judge) and verdict in VERDICTS:
                to_judge[n - 1].update(verdict=verdict, reason=str(v.get("reason", ""))[:160])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass        # leave them "unchecked" rather than guess
    return results


def tally(verdicts: list[dict]) -> dict:
    counts = {v: sum(1 for x in verdicts if x["verdict"] == v) for v in (*VERDICTS, "unchecked")}
    checked = len(verdicts) - counts["unchecked"]
    return {"supported": counts["supported"], "partial": counts["partial"],
            "not_supported": counts["unsupported"], "unchecked": counts["unchecked"],
            "faithfulness": round(counts["supported"] / checked, 2) if checked else None}
