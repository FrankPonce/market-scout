"""The agent, as a LangGraph state machine.

    plan -> research <-> tools -> write <-> critique -> judge -> END

- plan:     turns "category + country" into research questions.
- research: the LLM picks which tool to call next (this is the agentic part:
            the path is chosen at run time, not wired in advance).
- tools:    runs the calls; each success becomes a numbered piece of evidence.
- write:    drafts the brief; every finding must cite evidence like [E2].
- critique: checks every finding is cited, and sends the draft back if not.
- judge:    reads each claim against the evidence it cites (see judge.py), and
            sends the draft back once if any claim is not backed.

`run_chain` is the v1 baseline: one prompt, no tools, like the original n8n chain.
"""
from __future__ import annotations

import json
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import extern  # noqa: F401  (registers the optional rug case-study tools)
from . import tools as toolbox
from .judge import CITATION, claim_lines, judge_claims, tally

MAX_RESEARCH_STEPS = 8
MAX_REVISIONS = 2
MAX_JUDGE_REWRITES = 1      # how many times the judge may send a draft back
# Free model tiers cap tokens per minute, so long sources are trimmed in the prompts.
# The full text stays in the evidence record.
RESEARCH_CHARS = 900        # of each tool result echoed back to the researcher
WRITER_CHARS = 1500         # of each piece of evidence shown to the writer


class RunState(TypedDict, total=False):
    category: str
    country: str
    questions: list[str]
    messages: list[dict]        # the researcher's conversation with its tools
    evidence: list[dict]        # {id, tool, args, source, summary}
    research_steps: int
    tool_calls: int
    tool_errors: int
    tool_log: list[dict]        # every call: {tool, args, ok, detail}
    brief: str
    feedback: str
    revisions: int
    verdicts: list[dict]        # the judge's ruling on each claim
    judge_rewrites: int


def check_citations(brief: str, evidence: list[dict]) -> dict:
    """Every claim needs a citation to evidence that exists. Says nothing about accuracy."""
    known = {e["id"] for e in evidence}
    lines = claim_lines(brief)
    unsupported = [line for line in lines
                   if not CITATION.findall(line) or any(c not in known for c in CITATION.findall(line))]
    return {"claims": len(lines), "unsupported": unsupported,
            "coverage": (len(lines) - len(unsupported)) / len(lines) if lines else 0.0}


def build_graph(llm):
    def ctx(state: RunState) -> dict:
        return {"category": state["category"], "country": state["country"],
                "evidence": state.get("evidence", [])}

    def plan(state: RunState) -> dict:
        reply = llm.chat("planner", [
            {"role": "system", "content":
                "You plan market research. Reply with JSON: {\"questions\": [3 to 5 short, "
                "answerable research questions]}."},
            {"role": "user", "content":
                f"Product category: {state['category']}\nTarget country: {state['country']}"},
        ], json_mode=True, context=ctx(state))
        try:
            questions = json.loads(reply["content"])["questions"]
        except (json.JSONDecodeError, KeyError, TypeError):
            questions = [f"What is the market outlook for {state['category']} in {state['country']}?"]
        messages = [
            {"role": "system", "content":
                "You are a market researcher. Use the tools to gather evidence for the "
                "questions. Call one tool at a time. First call EVERY structured data tool "
                "that applies to this category and country (prices, catalogues, statistics, "
                "interest, social trends), one after another; only then use web search to "
                "fill gaps. Only fetch a "
                "URL that a tool result gave you; never guess or invent URLs. If a tool "
                "fails, try a different angle, such as a related or competing product. "
                "When you have enough evidence, or tools stop helping, reply 'Research complete.'"},
            {"role": "user", "content":
                f"Category: {state['category']}\nCountry: {state['country']}\nQuestions:\n"
                + "\n".join(f"- {q}" for q in questions)},
        ]
        return {"questions": questions, "messages": messages, "evidence": [],
                "research_steps": 0, "tool_calls": 0, "tool_errors": 0,
                "tool_log": [], "revisions": 0, "judge_rewrites": 0}

    def research(state: RunState) -> dict:
        reply = llm.chat("researcher", state["messages"], tools=toolbox.available_schemas(),
                         context=ctx(state))
        return {"messages": state["messages"] + [reply],
                "research_steps": state["research_steps"] + 1}

    def after_research(state: RunState) -> str:
        wants_tools = bool(state["messages"][-1].get("tool_calls"))
        return "tools" if wants_tools and state["research_steps"] <= MAX_RESEARCH_STEPS else "write"

    def run_tools(state: RunState) -> dict:
        messages, evidence = list(state["messages"]), list(state["evidence"])
        calls, errors = state["tool_calls"], state["tool_errors"]
        log = list(state["tool_log"])
        for call in messages[-1]["tool_calls"]:
            name, raw_args = call["function"]["name"], call["function"]["arguments"]
            calls += 1
            try:
                args = json.loads(raw_args or "{}")
                result = toolbox.call_tool(name, args)
                item = {"id": f"E{len(evidence) + 1}", "tool": name, "args": args, **result}
                evidence.append(item)
                content = f"[{item['id']}] {item['summary'][:RESEARCH_CHARS]}\nsource: {item['source']}"
                log.append({"tool": name, "args": raw_args, "ok": True, "detail": item["id"]})
            except Exception as e:  # report the failure back so the agent can adapt
                errors += 1
                content = f"ERROR from {name}: {type(e).__name__}: {e}"
                log.append({"tool": name, "args": raw_args, "ok": False,
                            "detail": f"{type(e).__name__}: {e}"})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        return {"messages": messages, "evidence": evidence,
                "tool_calls": calls, "tool_errors": errors, "tool_log": log}

    def write(state: RunState) -> dict:
        evidence_text = "\n".join(f"[{e['id']}] ({e['tool']}) {e['summary'][:WRITER_CHARS]}"
                                  for e in state["evidence"]) or "(no evidence gathered)"
        prompt = (
            f"Write a market brief on {state['category']} in {state['country']} in Markdown.\n"
            "Sections, in this order: '## Summary' (two or three plain sentences a busy "
            "manager could act on, no bullets), '## Findings', '## Risks', "
            "'## Recommendations', '## Open questions'.\n"
            "Rules: every bullet in Findings, Risks and Recommendations must end with the "
            "evidence id(s) it rests on, in plain square brackets like [E1]. Compare items "
            "where the evidence allows (brands, products, periods). Use ONLY the evidence "
            "below; never invent numbers. A risk must follow from the evidence, not from "
            "general knowledge. Put anything the evidence cannot answer under Open questions.\n\n"
            f"Research questions:\n" + "\n".join(f"- {q}" for q in state["questions"])
            + f"\n\nEvidence:\n{evidence_text}"
        )
        if state.get("feedback"):
            prompt += f"\n\nYour previous draft was rejected:\n{state['feedback']}\n\nPrevious draft:\n{state['brief']}"
        reply = llm.chat("writer", [{"role": "user", "content": prompt}], context=ctx(state))
        return {"brief": reply["content"] or ""}

    def critique(state: RunState) -> dict:
        check = check_citations(state["brief"], state["evidence"])
        if not check["unsupported"]:
            return {"feedback": ""}
        return {"revisions": state["revisions"] + 1,
                "feedback": "These bullets have no valid evidence citation; cite them or move "
                            "them to Open questions:\n" + "\n".join(check["unsupported"])}

    def after_critique(state: RunState) -> str:
        if state["feedback"] and state["revisions"] <= MAX_REVISIONS:
            return "write"
        return "judge"

    def judge(state: RunState) -> dict:
        """Read each claim against its evidence; send the draft back once if any fail."""
        verdicts = judge_claims(llm, state["brief"], state["evidence"])
        failed = [v for v in verdicts if v["verdict"] == "unsupported"]
        if not failed or state["judge_rewrites"] >= MAX_JUDGE_REWRITES:
            return {"verdicts": verdicts, "feedback": ""}
        return {"verdicts": verdicts, "judge_rewrites": state["judge_rewrites"] + 1,
                "revisions": state["revisions"] + 1,
                "feedback": "An independent check found these claims are NOT backed by the "
                            "evidence they cite. Rewrite each to say only what the evidence "
                            "states, or move it to Open questions:\n"
                            + "\n".join(f"- {v['claim']} (problem: {v['reason']})" for v in failed)}

    def after_judge(state: RunState) -> str:
        return "write" if state["feedback"] else END

    g = StateGraph(RunState)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("tools", run_tools)
    g.add_node("write", write)
    g.add_node("critique", critique)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_conditional_edges("research", after_research, {"tools": "tools", "write": "write"})
    g.add_edge("tools", "research")
    g.add_edge("write", "critique")
    g.add_node("judge", judge)
    g.add_conditional_edges("critique", after_critique, {"write": "write", "judge": "judge"})
    g.add_conditional_edges("judge", after_judge, {"write": "write", END: END})
    return g.compile()


def run_agent(llm, category: str, country: str) -> dict:
    started = time.time()
    state = build_graph(llm).invoke({"category": category, "country": country.upper()},
                                    {"recursion_limit": 60})
    return _finish(state, llm, "agent", started)


def run_chain(llm, category: str, country: str) -> dict:
    """v1 baseline: a single prompt with no tools and no evidence."""
    started = time.time()
    reply = llm.chat("chain", [{"role": "user", "content":
        f"Write a market brief on {category} in {country} in Markdown with sections "
        "'## Findings', '## Recommendations', '## Open questions'. Use bullets."}],
        context={"category": category, "country": country.upper()})
    state = {"category": category, "country": country.upper(), "brief": reply["content"] or "",
             "evidence": [], "tool_calls": 0, "tool_errors": 0, "revisions": 0}
    state["verdicts"] = judge_claims(llm, state["brief"], [])    # nothing cited: no model call
    return _finish(state, llm, "chain", started)


def _finish(state: dict, llm, mode: str, started: float) -> dict:
    check = check_citations(state["brief"], state["evidence"])
    state.setdefault("verdicts", [])
    state["metrics"] = {
        "mode": mode, "llm": llm.name,
        "claims": check["claims"], "unsupported_claims": len(check["unsupported"]),
        "citation_coverage": round(check["coverage"], 2),
        **tally(state["verdicts"]),
        "evidence": len(state["evidence"]),
        "distinct_tools": len({e["tool"] for e in state["evidence"]}),
        "tool_calls": state["tool_calls"], "tool_errors": state["tool_errors"],
        "revisions": state["revisions"],
        "llm_calls": llm.usage["calls"], "tokens": llm.usage["tokens"],
        "seconds": round(time.time() - started, 1),
    }
    return state
