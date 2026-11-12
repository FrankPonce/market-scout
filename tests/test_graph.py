"""Offline tests: scripted LLM and faked tools, so no network or API key is needed."""
import pytest

from marketscout import tools as toolbox
from marketscout.graph import check_citations, run_agent, run_chain
from marketscout.judge import judge_claims, tally
from marketscout.llm import OfflineLLM
from marketscout.tools import summarise_eurostat, summarise_pageviews


@pytest.fixture
def fake_tools(monkeypatch):
    def fake(name, args):
        if name == "open_food_catalogue":
            raise LookupError("service down")
        return {"source": f"https://example.org/{name}", "summary": f"{name} result."}
    monkeypatch.setattr(toolbox, "call_tool", fake)


def test_agent_cites_every_claim_and_survives_a_tool_failure(fake_tools):
    state = run_agent(OfflineLLM(), "oat milk", "fi")
    m = state["metrics"]
    assert m["evidence"] == 2 and m["tool_calls"] == 3 and m["tool_errors"] == 1
    assert m["claims"] == 2 and m["citation_coverage"] == 1.0
    assert "[E1]" in state["brief"] and state["country"] == "FI"


def test_chain_baseline_has_no_evidence():
    m = run_chain(OfflineLLM(), "oat milk", "FI")["metrics"]
    assert m["claims"] == 2 and m["citation_coverage"] == 0.0 and m["evidence"] == 0


def test_check_citations_flags_missing_and_unknown_ids():
    brief = "## Findings\n- cited [E1]\n- uncited\n- bad id [E9]\n## Open questions\n- free text"
    check = check_citations(brief, [{"id": "E1"}])
    assert check["claims"] == 3 and len(check["unsupported"]) == 2


def test_critic_sends_uncited_draft_back(fake_tools):
    class SloppyThenFixed(OfflineLLM):
        drafts = 0
        def chat(self, role, messages, **kw):
            if role == "writer":
                self.drafts += 1
                if self.drafts == 1:
                    return {"role": "assistant", "content": "## Findings\n- no citation here"}
            return super().chat(role, messages, **kw)
    llm = SloppyThenFixed()
    m = run_agent(llm, "oat milk", "FI")["metrics"]
    assert llm.drafts == 2 and m["revisions"] == 1 and m["citation_coverage"] == 1.0


def test_summaries():
    data = {"value": {"0": 75.8, "1": 82.0},
            "dimension": {"time": {"category": {"index": {"2020": 0, "2025": 1}}}}}
    assert "82% of people" in summarise_eurostat("FI", data) and "76% in 2020" in summarise_eurostat("FI", data)
    items = [{"timestamp": f"2025{m:02d}0100", "views": 100} for m in range(1, 13)]
    items += [{"timestamp": f"2026{m:02d}0100", "views": 150 if m != 7 else 400} for m in range(1, 13)]
    text = summarise_pageviews("Kaurajuoma", "fi", items)
    assert "peaking in July" in text and "+71%" in text


def test_full_width_citation_brackets_count():
    brief = "## Findings\n- cited 【E1】【E2】"
    check = check_citations(brief, [{"id": "E1"}, {"id": "E2"}])
    assert check["claims"] == 1 and not check["unsupported"]


class JudgingLLM(OfflineLLM):
    """Rules the first claim unsupported on the first audit, then passes everything."""
    audits = 0
    def chat(self, role, messages, **kw):
        if role == "judge":
            self.audits += 1
            n = kw["context"]["claims"]
            rulings = [{"n": i, "verdict": "supported", "reason": "stated"} for i in range(1, n + 1)]
            if self.audits == 1:
                rulings[0] = {"n": 1, "verdict": "unsupported", "reason": "number not in evidence"}
            import json
            return {"role": "assistant", "content": json.dumps({"verdicts": rulings})}
        return super().chat(role, messages, **kw)


def test_judge_sends_an_unbacked_claim_back_once(fake_tools):
    llm = JudgingLLM()
    state = run_agent(llm, "oat milk", "FI")
    m = state["metrics"]
    assert llm.audits == 2 and m["revisions"] == 1
    assert m["supported"] == m["claims"] == 2 and m["not_supported"] == 0 and m["faithfulness"] == 1.0


def test_uncited_claims_are_unsupported_without_a_model_call():
    llm = OfflineLLM()
    verdicts = judge_claims(llm, "## Findings\n- no citation\n- bad id [E9]", [{"id": "E1", "summary": "x"}])
    assert [v["verdict"] for v in verdicts] == ["unsupported", "unsupported"] and llm.usage["calls"] == 0
    assert run_chain(OfflineLLM(), "oat milk", "FI")["metrics"]["not_supported"] == 2


def test_judge_ignores_malformed_rulings():
    class Garbled(OfflineLLM):
        def chat(self, role, messages, **kw):
            return {"role": "assistant", "content": '{"verdicts": [{"n": 9, "verdict": "supported"}, {"n": 1, "verdict": "maybe"}]}'}
    verdicts = judge_claims(Garbled(), "## Findings\n- a claim [E1]", [{"id": "E1", "summary": "x"}])
    assert verdicts[0]["verdict"] == "unchecked" and tally(verdicts)["faithfulness"] is None
