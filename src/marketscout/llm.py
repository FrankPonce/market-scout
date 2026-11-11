"""LLM clients.

`ChatLLM` talks to any OpenAI-compatible endpoint (Groq, then OpenRouter as fallback),
both of which have free tiers. `OfflineLLM` is a scripted stand-in so the whole graph
runs, and is testable, with no API key: the data tools are still real, only the
"thinking" is canned.
"""
from __future__ import annotations

import json
import os
import time

import httpx

PROVIDERS = [
    # (env var with key, endpoint, env var with model, default model)
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions",
     "GROQ_MODEL", "openai/gpt-oss-120b"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions",
     "OPENROUTER_MODEL", "openrouter/free"),
]


class ChatLLM:
    name = "live"

    def __init__(self) -> None:
        self.usage = {"calls": 0, "tokens": 0}
        self.providers = [
            (os.environ[key], url, os.environ.get(model_env) or default)
            for key, url, model_env, default in PROVIDERS
            if os.environ.get(key)
        ]
        if not self.providers:
            raise RuntimeError("No LLM API key set (GROQ_API_KEY or OPENROUTER_API_KEY).")

    def chat(self, role: str, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, context: dict | None = None) -> dict:
        """Return the assistant message dict (may contain `tool_calls`).

        `role` and `context` are only used by OfflineLLM; they are ignored here.
        """
        body: dict = {"messages": messages, "temperature": 0.2}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for key, url, model in self.providers:
            try:
                for _ in range(6):      # free tiers rate-limit: wait, then retry
                    r = httpx.post(url, json={**body, "model": model},
                                   headers={"Authorization": f"Bearer {key}"}, timeout=90)
                    if r.status_code != 429:
                        break
                    time.sleep(min(float(r.headers.get("retry-after", 15)), 60) + 1)
                r.raise_for_status()
                data = r.json()
                self.usage["calls"] += 1
                self.usage["tokens"] += (data.get("usage") or {}).get("total_tokens", 0)
                return data["choices"][0]["message"]
            except Exception as e:  # try the next provider
                last_error = e
        raise RuntimeError(f"All LLM providers failed: {last_error}")


class OfflineLLM:
    """Scripted responses, one behaviour per graph role."""

    name = "offline"
    SCRIPT = ["wikipedia_interest", "eurostat_online_shopping", "open_food_catalogue"]

    def __init__(self) -> None:
        self.usage = {"calls": 0, "tokens": 0}

    def chat(self, role: str, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, context: dict | None = None) -> dict:
        self.usage["calls"] += 1
        ctx = context or {}
        category, country = ctx.get("category", ""), ctx.get("country", "")

        if role == "planner":
            return _msg(json.dumps({"questions": [
                f"Is consumer interest in {category} growing in {country}?",
                f"How mature is online shopping in {country}?",
                f"Which brands already sell {category} in {country}?",
            ]}))

        if role == "researcher":
            done = sum(1 for m in messages if m["role"] == "tool")
            if done >= len(self.SCRIPT):
                return _msg("Research complete.")
            name = self.SCRIPT[done]
            args = {"country": country}
            if name == "wikipedia_interest":
                args["topic"] = category
            elif name == "open_food_catalogue":
                args["category"] = category
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{done + 1}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }]}

        if role == "writer":
            evidence = ctx.get("evidence", [])
            lines = [f"# {category.title()} in {country}", "", "## Findings"]
            lines += [f"- {e['summary']} [{e['id']}]" for e in evidence]
            lines += ["", "## Open questions",
                      "- Offline demo mode: no interpretation was done, findings are restated evidence."]
            return _msg("\n".join(lines))

        if role == "chain":  # v1 baseline: one prompt, no tools, no evidence
            return _msg(f"# {category.title()} in {country}\n\n## Findings\n"
                        f"- Demand for {category} is growing strongly in {country}.\n"
                        f"- Competitors leave a gap in the premium segment.\n")

        raise ValueError(f"OfflineLLM has no script for role {role!r}")


def _msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


def get_llm(offline: bool = False):
    if offline:
        return OfflineLLM()
    try:
        return ChatLLM()
    except RuntimeError:
        return OfflineLLM()
