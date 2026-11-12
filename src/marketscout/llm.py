"""LLM clients.

`ChatLLM` talks to OpenAI-compatible endpoints (Groq, Mistral, OpenRouter), all of which
have free tiers, and rotates between whichever ones have a key set. `OfflineLLM` is a scripted stand-in so the whole graph
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
    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1/chat/completions",
     "MISTRAL_MODEL", "mistral-large-latest"),
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
            raise RuntimeError("No LLM API key set (GROQ_API_KEY, MISTRAL_API_KEY or OPENROUTER_API_KEY).")

    def chat(self, role: str, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, context: dict | None = None) -> dict:
        """Return the assistant message dict (may contain `tool_calls`).

        Providers are tried in order; one that is rate-limited is skipped at once, so
        several free tiers share the load. The judge starts from the *second* provider
        when there is one, so the brief is audited by a different model from its writer.
        `context` is only used by OfflineLLM.
        """
        body: dict = {"messages": messages, "temperature": 0.2}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        order = self.providers[1:] + self.providers[:1] if role == "judge" else self.providers
        last_error: Exception | None = None
        for _ in range(8):              # rounds: only wait once every provider is busy
            wait = None
            for key, url, model in order:
                try:
                    r = httpx.post(url, json={**body, "model": model},
                                   headers={"Authorization": f"Bearer {key}"}, timeout=90)
                    if r.status_code == 429:
                        retry = float(r.headers.get("retry-after", 20))
                        if retry <= 120:        # a per-minute cap; a daily cap is not worth waiting for
                            wait = retry if wait is None else min(wait, retry)
                        try:
                            reason = r.json()["error"]["message"].split(". Please")[0]
                        except Exception:
                            reason = "rate limited"
                        last_error = RuntimeError(f"{url.split('/')[2]}: {reason}"[:240])
                        continue
                    r.raise_for_status()
                    data = r.json()
                    self.usage["calls"] += 1
                    self.usage["tokens"] += (data.get("usage") or {}).get("total_tokens", 0)
                    message = data["choices"][0]["message"]
                    if isinstance(message.get("content"), list):    # some providers send text in parts
                        message["content"] = "".join(
                            part.get("text", "") for part in message["content"]
                            if isinstance(part, dict) and part.get("type", "text") == "text")
                    return message
                except Exception as e:  # a broken provider: try the next one
                    last_error = e
            if wait is None:            # every provider failed outright or is out for the day
                break
            time.sleep(wait + 1)
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

        if role == "judge":   # the offline stand-in cannot read, so it rules on nothing
            return _msg(json.dumps({"verdicts": []}))

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
