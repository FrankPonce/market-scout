"""Market Scout as a Model Context Protocol (MCP) server.

Exposes every data tool, plus the whole research agent, to any MCP client
(Claude Desktop, Claude Code, an IDE). The client's own model can then call
`search_trends` or `store_prices` directly, or hand a full question to
`research_market` and get back a cited, audited brief.

    pip install -e ".[mcp]"
    marketscout-mcp                      # speaks MCP over stdio

Example client entry:

    {"mcpServers": {"market-scout": {"command": "marketscout-mcp"}}}
"""
from __future__ import annotations

import asyncio
import json

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from . import graph  # noqa: F401  (importing it registers every tool)
from . import tools as toolbox
from .llm import get_llm

RESEARCH = types.Tool(
    name="research_market",
    description=("Run the full Market Scout agent: it plans the research, calls the data tools, "
                 "writes a brief for a seller in which every claim cites its evidence, and has a "
                 "second model audit each claim. Takes one to two minutes. Give `versus` to "
                 "compare two concepts head to head."),
    input_schema={"type": "object", "required": ["category", "country"], "properties": {
        "category": {"type": "string", "description": "Product category or concept, e.g. 'oat milk'"},
        "country": {"type": "string", "description": "ISO country code, e.g. FI"},
        "versus": {"type": "string", "description": "Optional second concept to compare against"}}},
)

def _research(arguments: dict) -> str:
    state = graph.run_agent(get_llm(), arguments["category"], arguments["country"],
                            versus=arguments.get("versus"))
    sources = "\n".join(f"[{e['id']}] {e['source']}" for e in state["evidence"])
    audit = "\n".join(f"- {v['verdict']}: {v['claim'][:120]}" for v in state.get("verdicts", []))
    return f"{state['brief']}\n\n## Sources\n{sources}\n\n## Audit of each claim\n{audit}"


async def list_tools(ctx, params) -> types.ListToolsResult:
    data_tools = [types.Tool(name=f["name"], description=f["description"], input_schema=f["parameters"])
                  for f in (schema["function"] for schema in toolbox.available_schemas())]
    return types.ListToolsResult(tools=[RESEARCH, *data_tools])


async def call_tool(ctx, params) -> types.CallToolResult:
    arguments = params.arguments or {}
    try:    # the tools use blocking network calls, so run them off the event loop
        if params.name == "research_market":
            text = await asyncio.to_thread(_research, arguments)
        else:
            result = await asyncio.to_thread(toolbox.call_tool, params.name, arguments)
            text = json.dumps({k: result[k] for k in ("summary", "source", "data") if k in result},
                              ensure_ascii=False)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])
    except Exception as e:      # report the failure to the client's model so it can adapt
        return types.CallToolResult(is_error=True, content=[
            types.TextContent(type="text", text=f"{type(e).__name__}: {e}")])


server = Server("market-scout", version="0.1.0", on_list_tools=list_tools, on_call_tool=call_tool)


async def serve() -> None:
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
