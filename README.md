<div align="center">

<img src="assets/banner.svg" alt="Market Scout: an agentic AI workflow for business, printed as a merchant's catalogue" width="100%">

<br>

![Type](https://img.shields.io/badge/Agentic%20AI-workflow%20for%20business-d9432b?style=for-the-badge&labelColor=1f1b2d)
![Built with](https://img.shields.io/badge/built%20with-LangGraph-1e7a74?style=for-the-badge&labelColor=1f1b2d)
![Python](https://img.shields.io/badge/python-3.10%2B-f0b429?style=for-the-badge&logo=python&logoColor=white&labelColor=1f1b2d)
![Tests](https://github.com/FrankPonce/market-scout/actions/workflows/tests.yml/badge.svg)
![MCP](https://img.shields.io/badge/MCP-server%20included-1e7a74?style=for-the-badge&labelColor=1f1b2d)
![Origin](https://img.shields.io/badge/evolved%20from-Wayfair%20externship%20%C2%B7%20n8n-8a5a00?style=for-the-badge&labelColor=1f1b2d)

[**The workflow**](#-the-agentic-workflow) · [**Origin**](#-where-this-came-from) · [**The catalogue**](#-the-catalogue) · [**Evals**](#-evals) · [**Quickstart**](#-quickstart) · [**Repo map**](#-repository-map) · [**Limits**](#-limits)

</div>

---

## 🎯 Overview

A category manager asks a simple question: *should we sell this product in this country,
and at what price?* Answering it normally takes an analyst days of gathering prices,
competitors and trends.

Market Scout is an **agentic AI workflow** that does that research on its own. Given a
product category and a country, the agent plans what to find out, chooses which data
sources to query, recovers when one fails, writes a brief in which every finding is
footnoted, has a second model audit each claim against its source, and prints the
result as a page from a vintage trade catalogue.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#0d1117','primaryColor':'#161b22','primaryTextColor':'#e6edf3','primaryBorderColor':'#30363d','secondaryColor':'#161b22','tertiaryColor':'#0d1117','lineColor':'#f0b429','textColor':'#e6edf3','titleColor':'#e6edf3','clusterBkg':'#0d1117','clusterBorder':'#30363d','edgeLabelBackground':'#0d1117','fontFamily':'Segoe UI, Inter, sans-serif'}}}%%
flowchart TB
    subgraph R ["🤖 Research loop: the agent decides each step"]
        direction LR
        Q([🛒 Category<br/>+ country]) --> A[🧭 Plan<br/>research questions]
        A --> B[🔎 Research<br/>agent picks a tool]
        B <-->|calls, adapts<br/>on failure| C[🧰 Tools<br/>prices · brands<br/>trends · web]
    end
    subgraph W ["✅ Write and check: nothing printed unchecked"]
        direction LR
        D[✍️ Write<br/>footnoted brief] --> E{🧐 Critique<br/>every claim<br/>cited?}
        E -- no --> D
        E -- yes --> F{⚖️ Judge<br/>claim matches<br/>its source?}
        F -- "no, rewrite once" --> D
        F -- yes --> G[📰 Print<br/>merchant's<br/>catalogue]
    end
    R -- research complete --> W
```

| | |
|---|---|
| **Problem** | Category and market research is slow, manual, and rarely shows where its numbers came from. |
| **Goal** | An agent that produces a seller's market brief in about a minute, with every claim traceable and checked. |
| **Who it's for** | Category managers, brand owners, e-commerce sellers, anyone deciding what to stock and how to price it. |
| **What makes it agentic** | It decides its own next step. Nothing in the research path is wired in advance. |
| **Origin** | Rebuilt from the n8n workflows I made during an externship with Wayfair. |

```bash
python -m marketscout "oat milk" --country FI
python -m marketscout.dashboard runs/oat-milk-fi-agent
```

## 🤖 The agentic workflow

"Agentic" is an overused word, so here is exactly what the agent decides for itself and
what is fixed.

| The agent decides | Fixed by design |
|---|---|
| Which questions the brief should answer | The stages: plan, research, write, critique, judge |
| Which tool to call next, and with what arguments | The rule that every claim must cite evidence |
| What to try when a source fails or comes back empty | How many retries and rewrites are allowed |
| When it has enough evidence to stop | The format of the brief |
| How to rewrite a claim the judge rejected | Who the reader is: the seller |

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph) state machine.

| Step | What happens |
|---|---|
| 🧭 **Plan** | Turns "oat milk, FI" into three to five research questions |
| 🔎 **Research** | The model picks which tool to call next: structured data first, then web search |
| 🧰 **Tools** | Runs the call. A success becomes numbered evidence (E1, E2, ...). A failure is sent back so the agent can adapt |
| ✍️ **Write** | Drafts a brief: summary, findings, risks, recommendations, open questions |
| 🧐 **Critique** | Rejects the draft if any claim lacks a citation to evidence that exists |
| ⚖️ **Judge** | Reads each claim beside the evidence it cites and rules it supported, partly supported or unsupported. Unsupported claims send the draft back once |
| 📰 **Print** | Lays the brief out as a catalogue page, stamping any claim the judge did not fully back |

When more than one model provider has a key set, calls rotate between them as each hits
its rate limit, and the judge runs on a different provider from the writer, so the
writer does not mark its own homework.

### 🧰 Tools

| Tool | Source | Tells the seller |
|---|---|---|
| `wikipedia_interest` | Wikimedia pageviews, in the local language | Whether public interest is rising, and when it peaks |
| `eurostat_online_shopping` | Eurostat | How used the country is to buying online |
| `open_food_catalogue` | Open Food Facts | Which brands are already on sale (food and drink) |
| `search_trends` | Google Trends | What people search for, year by year, and which related searches are rising |
| `search_suggestions` | Google autocomplete | What shoppers type next: price, where to buy, which brand |
| `store_prices` | Public product feeds of online shops (Shopify) | How many products competitors sell and at what price |
| `youtube_search` | YouTube, via yt-dlp | How much attention a topic gets, and how creators talk about it |
| `web_search`, `fetch_page` | Tavily, then the page itself (robots.txt respected) | Market reports, news, competitor pages |
| `retail_price_comparison`, `social_trends`, `trend_articles` | Extern course data (optional) | Prices, best sellers and trends for rugs at US retailers |

All of them are free and none needs a login. Adding a tool is one decorated function;
the agent discovers it from its description. A source that returns no readable text,
such as a paywalled page, is treated as a failed lookup, not as evidence.

### ⚔️ Compare mode

A seller often chooses between two concepts, not one. `--versus` makes the agent gather
the same evidence for both and say which the evidence favours:

```bash
python -m marketscout "y2k scene graphic tees" --versus "dark luxury streetwear tees" --country FI
```

### 🔌 MCP server

Market Scout is also a [Model Context Protocol](https://modelcontextprotocol.io) server,
so any MCP client (Claude Desktop, Claude Code, an IDE) can use its tools. The client's
own model can call `search_trends` or `store_prices` directly, or hand a whole question
to `research_market` and get back a cited, audited brief.

```bash
pip install -e ".[mcp]"
marketscout-mcp
```

```json
{"mcpServers": {"market-scout": {"command": "marketscout-mcp"}}}
```

## 🧵 Where this came from

In August 2026 I completed an externship with **Wayfair** through Extern, building AI
workflows for category intelligence. The result was four **n8n** workflows that tracked
area-rug trends and competitors across Amazon, Walmart and Wayfair and produced a
content strategy and a dashboard.

Those workflows worked, but looking back at them honestly:

- They were **fixed prompt chains**. Each "agent" node was one LLM call in a set order,
  with no tools and no decisions about what to do next.
- They covered **one category** (rugs) for one retailer, on data served by the course.
- Nothing checked whether the output was **true**. The reports read well either way.
- A full run took around 15 minutes, most of it fixed waits between steps to stay
  under rate limits.

Market Scout is the same business idea rebuilt to fix those four things.

| | n8n workflows (v1) | Market Scout (v2) |
|---|---|---|
| Control flow | Fixed chain of prompts | Agent chooses its next tool at run time |
| Scope | Area rugs | Any category, 13 countries, or two concepts head to head |
| Data | Course data server | Search trends, shop price feeds, open statistics, web search |
| When a source fails | Run breaks or continues blind | Failure is reported back and the agent tries another angle |
| Evidence | None attached | Every claim cites a numbered source |
| Checking | None | Citation check, then a judge model reads each claim against its source |
| Measured | No | Eval scorecard comparing chain and agent |
| Typical run | About 15 minutes | Around one to two minutes |

The v1 design survives as `--mode chain`, a single prompt with no tools, so both can be
scored by the same harness. The optional rug tools exist to re-run the original Wayfair
scenario through the new agent; that data is a frozen snapshot from May 2026 and is not
redistributed here.

## 📰 The catalogue

`python -m marketscout.dashboard <run folder>` turns a saved run into one standalone
HTML page, addressed to the seller and styled as a mail-order trade catalogue:

- a drawn product on a starburst, animated at a deliberately choppy, hand-drawn rate
- "for the Finnish table", "for the American home": a line that follows the country
- key figures computed from the evidence data, not written by the model
- a competitor price list, and graphs printed in halftone ink
- risks as a caution notice, advice as a clip-out coupon
- the agent's lookups as numbered order slips, stamped received or returned empty
- claims the judge did not fully back are stamped where they appear
- one openly licensed photo from Wikimedia Commons, with its credit

Why a catalogue: the subject is retail, and a trade catalogue is the form in which
sellers have always been told what to stock and what it costs.

## 📊 Evals

```bash
python evals/run_evals.py
```

Runs each case in `evals/cases.json` through both modes and writes
[`evals/scorecard.md`](evals/scorecard.md).

| Metric | Meaning |
|---|---|
| claims | Bullets in findings, risks and recommendations |
| citation_coverage | Share of claims that cite evidence that exists |
| supported / partial / not_supported | The judge's rulings |
| faithfulness | Supported claims as a share of claims judged |
| tool_errors, revisions, tokens, seconds | What the run cost |

Results so far, on Groq's free `openai/gpt-oss-120b`:

| Run | Mode | Claims | Cited | Fully backed | Partly backed | Not backed |
|---|---|---|---|---|---|---|
| Oat milk, Finland | chain | 34 | 0 | not judged | | |
| Oat milk, Finland | agent | 8 | 8 | not judged | | |
| Olive oil, Spain | chain | 33 | 0 | not judged | | |
| Olive oil, Spain | agent | 3 | 3 | not judged | | |
| Area rug, Germany | chain | 40 | 0 | not judged | | |
| Area rug, Germany | agent | 4 | 4 | not judged | | |
| Area rug, United States | agent | 13 | 13 | 7 | 6 | 0 |

The first six rows are the last full scorecard, which predates the judge. The last row
is the first judged run: every claim was cited, but the judge ruled only 7 of 13 fully
backed. The other 6 were inferences the source did not state, such as a recommended
price tier drawn from a purchase-intent figure. That gap between "cited" and "backed" is
exactly what the citation check alone could not see. A full judged scorecard is the next
thing to run.

What these numbers do not show: the judge is itself a model and has not yet been
checked against human labels, and three cases is a demonstration, not a benchmark.

## ⚡ Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m marketscout "oat milk" --country FI --offline
pytest
```

`--offline` uses a scripted stand-in for the model (the data tools are still real), so
it works with no API key. For real runs, copy `.env.example` to `.env` and add a free
key from [Groq](https://console.groq.com), [Mistral](https://console.mistral.ai) or
[OpenRouter](https://openrouter.ai), plus [Tavily](https://tavily.com) for web search.
Each run saves `brief.md` and `run.json` under `runs/`.

## 📁 Repository map

```
src/marketscout/
  graph.py       the agent, and the chain baseline
  judge.py       the claim-against-evidence check
  tools.py       open data tools; add one with the @tool decorator
  signals.py     search trends, autocomplete, shop price feeds, YouTube
  extern.py      optional rug case-study tools
  mcp_server.py  the same tools and the agent, served over MCP
  llm.py         model client (Groq, Mistral, OpenRouter) and the offline stand-in
  dashboard.py   builds the catalogue page from a saved run
  cli.py         command line
evals/           cases, runner, scorecard
tests/           offline tests, no key needed; run on every push by GitHub Actions
design/          the style prototypes that led to the catalogue
baseline/        where the original n8n workflow exports go
```

## 🚧 Limits

- Shop price feeds only exist for shops built on Shopify that allow robots to read
  them, and prices are not converted between currencies.
- Google Trends is read through an unofficial library and can be rate-limited. Its
  scale is relative interest, not a count of searches.
- No TikTok, Instagram or Pinterest data: none can be read without a login, and this
  project only uses sources it is allowed to use.
- Wikipedia pageviews measure curiosity, not sales, and the page says so.
- The judge catches claims that outrun their evidence. It does not check that the
  evidence itself is right.
- Free model tiers cap tokens per minute, so long sources are trimmed in prompts.
- The catalogue has five product drawings (carton, bottle, rug, t-shirt, box). Everything else
  gets the box.

## 🗺️ Roadmap

1. Hand-label a sample of claims to measure how often the judge agrees with a person.
2. Live mode: watch the catalogue print section by section as the agent works.
3. Reddit through its official API, and a licensed marketplace price source such as
   the eBay Browse API.
4. Cross-country mode: one category ranked across several countries.
5. A larger eval set and a written comparison of chain and agent.

## 🙏 Credits

Built on [LangGraph](https://github.com/langchain-ai/langgraph). Data from Wikimedia,
Eurostat and Open Food Facts under their open licences. Photos from Wikimedia Commons,
credited on each page. The original workflows were built during an externship with
Wayfair, run by Extern.
