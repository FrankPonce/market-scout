# marketscout

An AI agent that researches **any product category in any European country** and writes
a market brief in which every finding cites the evidence it rests on.

It is the code-first successor to a set of n8n workflows built for one category (area
rugs) at one retailer. Those were fixed prompt chains; this is an agent that chooses its
own tools, recovers when one fails, and is scored by an eval harness.

> Status: working skeleton. The graph, tools, critic and evals run end to end with a
> live model (Groq `openai/gpt-oss-120b`, free tier). Without a web-search key the agent
> guesses URLs and many fetches fail, so set `TAVILY_API_KEY` for useful runs.

## How it works

```
plan -> research <-> tools -> write <-> critique -> done
```

| Step | What happens |
|---|---|
| plan | Turns "oat milk, FI" into 3 to 5 research questions |
| research | The LLM decides which tool to call next, one at a time |
| tools | Runs the call; each success becomes numbered evidence (E1, E2, ...); failures are reported back so the agent can try another angle |
| write | Drafts the brief; every finding must end with a citation like `[E2]` |
| critique | Rejects the draft if any finding is uncited or cites evidence that does not exist |

Tools (all free, public, documented sources):

| Tool | Source | Tells you |
|---|---|---|
| `wikipedia_interest` | Wikimedia pageviews, in the local language | Interest trend and seasonality |
| `eurostat_online_shopping` | Eurostat `isoc_ec_ib20` | E-commerce maturity of the country |
| `open_food_catalogue` | Open Food Facts | Brands present (food and drink only) |
| `web_search` | Tavily (needs free key) | News, reports, competitor pages |
| `fetch_page` | Any public page, respecting robots.txt | Page text |

Optional, for the rug case study only (needs `EXTERN_API_URL` in `.env`): `retail_price_comparison`,
`social_trends` and `trend_articles` read a frozen May 2026 snapshot of US rug listings
and social posts from the Extern course server. Try `python -m marketscout "area rug" --country US`.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m marketscout "oat milk" --country FI --offline
```

`--offline` uses a scripted stand-in for the LLM (the data tools are still real), so it
works with no API key. For a real model, copy `.env.example` to `.env`, add a free
[Groq](https://console.groq.com) or [OpenRouter](https://openrouter.ai) key, and drop
`--offline`. Each run saves `brief.md` and `run.json` under `runs/`.

## Dashboard

```bash
python -m marketscout.dashboard runs/oat-milk-fi-agent
```

Writes a standalone `dashboard.html` into the run folder, styled as a page from a vintage
trade catalogue addressed to the seller: a drawn product on a starburst with choppy,
hand-animated motion, key figures, a competitor price list, graphs printed in halftone
ink, risks, a clip-out advice coupon, the agent's enquiries as order slips, and footnoted
sources. It also pulls one openly licensed photo from Wikimedia Commons, with its credit.

## Evals

```bash
python evals/run_evals.py
```

Runs every case in `evals/cases.json` through two modes and writes `evals/scorecard.md`:

- **chain**: one prompt, no tools. The shape of the original n8n workflows.
- **agent**: the graph above.

Metrics today: claims made, unsupported claims, citation coverage, evidence gathered,
distinct tools used, tool errors, revisions forced by the critic, tokens, seconds.

Honest limit: citation coverage only checks that a citation *exists*, not that the
evidence actually *supports* the claim, and a no-tools chain scores zero by construction.
The faithfulness check below is what makes the comparison meaningful.

## Roadmap

1. **Add web search** (Tavily key) and re-run the evals; today the agent has too little evidence.
2. **Faithfulness judge**: a second LLM pass that reads each claim next to its cited
   evidence and labels it supported / not supported. Hand-label 50 claims to measure
   how often the judge agrees with a human.
3. **More tools**: Google Trends, Eurostat trade and price data, a non-food product
   catalogue, optionally the hiring-signals dataset from the competitor-demand project.
4. **Cross-country mode**: same category across several countries, ranked.
5. **Real 3D models** for the dashboard cover, fetched from an openly licensed library.
6. **Human approval step** before the brief is finalised (LangGraph interrupt).
7. **Tracing** with Langfuse, for run screenshots in the case study.
8. **Bigger eval set** (20+ categories) and the v1 vs v2 write-up.

## Layout

```
src/marketscout/
  graph.py   the agent (LangGraph) and the chain baseline
  tools.py   data tools; add one with the @tool decorator
  llm.py     Groq / OpenRouter client, plus the offline stand-in
  cli.py     command line
evals/       cases, runner, scorecard
tests/       offline tests (no network, no key)
baseline/    the original n8n workflow exports go here
```
