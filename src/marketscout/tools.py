"""Tools the agent can call. All use free, public, documented data sources.

Every tool returns {"source": <url a human can open>, "summary": <one or two sentences>}.
The summary is what becomes a piece of evidence the brief has to cite.
To add a tool: write the function and decorate it with @tool(description, parameters).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import time
import urllib.robotparser
from urllib.parse import quote, urlparse

import httpx

# Wikimedia rejects clients that give no contact URL. Set MARKETSCOUT_CONTACT to your
# repo URL in .env so data providers can reach you.
USER_AGENT = f"marketscout/0.1 ({os.environ.get('MARKETSCOUT_CONTACT') or 'https://example.org/marketscout'})"
http = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=25, follow_redirects=True)

# ISO country code -> (Wikipedia language, English country name)
COUNTRIES = {
    "FI": ("fi", "finland"), "SE": ("sv", "sweden"), "DK": ("da", "denmark"),
    "DE": ("de", "germany"), "FR": ("fr", "france"), "ES": ("es", "spain"),
    "IT": ("it", "italy"), "NL": ("nl", "netherlands"), "PL": ("pl", "poland"),
    "PT": ("pt", "portugal"), "AT": ("de", "austria"), "IE": ("en", "ireland"),
    "US": ("en", "united states"),      # for the rug case study; Eurostat does not cover it
}

REGISTRY: dict[str, dict] = {}


def tool(description: str, parameters: dict, requires_env: str | None = None):
    def wrap(fn):
        REGISTRY[fn.__name__] = {
            "fn": fn, "requires_env": requires_env,
            "schema": {"type": "function", "function": {
                "name": fn.__name__, "description": description,
                "parameters": {"type": "object", "properties": parameters,
                               "required": list(parameters)},
            }},
        }
        return fn
    return wrap


def available_schemas() -> list[dict]:
    return [t["schema"] for t in REGISTRY.values()
            if not t["requires_env"] or os.environ.get(t["requires_env"])]


def call_tool(name: str, args: dict) -> dict:
    if name not in REGISTRY:
        raise ValueError(f"Unknown tool {name!r}")
    return REGISTRY[name]["fn"](**args)


def _country(code: str) -> tuple[str, str]:
    code = code.upper()
    if code not in COUNTRIES:
        raise ValueError(f"Unsupported country {code!r}; supported: {', '.join(COUNTRIES)}")
    return COUNTRIES[code]


_STR = {"type": "string"}
_COUNTRY = {"type": "string", "description": "ISO country code, e.g. FI, DE, ES"}


@tool("Consumer-interest proxy: monthly Wikipedia pageviews for a topic in the country's "
      "own language, last 24 months. Shows trend and seasonality, not sales.",
      {"topic": {**_STR, "description": "A common noun for the product, in English or the "
                                        "local language, e.g. 'oat milk', 'carpet', 'Teppich'"},
       "country": _COUNTRY})
def wikipedia_interest(topic: str, country: str) -> dict:
    lang, _ = _country(country)
    title = _wiki_title(lang, topic)            # topic given in the local language?
    if not title:
        api = "https://en.wikipedia.org/w/api.php"
        en_title = _wiki_title("en", topic)     # exact English title or redirect
        if not en_title:
            hits = http.get(api, params={"action": "query", "list": "search", "srsearch": topic,
                                         "srlimit": 1, "format": "json"}).json()["query"]["search"]
            if not hits:
                raise LookupError(f"No Wikipedia article found for {topic!r}")
            en_title = hits[0]["title"]
        pages = http.get(api, params={"action": "query", "prop": "langlinks", "titles": en_title,
                                      "lllang": lang, "format": "json"}).json()["query"]["pages"]
        links = next(iter(pages.values())).get("langlinks")
        if not links:
            raise LookupError(f"'{en_title}' has no {lang}.wikipedia article; try another term")
        title = links[0]["*"]

    end = dt.date.today().replace(day=1) - dt.timedelta(days=1)   # last full month
    start = dt.date(end.year - 2, end.month, 1) + dt.timedelta(days=32)
    start = start.replace(day=1)
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"{lang}.wikipedia/all-access/user/{quote(title.replace(' ', '_'), safe='')}/monthly/"
           f"{start:%Y%m%d}00/{end:%Y%m%d}00")
    r = http.get(url)
    r.raise_for_status()
    return {"source": f"https://pageviews.wmcloud.org/?project={lang}.wikipedia.org&pages={quote(title)}",
            "summary": summarise_pageviews(title, lang, r.json()["items"]),
            "data": {"kind": "monthly_views", "title": title, "lang": lang,
                     "points": [[i["timestamp"][:6], i["views"]] for i in r.json()["items"]]}}


def _wiki_title(lang: str, topic: str) -> str | None:
    """Exact article title on a wiki, following redirects; None if there is no such page."""
    pages = http.get(f"https://{lang}.wikipedia.org/w/api.php",
                     params={"action": "query", "titles": topic, "redirects": 1,
                             "format": "json"}).json()["query"]["pages"]
    page = next(iter(pages.values()))
    return None if "missing" in page or "invalid" in page else page["title"]


def summarise_pageviews(title: str, lang: str, items: list[dict]) -> str:
    views = [i["views"] for i in items]
    recent, earlier = views[-12:], views[-24:-12]
    peak = max(items[-12:], key=lambda i: i["views"])
    peak_month = dt.datetime.strptime(peak["timestamp"][:6], "%Y%m").strftime("%B")
    text = (f"The {lang}.wikipedia article '{title}' averaged {sum(recent) / len(recent):,.0f} "
            f"views per month over the last {len(recent)} months, peaking in {peak_month}")
    if len(earlier) == 12 and sum(earlier):
        change = (sum(recent) - sum(earlier)) / sum(earlier) * 100
        text += f", {change:+.0f}% versus the 12 months before"
    return text + "."


@tool("European countries only. Eurostat: share of people in a country who bought something online in the last "
      "12 months, by year. Measures e-commerce maturity of the market.",
      {"country": _COUNTRY})
def eurostat_online_shopping(country: str) -> dict:
    _country(country)
    r = http.get("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/isoc_ec_ib20",
                 params={"format": "JSON", "lang": "EN", "geo": country.upper(),
                         "ind_type": "IND_TOTAL", "indic_is": "I_BLT12", "unit": "PC_IND"})
    r.raise_for_status()
    return {"source": "https://ec.europa.eu/eurostat/databrowser/view/isoc_ec_ib20/default/table",
            "summary": summarise_eurostat(country.upper(), r.json()),
            "data": {"kind": "yearly_percent", "points": eurostat_series(r.json())}}


def eurostat_series(data: dict) -> list[list]:
    years = data["dimension"]["time"]["category"]["index"]      # {"2020": 0, ...}
    return [[y, data["value"][str(i)]] for y, i in sorted(years.items()) if str(i) in data["value"]]


def summarise_eurostat(country: str, data: dict) -> str:
    series = dict(eurostat_series(data))
    if not series:
        raise LookupError(f"Eurostat has no online-shopping data for {country}")
    first, last = min(series), max(series)
    return (f"In {country}, {series[last]:.0f}% of people bought something online in the "
            f"12 months before the {last} survey, compared with {series[first]:.0f}% in {first} (Eurostat).")


def _off_search(tag: str, country_name: str) -> dict:
    for attempt in range(3):        # the public server is often briefly overloaded
        r = http.get("https://world.openfoodfacts.org/api/v2/search",
                     params={"categories_tags_en": tag, "countries_tags_en": country_name.replace(" ", "-"),
                             "fields": "brands", "page_size": 100})
        if r.status_code < 500:
            break
        time.sleep(4 * (attempt + 1))
    r.raise_for_status()
    return r.json()


@tool("Open Food Facts: brands with products listed in a FOOD or DRINK category in a "
      "country. Crowd-sourced, so it shows presence, not market share.",
      {"category": {**_STR, "description": "Food/drink category in English plural, e.g. 'oat milks'"},
       "country": _COUNTRY})
def open_food_catalogue(category: str, country: str) -> dict:
    _, name = _country(country)
    tag = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")
    data = _off_search(tag, name)
    if not data.get("products") and not tag.endswith("s"):   # tags are plural: oat-milks
        tag += "s"
        data = _off_search(tag, name)
    counts: dict[str, int] = {}
    for p in data.get("products", []):
        brand = (p.get("brands") or "").split(",")[0].strip()
        if brand:
            counts[brand] = counts.get(brand, 0) + 1
    if not counts:
        raise LookupError(f"No Open Food Facts products for '{tag}' in {name}")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    return {"source": f"https://world.openfoodfacts.org/category/{tag}/country/{name.replace(' ', '-')}",
            "summary": (f"Open Food Facts lists {data.get('count', len(counts))} '{tag}' products "
                        f"sold in {name.title()}; brands with the most listings: "
                        + ", ".join(f"{b} ({n})" for b, n in top) + "."),
            "data": {"kind": "brand_counts", "label": tag.replace("-", " "), "total": data.get("count", len(counts)),
                     "points": [[b, n] for b, n in top]}}


@tool("Web search for recent news, reports and competitor pages. Returns top results.",
      {"query": _STR}, requires_env="TAVILY_API_KEY")
def web_search(query: str) -> dict:
    r = http.post("https://api.tavily.com/search",
                  json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": 5})
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise LookupError(f"No search results for {query!r}")
    return {"source": results[0]["url"],
            "summary": " | ".join(f"{x['title']}: {x['content'][:200]} ({x['url']})" for x in results)}


# Only offered alongside web_search: with no search results to follow, models invent URLs.
@tool("Fetch a public web page found via web_search and return its text. Respects robots.txt.",
      {"url": _STR}, requires_env="TAVILY_API_KEY")
def fetch_page(url: str) -> dict:
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("Only http(s) URLs can be fetched")
    robots = urllib.robotparser.RobotFileParser()
    try:
        robots.parse(http.get(f"{parts.scheme}://{parts.netloc}/robots.txt").text.splitlines())
    except httpx.HTTPError:
        robots.parse([])
    if not robots.can_fetch(USER_AGENT, url):
        raise PermissionError(f"robots.txt disallows fetching {url}")
    r = http.get(url)
    r.raise_for_status()
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", r.text)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    return {"source": url, "summary": text[:2500]}
