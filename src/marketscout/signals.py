"""Demand and competitor signals for any consumer product, from free public sources.

- search_trends       Google Trends: what people in a country search for, year by year
- search_suggestions  Google autocomplete: what people type after a phrase
- store_prices        public product feeds of online shops: what competitors charge
- youtube_search      what gets watched on a topic

None needs an API key. Google Trends is read through the unofficial `pytrends`
library, which can be rate-limited or break when Google changes its site.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import urllib.robotparser
from pathlib import Path

from .tools import USER_AGENT, _COUNTRY, _STR, _country, http, tool

_LIST = {"type": "array", "items": {"type": "string"}}
CACHE = Path(".marketscout_cache")     # Google rate-limits Trends hard, so answers are kept for a day
CACHE_SECONDS = 24 * 3600


def _cached(key: str, fetch):
    path = CACHE / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS:
        return json.loads(path.read_text(encoding="utf-8"))
    value = fetch()
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "SEK": "kr ", "DKK": "kr ", "NOK": "kr "}


def _as_list(value) -> list[str]:
    """Models sometimes send 'a, b' where a list was asked for."""
    if isinstance(value, str):
        value = value.split(",")
    return [str(v).strip() for v in value if str(v).strip()]


@tool("Google Trends: compare how much people in a country search for up to 5 terms, year "
      "by year over five years, plus the fastest-rising related searches. Best single "
      "measure of consumer demand. Terms must be SHORT keywords or brand names people "
      "really type (one or two words, e.g. 'y2k', 'streetwear', 'Supreme'), never a long "
      "description. To compare concepts, put a keyword for each in the same call. Set "
      "clothing_only true for fashion terms, to exclude unrelated meanings.",
      {"terms": {**_LIST, "description": "1 to 5 search terms, e.g. ['y2k', 'streetwear']"},
       "country": _COUNTRY,
       "clothing_only": {"type": "boolean", "description": "Count only searches in the Apparel category"}})
def search_trends(terms, country: str, clothing_only: bool = False) -> dict:
    from pytrends.request import TrendReq

    _country(country)
    terms = _as_list(terms)[:5]
    if not terms:
        raise ValueError("Give at least one search term")
    category = 68 if clothing_only else 0

    def fetch() -> dict:
        for attempt in range(3):
            try:
                trends = TrendReq(hl="en-US", tz=0, timeout=(5, 25))
                trends.build_payload(terms, geo=country.upper(), timeframe="today 5-y", cat=category)
                frame = trends.interest_over_time()
                break
            except Exception as e:
                if "429" not in str(e) or attempt == 2:
                    raise
                time.sleep(20 * (attempt + 1))      # rate limited: back off and retry
        if frame.empty:
            raise LookupError(f"Google Trends has too little data for {terms} in {country}")
        yearly = frame.drop(columns="isPartial").groupby(frame.index.year).mean().round(1)
        if float(yearly.to_numpy().max()) < 1:
            raise LookupError(f"Almost nobody in {country} searches for {terms}. Use shorter, more common "
                              "keywords or brand names (one or two words each).")
        rising = []
        try:
            related = trends.related_queries().get(terms[0], {}).get("rising")
            if related is not None:
                rising = [str(q) for q in related["query"][:6]]
        except Exception:
            pass        # related searches are a bonus
        return {"points": [[str(year), *[float(row[t]) for t in terms]] for year, row in yearly.iterrows()],
                "rising": rising}

    found = _cached(f"trends|{country.upper()}|{category}|{'|'.join(terms)}", fetch)
    points, rising = found["points"], found["rising"]
    return {"source": "https://trends.google.com/trends/explore?date=today%205-y&geo="
                      f"{country.upper()}&q={','.join(terms).replace(' ', '%20')}",
            "summary": summarise_trends(terms, country.upper(), points, clothing_only, rising),
            "data": {"kind": "trend_lines", "terms": terms, "points": points,
                     "clothing_only": clothing_only}}


def summarise_trends(terms: list[str], country: str, points: list[list], clothing_only: bool,
                     rising: list[str]) -> str:
    first, last = points[0], points[-1]
    parts = [f"'{t}' {first[i + 1]:g} in {first[0]} to {last[i + 1]:g} in {last[0]}" for i, t in enumerate(terms)]
    leader = terms[max(range(len(terms)), key=lambda i: last[i + 1])]
    text = (f"Google search interest in {country}{' (clothing searches only)' if clothing_only else ''}, "
            f"yearly average on a 0 to 100 scale shared by these terms: " + "; ".join(parts)
            + f". Highest in {last[0]}: '{leader}'.")
    if rising:
        text += f" Fastest-rising searches related to '{terms[0]}': " + ", ".join(rising) + "."
    return text + " The scale is relative, not a count of searches, and shows interest, not sales."


@tool("Google autocomplete: the phrases people in a country most often type after a "
      "starting phrase. Shows what shoppers want to know (price, where to buy, which brand).",
      {"query": {**_STR, "description": "Starting phrase, ideally in the local language"}, "country": _COUNTRY})
def search_suggestions(query: str, country: str) -> dict:
    lang, _ = _country(country)
    r = http.get("https://suggestqueries.google.com/complete/search",
                 params={"client": "firefox", "hl": lang, "gl": country.lower(), "q": query})
    r.raise_for_status()
    found = [s for s in json.loads(r.text)[1] if s.lower() != query.lower()][:10]
    if not found:
        raise LookupError(f"No suggestions for {query!r}; try a shorter or local-language phrase")
    return {"source": f"https://www.google.com/search?q={query.replace(' ', '+')}&gl={country.lower()}",
            "summary": f"People in {country.upper()} who start typing '{query}' are offered: " + "; ".join(found) + "."}


@tool("Competitor prices: read the public product feed of online shops (works for shops "
      "built on Shopify, which most independent brands are) and report how many matching "
      "products each sells and at what price. Give the shops' web domains; if you do not "
      "know the competitors' domains, find them with web_search first.",
      {"domains": {**_LIST, "description": "Shop domains, e.g. ['makia.com', 'billebeino.com']"},
       "product_word": {**_STR, "description": "Word that must appear in the product name or type, e.g. 'tee'"}})
def store_prices(domains, product_word: str) -> dict:
    words = {product_word.lower(), *{"tee": ["t-shirt", "tshirt", "t-paita"], "t-shirt": ["tee", "tshirt", "t-paita"],
                                     "hoodie": ["hood", "huppari"]}.get(product_word.lower(), [])}
    stores, skipped = [], []
    for domain in _as_list(domains)[:8]:
        domain = domain.replace("https://", "").replace("http://", "").strip("/")
        try:
            robots = urllib.robotparser.RobotFileParser()
            robots.parse(http.get(f"https://{domain}/robots.txt").text.splitlines())
            if not robots.can_fetch(USER_AGENT, f"https://{domain}/products.json"):
                skipped.append(f"{domain} (asks robots to stay out)")
                continue
            r = http.get(f"https://{domain}/products.json", params={"limit": 250})
            if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
                skipped.append(f"{domain} (no public feed)")
                continue
            prices = sorted(
                float(p["variants"][0]["price"]) for p in r.json().get("products", [])
                if p.get("variants") and any(w in f"{p.get('product_type', '')} {p['title']}".lower() for w in words))
            prices = [p for p in prices if p > 0]
            if len(prices) < 2:
                skipped.append(f"{domain} (fewer than two matching products)")
                continue
            currency = ""
            try:
                currency = http.get(f"https://{domain}/meta.json").json().get("currency", "")
            except Exception:
                pass
            q1, _, q3 = statistics.quantiles(prices, n=4) if len(prices) >= 4 else (prices[0], 0, prices[-1])
            stores.append({"name": domain.removeprefix("www."), "n": len(prices), "min": prices[0], "q1": round(q1, 2),
                           "median": round(statistics.median(prices), 2), "q3": round(q3, 2), "max": prices[-1],
                           "rating": None, "currency": currency})
        except Exception as e:
            skipped.append(f"{domain} ({type(e).__name__})")
        time.sleep(0.5)
    if not stores:
        raise LookupError("No shop gave a readable product feed: " + "; ".join(skipped))
    parts = [f"{s['name']}: {s['n']} products, median {s['currency']} {s['median']:.0f} "
             f"(range {s['min']:.0f} to {s['max']:.0f})" for s in stores]
    return {"source": f"https://{stores[0]['name']}/collections/all",
            "summary": (f"Prices of products matching '{product_word}' in public shop feeds. " + "; ".join(parts) + "."
                        + (" Not readable: " + "; ".join(skipped) + "." if skipped else "")
                        + " Prices are in each shop's own currency and were not converted."),
            "data": {"kind": "retailer_prices", "label": product_word, "points": stores, "products": []}}


@tool("YouTube: the most relevant videos for a search, with view counts. Shows how much "
      "attention a topic or style gets and how creators talk about it.",
      {"query": {**_STR, "description": "What to search for, e.g. 'y2k outfit ideas'"}})
def youtube_search(query: str) -> dict:
    import yt_dlp

    options = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True,
               "socket_timeout": 15, "retries": 1}
    with yt_dlp.YoutubeDL(options) as ydl:
        entries = (ydl.extract_info(f"ytsearch12:{query}", download=False) or {}).get("entries") or []
    videos = [{"title": (e.get("title") or "")[:80], "views": int(e.get("view_count") or 0),
               "channel": e.get("channel") or e.get("uploader") or ""} for e in entries if e]
    if not videos:
        raise LookupError(f"No YouTube results for {query!r}")
    videos.sort(key=lambda v: -v["views"])
    return {"source": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
            "summary": (f"Top {len(videos)} YouTube results for '{query}' have {sum(v['views'] for v in videos):,} "
                        "views in total. Most watched: "
                        + "; ".join(f"'{v['title']}' by {v['channel']} ({v['views']:,} views)" for v in videos[:4])
                        + ". View counts are worldwide and all-time, not specific to one country."),
            "data": {"kind": "videos", "points": [[v["title"], v["views"]] for v in videos[:6]]}}
