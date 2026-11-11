"""Optional tools backed by the Extern course data server (rugs only, US retailers).

These exist for the retail case study: re-running the original Wayfair externship
scenario through the agent. They are only offered when EXTERN_API_URL is set in .env.
The server holds a frozen snapshot (scraped May 2026) of four rug categories, so this
is not a general data source and its address is kept out of the repo.
"""
from __future__ import annotations

import os
import re
import statistics

from .tools import http, tool

ENV = "EXTERN_API_URL"
RETAILERS = ("wayfair", "amazon", "walmart")
_CATEGORY = {"type": "string", "enum": ["area_rug", "outdoor_rug", "hallway_runner", "shag_rug"],
             "description": "Rug category key"}
_SNAPSHOT = "Extern course dataset, scraped May 2026"


def _get(path: str, **params) -> dict:
    r = http.get(os.environ[ENV].rstrip("/") + path, params=params, timeout=40)
    r.raise_for_status()
    return r.json()


def _number(text) -> float | None:
    match = re.search(r"\d[\d,]*\.?\d*", str(text or ""))
    return float(match.group().replace(",", "")) if match else None


@tool("Rugs only, US retailers. Compare Wayfair, Amazon and Walmart listings for a rug "
      "category: price range, median price, ratings and the most-reviewed products.",
      {"category": _CATEGORY}, requires_env=ENV)
def retail_price_comparison(category: str) -> dict:
    retailers, products = [], []
    for name in RETAILERS:
        rows = []
        for p in _get(f"/api/products/{name}", category=category, limit=50).get("products", []):
            price = _number(p.get("price"))
            if price:
                rows.append({"retailer": name.title(), "name": p["name"][:90], "price": price,
                             "rating": _number(p.get("rating")), "reviews": int(_number(p.get("review_count")) or 0),
                             "url": p.get("url", "")})
        if len(rows) < 2:
            continue
        prices = sorted(r["price"] for r in rows)
        q1, _, q3 = statistics.quantiles(prices, n=4)
        rated = [r["rating"] for r in rows if r["rating"]]
        retailers.append({"name": name.title(), "n": len(rows), "min": prices[0], "q1": round(q1, 2),
                          "median": round(statistics.median(prices), 2), "q3": round(q3, 2), "max": prices[-1],
                          "rating": round(statistics.mean(rated), 2) if rated else None})
        products += sorted(rows, key=lambda r: -r["reviews"])[:2]
    if not retailers:
        raise LookupError(f"No priced products for {category}")
    parts = [f"{r['name']}: {r['n']} listings, median ${r['median']:.0f} (range ${r['min']:.0f} to "
             f"${r['max']:.0f}), average rating {r['rating']}" for r in retailers]
    top = max(products, key=lambda r: r["reviews"])
    return {"source": top["url"],
            "summary": (f"{category.replace('_', ' ').title()} listings by US retailer. " + "; ".join(parts)
                        + f". Most-reviewed product: '{top['name']}' at {top['retailer']} "
                        f"(${top['price']:.0f}, {top['reviews']:,} reviews). Prices mix rug sizes, "
                        f"so medians are indicative only. Source: {_SNAPSHOT}."),
            "data": {"kind": "retailer_prices", "label": category.replace("_", " "),
                     "points": retailers, "products": products}}


@tool("Rugs only. Instagram and Pinterest posts for a rug category: which hashtags and "
      "styles draw the most engagement.", {"category": _CATEGORY}, requires_env=ENV)
def social_trends(category: str) -> dict:
    tags: dict[str, int] = {}
    posts = []
    for platform in ("instagram", "pinterest"):
        for item in _get(f"/api/trends/{platform}", category=category, limit=50).get("items", []):
            engagement = int(_number(item.get("engagement")) or 0)
            posts.append({"platform": platform, "engagement": engagement,
                          "caption": (item.get("caption") or "")[:160], "url": item.get("url", "")})
            for tag in item.get("hashtags", []):
                tag = tag.lower()
                if tag.strip("#") != category.replace("_", ""):       # skip the category's own tag
                    tags[tag] = tags.get(tag, 0) + engagement
    if not posts:
        raise LookupError(f"No social posts for {category}")
    top_tags = sorted(tags.items(), key=lambda kv: -kv[1])[:6]
    best = sorted(posts, key=lambda p: -p["engagement"])[:3]
    return {"source": best[0]["url"],
            "summary": (f"{len(posts)} Instagram and Pinterest posts about {category.replace('_', ' ')}s. "
                        "Hashtags by total engagement: " + ", ".join(f"{t} ({n:,})" for t, n in top_tags)
                        + ". Top posts: " + " | ".join(f"[{p['platform']}, {p['engagement']:,}] {p['caption']}" for p in best)
                        + f" Source: {_SNAPSHOT}."),
            "data": {"kind": "hashtags", "points": [[t, n] for t, n in top_tags]}}


@tool("Rugs only. Editorial articles about a rug category: trend forecasts, market "
      "overviews and buying guides.", {"category": _CATEGORY}, requires_env=ENV)
def trend_articles(category: str) -> dict:
    items = _get("/api/trends/blogs", category=category, limit=10).get("items", [])
    if not items:
        raise LookupError(f"No articles for {category}")
    return {"source": items[0].get("url", ""),
            "summary": " || ".join(
                f"{i.get('title')} ({i.get('source_name')}, {i.get('published_at')}): "
                + re.sub(r"\s+", " ", i.get("content") or i.get("excerpt") or "")[:700]
                for i in items[:3]) + f" Source: {_SNAPSHOT}."}
