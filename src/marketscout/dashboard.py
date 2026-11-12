"""Turn a saved run into a standalone dashboard page.

    python -m marketscout.dashboard runs/oat-milk-fi-agent

Writes dashboard.html into the run folder. The page is one self-contained file
(data inlined), so it can be hosted anywhere static.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from .tools import COUNTRIES, http

TEMPLATE = Path(__file__).parent / "dashboard_template.html"
CITATION = re.compile(r"[\[【](E\d+)[\]】]")   # [E1] or the full-width 【E1】 some models emit
OUTLINES = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/{}.geo.json"
ISO3 = {"FI": "FIN", "SE": "SWE", "DK": "DNK", "DE": "DEU", "FR": "FRA", "ES": "ESP",
        "IT": "ITA", "NL": "NLD", "PL": "POL", "PT": "PRT", "AT": "AUT", "IE": "IRL", "US": "USA"}

# Stand-in 3D shapes until real models are fetched: keyword -> shape drawn by the page.
SHAPES = [
    (("milk", "juice", "drink", "cream", "yogurt", "yoghurt"), "carton"),
    (("oil", "wine", "beer", "water", "sauce", "vinegar", "soda", "bottle", "shampoo"), "bottle"),
    (("rug", "carpet", "mat", "runner"), "rug"),
]


DEMONYM = {"US": "American", "FI": "Finnish", "SE": "Swedish", "DK": "Danish", "DE": "German",
           "FR": "French", "ES": "Spanish", "IT": "Italian", "NL": "Dutch", "PL": "Polish",
           "PT": "Portuguese", "AT": "Austrian", "IE": "Irish"}
OPEN_LICENCES = ("public domain", "cc0", "cc by", "pd")


def tagline(country: str, shape: str) -> str | None:
    """'for the American home', 'for the Finnish table': the line under the title."""
    if country not in DEMONYM:
        return None
    return f"for the {DEMONYM[country]} {'table' if shape in ('carton', 'bottle') else 'home'}"


def stock_photo(category: str) -> dict | None:
    """An openly licensed photo from Wikimedia Commons, with its credit. None if nothing fits."""
    try:
        pages = http.get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "generator": "search", "gsrsearch": f"{category} filetype:bitmap",
            "gsrnamespace": 6, "gsrlimit": 12, "prop": "imageinfo",
            "iiprop": "url|extmetadata|size", "iiurlwidth": 900, "format": "json",
        }).json().get("query", {}).get("pages", {})
        for page in sorted(pages.values(), key=lambda x: x.get("index", 99)):
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})
            licence = meta.get("LicenseShortName", {}).get("value", "")
            if info.get("width", 0) < 600 or not licence.lower().startswith(OPEN_LICENCES):
                continue
            artist = re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
            artist = re.sub(" +", " ", artist.replace(chr(10), " "))[:60] or "Unknown author"
            title = page["title"].removeprefix("File:").rsplit(".", 1)[0].replace("_", " ")[:80]
            return {"url": info.get("thumburl", info["url"]).split("?")[0], "title": title,
                    "credit": f"{artist}, {licence}", "page": info.get("descriptionurl", "")}
    except Exception:
        pass
    return None


def pick_shape(category: str) -> str:
    words = category.lower()
    return next((shape for keys, shape in SHAPES if any(k in words for k in keys)), "box")


def _clean(text: str) -> dict:
    cites = list(dict.fromkeys(CITATION.findall(text)))
    text = CITATION.sub("", text).replace("**", "")
    text = re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s+", " ", text)).strip()
    return {"text": text, "cites": cites}


def parse_brief(brief: str, verdicts: list[dict] | None = None) -> dict:
    """Split the Markdown brief into its sections, attaching the judge's ruling to each claim."""
    verdicts, n, exempt = verdicts or [], 0, False

    def ruled(item: dict) -> dict:
        nonlocal n
        if n < len(verdicts):       # same order as judge.claim_lines
            item.update(verdict=verdicts[n]["verdict"], reason=verdicts[n]["reason"])
        n += 1
        return item

    sections = {"summary": [], "findings": [], "risks": [], "recommendations": [], "open": []}
    current = None
    for line in brief.splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("# ").lower()
            exempt = "open questions" in title
            current = next((key for word, key in (("summar", "summary"), ("finding", "findings"),
                                                  ("risk", "risks"), ("recommend", "recommendations"),
                                                  ("open", "open")) if word in title), None)
        elif line.startswith("- ") and not exempt:
            item = ruled(_clean(line[2:]))
            if current:
                sections[current].append(item)
        elif current == "summary" and line:
            sections["summary"].append(_clean(line))
        elif line.startswith("- ") and current:
            sections[current].append(_clean(line[2:]))
    return sections


def country_outline(code: str) -> dict | None:
    """Mainland outline as an SVG path, 100 units tall. None if it cannot be fetched."""
    try:
        geometry = http.get(OUTLINES.format(ISO3[code])).json()["features"][0]["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        rings = [p[0] for p in polygons]

        def extent(ring):
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        main = max(rings, key=extent)
        cx = sum(p[0] for p in main) / len(main)
        cy = sum(p[1] for p in main) / len(main)
        # keep the mainland and nearby islands; drop specks and overseas territories
        rings = [r for r in rings if extent(r) >= extent(main) * 0.01
                 and abs(r[0][0] - cx) < 15 and abs(r[0][1] - cy) < 15]
        squeeze = math.cos(math.radians(cy))        # longitudes shrink towards the pole
        points = [[(lon * squeeze, -lat) for lon, lat in r] for r in rings]
        xs = [x for r in points for x, _ in r]
        ys = [y for r in points for _, y in r]
        scale = 100 / (max(ys) - min(ys))
        path = "".join("M" + "L".join(f"{(x - min(xs)) * scale:.1f},{(y - min(ys)) * scale:.1f}"
                                      for x, y in r) + "Z" for r in points)
        return {"path": path, "w": round((max(xs) - min(xs)) * scale, 1), "h": 100}
    except Exception:
        return None


def build(run_dir: Path) -> Path:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    country = run["country"]
    payload = {
        "category": run["category"],
        "country": country,
        "countryName": "the United States" if country == "US" else COUNTRIES.get(country, ("", country))[1].title(),
        "outline": country_outline(country),
        "shape": pick_shape(run["category"]),
        "tagline": tagline(country, pick_shape(run["category"])),
        "photo": stock_photo(run["category"]),
        "questions": run.get("questions", []),
        "sections": parse_brief(run.get("brief", ""), run.get("verdicts")),
        "evidence": [{**e, "summary": e["summary"][:420]} for e in run.get("evidence", [])],
        "toolLog": run.get("tool_log", []),
        "metrics": run["metrics"],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__RUN_DATA__*/null", data)
    out = run_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(f"Wrote {build(Path(sys.argv[1]))}")
