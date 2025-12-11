# data_gnews.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone  # kept if we later need it
from typing import Dict, List, Optional
import os

import pandas as pd
import requests

from cache_disk import DiskCache
from utils import dedup_preserve_order, quote_phrase


# We now implement the "GNews" abstraction on top of Google Programmable Search
# (Custom Search JSON API). The rest of the app still calls these helpers as if
# they were backed by gnews.io; only the implementation changed.
CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

# ID of the Programmable Search Engine (cx) comes from the environment.
# The API key is still passed in from the caller (GNEWS_API_KEY).
PSE_SEARCH_ENGINE_ID = os.getenv("PSE_SEARCH_ENGINE_ID", "").strip()

# Simple domain-based groups so we can distinguish "retail/gossip" vs "official"
NEWS_SOURCE_GROUPS = {
    # Retail / trader-ish
    "cnbcindonesia.com": "retail",
    "kontan.co.id": "retail",
    "emitennews.com": "retail",
    "idnfinancials.com": "retail",
    "indopremier.com": "retail",  # IPOTNEWS
    "bisnis.com": "retail",
    "finance.detik.com": "retail",

    # Official / semi-official
    "idx.co.id": "official",
    "ojk.go.id": "official",
    "indonesia-investments.com": "official",
    "thejakartapost.com": "official",
}


def classify_source(url_or_host: str) -> str:
    """
    Map a URL or host to a simple group: 'retail', 'official', or 'other'.
    """
    host = (url_or_host or "").lower()
    for key, group in NEWS_SOURCE_GROUPS.items():
        if key in host:
            return group
    return "other"


def build_gnews_query(ticker: str, aliases: Dict[str, List[str]], max_phrases: int = 6) -> str:
    """
    Build a reasonably strict search query for news/articles about a given ticker.

    - Use company aliases from config.ALIASES
    - Prefer multi-word phrases ("Barito Pacific" etc.)
    - Always include the raw ticker
    - AND them with generic stock/IDX keywords (English + a bit Indonesian)
    """
    t = ticker.strip().upper()
    names = aliases.get(t, [])

    phrases: list[str] = []
    for n in names:
        n = str(n).strip()
        if not n:
            continue
        # Prefer multi-word names and trim leading "PT" variants
        if " " in n:
            low = n.lower()
            if low.startswith("pt. "):
                n = n[4:].strip()
            elif low.startswith("pt "):
                n = n[3:].strip()
            phrases.append(n)

    # Always include the raw ticker as well
    phrases.append(t)
    phrases = dedup_preserve_order(phrases)[:max_phrases]
    core = " OR ".join([quote_phrase(p) for p in phrases])

    # Anchor the query in a stock/IDX context.
    anchor = (
        '(stock OR shares OR IDX OR "Jakarta" OR Indonesia '
        'OR earnings OR dividend OR market OR saham OR "Bursa Efek")'
    )
    return f"({core}) {anchor}"


def _parse_published_at(item: dict) -> str | None:
    """
    Try to extract a publication timestamp from the Custom Search result item.

    The Custom Search API doesn't provide a standard date field, but many news
    sites expose it in meta tags (article:published_time, datePublished, etc.).
    We scan pagemap.metatags and pick the first plausible value.
    """
    pagemap = item.get("pagemap") or {}
    meta_list = pagemap.get("metatags") or []
    candidates: list[str] = []

    for meta in meta_list:
        if not isinstance(meta, dict):
            continue
        for k, v in meta.items():
            lk = str(k).lower()
            if any(s in lk for s in ["publish", "modified", "date", "time"]):
                v_str = str(v).strip()
                if v_str:
                    candidates.append(v_str)

    return candidates[0] if candidates else None


def fetch_gnews_articles(
    ticker: str,
    query: str,
    api_key: str,
    lookback_days: int,
    max_results: int,
    cache: DiskCache,
    ttl_seconds: int,
    lang: str = "en",
    errors: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch articles for a given ticker using Google Programmable Search.

    Returns a dataframe with at least:
      publishedAt, title, description, url, source_name, source_url,
      source_group, ticker
    """
    t = ticker.strip().upper()
    q = query.strip()
    max_results = int(max_results)

    cache_key = f"pse|t={t}|q={q}|d={int(lookback_days)}|n={max_results}|lang={lang}"
    cached = cache.get("pse_articles", cache_key, ttl_seconds=ttl_seconds)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached

    if not api_key:
        # Caller passes whatever is in GNEWS_API_KEY; if it's empty, just bail.
        return pd.DataFrame()
    if not PSE_SEARCH_ENGINE_ID:
        if errors is not None:
            errors.append("PSE_SEARCH_ENGINE_ID not set; news search backend is disabled.")
        return pd.DataFrame()

    params = {
        "key": api_key,
        "cx": PSE_SEARCH_ENGINE_ID,
        "q": q,
        "num": min(max_results, 10),
    }

    # dateRestrict supports dN / wN / mN / yN; we use days.
    if lookback_days and lookback_days > 0:
        params["dateRestrict"] = f"d{int(lookback_days)}"
        params["sort"] = "date"  # ask for newest first when possible

    try:
        r = requests.get(CSE_ENDPOINT, params=params, timeout=20)
        if r.status_code != 200:
            if errors is not None:
                errors.append(f"Custom Search API HTTP {r.status_code} for {t}: {r.text[:200]}")
            return pd.DataFrame()
        data = r.json()
    except Exception as exc:
        if errors is not None:
            errors.append(f"Custom Search API error for {t}: {exc}")
        return pd.DataFrame()

    items = data.get("items") or []
    if not items:
        return pd.DataFrame()

    rows: list[dict] = []
    for it in items:
        title = it.get("title")
        snippet = it.get("snippet")
        link = it.get("link")
        display_link = it.get("displayLink")

        pub = _parse_published_at(it)

        src_name = display_link or ""
        src_url = None
        if display_link:
            if display_link.startswith("http://") or display_link.startswith("https://"):
                src_url = display_link
            else:
                src_url = f"https://{display_link}"

        group = classify_source(display_link or link or "")

        rows.append(
            {
                "publishedAt": pub,
                "title": title,
                "description": snippet,
                "url": link,
                "source_name": src_name,
                "source_url": src_url,
                "source_group": group,
                "ticker": t,
            }
        )

    df = pd.DataFrame(rows)
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    cache.set("pse_articles", cache_key, df)
    return df


def fetch_latest_headlines(
    ticker: str,
    query: str,
    api_key: str,
    lookback_days: int,
    max_records: int,
    cache: DiskCache,
    ttl_seconds: int,
    lang: str = "en",
    errors: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Light wrapper used by the Streamlit "Headlines" section.

    It just calls fetch_gnews_articles() (now backed by Programmable Search)
    and trims the columns to what the UI expects.
    """
    df = fetch_gnews_articles(
        ticker=ticker,
        query=query,
        api_key=api_key,
        lookback_days=lookback_days,
        max_results=max_records,
        cache=cache,
        ttl_seconds=ttl_seconds,
        lang=lang,
        errors=errors,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["publishedAt", "title", "source_name", "url"])
    cols = [c for c in ["publishedAt", "title", "source_name", "url"] if c in df.columns]
    return df[cols].head(int(max_records)).copy()
