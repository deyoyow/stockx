# data_gnews.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from cache_disk import DiskCache
from utils import dedup_preserve_order, quote_phrase


GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"


def build_gnews_query(ticker: str, aliases: Dict[str, List[str]]) -> str:
    """
    English-only for now.
    We anchor with stock/market terms to reduce false positives.
    """
    t = ticker.strip().upper()
    names = aliases.get(t, [])

    phrases: list[str] = []
    # Prefer multi-word company names, avoid single generic words like "Astra"
    for n in names:
        n = str(n).strip()
        if not n:
            continue
        if " " in n:  # only multi-word
            # strip PT / PT.
            low = n.lower()
            if low.startswith("pt. "):
                n = n[4:].strip()
            elif low.startswith("pt "):
                n = n[3:].strip()
            phrases.append(n)

    phrases += [t]  # ticker always
    phrases = dedup_preserve_order(phrases)[:5]
    parts = [quote_phrase(p) for p in phrases]
    core = " OR ".join(parts)

    # Stock context anchors (English for now)
    anchor = '(stock OR shares OR "IDX" OR "Jakarta" OR "Indonesia" OR market OR earnings OR dividend OR IPO)'
    return f"({core}) AND {anchor}"


def fetch_gnews_articles(
    ticker: str,
    query: str,
    api_key: str,
    lookback_days: int,
    max_results: int,
    cache: DiskCache,
    ttl_seconds: int,
    lang: str = "en",
) -> pd.DataFrame:
    """
    Returns articles dataframe with:
    publishedAt,title,description,url,source_name,source_url,ticker
    """
    t = ticker.strip().upper()
    cache_key = f"gnews|t={t}|q={query}|days={int(lookback_days)}|n={int(max_results)}|lang={lang}"
    cached = cache.get("gnews_articles", cache_key, ttl_seconds=ttl_seconds)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached

    if not api_key:
        return pd.DataFrame()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days))
    # GNews expects RFC3339
    from_dt = start_dt.isoformat().replace("+00:00", "Z")
    to_dt = end_dt.isoformat().replace("+00:00", "Z")

    params = {
        "q": query,
        "lang": lang,
        "from": from_dt,
        "to": to_dt,
        "max": int(max_results),
        "token": api_key,
    }

    try:
        r = requests.get(GNEWS_ENDPOINT, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    articles = data.get("articles", []) or []
    if not articles:
        return pd.DataFrame()

    rows = []
    for a in articles:
        src = a.get("source") or {}
        rows.append(
            {
                "publishedAt": a.get("publishedAt"),
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "source_name": src.get("name"),
                "source_url": src.get("url"),
                "ticker": t,
            }
        )

    df = pd.DataFrame(rows)
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    cache.set("gnews_articles", cache_key, df)
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
) -> pd.DataFrame:
    df = fetch_gnews_articles(
        ticker=ticker,
        query=query,
        api_key=api_key,
        lookback_days=lookback_days,
        max_results=max_records,
        cache=cache,
        ttl_seconds=ttl_seconds,
        lang=lang,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["publishedAt", "title", "source_name", "url"])
    cols = [c for c in ["publishedAt", "title", "source_name", "url"] if c in df.columns]
    return df[cols].head(int(max_records)).copy()
