# data_x.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import requests

from cache_disk import DiskCache
from utils import dedup_preserve_order, quote_phrase


X_RECENT_SEARCH = "https://api.twitter.com/2/tweets/search/recent"


def build_x_query(ticker: str, aliases: Dict[str, List[str]]) -> str:
    """
    Keep it relatively strict to reduce noise.
    English now; later we can add Indonesian anchors.
    """
    t = ticker.strip().upper()
    names = aliases.get(t, [])

    phrases: list[str] = []
    for n in names:
        n = str(n).strip()
        if not n:
            continue
        if " " in n:  # multi-word only (avoid generic single words)
            low = n.lower()
            if low.startswith("pt. "):
                n = n[4:].strip()
            elif low.startswith("pt "):
                n = n[3:].strip()
            phrases.append(n)

    phrases += [t]
    phrases = dedup_preserve_order(phrases)[:5]
    core = " OR ".join([quote_phrase(p) for p in phrases])

    # stock anchor terms
    anchor = '(stock OR shares OR IDX OR "Jakarta" OR Indonesia OR earnings OR dividend OR market)'
    # remove retweets
    return f"({core}) {anchor} -is:retweet"


def fetch_x_posts(
    ticker: str,
    query: str,
    bearer_token: str,
    lookback_hours: int,
    max_results: int,
    cache: DiskCache,
    ttl_seconds: int,
    endpoint: str = X_RECENT_SEARCH,
) -> pd.DataFrame:
    """
    Returns dataframe with:
    created_at,text,lang,like_count,retweet_count,reply_count,quote_count,ticker
    """
    t = ticker.strip().upper()
    cache_key = f"x|t={t}|q={query}|h={int(lookback_hours)}|n={int(max_results)}"
    cached = cache.get("x_posts", cache_key, ttl_seconds=ttl_seconds)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached

    if not bearer_token:
        return pd.DataFrame()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=int(lookback_hours))
    start_time = start_dt.isoformat().replace("+00:00", "Z")

    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "max_results": min(int(max_results), 100),
        "start_time": start_time,
        "tweet.fields": "created_at,lang,public_metrics",
    }

    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    items = data.get("data", []) or []
    if not items:
        return pd.DataFrame()

    rows = []
    for it in items:
        pm = it.get("public_metrics") or {}
        rows.append(
            {
                "created_at": it.get("created_at"),
                "text": it.get("text"),
                "lang": it.get("lang"),
                "like_count": pm.get("like_count", 0),
                "retweet_count": pm.get("retweet_count", 0),
                "reply_count": pm.get("reply_count", 0),
                "quote_count": pm.get("quote_count", 0),
                "ticker": t,
            }
        )

    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    cache.set("x_posts", cache_key, df)
    return df
