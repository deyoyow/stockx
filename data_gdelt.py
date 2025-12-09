# data_gdelt.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import time
import numpy as np
import pandas as pd
from gdeltdoc import GdeltDoc, Filters

from cache_disk import DiskCache
from utils import dedup_preserve_order, quote_phrase, is_all_caps_acronym


gd = GdeltDoc()

def build_keyword_query(ticker: str, aliases: Dict[str, List[str]], max_phrases: int = 6) -> str:
    """
    Build a GDELT keyword query for one IDX ticker.

    Strategy:
    - For each alias:
        * Strip leading 'PT ' / 'PT. ' → core name
        * For multi-word names (e.g. 'Astra International', 'Bank Mandiri'):
              include the plain core name  → "Astra International"
              plus 'PT {core} Tbk' and '{core} Tbk'
        * For all-caps acronyms (BCA, BRI, BNI, etc.):
              include the acronym itself  → BCA
    - Always add '{TICKER} Tbk' and '{TICKER}' (e.g. 'ASII Tbk', 'ASII').
    - Then deduplicate and cap to max_phrases.
    """
    t = ticker.strip().upper()
    names = aliases.get(t, [])

    phrases: list[str] = []

    for raw in names:
        n = raw.strip()
        if not n:
            continue

        # Accept multi-word names, or all-caps acronyms, or exact ticker
        if (" " not in n) and (not is_all_caps_acronym(n)) and (n.upper() != t):
            continue

        low = n.lower()
        if low.startswith("pt. "):
            core = n[4:].strip()
        elif low.startswith("pt "):
            core = n[3:].strip()
        else:
            core = n.strip()

        if not core:
            continue

        # 1) Plain company name (no PT/Tbk) – safe because it's multi-word or acronym
        phrases.append(core)

        # 2) Tbk variants
        if "tbk" in core.lower():
            # already has Tbk
            phrases.append(core)
        else:
            phrases.append(f"PT {core} Tbk")
            phrases.append(f"{core} Tbk")

        # 3) Acronym itself (e.g. BCA, BRI)
        if is_all_caps_acronym(core):
            phrases.append(core)

    # Always include ticker forms
    phrases.append(f"{t} Tbk")
    phrases.append(t)

    # Deduplicate + cap length
    phrases = dedup_preserve_order(phrases)[:max_phrases]

    # Quote multi-word phrases
    parts = [quote_phrase(p) for p in phrases]
    return " OR ".join(parts)

def _pick_col(df: pd.DataFrame, preferred: list[str], contains: list[str]) -> str | None:
    cols = [c for c in df.columns if c.lower() != "datetime"]
    lower_map = {c.lower(): c for c in cols}

    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]

    for c in cols:
        cl = c.lower()
        if any(sub in cl for sub in contains):
            return c

    return None

def _retry(call, retries: int = 2, base_sleep: float = 0.6):
    last = None
    for i in range(retries + 1):
        try:
            return call()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** i))
    raise last  # type: ignore

def fetch_gdelt_metrics(
    query: str,
    lookback_days_news: int,
    include_vol: bool,
    cache: DiskCache,
    ttl_seconds: int,
) -> dict:
    key = f"metrics|q={query}|days={int(lookback_days_news)}|vol={bool(include_vol)}"
    cached = cache.get("gdelt_metrics", key, ttl_seconds=ttl_seconds)
    if isinstance(cached, dict) and cached:
        return cached

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days_news))

    f = Filters(
        keyword=query,
        start_date=start_dt.date().isoformat(),
        end_date=end_dt.date().isoformat(),
    )

    res = {"tone_latest": np.nan, "vol_latest": np.nan}

    # Tone
    try:
        tone_df = _retry(lambda: gd.timeline_search("timelinetone", f), retries=2)
        if tone_df is not None and not tone_df.empty:
            col = _pick_col(tone_df, preferred=["AverageTone"], contains=["tone"])
            if col is not None:
                res["tone_latest"] = float(pd.to_numeric(tone_df[col].iloc[-1], errors="coerce"))
            else:
                last = tone_df.iloc[-1].drop(labels=["datetime"], errors="ignore")
                res["tone_latest"] = float(pd.to_numeric(last, errors="coerce").mean())
    except Exception:
        pass

    # Volume
    if include_vol:
        try:
            vol_df = _retry(lambda: gd.timeline_search("timelinevolraw", f), retries=2)
            if vol_df is not None and not vol_df.empty:
                col = _pick_col(vol_df, preferred=["Volume"], contains=["vol", "count"])
                if col is not None:
                    res["vol_latest"] = float(pd.to_numeric(vol_df[col].iloc[-1], errors="coerce"))
                else:
                    last = vol_df.iloc[-1].drop(labels=["datetime"], errors="ignore")
                    res["vol_latest"] = float(pd.to_numeric(last, errors="coerce").sum())
        except Exception:
            pass

    cache.set("gdelt_metrics", key, res)
    return res

def fetch_latest_headlines(
    query: str,
    lookback_days_news: int,
    max_records: int,
    cache: DiskCache,
    ttl_seconds: int,
) -> pd.DataFrame:
    key = f"headlines|q={query}|days={int(lookback_days_news)}|n={int(max_records)}"
    cached = cache.get("gdelt_headlines", key, ttl_seconds=ttl_seconds)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days_news))

    f = Filters(
        keyword=query,
        start_date=start_dt.date().isoformat(),
        end_date=end_dt.date().isoformat(),
        num_records=min(int(max_records), 250),
    )

    try:
        articles = _retry(lambda: gd.article_search(f), retries=2)
        if articles is None or articles.empty:
            return pd.DataFrame(columns=["seendate", "title", "domain", "url"])

        a = articles.copy()
        a.columns = [c.lower() for c in a.columns]
        cols = [c for c in ["seendate", "title", "domain", "url"] if c in a.columns]
        out = a[cols].head(int(max_records)).copy()
        cache.set("gdelt_headlines", key, out)
        return out
    except Exception:
        return pd.DataFrame(columns=["seendate", "title", "domain", "url"])
