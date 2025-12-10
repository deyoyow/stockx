# data_idx.py
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List

import cloudscraper
import pandas as pd

from cache_disk import DiskCache

IDX_URL_TEMPLATE = (
    "https://idx.co.id/umbraco/Surface/ListedCompany/GetTradingInfoSS?"
    "code={code}&length={length}"
)


def _fetch_idx_one(
    ticker_jk: str,
    length: int,
    cache: DiskCache,
    ttl_seconds: int,
) -> pd.DataFrame:
    """
    Fetch raw daily data for a single IDX code using the same endpoint
    as antonizer/IDX-Scrapper.

    Returns a DataFrame with columns: date, close, volume.
    """
    base = ticker_jk.split(".")[0].upper().strip()
    if not base:
        return pd.DataFrame()

    key = f"idx|code={base}|len={int(length)}"
    cached = cache.get("prices_idx", key, ttl_seconds=ttl_seconds)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()

    scraper = cloudscraper.CloudScraper()
    url = IDX_URL_TEMPLATE.format(code=base, length=int(length))

    try:
        text = scraper.get(url, timeout=15).text
        payload = json.loads(text)
    except Exception:
        return pd.DataFrame()

    replies = payload.get("replies") or []
    if not replies:
        return pd.DataFrame()

    rows = []
    for row in replies:
        # IDX returns a string date, let pandas parse it
        date_raw = row.get("Date")
        if not date_raw:
            continue

        dt = pd.to_datetime(date_raw, errors="coerce")
        if pd.isna(dt):
            continue

        rows.append(
            {
                "date": dt,
                "close": row.get("Close"),
                "volume": row.get("Volume"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cache.set("prices_idx", key, df)
    return df


def fetch_idx_prices(
    tickers_jk: List[str],
    lookback_days_price: int,
    cache: DiskCache,
    ttl_seconds: int,
) -> pd.DataFrame:
    """
    Fetch prices from IDX for the given tickers and lookback.

    Returns a long dataframe with: ticker, date, close, volume.
    """
    tickers_jk = [t.strip() for t in tickers_jk if t and str(t).strip()]
    tickers_jk = sorted(set(tickers_jk))
    if not tickers_jk:
        return pd.DataFrame()

    # Slightly longer than lookback to be safe
    length = max(1, int(lookback_days_price) + 3)

    out_frames = []
    for t in tickers_jk:
        df_t = _fetch_idx_one(
            ticker_jk=t,
            length=length,
            cache=cache,
            ttl_seconds=ttl_seconds,
        )
        if df_t is None or df_t.empty:
            continue

        df_t = df_t.copy()
        df_t["ticker"] = t.upper()

        # Filter lookback window (defensive)
        start = datetime.utcnow() - timedelta(days=int(lookback_days_price) + 3)
        df_t = df_t[df_t["date"] >= pd.Timestamp(start)]
        if df_t.empty:
            continue

        out_frames.append(df_t)

    if not out_frames:
        return pd.DataFrame()

    combo = pd.concat(out_frames, ignore_index=True)
    combo["date"] = pd.to_datetime(combo["date"], errors="coerce")
    combo = combo.dropna(subset=["date"])
    return combo
