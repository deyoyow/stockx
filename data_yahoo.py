# data_yahoo.py
from __future__ import annotations
from typing import List
import pandas as pd
import yfinance as yf
from cache_disk import DiskCache

def _normalize_yf_download(df: pd.DataFrame, tickers_jk: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = []

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))

        for t in tickers_jk:
            sub = None
            if t in lvl0:
                sub = df[t].copy()
            elif t in lvl1:
                sub = df.xs(t, level=1, axis=1).copy()

            if sub is None or sub.empty:
                continue

            sub.columns = [c.lower() for c in sub.columns]
            sub = sub.reset_index()

            if "Datetime" in sub.columns and "Date" not in sub.columns:
                sub = sub.rename(columns={"Datetime": "Date"})
            if "date" in sub.columns and "Date" not in sub.columns:
                sub = sub.rename(columns={"date": "Date"})

            sub["ticker"] = t
            out.append(sub)

    else:
        sub = df.copy()
        sub.columns = [c.lower() for c in sub.columns]
        sub = sub.reset_index()
        if "Datetime" in sub.columns and "Date" not in sub.columns:
            sub = sub.rename(columns={"Datetime": "Date"})
        if "date" in sub.columns and "Date" not in sub.columns:
            sub = sub.rename(columns={"date": "Date"})
        sub["ticker"] = tickers_jk[0] if tickers_jk else ""
        out.append(sub)

    if not out:
        return pd.DataFrame()

    res = pd.concat(out, ignore_index=True)
    if "Date" in res.columns:
        res["Date"] = pd.to_datetime(res["Date"], errors="coerce")
    return res

def fetch_prices_fast(
    tickers_jk: List[str],
    lookback_days_price: int,
    cache: DiskCache,
    ttl_seconds: int,
) -> pd.DataFrame:
    tickers_jk = [t.strip() for t in tickers_jk if t and str(t).strip()]
    if not tickers_jk:
        return pd.DataFrame()

    key = f"fast|days={int(lookback_days_price)}|tickers={' '.join(sorted(tickers_jk))}"
    cached = cache.get("prices", key, ttl_seconds=ttl_seconds)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached

    tickers_str = " ".join(tickers_jk)
    df = yf.download(
        tickers=tickers_str,
        period=f"{int(lookback_days_price)}d",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,   # speed
    )

    out = _normalize_yf_download(df, tickers_jk)
    if out is not None and not out.empty:
        cache.set("prices", key, out)
    return out
