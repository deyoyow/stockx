# scoring.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np
import pandas as pd

from utils import zscore
from data_gnews import build_gnews_query, fetch_gnews_articles
from data_x import build_x_query, fetch_x_posts
from sentiment import aggregate_sentiment


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "ticker" not in df.columns or "close" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()

    feats = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date")
        close = pd.to_numeric(g["close"], errors="coerce")
        vol = pd.to_numeric(g.get("volume", np.nan), errors="coerce")

        if close.dropna().shape[0] < 10:
            continue

        last_close = float(close.iloc[-1])
        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else np.nan
        avg_vol_20 = float(vol.tail(20).mean()) if vol.notna().any() else np.nan

        feats.append(
            {
                "ticker": str(t).replace(".jk", "").replace(".JK", ""),
                "ticker_jk": str(t),
                "last_close": last_close,
                "ret_5d": ret_5d,
                "avg_vol_20d": avg_vol_20,
            }
        )
    return pd.DataFrame(feats)


def fetch_gnews_for_universe(
    tickers: list[str],
    aliases: dict,
    gnews_api_key: str,
    lookback_days_news: int,
    max_articles_per_ticker: int,
    cache,
    ttl_seconds: int,
    max_workers: int,
    progress_cb: Optional[Callable[[int, int, str | None], None]] = None,
) -> pd.DataFrame:
    tasks = []
    for t in tickers:
        q = build_gnews_query(t, aliases)
        tasks.append((t, q))

    rows = []
    total = max(1, len(tasks))
    done = 0

    def _one(t: str, q: str) -> dict:
        df = fetch_gnews_articles(
            ticker=t,
            query=q,
            api_key=gnews_api_key,
            lookback_days=lookback_days_news,
            max_results=max_articles_per_ticker,
            cache=cache,
            ttl_seconds=ttl_seconds,
            lang="en",
        )
        # sentiment on title + description
        if df is None or df.empty:
            return {"ticker": t, "kw_used": q, "gnews_sent": np.nan, "gnews_buzz": 0}

        text = (df.get("title", "").fillna("").astype(str) + ". " + df.get("description", "").fillna("").astype(str))
        df2 = pd.DataFrame({"text": text})
        sent, n = aggregate_sentiment(df2, "text")
        buzz = int(df.shape[0])
        return {"ticker": t, "kw_used": q, "gnews_sent": sent, "gnews_buzz": buzz}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_one, t, q): (t, q) for (t, q) in tasks}
        for fut in as_completed(fut_map):
            t, _ = fut_map[fut]
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total, t)
                except Exception:
                    pass
            try:
                rows.append(fut.result())
            except Exception:
                rows.append({"ticker": t, "kw_used": "", "gnews_sent": np.nan, "gnews_buzz": 0})

    return pd.DataFrame(rows)


def fetch_x_for_tickers(
    tickers: list[str],
    aliases: dict,
    x_bearer_token: str,
    lookback_hours: int,
    max_posts_per_ticker: int,
    cache,
    ttl_seconds: int,
    max_workers: int,
    progress_cb: Optional[Callable[[int, int, str | None], None]] = None,
) -> pd.DataFrame:
    tasks = []
    for t in tickers:
        q = build_x_query(t, aliases)
        tasks.append((t, q))

    rows = []
    total = max(1, len(tasks))
    done = 0

    def _one(t: str, q: str) -> dict:
        df = fetch_x_posts(
            ticker=t,
            query=q,
            bearer_token=x_bearer_token,
            lookback_hours=lookback_hours,
            max_results=max_posts_per_ticker,
            cache=cache,
            ttl_seconds=ttl_seconds,
        )
        if df is None or df.empty:
            return {"ticker": t, "x_query": q, "x_sent": np.nan, "x_buzz": 0}

        # sentiment on tweet text
        df2 = pd.DataFrame({"text": df.get("text", "").fillna("").astype(str)})
        sent, n = aggregate_sentiment(df2, "text")
        buzz = int(df.shape[0])
        return {"ticker": t, "x_query": q, "x_sent": sent, "x_buzz": buzz}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_one, t, q): (t, q) for (t, q) in tasks}
        for fut in as_completed(fut_map):
            t, _ = fut_map[fut]
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total, t)
                except Exception:
                    pass
            try:
                rows.append(fut.result())
            except Exception:
                rows.append({"ticker": t, "x_query": "", "x_sent": np.nan, "x_buzz": 0})

    return pd.DataFrame(rows)


def rank_stage1(df: pd.DataFrame, w_sent: float, w_buzz: float, w_mom: float) -> pd.DataFrame:
    out = df.copy()
    out["gnews_sent"] = pd.to_numeric(out["gnews_sent"], errors="coerce")
    out["gnews_buzz"] = pd.to_numeric(out["gnews_buzz"], errors="coerce")
    out["ret_5d"] = pd.to_numeric(out.get("ret_5d", 0.0), errors="coerce")

    out["sent_z"] = zscore(out["gnews_sent"].fillna(0.0))
    out["buzz_z"] = zscore(np.log1p(out["gnews_buzz"].fillna(0.0)))
    out["mom_z"] = zscore(out["ret_5d"].fillna(0.0))

    out["stage1_score"] = (w_sent * out["sent_z"]) + (w_buzz * out["buzz_z"]) + (w_mom * out["mom_z"])
    return out.sort_values("stage1_score", ascending=False).reset_index(drop=True)


def rank_final(df: pd.DataFrame, w_x_sent: float, w_x_buzz: float) -> pd.DataFrame:
    out = df.copy()
    out["x_sent"] = pd.to_numeric(out.get("x_sent", np.nan), errors="coerce")
    out["x_buzz"] = pd.to_numeric(out.get("x_buzz", 0), errors="coerce")

    out["x_sent_z"] = zscore(out["x_sent"].fillna(0.0))
    out["x_buzz_z"] = zscore(np.log1p(out["x_buzz"].fillna(0.0)))

    out["score"] = out["stage1_score"].fillna(0.0) + (w_x_sent * out["x_sent_z"]) + (w_x_buzz * out["x_buzz_z"])
    return out.sort_values("score", ascending=False).reset_index(drop=True)
