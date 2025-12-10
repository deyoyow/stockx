# data_prices.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

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


def _is_yahoo_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("rate limit" in msg) or ("too many requests" in msg) or ("429" in msg)


def _stooq_symbol_candidates(ticker_jk: str) -> list[str]:
    """
    Stooq symbol formats vary by exchange. For IDX it may not be available.
    We try a few common patterns. This is best-effort.
    """
    base = ticker_jk.split(".")[0].lower()
    return [
        f"{base}.id",
        f"{base}.jk",
        base,
    ]


def _fetch_stooq_one(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Expected columns: Date, Open, High, Low, Close, Volume
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def fetch_prices(
    tickers_jk: List[str],
    lookback_days_price: int,
    cache: DiskCache,
    ttl_seconds: int,
    mode: str = "yahoo_fallback_stooq",  # "yahoo", "stooq", "yahoo_fallback_stooq"
    yf_threads: bool = True,
    stooq_overrides: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Returns:
      - price_df: normalized long dataframe with columns including Date/open/high/low/close/volume/ticker
      - source_map: { "BBCA.JK": "yahoo" | "stooq" | "none" }

    Notes:
    - Yahoo can rate-limit by IP. Stooq may not have IDX tickers. This is best-effort fallback.
    """
    tickers_jk = [t.strip() for t in tickers_jk if t and str(t).strip()]
    tickers_jk = sorted(set(tickers_jk))
    if not tickers_jk:
        return pd.DataFrame(), {}

    stooq_overrides = stooq_overrides or {}
    source_map: Dict[str, str] = {t: "none" for t in tickers_jk}

    # Cache key should NOT depend on UI knobs like threads
    key = f"prices|mode={mode}|days={int(lookback_days_price)}|tickers={' '.join(tickers_jk)}"
    cached = cache.get("prices_combo", key, ttl_seconds=ttl_seconds)
    if isinstance(cached, dict) and "df" in cached and "src" in cached:
        df_cached = cached["df"]
        src_cached = cached["src"]
        if isinstance(df_cached, pd.DataFrame) and isinstance(src_cached, dict):
            return df_cached, src_cached

    out_frames: list[pd.DataFrame] = []

    # 1) Yahoo attempt
    missing_after_yahoo = tickers_jk[:]
    if mode in ("yahoo", "yahoo_fallback_stooq"):
        try:
            df = yf.download(
                tickers=" ".join(tickers_jk),
                period=f"{int(lookback_days_price)}d",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=bool(yf_threads),
            )
            norm = _normalize_yf_download(df, tickers_jk)
        except Exception as e:
            norm = pd.DataFrame()
            # Mark yahoo failure for all, we will fallback if allowed
            if not _is_yahoo_rate_limit_error(e):
                # still allow fallback, but this hints a different issue
                pass

        if norm is not None and not norm.empty:
            # Determine which tickers succeeded
            got = set(norm["ticker"].dropna().astype(str).unique().tolist())
            for t in tickers_jk:
                if t in got:
                    source_map[t] = "yahoo"
            out_frames.append(norm)
            missing_after_yahoo = [t for t in tickers_jk if t not in got]

        if mode == "yahoo":
            combo = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
            cache.set("prices_combo", key, {"df": combo, "src": source_map})
            return combo, source_map

    # 2) Stooq fallback (only missing tickers)
    if mode in ("stooq", "yahoo_fallback_stooq"):
        for t in missing_after_yahoo if mode == "yahoo_fallback_stooq" else tickers_jk:
            # Per-ticker cache
            t_key = f"stooq|t={t}|days={int(lookback_days_price)}"
            cached_t = cache.get("prices_stooq", t_key, ttl_seconds=ttl_seconds)
            if isinstance(cached_t, pd.DataFrame) and not cached_t.empty:
                df_t = cached_t.copy()
            else:
                sym_override = stooq_overrides.get(t, "").strip()
                candidates = [sym_override] if sym_override else _stooq_symbol_candidates(t)
                df_t = pd.DataFrame()

                for sym in [c for c in candidates if c]:
                    df_try = _fetch_stooq_one(sym)
                    if df_try is None or df_try.empty:
                        continue
                    df_t = df_try
                    break

                if df_t is not None and not df_t.empty:
                    cache.set("prices_stooq", t_key, df_t)

            if df_t is None or df_t.empty:
                continue

            # Filter lookback window
            start = datetime.utcnow() - timedelta(days=int(lookback_days_price) + 3)
            df_t = df_t[df_t["date"] >= pd.Timestamp(start)]

            # Normalize to match yfinance output schema as much as possible
            df_t = df_t.copy()
            df_t["ticker"] = t
            # create Date column to match your existing downstream usage
            df_t["Date"] = df_t["date"]
            out_frames.append(df_t)
            source_map[t] = "stooq"

    combo = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    if "Date" in combo.columns:
        combo["Date"] = pd.to_datetime(combo["Date"], errors="coerce")

    cache.set("prices_combo", key, {"df": combo, "src": source_map})
    return combo, source_map


def load_prices_from_csv(
    path: Path, tickers_jk: list[str], lookback_days_price: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load long-format price data from CSV with columns: ticker,date,close[,volume]
    Returns (df, src_map) where src_map maps ticker_jk -> "csv" or "none".
    """

    tickers_jk = [t.strip() for t in tickers_jk if t and str(t).strip()]
    tickers_jk = sorted(set(tickers_jk))
    if not tickers_jk:
        return pd.DataFrame(), {}

    src_map: dict[str, str] = {t: "none" for t in tickers_jk}
    if path is None or not Path(path).exists():
        return pd.DataFrame(), src_map

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), src_map

    if df is None or df.empty:
        return pd.DataFrame(), src_map

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns or "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(), src_map

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close", "ticker"])

    # Filter to requested tickers and lookback window
    df = df[df["ticker"].isin(tickers_jk)]
    if df.empty:
        return pd.DataFrame(), src_map

    start = datetime.utcnow() - timedelta(days=int(lookback_days_price) + 3)
    df = df[df["date"] >= pd.Timestamp(start)]
    if df.empty:
        return pd.DataFrame(), src_map

    df = df[[c for c in ["ticker", "date", "close", "volume"] if c in df.columns]].copy()
    df.sort_values(["ticker", "date"], inplace=True)

    for t in df["ticker"].dropna().astype(str).unique():
        if t in src_map:
            src_map[t] = "csv"

    return df, src_map
