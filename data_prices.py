# data_prices.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from cache_disk import DiskCache
from data_idx import fetch_idx_prices  # <-- add this line

import time
import pandas as pd
import yfinance as yf

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
    mode: str = "yahoo_fallback_stooq",  # also supports "yahoo_fallback_idx"
    yf_threads: bool = True,
    stooq_overrides: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch prices with Yahoo as primary, optional IDX and/or Stooq fallback.

    Returns:
      - price_df: long-format dataframe (any columns; will be normalized later)
      - source_map: { "BBCA.JK": "yahoo" | "idx" | "stooq" | "none" }

    Notes:
    - Yahoo is called in batches of 10 tickers to reduce rate-limit issues.
    - When mode is 'yahoo_fallback_idx', only missing tickers go to IDX.
    - When mode is 'yahoo_fallback_stooq', only missing tickers go to Stooq.
    """
    # -------- 0) Clean inputs --------
    tickers_jk = [t.strip() for t in tickers_jk if t and str(t).strip()]
    tickers_jk = sorted(set(tickers_jk))
    if not tickers_jk:
        return pd.DataFrame(), {}

    stooq_overrides = stooq_overrides or {}
    source_map: Dict[str, str] = {t: "none" for t in tickers_jk}

    # Cache key does not depend on threads or other transient options
    key = (
        f"prices|mode={mode}|days={int(lookback_days_price)}|"
        f"tickers={' '.join(tickers_jk)}"
    )
    cached = cache.get("prices_combo", key, ttl_seconds=ttl_seconds)
    if isinstance(cached, dict) and "df" in cached and "src" in cached:
        df_cached = cached.get("df")
        src_cached = cached.get("src")
        if isinstance(df_cached, pd.DataFrame) and isinstance(src_cached, dict):
            return df_cached, src_cached

    out_frames: list[pd.DataFrame] = []
    missing_after_yahoo = list(tickers_jk)  # default if Yahoo not used

    # -------- 1) Yahoo (primary) --------
    if mode in ("yahoo", "yahoo_fallback_stooq", "yahoo_fallback_idx"):
        got_yahoo: set[str] = set()

        # Call yfinance in batches of 10 tickers
        batch_size = 1
        for i in range(0, len(tickers_jk), batch_size):
            batch = tickers_jk[i : i + batch_size]
            if not batch:
                continue

            # Extra margin on lookback days to cover weekends/holidays
            margin_days = 3
            period_days = int(max(lookback_days_price + margin_days, 10))

            try:
                df_raw = yf.download(
                    tickers=batch,
                    period=f"{period_days}d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=yf_threads,
                    progress=False,
                )
            except Exception as e:
                # If it's a rate-limit, stop hammering Yahoo and break;
                # fallback sources will take over for everyone.
                if _is_yahoo_rate_limit_error(e):
                    break
                # For non-rate-limit errors, just skip this batch
                continue

            df_norm = _normalize_yf_download(df_raw, batch)
            if df_norm is None or df_norm.empty:
                continue

            # Filter to lookback window
            start = pd.Timestamp.utcnow() - pd.Timedelta(
                days=int(lookback_days_price) + 3
            )
            if "Date" in df_norm.columns:
                df_norm = df_norm[df_norm["Date"] >= start]

            if df_norm.empty:
                continue

            out_frames.append(df_norm)

            present_batch = (
                df_norm.get("ticker", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            for t in present_batch:
                got_yahoo.add(t)

        # Update source map for those which Yahoo actually returned
        for t in got_yahoo:
            if t in source_map:
                source_map[t] = "yahoo"

        missing_after_yahoo = [t for t in tickers_jk if t not in got_yahoo]

        # If user requested Yahoo only, we're done here
        if mode == "yahoo":
            combo = (
                pd.concat(out_frames, ignore_index=True)
                if out_frames
                else pd.DataFrame()
            )
            cache.set("prices_combo", key, {"df": combo, "src": source_map})
            return combo, source_map

    # -------- 2) IDX fallback (for missing tickers or whole universe) --------
    if mode in ("idx", "yahoo_fallback_idx"):
        if mode == "yahoo_fallback_idx":
            idx_targets = [t for t in missing_after_yahoo]
        else:
            idx_targets = list(tickers_jk)

        if idx_targets:
            try:
                from data_idx import fetch_idx_prices  # local import to avoid cycles
            except Exception:
                fetch_idx_prices = None  # type: ignore

            df_idx = pd.DataFrame()
            if fetch_idx_prices is not None:
                try:
                    df_idx = fetch_idx_prices(
                        tickers_jk=idx_targets,
                        lookback_days_price=int(lookback_days_price),
                    )
                except Exception:
                    df_idx = pd.DataFrame()

            if df_idx is not None and not df_idx.empty:
                # Ensure at least ticker/date/close columns exist as expected
                df_idx = df_idx.copy()
                # (columns already ticker, date, close[, volume] from data_idx)
                out_frames.append(df_idx)

                present_idx = (
                    df_idx.get("ticker", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                for t in present_idx:
                    # Only set to IDX if Yahoo didn't already fill it
                    if t in source_map and source_map[t] == "none":
                        source_map[t] = "idx"

        # If pure IDX mode, we skip Stooq and finish
        if mode == "idx":
            combo = (
                pd.concat(out_frames, ignore_index=True)
                if out_frames
                else pd.DataFrame()
            )
            cache.set("prices_combo", key, {"df": combo, "src": source_map})
            return combo, source_map

    # -------- 3) Stooq fallback (for remaining missing) --------
    if mode in ("stooq", "yahoo_fallback_stooq"):
        if mode == "yahoo_fallback_stooq":
            stooq_targets = [t for t in missing_after_yahoo]
        else:
            stooq_targets = list(tickers_jk)

        start = pd.Timestamp.utcnow() - pd.Timedelta(
            days=int(lookback_days_price) + 3
        )

        for t in stooq_targets:
            custom_symbol = stooq_overrides.get(t)
            if custom_symbol:
                candidates = [custom_symbol]
            else:
                candidates = _stooq_symbol_candidates(t)

            df_t = pd.DataFrame()
            for sym in candidates:
                df_candidate = _fetch_stooq_one(sym)
                if df_candidate is not None and not df_candidate.empty:
                    df_t = df_candidate
                    break

            if df_t is None or df_t.empty:
                continue

            df_t = df_t.copy()
            # Filter by lookback
            df_t["date"] = pd.to_datetime(df_t["date"], errors="coerce")
            df_t = df_t[df_t["date"] >= start]
            if df_t.empty:
                continue

            df_t["ticker"] = t
            out_frames.append(df_t)

            if t in source_map and source_map[t] == "none":
                source_map[t] = "stooq"

    # -------- 4) Combine & cache --------
    combo = (
        pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    )

    # Optional: standardize datetime column name for downstream normalization
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
        df_raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), src_map

    df = normalize_price_frame(df_raw)
    if df is None or df.empty:
        return pd.DataFrame(), src_map

    # Filter to requested tickers and lookback window
    df = df[df["ticker"].isin(tickers_jk)]
    if df.empty:
        return pd.DataFrame(), src_map

    start = datetime.utcnow() - timedelta(days=int(lookback_days_price) + 3)
    df = df[df["date"] >= pd.Timestamp(start)]
    if df.empty:
        return pd.DataFrame(), src_map
    df.sort_values(["ticker", "date"], inplace=True)

    for t in df["ticker"].dropna().astype(str).unique():
        if t in src_map:
            src_map[t] = "csv"

    return df, src_map


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any dataframe containing price data into the expected long format:
    ticker, date, close[, volume]. Returns an empty frame when unusable.
    """

    if isinstance(df, tuple):
        # Defensive: callers sometimes pass (df, src_map)
        df = df[0] if df else None

    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    if "close" not in out.columns and "adj close" in out.columns:
        out["close"] = out["adj close"]

    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    if "date" not in out.columns and "Date" in df.columns:
        out["date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "date" not in out.columns:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    required_cols = {"ticker", "close", "date"}
    if not required_cols.issubset(set(out.columns)):
        return pd.DataFrame()

    keep_cols = [c for c in ["ticker", "date", "close", "volume"] if c in out.columns]
    out = out[keep_cols]
    out = out.dropna(subset=["ticker", "date", "close"])
    if out.empty:
        return pd.DataFrame()

    out["ticker"] = out["ticker"].astype(str).str.strip()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def fetch_prices_infosaham_style(
    tickers_jk: list[str], lookback_days_price: int
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Best-effort sequential fetch that mirrors the infoSaham scraper style.

    This uses yfinance per ticker (serially) to reduce rate-limit pressure and
    returns a normalized long dataframe plus a source map labelled "infosaham".
    It is intentionally tolerant of per-ticker failures and reports them back
    to the caller.
    """

    tickers_jk = sorted(set([t for t in tickers_jk if t]))
    if not tickers_jk:
        return pd.DataFrame(), {}, []

    frames: list[pd.DataFrame] = []
    src_map: dict[str, str] = {t: "none" for t in tickers_jk}
    failures: list[str] = []
    start = datetime.utcnow() - timedelta(days=int(lookback_days_price) + 3)

    for t in tickers_jk:
        try:
            df = yf.download(
                tickers=t,
                period=f"{int(lookback_days_price)}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as e:  # pragma: no cover - defensive network guard
            failures.append(f"{t}: {e}")
            continue

        norm = normalize_price_frame(_normalize_yf_download(df, [t]))
        if norm.empty:
            failures.append(f"{t}: no data returned")
            continue

        norm = norm[norm["date"] >= pd.Timestamp(start)]
        if norm.empty:
            failures.append(f"{t}: no rows within lookback")
            continue

        frames.append(norm)
        src_map[t] = "infosaham"

    combo = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combo, src_map, failures
