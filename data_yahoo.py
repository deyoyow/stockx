# data_yahoo.py
from __future__ import annotations
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

def fetch_prices_fast(
    tickers_jk: List[str],
    lookback_days_price: int,
    cache: DiskCache,
    ttl_seconds: int,

    chunk_size: int = 8,
    ) -> Tuple[pd.DataFrame, Dict[str, str], list[str]]:
    """yfinance-only price fetch with cache and explicit source map.

    The fetch is chunked to reduce the likelihood of Yahoo rate limits. We
    return a best-effort dataframe, a source map, and a list of error notes
    describing any symbols that failed.
    """

    tickers_jk = sorted(set([t.strip() for t in tickers_jk if t and str(t).strip()]))
    if not tickers_jk:
        return pd.DataFrame(), {}, []

    key = f"fast|days={int(lookback_days_price)}|tickers={' '.join(tickers_jk)}|chunk={int(chunk_size)}"
    cached = cache.get("prices", key, ttl_seconds=ttl_seconds)
    if isinstance(cached, dict) and {"df", "src", "errors"}.issubset(cached.keys()):
        df_cached = cached["df"]
        src_cached = cached["src"]
        err_cached = cached.get("errors", []) or []
        if isinstance(df_cached, pd.DataFrame) and isinstance(src_cached, dict):
            return df_cached, src_cached, err_cached

    frames: list[pd.DataFrame] = []
    got: set[str] = set()
    errors: list[str] = []

    def _chunks(seq: list[str], size: int) -> list[list[str]]:
        size = max(1, int(size))
        return [seq[i : i + size] for i in range(0, len(seq), size)]

    for chunk in _chunks(tickers_jk, chunk_size):
        try:
            df = yf.download(
                tickers=" ".join(chunk),
                period=f"{int(lookback_days_price)}d",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=False,
            )
        except Exception as e:  # pragma: no cover - network guard
            errors.append(f"{', '.join(chunk)}: {e}")
            continue

        norm = _normalize_yf_download(df, chunk)
        if norm is None or norm.empty:
            errors.append(f"{', '.join(chunk)}: no data returned")
            continue

        frames.append(norm)
        got.update(norm["ticker"].dropna().astype(str).unique().tolist())

    if not frames and len(tickers_jk) > 1:
        # Fallback to serial fetch per ticker to soften rate limits
        for t in tickers_jk:
            try:
                df_one = yf.download(
                    tickers=t,
                    period=f"{int(lookback_days_price)}d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception as e:  # pragma: no cover
                errors.append(f"{t}: {e}")
                continue

            norm_one = _normalize_yf_download(df_one, [t])
            if norm_one is None or norm_one.empty:
                errors.append(f"{t}: no data returned")
                continue

            frames.append(norm_one)
            got.update(norm_one["ticker"].dropna().astype(str).unique().tolist())

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    src_map = {t: ("yahoo" if t in got else "none") for t in tickers_jk}

    cache.set("prices", key, {"df": out, "src": src_map, "errors": errors})
    return out, src_map, errors
