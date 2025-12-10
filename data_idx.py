# data_idx.py
from __future__ import annotations

from typing import List
import pandas as pd
import requests


IDX_ENDPOINT = "https://idx.co.id/umbraco/Surface/ListedCompany/GetTradingInfoSS"


def _fetch_idx_one(code: str, length: int) -> pd.DataFrame:
    """
    Low-level call to the IDX endpoint for a single code (e.g. 'BBRI').

    Returns a DataFrame with at least: Date, Close, Volume
    or an empty frame if nothing usable comes back.
    """
    params = {"code": code, "length": int(length)}

    try:
        r = requests.get(IDX_ENDPOINT, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    replies = data.get("replies") or []
    if not replies:
        return pd.DataFrame()

    df = pd.DataFrame(replies)

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Expect date/close/volume-ish columns
    date_col = "date"
    close_col = "close"
    vol_col = "volume" if "volume" in df.columns else None

    if date_col not in df.columns or close_col not in df.columns:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    keep = [date_col, close_col]
    if vol_col:
        keep.append(vol_col)

    return df[keep].copy()


def fetch_idx_prices(
    tickers_jk: List[str],
    lookback_days_price: int,
) -> pd.DataFrame:
    """
    High-level IDX scraper: accepts a list of 'BBRI.JK' style tickers and
    returns a long-format DataFrame with columns: ticker, date, close[, volume].

    This is meant to feed directly into normalize_price_frame / price_idx.csv.
    """
    tickers_jk = [t.strip() for t in tickers_jk if t and str(t).strip()]
    if not tickers_jk:
        return pd.DataFrame()

    rows = []

    for t in sorted(set(tickers_jk)):
        base = t.split(".")[0].upper()  # 'BBRI.JK' -> 'BBRI'
        df_one = _fetch_idx_one(base, length=int(lookback_days_price))
        if df_one is None or df_one.empty:
            continue

        df_one = df_one.copy()
        df_one["ticker"] = t
        rows.append(df_one)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df.rename(
        columns={
            "date": "date",
            "close": "close",
            "volume": "volume",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df[["ticker", "date", "close"] + (["volume"] if "volume" in df.columns else [])]
