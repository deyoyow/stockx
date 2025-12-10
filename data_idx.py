# data_idx.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

IDX_ENDPOINT = "https://www.idx.co.id/umbraco/Surface/TradingInfo/GetTradingInfoSS"


def fetch_idx_prices_for_one(ticker_jk: str, lookback_days: int) -> pd.DataFrame:
    """Best-effort IDX fetch for a single ticker.

    Returns a long dataframe with columns ticker,date,close,volume. If the IDX
    endpoint does not respond with JSON or lacks the expected fields, an empty
    dataframe is returned instead of raising.
    """

    ticker_base = (ticker_jk or "").upper().replace(".JK", "").strip()
    if not ticker_base:
        return pd.DataFrame()

    try:
        resp = requests.get(IDX_ENDPOINT, params={"code": ticker_base}, timeout=15)
    except Exception:
        return pd.DataFrame()

    if resp.status_code != 200:
        return pd.DataFrame()

    try:
        payload = resp.json()
    except Exception:
        return pd.DataFrame()

    replies = payload.get("replies") if isinstance(payload, dict) else None
    if not isinstance(replies, list):
        return pd.DataFrame()

    rows = []
    start_dt = datetime.utcnow() - timedelta(days=int(lookback_days) + 3)
    for item in replies:
        if not isinstance(item, dict):
            continue
        dt_raw: Optional[str] = item.get("Date") or item.get("date")
        close = item.get("Close") or item.get("close")
        volume = item.get("Volume") if "Volume" in item else item.get("volume")

        try:
            dt_parsed = pd.to_datetime(dt_raw, errors="coerce")
        except Exception:
            dt_parsed = pd.NaT

        if pd.isna(dt_parsed) or dt_parsed < start_dt:
            continue

        rows.append(
            {
                "ticker": f"{ticker_base}.JK",
                "date": dt_parsed,
                "close": close,
                "volume": volume,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df = df.dropna(subset=["close", "date"])
    return df.sort_values("date").reset_index(drop=True)
