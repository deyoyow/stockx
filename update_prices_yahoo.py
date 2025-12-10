"""CLI to fetch IDX prices politely via yfinance with IDX fallback.

Usage:
    python update_prices_yahoo.py [--lookback 120]

This script pulls the watchlist from config.WATCHLIST (no .JK suffix), fetches
prices sequentially with small sleeps, and writes a normalized CSV to
`data/price_idx.csv`. If Yahoo Finance signals a rate limit, the script stops
immediately and leaves any existing CSV untouched.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yfinance as yf

import config
from data_idx import fetch_idx_prices_for_one
from data_prices import normalize_price_frame

DATA_PATH = Path(__file__).resolve().parent / "data" / "price_idx.csv"


def _is_rate_limit_error(msg: str) -> bool:
    msg_low = str(msg).lower()
    return "rate limit" in msg_low or "too many requests" in msg_low or "429" in msg_low


def _fetch_one_yahoo(ticker_jk: str, lookback_days: int) -> Tuple[pd.DataFrame, str | None]:
    try:
        df = yf.download(
            tickers=ticker_jk,
            period=f"{int(lookback_days)}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:  # pragma: no cover - network guard
        return pd.DataFrame(), str(e)

    if df is None or df.empty:
        return pd.DataFrame(), "empty"

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    if "date" not in df.columns:
        return pd.DataFrame(), "missing date"

    df["ticker"] = ticker_jk
    return df, None


def update_prices(lookback_days: int) -> None:
    tickers = [t.strip().upper() for t in config.WATCHLIST if t and str(t).strip()]
    tickers_jk = [f"{t}.JK" for t in tickers]

    frames: List[pd.DataFrame] = []
    failed: List[str] = []
    rate_limited = False

    for idx, t in enumerate(tickers_jk, start=1):
        print(f"[{idx}/{len(tickers_jk)}] Fetching {t} from Yahoo...")
        df_raw, err = _fetch_one_yahoo(t, lookback_days)

        if err and _is_rate_limit_error(err):
            print("Detected Yahoo Finance rate limit; stopping downloads. Existing CSV is left unchanged.")
            rate_limited = True
            break

        if err or df_raw.empty:
            if err and err != "empty":
                print(f"  Yahoo error: {err}")
            print(f"  Trying IDX fallback for {t}...")
            df_raw = fetch_idx_prices_for_one(t, lookback_days)

        if df_raw is None or df_raw.empty:
            failed.append(t)
        else:
            frames.append(df_raw)

        time.sleep(1.5)

    if rate_limited:
        return

    if not frames:
        print("No price data fetched; CSV not updated.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = normalize_price_frame(combined)
    if combined is None or combined.empty:
        print("Fetched data could not be normalized; CSV not updated.")
        return

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(combined)} rows to {DATA_PATH}.")

    if failed:
        print("No data for: " + ", ".join(failed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Update price_idx.csv using Yahoo Finance with IDX fallback.")
    parser.add_argument("--lookback", type=int, default=120, help="Number of days to fetch (default: 120)")
    args = parser.parse_args()
    update_prices(args.lookback)


if __name__ == "__main__":
    main()
