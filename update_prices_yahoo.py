"""CLI to fetch IDX prices politely via yfinance with IDX fallback.

Usage:
    python update_prices_yahoo.py [--lookback 120]

This script pulls the watchlist from Google Sheets (all worksheet tabs) when
configured, falling back to config.WATCHLIST (no .JK suffix). It fetches prices
sequentially with small sleeps, appends new rows into `data/price_idx.csv`
without duplicates, and rotates out an archive file if the CSV grows beyond
10MB. If Yahoo Finance signals a rate limit, the script stops immediately and
leaves any existing CSV untouched.
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
from data_google import load_sheet_universes
from data_prices import normalize_price_frame

DATA_PATH = Path(__file__).resolve().parent / "data" / "price_idx.csv"
MAX_BYTES = 10 * 1024 * 1024


def _is_rate_limit_error(msg: str) -> bool:
    msg_low = str(msg).lower()
    return "rate limit" in msg_low or "too many requests" in msg_low or "429" in msg_low


def _load_watchlist_from_sheet() -> Tuple[List[str], str | None]:
    universes, err = load_sheet_universes(
        sheet_id=config.GOOGLE_SHEET_ID,
        creds_path=config.GOOGLE_SERVICE_ACCOUNT_FILE,
        creds_json=config.GOOGLE_SERVICE_ACCOUNT_JSON,
    )
    if err or not universes:
        return [], err or None

    tickers: List[str] = []
    for vals in universes.values():
        tickers.extend([v.strip().upper() for v in vals if v])
    tickers = sorted(set(tickers))
    return tickers, None


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
    sheet_tickers, sheet_err = _load_watchlist_from_sheet()
    if sheet_tickers:
        print(f"Loaded {len(sheet_tickers)} tickers from Google Sheet across all tabs.")
    elif sheet_err:
        print(f"Sheet watchlist unavailable: {sheet_err}. Falling back to config.WATCHLIST")

    tickers = sheet_tickers or [t.strip().upper() for t in config.WATCHLIST if t and str(t).strip()]
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

    existing = pd.DataFrame()
    if DATA_PATH.exists():
        try:
            existing = pd.read_csv(DATA_PATH)
            existing = normalize_price_frame(existing)
        except Exception:
            existing = pd.DataFrame()

    merged = pd.concat([existing, combined], ignore_index=True)
    merged = normalize_price_frame(merged)
    if merged is None or merged.empty:
        print("Merged data unusable; CSV not updated.")
        return

    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="last")
    merged.sort_values(["ticker", "date"], inplace=True)

    # Size-aware rotation
    csv_bytes = len(merged.to_csv(index=False).encode("utf-8"))
    if csv_bytes > MAX_BYTES:
        archive_path = DATA_PATH.with_name(
            f"{DATA_PATH.stem}_{pd.Timestamp.utcnow():%Y%m%d%H%M%S}{DATA_PATH.suffix}"
        )
        merged.to_csv(archive_path, index=False)
        print(f"Archive written to {archive_path} (full dataset, {csv_bytes/1_048_576:.2f} MB).")

        # Trim oldest rows until under limit for the working CSV
        trimmed = merged.copy()
        while len(trimmed) > 0 and len(trimmed.to_csv(index=False).encode("utf-8")) > MAX_BYTES:
            cutoff = max(1, int(len(trimmed) * 0.9))
            trimmed = trimmed.iloc[len(trimmed) - cutoff :]

        trimmed.to_csv(DATA_PATH, index=False)
        print(f"Working price_idx.csv trimmed to {len(trimmed)} rows to stay under 10MB.")
    else:
        merged.to_csv(DATA_PATH, index=False)
        print(f"Wrote {len(merged)} rows to {DATA_PATH}.")

    if failed:
        print("No data for: " + ", ".join(failed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Update price_idx.csv using Yahoo Finance with IDX fallback.")
    parser.add_argument("--lookback", type=int, default=120, help="Number of days to fetch (default: 120)")
    args = parser.parse_args()
    update_prices(args.lookback)


if __name__ == "__main__":
    main()
