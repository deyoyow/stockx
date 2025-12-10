# update_prices_from_sheets.py
from __future__ import annotations

from pathlib import Path
from typing import List
import time

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ========= CONFIG =========
# From your sheet URL: https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0
SHEET_ID = "1J5mm1AvJInP_C_0rMBNkm5OF_apAJQqzYj8MGkXnnXg"

# Path to your service-account JSON key (relative to repo root)
SERVICE_ACCOUNT_FILE = "secrets/google-sheets-sa.json"

# Tab names at the bottom of the sheet. One tab per ticker.
SHEETS_TICKERS: List[str] = [
    "BBCA","BBRI","BMRI","BBNI","BBTN","BRIS","BNGA","BTPS","MEGA","NISP",
    "TLKM","ISAT","EXCL","MTEL","TBIG","TOWR",
    "ASII","UNVR","ICBP","INDF","MYOR","AMRT","SIDO","KLBF","MAPI","ERAA",
    "ADRO","PTBA","ITMG","HRUM","INDY","MEDC","PGAS","PGEO",
    "ANTM","TINS","INCO","MDKA","NCKL",
    "JSMR","WIKA","PTPP","ADHI","WSKT",
    "BSDE","CTRA","PWON","SMRA","ASRI",
    "GOTO","BUKA","SRTG","BRPT","EMTK",
    "MIKA","SILO","HEAL",
    # add/remove tickers as you create tabs
]

# Output CSV used by your Streamlit app
OUT_PATH = Path("data") / "price_idx.csv"

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Delay between reading each sheet (in seconds) to avoid rate limits
REQUEST_DELAY_SEC = 10.0
# ==========================


def make_client() -> gspread.Client:
    """Authorize a Sheets client using the service account JSON."""
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return gspread.authorize(creds)


def read_one_ticker(sh: gspread.Spreadsheet, ticker_base: str) -> pd.DataFrame:
    """
    Read one ticker tab (e.g. 'HEAL').

    Expects header row: Date | Open | High | Low | Close | Volume
    Returns a DataFrame with: ticker, date, close, volume
    """
    try:
        ws = sh.worksheet(ticker_base)
    except gspread.WorksheetNotFound:
        print(f"  worksheet '{ticker_base}' not found, skipping.")
        return pd.DataFrame()

    # get_all_records() uses first row as header and returns list of dicts
    rows = ws.get_all_records()
    if not rows:
        print("  sheet is empty, skipping.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        print(f"  sheet {ticker_base}: missing date/close columns, skipping.")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    vol_col = "volume" if "volume" in df.columns else None
    if vol_col:
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    if df.empty:
        print("  all rows invalid after cleaning, skipping.")
        return pd.DataFrame()

    # Build output in the exact shape we want
    out = pd.DataFrame()
    out["ticker"] = [f"{ticker_base}.JK"] * len(df)  # <-- IMPORTANT FIX
    out["date"] = df["date"].values
    out["close"] = df["close"].values
    out["volume"] = df[vol_col].values if vol_col else pd.NA

    # Debug: show a sample of tickers so we know the column is filled
    uniq = pd.unique(out["ticker"])
    print(f"  sample ticker values: {uniq[:3]}")

    return out


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = make_client()
    sh = client.open_by_key(SHEET_ID)

    frames: list[pd.DataFrame] = []

    for base in SHEETS_TICKERS:
        print(f"Reading {base} ...")
        try:
            df_one = read_one_ticker(sh, base)
        except Exception as e:
            print(f"  error while reading {base}: {e}")
            df_one = pd.DataFrame()

        if df_one is None or df_one.empty:
            print("  no data")
        else:
            print(f"  {len(df_one)} rows")
            frames.append(df_one)

        time.sleep(REQUEST_DELAY_SEC)

    if not frames:
        print("No data fetched from any sheet; not updating CSV.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    try:
        all_df.to_csv(OUT_PATH, index=False)
    except PermissionError as e:
        tmp = OUT_PATH.with_suffix(".tmp.csv")
        all_df.to_csv(tmp, index=False)
        print(f"Could not overwrite {OUT_PATH}; wrote {tmp} instead. Error: {e}")
        return

    print(
        f"Saved {len(all_df)} rows for {all_df['ticker'].nunique()} tickers "
        f"to {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
