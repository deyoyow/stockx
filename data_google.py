"""Helpers for loading universe tickers from Google Sheets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import gspread
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def _load_credentials(creds_path: str | None, creds_json: str | None) -> Credentials | None:
    """Load service account credentials from a file path or JSON string."""
    if creds_json:
        try:
            data = json.loads(creds_json)
            return Credentials.from_service_account_info(data, scopes=SCOPES)
        except Exception:
            return None

    if creds_path:
        path_obj = Path(creds_path).expanduser()
        if path_obj.exists():
            try:
                return Credentials.from_service_account_file(str(path_obj), scopes=SCOPES)
            except Exception:
                return None
    return None


def load_sheet_universes(
    sheet_id: str,
    creds_path: str | None = None,
    creds_json: str | None = None,
) -> Tuple[Dict[str, List[str]], str | None]:
    """
    Load all worksheets within a Google Sheet as named universes.

    Returns a mapping of worksheet title -> list of tickers (uppercased, first column)
    and an optional error message.
    """
    sheet_id = (sheet_id or "").strip()
    if not sheet_id:
        return {}, "GOOGLE_SHEET_ID is not configured."

    creds = _load_credentials(creds_path, creds_json)
    if creds is None:
        return {}, "Google credentials missing or invalid. Set GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE."

    try:
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)
    except Exception as exc:  # pragma: no cover - network
        return {}, f"Failed to open Google Sheet: {exc}"

    universes: Dict[str, List[str]] = {}

    try:
        worksheets = sh.worksheets()
    except Exception as exc:  # pragma: no cover - network
        return {}, f"Failed to list worksheets: {exc}"

    for ws in worksheets:
        try:
            col_vals = ws.col_values(1)
        except Exception:
            col_vals = []
        tickers = [v.strip().upper() for v in col_vals if v and isinstance(v, str)]
        tickers = [t.replace(".JK", "") for t in tickers]
        if tickers:
            universes[ws.title] = tickers

    if not universes:
        return {}, "No tickers found in the Google Sheet."
    return universes, None
