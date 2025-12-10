# Agent Guide for the `stockx` Repository

This document defines how coding agents should work with this project.

The goal of this repo is an **IDX stock sentiment dashboard** built with **Streamlit**, combining:
- News sentiment (GNews + GDELT)
- X/Twitter sentiment
- Simple price-based momentum
- Optional portfolio P&L

The app must remain **robust**, **API-friendly**, and **simple to run** for the user.

---

## 1. Tech Stack & Entry Points

- **Language**: Python
- **App framework**: Streamlit
- **Key entry points**:
  - `app.py` – main Streamlit app
  - `update_prices_yahoo.py` (to be created/maintained) – CLI script for updating prices
- **Important modules**:
  - `config.py` – configuration (watchlist, defaults, etc.)
  - `data_prices.py` – price loaders, CSV helpers, normalization
  - `data_yahoo.py` – Yahoo Finance utilities
  - `data_idx.py` – IDX price scraping (backup only)
  - `data_gnews.py`, `data_gdelt.py` – news data
  - `data_x.py` – X/Twitter data
  - `sentiment.py`, `scoring.py` – NLP / scoring pipeline
  - `cache_disk.py` – disk-based caching
  - `utils.py` – miscellaneous helpers
- **Data directory**:
  - `data/price_idx.csv` – canonical price cache for IDX stocks
  - `data/portfolio.csv` – user portfolio (if present)

**Do not commit secrets**. Environment variables and keys live in `.env` / `.env.example`.

---

## 2. Critical Design Rules (Must Follow)

### 2.1 Price Data Architecture

1. **Single source of truth for prices**  
   - All price-based features in the app must read from **`data/price_idx.csv`**.
   - This CSV is long-format:

     ```text
     ticker   (e.g. BBCA.JK)
     date     (datetime / YYYY-MM-DD)
     close    (float)
     volume   (float or NaN)
     ```

2. **No direct yfinance calls inside Streamlit**  
   - `app.py` (and anything it imports for the main run) **must not** call `yfinance.download` or similar.
   - Streamlit reruns would otherwise hammer Yahoo and trigger rate-limits.
   - All Yahoo calls must be in a **separate CLI script** (see 2.2).

3. **Offline price updater script (CLI)**  
   - Maintain a script (name: `update_prices_yahoo.py`) that:
     - Uses a defined universe (e.g. `config.WATCHLIST`) of IDX tickers (without `.JK`).
     - For each ticker:
       - Try `yfinance.download` (single ticker at a time).
       - Sleep 1–2 seconds between tickers.
       - Detect rate-limit (`Too Many Requests`, `rate limit`, `429`, etc.). On first detection, **stop** further downloads and leave existing CSV untouched.
       - If Yahoo returns an empty frame or fails for a ticker, optionally try IDX backup (see 2.3).
     - Writes / updates `data/price_idx.csv` as long-format with `ticker,date,close,volume`.

4. **Streamlit price modes**

   In the Screener tab of `app.py`, price choices should be:

   - `"From CSV (offline)"`  
     - Load prices via `load_prices_from_csv(PRICE_CSV_PATH, tickers_jk, lookback_days_price)`.
     - If CSV missing or contains no usable rows for the chosen tickers/lookback:
       - Show a clear warning.
       - Fall back to **sentiment-only** mode (momentum weight = 0).
   - `"Skip prices (sentiment-only)"`  
     - Do not attempt to load any price data.
     - Build a minimal `feat_df` with NaNs for price fields and `w_mom = 0`.

   **Do not** add live “fetch from yfinance” options back into the Streamlit UI.

5. **Portfolio tab price usage**

   - Portfolio calculations **must** use `data/price_idx.csv` (no live API).
   - Load latest closes via a helper (`load_latest_prices` or equivalent) that reads from the CSV only.
   - If prices for some tickers are missing, handle gracefully (NaN P&L, warning).

---

### 2.2 Using Yahoo Finance (yfinance)

- Allowed **only** in:
  - `update_prices_yahoo.py`
  - Low-level helpers in `data_yahoo.py` that are called from that script.

- When implementing or updating this script:
  - Use **single-ticker** calls per loop, not large multi-ticker batches.
  - Add a delay between calls (`time.sleep(1.0–2.0)`).
  - If rate-limit is detected, stop immediately and log a short message.
  - Do not loop endlessly on errors.

---

### 2.3 IDX Scraper (Backup Only)

- `data_idx.py` wraps IDX endpoints (inspired by `antonizer/IDX-Scrapper`).
- Rules:
  - Use simple HTTP client (`requests` or `cloudscraper`).
  - For each ticker, return a DataFrame with at least `date,close,volume`.
  - If the endpoint returns non-JSON or does not include the expected fields:
    - Return **empty DataFrame**, not an exception.
  - **Do not** implement aggressive anti-bot or Cloudflare bypassing.

- IDX should be used **only as a fallback**:
  - Inside `update_prices_yahoo.py`, when Yahoo fails for a specific ticker.
  - Not as a default global live source in the Streamlit app.

---

## 3. Streamlit App Guidelines

1. **Main file**: `app.py`
2. **Tabs**:
   - Screener (main sentiment + ranking)
   - Prices / CSV preview (optional; must not call yfinance)
   - Portfolio / P&L

3. **Sentiment pipeline** (GNews, GDELT, X):
   - Preserve the current design:
     - GNews primary, GDELT fallback.
     - X sentiment and tweet listing behind appropriate toggles / conditions.
   - Handle missing API keys gracefully:
     - If X token is missing, disable X calls and show short explanation.

4. **Rerun-safe design**:
   - Avoid expensive operations on every rerun (no API loops, no large downloads).
   - Use `DiskCache` for any network data that still must be fetched at runtime (news/X), respecting TTLs defined in `config.py`.

5. **Deprecations**:
   - Replace all `use_container_width` usages with `width="stretch"` (or `"content"`), as per Streamlit’s current guidance.
   - Clean up any other Streamlit deprecations if they appear.

---

## 4. Coding Style & Dependency Rules

1. **General style**
   - Keep code readable and modular.
   - Prefer pure functions and small helpers over inline complex logic.
   - Add type hints where practical (`-> pd.DataFrame`, `-> dict[str, str]`, etc.).
   - Maintain or add docstrings that explain **purpose** and **inputs/outputs**.

2. **Dependencies**
   - Prefer using existing dependencies from `requirements.txt`.
   - If you add a new dependency:
     - Add it to `requirements.txt`.
     - Justify its use (e.g. in a short comment in the relevant module).

3. **Error handling**
   - For external services (Yahoo, IDX, GNews, X):
     - Catch exceptions and convert them into:
       - Empty DataFrames, or
       - Short, clear messages in the UI (via `st.warning` / `st.error`) rather than stack traces.
   - Do not silently swallow unexpected exceptions without at least logging or commenting.

4. **Caching**
   - Respect existing cache design in `cache_disk.py`.
   - Cache keys should be stable and reproducible (avoid including random values or timestamps unnecessarily).

---

## 5. What Not to Do

- Do **not**:
  - Add live yfinance calls inside Streamlit rerun paths.
  - Bypass Cloudflare or similar protections aggressively for IDX.
  - Hard-code secrets, API keys, or tokens in the code.
  - Break existing tab structure without a clear reason.
  - Create circular imports between core modules (`app.py`, `data_prices.py`, `data_yahoo.py`, `data_idx.py`, etc.).

---

## 6. How to Run (for reference)

- **App**:

  ```bash
  streamlit run app.py
