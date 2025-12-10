## Setup (Conda)
conda env create -f environment.yml
conda activate stockx
cp .env.example .env
streamlit run app.py

### Usage tips

- Prices are read from `data/price_idx.csv`. Update it offline with:

  ```bash
  python update_prices_yahoo.py --lookback 180
  ```

  The CLI fetches tickers one-by-one with a sleep between calls and stops if Yahoo responds with a rate limit. IDX is used as a best-effort fallback per ticker.
- In the Screener tab choose **From CSV (offline)** to use the cached prices or **Skip prices (sentiment-only)** if no CSV is available.
- If you see many GNews HTTP 429 errors, reduce the "Max tickers for GNews" cap in Advanced mode so fewer requests are sent; the remaining tickers will automatically fall back to GDELT headlines.

### Secrets

- Keep your real `GNEWS_API_KEY` and `X_BEARER_TOKEN` only in the `.env` file.
- The `.env` file is already gitignored so credentials are not committed.
- Never commit production keys directly into the repository history.

## Setup (pip)
python -m venv .venv
# activate venv...
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
