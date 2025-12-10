## Setup (Conda)
conda env create -f environment.yml
conda activate stockx
cp .env.example .env
streamlit run app.py

### Usage tips

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
