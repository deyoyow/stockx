## Setup (Conda)
conda env create -f environment.yml
conda activate stockx
cp .env.example .env
streamlit run app.py

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
