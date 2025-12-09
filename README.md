## Setup (Conda)
conda env create -f environment.yml
conda activate stockx
cp .env.example .env
streamlit run app.py

## Setup (pip)
python -m venv .venv
# activate venv...
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
