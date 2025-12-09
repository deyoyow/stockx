# sentiment.py
from __future__ import annotations

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_analyzer = SentimentIntensityAnalyzer()


def score_english_texts(texts: list[str]) -> pd.DataFrame:
    """
    Returns DataFrame with columns: compound,pos,neu,neg
    VADER is English-optimized and lightweight.
    """
    rows = []
    for t in texts:
        s = str(t or "").strip()
        if not s:
            rows.append({"compound": np.nan, "pos": np.nan, "neu": np.nan, "neg": np.nan})
            continue
        r = _analyzer.polarity_scores(s)
        rows.append({"compound": r["compound"], "pos": r["pos"], "neu": r["neu"], "neg": r["neg"]})
    return pd.DataFrame(rows)


def aggregate_sentiment(df: pd.DataFrame, text_col: str) -> tuple[float, int]:
    """
    Returns (mean_compound, count_used)
    """
    if df is None or df.empty or text_col not in df.columns:
        return (float("nan"), 0)

    texts = df[text_col].fillna("").astype(str).tolist()
    s = score_english_texts(texts)
    comp = pd.to_numeric(s["compound"], errors="coerce")
    comp = comp.dropna()
    if comp.empty:
        return (float("nan"), 0)
    return (float(comp.mean()), int(comp.shape[0]))
