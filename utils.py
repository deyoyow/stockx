# utils.py
from __future__ import annotations
import numpy as np
import pandas as pd

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

def dedup_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out

def quote_phrase(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if " " in s and not (s.startswith('"') and s.endswith('"')):
        return f'"{s}"'
    return s

def is_all_caps_acronym(s: str) -> bool:
    s = s.strip()
    return (3 <= len(s) <= 6) and s.isupper() and s.isalpha()
