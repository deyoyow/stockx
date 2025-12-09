# cache_disk.py
from __future__ import annotations
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
import pandas as pd

class DiskCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        ns = self.root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return ns / f"{h}.pkl"

    def get(self, namespace: str, key: str, ttl_seconds: int) -> Optional[Any]:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > ttl_seconds:
            return None
        try:
            return pd.read_pickle(p)
        except Exception:
            return None

    def set(self, namespace: str, key: str, obj: Any) -> None:
        p = self._path(namespace, key)
        try:
            pd.to_pickle(obj, p)
        except Exception:
            pass

    def clear(self) -> int:
        n = 0
        try:
            for p in self.root.rglob("*.pkl"):
                p.unlink(missing_ok=True)
                n += 1
        except Exception:
            pass
        return n
