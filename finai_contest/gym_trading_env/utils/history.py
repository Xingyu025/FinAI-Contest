# finai_contest/gym_trading_env/utils/history.py
from typing import List, Dict, Any
import collections

class History:
    def __init__(self, max_size: int = 10_000):
        self._max_size = max_size
        self._rows: List[Dict[str, Any]] = []
        self._columns: List[str] = []

    @property
    def columns(self):
        return list(self._columns)

    def _ensure_columns(self, keys):
        for k in keys:
            if k not in self._columns:
                self._columns.append(k)
        # maintain columns order stable

    def set(self, **kwargs):
        """Set the initial (index 0) row. Overwrites existing history."""
        self._rows = []
        self._ensure_columns(kwargs.keys())
        self._rows.append(dict(kwargs))

    def add(self, **kwargs):
        """Append a row; missing keys will be filled with None."""
        self._ensure_columns(kwargs.keys())
        row = {k: kwargs.get(k, None) for k in self._columns}
        # fill provided keys
        row.update(kwargs)
        self._rows.append(row)
        # optionally trim
        if len(self._rows) > self._max_size:
            self._rows.pop(0)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        # tuple access: ("col", idx)
        if isinstance(key, tuple) and len(key) == 2:
            col, idx = key
            idx = idx if idx >= 0 else len(self._rows) + idx
            return self._rows[idx].get(col)
        # integer row access: history[0] -> dict (row)
        if isinstance(key, int):
            idx = key if key >= 0 else len(self._rows) + key
            return self._rows[idx]
        # list of columns -> list of rows (each row is list of column values)
        if isinstance(key, (list, tuple)):
            cols = list(key)
            return [[row.get(c) for c in cols] for row in self._rows]
        # fallback: allow string to return full column as list
        if isinstance(key, str):
            return [row.get(key) for row in self._rows]
        raise KeyError(f"Unsupported key type: {type(key)}")