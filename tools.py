import io
import re
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pandasql import sqldf

class CSVTool:
    """Loads and holds a DataFrame from uploaded CSV bytes."""
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.name: str = ""

    def load(self, file_bytes: bytes, filename: str, **read_csv_kwargs) -> str:
        buf = io.BytesIO(file_bytes)
        df = pd.read_csv(buf, **read_csv_kwargs)
        self.df = df
        self.name = filename
        return f"Loaded '{filename}' with {len(df)} rows × {len(df.columns)} columns."

    def available(self) -> bool:
        return self.df is not None

    def head(self, n: int = 5) -> pd.DataFrame:
        if self.df is None: raise ValueError("No CSV loaded.")
        return self.df.head(n)

class ProfileTool:
    """Generates schema, missingness, stats, and correlations."""
    def __init__(self, csv_tool: CSVTool):
        self.csv = csv_tool

    def schema(self) -> pd.DataFrame:
        if not self.csv.available(): raise ValueError("No CSV loaded.")
        df = self.csv.df
        return pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": df.notnull().sum().values,
            "nulls": df.isnull().sum().values,
            "unique": [df[c].nunique(dropna=True) for c in df.columns],
        })

    def stats(self) -> pd.DataFrame:
        if not self.csv.available(): raise ValueError("No CSV loaded.")
        return self.csv.df.describe(include="all").T

    def missingness(self) -> pd.DataFrame:
        if not self.csv.available(): raise ValueError("No CSV loaded.")
        miss = self.csv.df.isnull().mean().reset_index()
        miss.columns = ["column", "missing_ratio"]
        return miss.sort_values("missing_ratio", ascending=False)

    def correlations(self) -> pd.DataFrame:
        if not self.csv.available(): raise ValueError("No CSV loaded.")
        num = self.csv.df.select_dtypes(include=[np.number])
        if num.empty:
            return pd.DataFrame(columns=["note"], data=[["No numeric columns for correlation"]])
        return num.corr(numeric_only=True)

class QueryTool:
    """
    Answers via:
      - Simple NL patterns (avg/sum/max/min/count, unique values, filter + top n)
      - SQL if the prompt starts with 'sql:'
    """
    SIMPLE_PATTERNS = {
        r"^average of ([\w\-\s]+)$": "avg",
        r"^mean of ([\w\-\s]+)$": "avg",
        r"^sum of ([\w\-\s]+)$": "sum",
        r"^max of ([\w\-\s]+)$": "max",
        r"^min of ([\w\-\s]+)$": "min",
        r"^count rows$": "count_rows",
        r"^unique values of ([\w\-\s]+)$": "unique",
    }

    FILTER_PATTERN = r"^filter ([\w\-\s]+)\s*(==|=|>=|<=|>|<|!=)\s*([^\s]+)\s*and show top\s*(\d+)$"

    def __init__(self, csv_tool: CSVTool):
        self.csv = csv_tool

    def _coerce_value(self, v: str):
        # Try int, float, bool, else strip quotes
        for caster in (int, float):
            try: return caster(v)
            except: pass
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v.strip("'\"")

    def sql(self, query: str) -> pd.DataFrame:
        if not self.csv.available(): raise ValueError("No CSV loaded.")
        df = self.csv.df
        return sqldf(query, {"df": df})

    def simple(self, prompt: str) -> Any:
        if not self.csv.available(): raise ValueError("No CSV loaded.")
        df = self.csv.df
        p = prompt.strip().lower()

        # Filter pattern
        m = re.match(self.FILTER_PATTERN, p)
        if m:
            col, op, val, topn = m.groups()
            col = col.strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            val = self._coerce_value(val)
            topn = int(topn)
            ops = {
                "==": lambda s, x: s == x, "=": lambda s, x: s == x,
                ">=": lambda s, x: s >= x, "<=": lambda s, x: s <= x,
                ">": lambda s, x: s > x, "<": lambda s, x: s < x,
                "!=": lambda s, x: s != x,
            }
            mask = ops[op](df[col], val)
            return df[mask].head(topn)

        # Simple aggregations
        for pat, kind in self.SIMPLE_PATTERNS.items():
            m = re.match(pat, p)
            if m:
                if kind == "count_rows":
                    return int(len(df))
                col = m.group(1).strip()
                if col not in df.columns:
                    return f"Column '{col}' not found."
                if kind == "avg":
                    return float(df[col].astype(float).mean())
                if kind == "sum":
                    return float(df[col].astype(float).sum())
                if kind == "max":
                    return df[col].max()
                if kind == "min":
                    return df[col].min()
                if kind == "unique":
                    return df[col].dropna().unique().tolist()

        return "Unrecognized pattern. Try: 'average of <col>', 'sum of <col>', 'count rows', 'unique values of <col>', or 'filter <col> > 10 and show top 5'. To use SQL, start with 'sql:'."

class ChartTool:
    """Returns instructions for plotting; actual plotting handled in Streamlit."""
    def __init__(self, csv_tool: CSVTool):
        self.csv = csv_tool

    def available_columns(self):
        if not self.csv.available(): return []
        return list(self.csv.df.columns)

    def data_for_chart(self, x: str, y: Optional[str] = None, kind: str = "line"):
        if not self.csv.available(): raise ValueError("No CSV loaded.")
        df = self.csv.df
        if x not in df.columns:
            raise ValueError(f"Column '{x}' not found.")
        if y is not None and y not in df.columns:
            raise ValueError(f"Column '{y}' not found.")
        return df[[c for c in [x, y] if c is not None]]
