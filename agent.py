from typing import List, Dict, Any
import pandas as pd
from tools import CSVTool, ProfileTool, QueryTool, ChartTool

class CSVAgent:
    """Lightweight agent orchestrating tools and keeping a short memory."""
    def __init__(self):
        self.csv = CSVTool()
        self.profile = ProfileTool(self.csv)
        self.query = QueryTool(self.csv)
        self.chart = ChartTool(self.csv)
        self.memory: List[str] = []

    # Memory helpers
    def _remember(self, role: str, content: str):
        self.memory.append(f"{role.upper()}: {content}")
        if len(self.memory) > 50:  # keep it short
            self.memory = self.memory[-50:]

    # Public API
    def load_csv(self, file_bytes: bytes, filename: str, **kwargs) -> str:
        msg = self.csv.load(file_bytes, filename, **kwargs)
        self._remember("system", msg)
        return msg

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.csv.head(n)

    def profile_schema(self) -> pd.DataFrame:
        return self.profile.schema()

    def profile_stats(self) -> pd.DataFrame:
        return self.profile.stats()

    def profile_missing(self) -> pd.DataFrame:
        return self.profile.missingness()

    def profile_corr(self) -> pd.DataFrame:
        return self.profile.correlations()

    def ask(self, text: str):
        self._remember("user", text)
        if text.strip().lower().startswith("sql:"):
            res = self.query.sql(text.strip()[4:].strip())
        else:
            res = self.query.simple(text)
        self._remember("agent", f"{type(res).__name__}")
        return res

    def chart_data(self, x: str, y: str | None, kind: str):
        return self.chart.data_for_chart(x, y, kind)

    def available_columns(self):
        return self.chart.available_columns()
