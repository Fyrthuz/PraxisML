"""
NLP Profiler — analiza datasets tabulares para detectar columnas de texto libre,
idioma predominante, longitud media, vocabulario estimado y sugiere un pipeline NLP.

Post-hook de `POST /api/v1/datasets/` (guarda el resultado en column_types_analysis).
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_MIN_AVG_WORDS = 3
_MIN_AVG_LENGTH = 15


class NLPProfiler:
    """Detecta el potencial NLP de un dataset tabular."""

    def analyze_df(
        self,
        df: pd.DataFrame,
        column_names: Optional[List[str]] = None,
        sample: int = 2000,
    ) -> Dict[str, Any]:
        cols = column_names or list(df.columns)
        if len(cols) == 0:
            return self._empty()
        df = df[cols].head(sample)

        text_columns: List[str] = []
        numeric_columns: List[str] = []
        categorical_columns: List[str] = []

        for col in cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(str(col))
                continue
            if pd.api.types.is_object_dtype(series):
                samples = series.dropna().astype(str)
                samples = samples[samples.str.strip() != ""]
                if len(samples) == 0:
                    categorical_columns.append(str(col))
                    continue
                avg_len = float(samples.str.len().mean())
                avg_words = float(samples.str.split().str.len().mean())
                if avg_words >= _MIN_AVG_WORDS and avg_len >= _MIN_AVG_LENGTH:
                    text_columns.append(str(col))
                else:
                    categorical_columns.append(str(col))
            else:
                categorical_columns.append(str(col))

        predominant_language = self._detect_language(df, text_columns, sample)
        avg_text_length_chars, avg_tokens_per_text, estimated_vocab_size = (
            self._text_stats(df, text_columns, sample)
        )
        suggested_task, suggested_label_columns = self._suggest(
            df, text_columns, numeric_columns, categorical_columns
        )

        return {
            "text_columns": text_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "predominant_language": predominant_language,
            "avg_text_length_chars": round(avg_text_length_chars, 2),
            "avg_tokens_per_text": round(avg_tokens_per_text, 2),
            "estimated_vocab_size": int(estimated_vocab_size),
            "suggested_task": suggested_task,
            "suggested_algorithm": "distilbert",
            "suggested_label_columns": suggested_label_columns,
        }

    def _detect_language(
        self, df: pd.DataFrame, text_columns: List[str], sample: int
    ) -> Optional[str]:
        if not text_columns:
            return None
        try:
            from langdetect import detect
        except Exception:
            return None

        counts: Dict[str, int] = {}
        for col in text_columns:
            texts = df[col].dropna().astype(str).head(min(sample, 200)).tolist()
            for t in texts:
                try:
                    lang = detect(t)
                    counts[lang] = counts.get(lang, 0) + 1
                except Exception:
                    continue
        if not counts:
            return None
        return max(counts, key=counts.get)

    def _text_stats(
        self, df: pd.DataFrame, text_columns: List[str], sample: int
    ) -> tuple:
        if not text_columns:
            return 0.0, 0.0, 0
        first = df[text_columns[0]].dropna().astype(str)
        first = first[first.str.strip() != ""]
        if len(first) == 0:
            return 0.0, 0.0, 0
        avg_len = float(first.str.len().mean())
        avg_tokens = float(first.str.split().str.len().mean())
        vocab: set = set()
        for t in first.head(min(sample, 500)).tolist():
            vocab.update(t.split())
        return avg_len, avg_tokens, len(vocab)

    def _suggest(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        numeric_columns: List[str],
        categorical_columns: List[str],
    ) -> tuple:
        if not text_columns:
            return None, []

        candidates = numeric_columns + categorical_columns
        for col in candidates:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            n_unique = series.nunique()
            # Etiqueta probable: cardinalidad baja (2-10), no parece un id
            if 2 <= n_unique <= 10 and n_unique < len(series) * 0.5:
                return "sentiment_analysis", [col]

        return "text_classification", candidates[:1]

    def _empty(self) -> Dict[str, Any]:
        return {
            "text_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "predominant_language": None,
            "avg_text_length_chars": 0.0,
            "avg_tokens_per_text": 0.0,
            "estimated_vocab_size": 0,
            "suggested_task": None,
            "suggested_algorithm": "distilbert",
            "suggested_label_columns": [],
        }
