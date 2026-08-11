"""
Tests del NLPProfiler: detección de columnas, idioma, longitud y sugerencias.
"""

import pandas as pd

from app.services.nlp_profiler import NLPProfiler


def _df_textual():
    return pd.DataFrame({
        "review": [
            "This product is absolutely amazing, I love it so much",
            "Terrible experience, would never buy this again",
            "Excellent quality and very fast shipping, highly recommend",
            "Waste of money, completely disappointed with the quality",
            "Great value for the price, works as described perfectly",
        ] * 4,
        "rating": [5, 1, 5, 1, 4] * 4,
        "product_id": ["A1", "B2", "C3", "A1", "B2"] * 4,
        "price": [10.5, 20.0, 15.0, 9.0, 12.0] * 4,
    })


def test_detects_text_numeric_categorical():
    result = NLPProfiler().analyze_df(_df_textual())
    assert "review" in result["text_columns"]
    assert "price" in result["numeric_columns"]
    assert result["suggested_task"] == "sentiment_analysis"
    assert result["suggested_label_columns"] == ["rating"]
    assert result["suggested_algorithm"] == "distilbert"


def test_avg_text_stats():
    result = NLPProfiler().analyze_df(_df_textual())
    assert result["avg_tokens_per_text"] >= 3
    assert result["avg_text_length_chars"] >= 15
    assert result["estimated_vocab_size"] > 0


def test_empty_df():
    result = NLPProfiler().analyze_df(pd.DataFrame({"a": []}))
    assert result["text_columns"] == []
    assert result["suggested_task"] is None


def test_short_texts_are_categorical():
    df = pd.DataFrame({"code": ["AA", "BB", "CC", "AA", "BB"]})
    result = NLPProfiler().analyze_df(df)
    assert result["text_columns"] == []
    assert "code" in result["categorical_columns"]
