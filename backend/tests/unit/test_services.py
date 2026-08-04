"""Tests unitarios para servicios compartidos."""

from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestModelCache:
    def test_model_cache_singleton(self):
        from app.services.model_cache import get_model_cache

        c1 = get_model_cache()
        c2 = get_model_cache()
        assert c1 is c2

    def test_model_cache_set_get(self):
        from app.services.model_cache import get_model_cache

        cache = get_model_cache()
        mock_model = MagicMock()
        cache.set("test-key-unique", mock_model, ttl=60)
        result = cache.get("test-key-unique")
        assert result is mock_model

    def test_model_cache_get_missing(self):
        from app.services.model_cache import get_model_cache

        cache = get_model_cache()
        result = cache.get("nonexistent-key-xyz")
        assert result is None

    def test_model_cache_stats(self):
        from app.services.model_cache import get_model_cache

        cache = get_model_cache()
        stats = cache.get_stats()
        assert isinstance(stats, dict)


class TestBackgroundCache:
    def test_background_cache_singleton(self):
        from app.services.background_cache import get_background_cache

        c1 = get_background_cache()
        c2 = get_background_cache()
        assert c1 is c2

    def test_background_cache_set_get(self):
        from app.services.background_cache import get_background_cache

        cache = get_background_cache()
        mock_data = MagicMock()
        cache.set("bg-test-key-unique", mock_data, ttl=60)
        result = cache.get("bg-test-key-unique")
        assert result is mock_data

    def test_background_cache_get_missing(self):
        from app.services.background_cache import get_background_cache

        cache = get_background_cache()
        result = cache.get("nonexistent-bg-key-xyz")
        assert result is None


class TestDataProfiler:
    def test_detect_column_types_numeric(self):
        from app.services.data_profiler import detect_column_types

        df = pd.DataFrame({"num_col": [1, 2, 3], "cat_col": ["a", "b", "c"]})
        types = detect_column_types(df)
        assert types["num_col"] == "numeric"
        assert types["cat_col"] == "categorical"

    def test_detect_column_types_datetime(self):
        from app.services.data_profiler import detect_column_types

        df = pd.DataFrame({"date_col": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        types = detect_column_types(df)
        assert types["date_col"] == "datetime"

    def test_profile_dataset_basic(self):
        from app.services.data_profiler import profile_dataset

        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
        })
        result = profile_dataset(df)
        assert result is not None
        assert isinstance(result, dict)
