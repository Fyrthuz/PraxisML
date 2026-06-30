import numpy as np
import pandas as pd
import pytest


class TestDetectColumnTypes:
    def test_detect_numeric(self):
        from app.services.data_profiler import detect_column_types

        df = pd.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]})
        types = detect_column_types(df)
        assert types["age"] == "numeric"
        assert types["income"] == "numeric"

    def test_detect_categorical(self):
        from app.services.data_profiler import detect_column_types

        df = pd.DataFrame({"city": ["NYC", "LA", "SF"], "country": ["USA", "USA", "USA"]})
        types = detect_column_types(df)
        assert types["city"] == "categorical"
        assert types["country"] == "categorical"

    def test_detect_datetime(self):
        from app.services.data_profiler import detect_column_types

        df = pd.DataFrame({"date": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"])})
        types = detect_column_types(df)
        assert types["date"] == "datetime"

    def test_detect_mixed_types(self):
        from app.services.data_profiler import detect_column_types

        df = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "SF"],
                "joined": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
            }
        )
        types = detect_column_types(df)
        assert types["age"] == "numeric"
        assert types["city"] == "categorical"
        assert types["joined"] == "datetime"


class TestProfileDataset:
    def test_profile_numeric_column(self):
        from app.services.data_profiler import profile_dataset

        df = pd.DataFrame({"age": [25, 30, 35, 40, 45]})
        profile = profile_dataset(df)
        assert "age" in profile
        col = profile["age"]
        assert col["type"] == "numeric"
        assert col["mean"] == 35.0
        assert col["min"] == 25.0
        assert col["max"] == 45.0
        assert col["null_count"] == 0
        assert col["distinct_count"] == 5

    def test_profile_categorical_column(self):
        from app.services.data_profiler import profile_dataset

        df = pd.DataFrame({"city": ["NYC", "LA", "NYC", "SF", "LA"]})
        profile = profile_dataset(df)
        assert "city" in profile
        col = profile["city"]
        assert col["type"] == "categorical"
        assert col["distinct_count"] == 3
        assert len(col["top_values"]) <= 5

    def test_profile_with_nulls(self):
        from app.services.data_profiler import profile_dataset

        df = pd.DataFrame({"age": [25, None, 35, None, 45]})
        profile = profile_dataset(df)
        assert profile["age"]["null_count"] == 2

    def test_profile_empty_dataframe(self):
        from app.services.data_profiler import profile_dataset

        df = pd.DataFrame()
        profile = profile_dataset(df)
        assert profile == {}

    def test_profile_returns_column_keys(self):
        from app.services.data_profiler import profile_dataset

        df = pd.DataFrame({"a": [1, 2, 3]})
        profile = profile_dataset(df)
        assert "a" in profile
        assert profile["a"]["type"] == "numeric"
