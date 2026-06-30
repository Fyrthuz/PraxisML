import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_path():
    content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,NYC\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


class TestReadTabular:
    def test_read_csv_from_path(self, sample_csv_path):
        from app.core_ml.tabular_parser import read_tabular

        df = read_tabular(sample_csv_path)
        assert len(df) == 3
        assert list(df.columns) == ["name", "age", "city"]

    def test_read_csv_from_bytesio(self):
        from app.core_ml.tabular_parser import read_tabular

        content = b"name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,NYC\n"
        df = read_tabular(io.BytesIO(content), file_type="csv")
        assert len(df) == 3
        assert list(df.columns) == ["name", "age", "city"]

    def test_read_csv_no_file_type_auto_detect(self, sample_csv_path):
        from app.core_ml.tabular_parser import read_tabular

        df = read_tabular(sample_csv_path, file_type="csv")
        assert len(df) == 3

    def test_read_invalid_file_raises(self):
        from app.core_ml.tabular_parser import read_tabular

        with pytest.raises(FileNotFoundError):
            read_tabular("/nonexistent/file.csv")


class TestDetectFileType:
    def test_detect_csv(self):
        from app.core_ml.tabular_parser import detect_file_type

        assert detect_file_type("data.csv") == "csv"
        assert detect_file_type("data.CSV") == "csv"

    def test_detect_xlsx(self):
        from app.core_ml.tabular_parser import detect_file_type

        assert detect_file_type("data.xlsx") == "xlsx"

    def test_detect_parquet(self):
        from app.core_ml.tabular_parser import detect_file_type

        assert detect_file_type("data.parquet") == "parquet"

    def test_detect_unknown(self):
        from app.core_ml.tabular_parser import detect_file_type

        assert detect_file_type("data.txt") is None


class TestExtractMetadata:
    def test_extract_metadata_basic(self, sample_csv_path):
        from app.core_ml.tabular_parser import extract_metadata

        meta = extract_metadata(sample_csv_path)
        assert meta["num_rows"] == 3
        assert meta["num_columns"] == 3
        assert "name" in meta["column_names"]

    def test_extract_metadata_dtypes(self, sample_csv_path):
        from app.core_ml.tabular_parser import extract_metadata

        meta = extract_metadata(sample_csv_path)
        assert "column_dtypes" in meta
        assert isinstance(meta["column_dtypes"], dict)


class TestGetPreview:
    def test_get_preview(self, sample_csv_path):
        from app.core_ml.tabular_parser import get_preview

        preview_df, meta = get_preview(sample_csv_path, max_rows=2)
        assert len(preview_df) == 2
        assert meta["num_rows"] == 3
        assert meta["column_names"] == ["name", "age", "city"]

    def test_get_preview_respects_max_rows(self, sample_csv_path):
        from app.core_ml.tabular_parser import get_preview

        preview_df, meta = get_preview(sample_csv_path, max_rows=2)
        assert len(preview_df) == 2
