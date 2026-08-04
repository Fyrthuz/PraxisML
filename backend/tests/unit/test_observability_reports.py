"""Tests unitarios para el módulo de reportes."""

import csv
import io
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


class TestGenerateCostCsv:
    def test_generate_csv_empty(self):
        from app.observability.reports import generate_cost_csv

        db = MagicMock()
        chain = db.query.return_value
        chain.filter.return_value = chain
        chain.filter.return_value = chain
        chain.order_by.return_value.all.return_value = []

        now = datetime.now(timezone.utc)
        result = generate_cost_csv(db, "tenant-1", now - timedelta(days=30), now)

        assert isinstance(result, bytes)
        text = result.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 1  # Solo header

    def test_generate_csv_with_records(self):
        from app.observability.reports import generate_cost_csv
        from app.observability.models import CostRecord

        now = datetime.now(timezone.utc)

        mock_record = MagicMock(spec=CostRecord)
        mock_record.period_start = now
        mock_record.period_end = now
        mock_record.cost_type = "inference"
        mock_record.amount = 0.50
        mock_record.currency = "USD"
        mock_record.model_id = "model-1"
        mock_record.source = "pred-1"

        db = MagicMock()
        chain = db.query.return_value
        chain.filter.return_value = chain
        chain.filter.return_value = chain
        chain.order_by.return_value.all.return_value = [mock_record]

        result = generate_cost_csv(db, "tenant-1", now - timedelta(days=30), now)
        text = result.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 record
        assert "inference" in rows[1]


class TestGenerateMonthlyReportPdf:
    def test_generate_pdf_with_reportlab(self):
        """Cuando reportlab está instalado, genera PDF real."""
        from app.observability.reports import generate_monthly_report_pdf

        db = MagicMock()

        def query_side_effect(*args, **kwargs):
            mock_chain = MagicMock()
            mock_chain.filter.return_value = mock_chain
            mock_chain.filter.return_value.filter.return_value.scalar.side_effect = [250.0, 50]
            return mock_chain

        db.query.side_effect = query_side_effect

        result = generate_monthly_report_pdf(db, "tenant-1", "2026-07")
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:5] == b'%PDF-'
