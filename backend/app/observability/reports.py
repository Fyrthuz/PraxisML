"""
Generación de reportes mensuales PDF y CSV.
"""

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.observability.models import CostRecord, UsageMetric

logger = logging.getLogger(__name__)


def generate_cost_csv(
    db: Session,
    tenant_id: str,
    from_date: datetime,
    to_date: datetime,
) -> bytes:
    """Genera CSV con desglose de costes."""
    records = (
        db.query(CostRecord)
        .filter(
            CostRecord.tenant_id == tenant_id,
            CostRecord.period_start >= from_date,
            CostRecord.period_start <= to_date,
        )
        .order_by(CostRecord.period_start)
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "period_start", "period_end", "cost_type", "amount",
        "currency", "model_id", "source",
    ])
    for r in records:
        writer.writerow([
            r.period_start.isoformat(),
            r.period_end.isoformat(),
            r.cost_type,
            r.amount,
            r.currency,
            r.model_id or "",
            r.source or "",
        ])

    return output.getvalue().encode("utf-8")


def generate_monthly_report_pdf(
    db: Session,
    tenant_id: str,
    month: str,
) -> bytes:
    """Genera reporte PDF mensual."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        logger.warning("reportlab no instalado, generando reporte CSV en su lugar")
        year_str, month_str = month.split("-")
        from_date = datetime(int(year_str), int(month_str), 1, tzinfo=timezone.utc)
        if int(month_str) == 12:
            to_date = datetime(int(year_str) + 1, 1, 1, tzinfo=timezone.utc)
        else:
            to_date = datetime(int(year_str), int(month_str) + 1, 1, tzinfo=timezone.utc)
        return generate_cost_csv(db, tenant_id, from_date, to_date)

    year_str, month_str = month.split("-")
    year, month_num = int(year_str), int(month_str)
    from_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
    if month_num == 12:
        to_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        to_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)

    total_cost = (
        db.query(func.coalesce(func.sum(CostRecord.amount), 0))
        .filter(
            CostRecord.tenant_id == tenant_id,
            CostRecord.period_start >= from_date,
            CostRecord.period_start < to_date,
        )
        .scalar()
    )

    total_predictions = (
        db.query(func.count(UsageMetric.id))
        .filter(
            UsageMetric.tenant_id == tenant_id,
            UsageMetric.metric_name == "inference_latency_ms",
            UsageMetric.recorded_at >= from_date,
            UsageMetric.recorded_at < to_date,
        )
        .scalar()
    )

    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setTitle(f"PraxisML - Reporte Mensual {month}")
    width, height = A4

    p.drawString(72, height - 72, f"Reporte Mensual - PraxisML")
    p.drawString(72, height - 96, f"Tenant: {tenant_id}")
    p.drawString(72, height - 120, f"Periodo: {month}")
    p.line(72, height - 130, width - 72, height - 130)

    p.drawString(72, height - 160, f"Coste Total: ${float(total_cost):.2f}")
    p.drawString(72, height - 184, f"Predicciones: {total_predictions}")
    p.drawString(
        72, height - 208,
        f"Coste Promedio por Prediccion: "
        f"${float(total_cost) / total_predictions:.4f}" if total_predictions else "N/A",
    )

    p.save()
    return buf.getvalue()
