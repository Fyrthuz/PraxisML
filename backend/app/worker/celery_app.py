import logging

from celery import Celery, signals
from celery.schedules import crontab

from app.core.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.worker.tasks.predict",
        "app.worker.tasks.single_predict",
        "app.worker.tasks.train",
        "app.worker.tasks.scheduled_checks",
    ],
)

# No explicit task routes; everything goes to the default 'celery' queue
celery_app.conf.task_routes = {}

# Configuramos timeouts e integraciones en caso de fallos
celery_app.conf.update(
    task_track_started=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

# ── Beat Schedule (tareas periódicas) ─────────────────────────────────────

celery_app.conf.beat_schedule = {
    "drift-check-every-6h": {
        "task": "app.worker.tasks.scheduled_checks.scheduled_drift_check",
        "schedule": crontab(minute=0, hour="*/6"),
    },
    "alert-evaluation-every-5m": {
        "task": "app.worker.tasks.scheduled_checks.evaluate_alerts",
        "schedule": crontab(minute="*/5"),
    },
    "zombie-model-check-daily": {
        "task": "app.worker.tasks.scheduled_checks.check_zombie_models",
        "schedule": crontab(hour=3, minute=0),
    },
    "budget-check-daily": {
        "task": "app.worker.tasks.scheduled_checks.check_budget_thresholds",
        "schedule": crontab(hour=9, minute=0),
    },
}


# ── Signal Handlers (Prometheus metrics) ────────────────────────────────────


@signals.task_failure.connect
def handle_task_failure(sender=None, exception=None, **kwargs):
    """Incrementa errors_total cuando una tarea Celery falla."""
    try:
        from app.observability.metrics import errors_total

        task_name = sender.name if sender else "unknown"
        errors_total.labels(
            tenant_id="system",
            error_type="CeleryTaskFailure",
            component=task_name,
        ).inc()
        logger.warning("Celery task failed: %s - %s", task_name, str(exception)[:200])
    except Exception:
        pass  # No fallar si metrics no está disponible


@signals.task_prerun.connect
def handle_task_prerun(sender=None, task_id=None, task=None, **kwargs):
    """Log cuando una tarea comienza."""
    logger.info("Task started: %s [%s]", task.name if task else "?", task_id)


@signals.task_success.connect
def handle_task_success(sender=None, **kwargs):
    """Log cuando una tarea se completa exitosamente."""
    task_name = sender.name if sender else "?"
    logger.info("Task completed: %s", task_name)
