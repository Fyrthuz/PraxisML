from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks.predict", "app.worker.tasks.single_predict", "app.worker.tasks.train"]
)

# No explicit task routes; everything goes to the default 'celery' queue
celery_app.conf.task_routes = {}

# Configuramos timeouts e integraciones en caso de fallos
celery_app.conf.update(
    task_track_started=True,
    task_reject_on_worker_lost=True,
    # Consume de a 1 tarea por worker (vital para ML para no ahogar RAM)
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)
