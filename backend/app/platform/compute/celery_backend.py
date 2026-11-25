"""
CeleryComputeBackend — Implementación de ComputeBackend sobre Celery + colas tipadas.
"""

from __future__ import annotations

from celery.result import AsyncResult

from app.platform.compute.base import (
    ComputeBackend,
    ComputeProfile,
    JobHandle,
    JobSpec,
    JobStatus,
)
from app.worker.celery_app import celery_app

PROFILE_TO_QUEUE = {
    ComputeProfile.CPU_LIGHT: "cpu-light",
    ComputeProfile.CPU_HEAVY: "cpu-heavy",
    ComputeProfile.GPU_TRAIN: "gpu-train",
    ComputeProfile.SYSTEM_BEAT: "system-beat",
}


class CeleryComputeBackend(ComputeBackend):
    backend_name = "celery"

    def submit(self, spec: JobSpec) -> JobHandle:
        queue = PROFILE_TO_QUEUE.get(spec.profile, "cpu-heavy")
        async_result = celery_app.send_task(
            spec.task_name,
            args=list(spec.args),
            kwargs=spec.kwargs,
            queue=queue,
        )
        return JobHandle(
            job_id=str(async_result.id),
            backend=self.backend_name,
            queue=queue,
        )

    def get_status(self, job_id: str) -> JobStatus:
        res = AsyncResult(job_id, app=celery_app)
        error = str(res.result) if res.failed() else None
        result = res.result if res.successful() else None
        return JobStatus(
            job_id=job_id,
            state=res.state,
            result=result,
            error=error,
        )

    def cancel(self, job_id: str) -> bool:
        celery_app.control.revoke(job_id, terminate=True)
        return True
