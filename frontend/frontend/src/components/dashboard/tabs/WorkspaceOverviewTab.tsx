"use client";
import React, { useEffect, useState } from "react";
import { ArrowUpRight, Check, CircleDashed, Server } from "lucide-react";
import { Dataset, MLModel, Prediction } from "@/lib/api";
import { config } from "@/lib/config";
import type { View } from "../Sidebar";

interface PluginMeta {
    name: string;
    version: string;
    default_queue: string;
    algorithms_count: number;
    copilot_tools_count: number;
}

interface PlatformSummary {
    services_health: {
        postgres: { status: string };
        redis: { status: string; queues?: Record<string, number> };
        celery_workers?: { status: string; workers?: number };
        storage: { status: string; backend: string };
        mlflow: { status: string; tracking_uri: string };
        prometheus?: { status: string; url?: string };
        grafana?: { status: string; url?: string };
        serving_plane?: { status: string; active_endpoints?: number; runtimes?: string[] };
        vector_search?: { status: string; indexed_vectors?: number; embedding_model?: string };
        llm_gateway?: { status: string; provider?: string; model?: string };
        event_bus?: { status: string; backend?: string };
    };
    serving: {
        total_endpoints: number;
        active_endpoints: number;
        paused_endpoints: number;
        total_requests: number;
        total_errors: number;
        error_rate_pct: number;
        avg_latency_ms: number;
        endpoints: {
            id: string;
            name: string;
            slug: string;
            status: string;
            traffic_split_pct: number;
            request_count: number;
            error_count: number;
            avg_latency_ms: number;
        }[];
    };
    tracking_and_data: {
        total_datasets: number;
        dvc_tracked_datasets: number;
        datasets_storage_mb: number;
        total_models: number;
        models_by_stage: Record<string, number>;
    };
    compute_and_governance: {
        runtime_environments: number;
        user_pipelines: number;
        workspaces: number;
        active_api_keys: number;
        audit_events_24h: number;
    };
    usage_by_domain: Record<string, number>;
}

interface WorkspaceOverviewTabProps {
    datasets: Dataset[];
    models: MLModel[];
    predictions: Prediction[];
    token: string | null;
    onNavigate: (view: View) => void;
    onOpenUploadDataset: () => void;
}

function StatusDot({ status }: { status?: string }) {
    const ok = status === "ok" || status === "healthy";
    const info = status === "rag_local" || status === "idle";
    return (
        <span
            className={`inline-block w-2 h-2 rounded-full ${
                ok
                    ? "bg-emerald-400"
                    : info
                    ? "bg-sky-400"
                    : status === "degraded"
                    ? "bg-amber-400"
                    : "bg-red-400"
            }`}
        />
    );
}

export default function WorkspaceOverviewTab({
    datasets,
    models,
    predictions,
    token,
    onNavigate,
    onOpenUploadDataset,
}: WorkspaceOverviewTabProps) {
    const [plugins, setPlugins] = useState<PluginMeta[]>([]);
    const [summary, setSummary] = useState<PlatformSummary | null>(null);

    useEffect(() => {
        if (!token) return;
        const headers = { Authorization: `Bearer ${token}` };
        Promise.all([
            fetch(config.getFullApiUrl("/copilot/plugins"), { headers }).then((r) => (r.ok ? r.json() : null)),
            fetch(config.getFullApiUrl("/observability/platform-summary"), { headers }).then((r) =>
                r.ok ? r.json() : null
            ),
        ])
            .then(([plugData, sumData]) => {
                if (plugData?.plugins) setPlugins(plugData.plugins);
                if (sumData) setSummary(sumData);
            })
            .catch(() => {});
    }, [token]);

    const prodModels = models.filter((m) => m.stage === "Production").length;
    const completedPredictions = predictions.filter((p) => p.status === "COMPLETED").length;
    const activeEndpoints = summary?.serving.active_endpoints ?? 0;
    const totalServingReqs = summary?.serving.total_requests ?? 0;
    const queuedJobs = summary?.services_health.redis.queues
        ? Object.values(summary.services_health.redis.queues).reduce((a, b) => a + b, 0)
        : 0;

    const pipelineStages = [
        {
            idx: "01",
            title: "Ingest & Track",
            summary: "Upload tabular/ZIP data to object storage with DVC versioning.",
            done: datasets.length > 0,
            cta: datasets.length > 0 ? "Datasets" : "Upload",
            action: () => (datasets.length > 0 ? onNavigate("datasets") : onOpenUploadDataset()),
        },
        {
            idx: "02",
            title: "Environments & Pipelines",
            summary: "Clean features with custom names or run Python code in isolated uv/Docker envs.",
            done: datasets.some((d) => (d.version || 1) > 1) || (summary?.compute_and_governance.user_pipelines ?? 0) > 0,
            cta: "Environments",
            action: () => onNavigate("environments"),
        },
        {
            idx: "03",
            title: "Train & Compare",
            summary: "Run Sklearn, PyTorch or NLP jobs via Celery + MLflow and compare side by side.",
            done: models.length > 0,
            cta: "Training",
            action: () => onNavigate("training"),
        },
        {
            idx: "04",
            title: "Deploy & Monitor",
            summary: "Serve endpoints with Canary splits, API keys, invocation logs and FinOps.",
            done: activeEndpoints > 0 || prodModels > 0 || completedPredictions > 0,
            cta: "Model Endpoints",
            action: () => onNavigate("endpoints"),
        },
    ];

    return (
        <div className="space-y-6">
            {/* Compact 6-KPI Strip */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 border border-zinc-800/90 rounded-lg bg-[#14171e] divide-y md:divide-y-0 md:divide-x divide-zinc-800/90 overflow-hidden">
                <button onClick={() => onNavigate("datasets")} className="p-4 text-left hover:bg-zinc-800/30 transition-colors">
                    <p className="text-[11px] font-medium text-zinc-500">Datasets</p>
                    <div className="flex items-baseline gap-2 mt-1">
                        <p className="text-2xl font-semibold tabular-nums text-zinc-100">{datasets.length}</p>
                        <span className="text-[11px] font-mono text-zinc-400">
                            {summary?.tracking_and_data.dvc_tracked_datasets ?? 0} dvc
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono tabular-nums">
                        {summary ? `${summary.tracking_and_data.datasets_storage_mb.toFixed(2)} MB` : "object storage"}
                    </p>
                </button>

                <button onClick={() => onNavigate("models")} className="p-4 text-left hover:bg-zinc-800/30 transition-colors">
                    <p className="text-[11px] font-medium text-zinc-500">Models</p>
                    <div className="flex items-baseline gap-2 mt-1">
                        <p className="text-2xl font-semibold tabular-nums text-zinc-100">{models.length}</p>
                        <span className="text-[11px] font-mono text-emerald-400">{prodModels} prod</span>
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono">mlflow registry</p>
                </button>

                <button onClick={() => onNavigate("endpoints")} className="p-4 text-left hover:bg-zinc-800/30 transition-colors">
                    <p className="text-[11px] font-medium text-zinc-500">Model Endpoints</p>
                    <div className="flex items-baseline gap-2 mt-1">
                        <p className="text-2xl font-semibold tabular-nums text-zinc-100">{activeEndpoints}</p>
                        <span className="text-[11px] font-mono text-orange-400">{totalServingReqs} req</span>
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono tabular-nums">
                        {summary ? `${summary.serving.avg_latency_ms} ms avg` : "serving plane"}
                    </p>
                </button>

                <button onClick={() => onNavigate("environments")} className="p-4 text-left hover:bg-zinc-800/30 transition-colors">
                    <p className="text-[11px] font-medium text-zinc-500">Envs & Pipelines</p>
                    <div className="flex items-baseline gap-2 mt-1">
                        <p className="text-2xl font-semibold tabular-nums text-zinc-100">
                            {summary?.compute_and_governance.runtime_environments ?? 0}
                        </p>
                        <span className="text-[11px] font-mono text-zinc-400">
                            {summary?.compute_and_governance.user_pipelines ?? 0} pipes
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono">uv / docker</p>
                </button>

                <button onClick={() => onNavigate("governance")} className="p-4 text-left hover:bg-zinc-800/30 transition-colors">
                    <p className="text-[11px] font-medium text-zinc-500">API Keys & Audit</p>
                    <div className="flex items-baseline gap-2 mt-1">
                        <p className="text-2xl font-semibold tabular-nums text-zinc-100">
                            {summary?.compute_and_governance.active_api_keys ?? 0}
                        </p>
                        <span className="text-[11px] font-mono text-zinc-400">
                            {summary?.compute_and_governance.workspaces ?? 1} ws
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono tabular-nums">
                        {summary?.compute_and_governance.audit_events_24h ?? 0} events / 24h
                    </p>
                </button>

                <button onClick={() => onNavigate("copilot")} className="p-4 text-left hover:bg-zinc-800/30 transition-colors">
                    <p className="text-[11px] font-medium text-zinc-500">Active Plugins</p>
                    <p className="text-2xl font-semibold tabular-nums text-zinc-100 mt-1">{plugins.length}</p>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono truncate">
                        {plugins.length > 0 ? plugins.map((p) => p.name).join(", ") : "core only"}
                    </p>
                </button>
            </div>

            {/* Live Platform Services Health Bar */}
            {summary && (
                <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] px-4 py-3 flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] font-mono">
                        <span className="font-sans font-semibold uppercase tracking-wider text-zinc-500">
                            Services
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.postgres.status} /> PostgreSQL
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.redis.status} /> Redis ({queuedJobs}q)
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.celery_workers?.status} /> Celery ({summary.services_health.celery_workers?.workers ?? 0}w)
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.serving_plane?.status} /> Serving ({summary.services_health.serving_plane?.active_endpoints ?? 0})
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.storage.status} /> Storage ({summary.services_health.storage.backend})
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.mlflow.status} /> MLflow
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.prometheus?.status} /> Prometheus
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.grafana?.status} /> Grafana
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.vector_search?.status} /> pgvector ({summary.services_health.vector_search?.indexed_vectors ?? 0})
                        </span>
                        <span className="flex items-center gap-1.5 text-zinc-300">
                            <StatusDot status={summary.services_health.llm_gateway?.status} /> Copilot
                        </span>
                    </div>
                    <button
                        onClick={() => onNavigate("observability")}
                        className="text-xs text-zinc-400 hover:text-zinc-100 inline-flex items-center gap-1 shrink-0"
                    >
                        Full Observability <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                </section>
            )}

            {/* Pipeline stages */}
            <section className="border border-zinc-800/90 rounded-lg bg-[#14171e]">
                <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                        ML Lifecycle
                    </h3>
                    <span className="text-[11px] font-mono text-zinc-500">
                        {completedPredictions} batch/single · {totalServingReqs} endpoint invocations
                    </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-zinc-800/80">
                    {pipelineStages.map((stage) => (
                        <div key={stage.idx} className="p-4 flex flex-col justify-between gap-3">
                            <div>
                                <div className="flex items-center justify-between mb-1.5">
                                    <span className="text-[11px] font-mono text-zinc-500">{stage.idx}</span>
                                    {stage.done ? (
                                        <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-400">
                                            <Check className="w-3 h-3" /> ready
                                        </span>
                                    ) : (
                                        <span className="inline-flex items-center gap-1 text-[10px] font-mono text-zinc-500">
                                            <CircleDashed className="w-3 h-3" /> pending
                                        </span>
                                    )}
                                </div>
                                <p className="text-sm font-medium text-zinc-100">{stage.title}</p>
                                <p className="text-xs text-zinc-500 mt-0.5 leading-relaxed">{stage.summary}</p>
                            </div>
                            <button
                                onClick={stage.action}
                                className="self-start text-xs font-medium text-zinc-300 hover:text-orange-400 inline-flex items-center gap-1 transition-colors"
                            >
                                {stage.cta} <ArrowUpRight className="w-3 h-3" />
                            </button>
                        </div>
                    ))}
                </div>
            </section>

            {/* Deployed Model Endpoints + Recent Assets */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                {/* Deployed Serving Endpoints */}
                <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] overflow-hidden">
                    <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                            <Server className="w-3.5 h-3.5 text-orange-400" /> Deployed Endpoints
                        </h3>
                        <button onClick={() => onNavigate("endpoints")} className="text-xs text-zinc-400 hover:text-zinc-100">
                            Manage
                        </button>
                    </div>
                    {!summary || summary.serving.endpoints.length === 0 ? (
                        <div className="px-4 py-8 text-center text-xs text-zinc-500">
                            No model endpoints deployed yet.
                        </div>
                    ) : (
                        <div className="divide-y divide-zinc-800/70">
                            {summary.serving.endpoints.slice(0, 5).map((ep) => (
                                <button
                                    key={ep.id}
                                    onClick={() => onNavigate("endpoints")}
                                    className="w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 hover:bg-zinc-800/30 transition-colors"
                                >
                                    <div className="min-w-0">
                                        <p className="text-sm text-zinc-200 truncate">{ep.name}</p>
                                        <p className="text-[11px] font-mono text-zinc-500 tabular-nums">
                                            /{ep.slug} · {ep.request_count} req · {ep.avg_latency_ms}ms
                                        </p>
                                    </div>
                                    <span
                                        className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border ${
                                            ep.status === "active"
                                                ? "border-emerald-800/80 bg-emerald-950/60 text-emerald-300"
                                                : "border-zinc-800 bg-zinc-900 text-zinc-500"
                                        }`}
                                    >
                                        {ep.status}
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}
                </section>

                {/* Recent Datasets */}
                <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] overflow-hidden">
                    <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                            Recent Datasets
                        </h3>
                        <button onClick={() => onNavigate("datasets")} className="text-xs text-zinc-400 hover:text-zinc-100">
                            View all
                        </button>
                    </div>
                    {datasets.length === 0 ? (
                        <div className="px-4 py-8 text-center text-xs text-zinc-500">No datasets registered.</div>
                    ) : (
                        <div className="divide-y divide-zinc-800/70">
                            {datasets.slice(0, 5).map((ds) => (
                                <div key={ds.id} className="px-4 py-2.5 flex items-center justify-between gap-3 hover:bg-zinc-800/30">
                                    <div className="min-w-0">
                                        <p className="text-sm text-zinc-200 truncate">{ds.name}</p>
                                        <p className="text-[11px] font-mono text-zinc-500 tabular-nums">
                                            {ds.num_rows != null
                                                ? `${ds.num_rows.toLocaleString()} × ${ds.num_columns}`
                                                : `${(ds.file_size_bytes / 1024).toFixed(1)} KB`}
                                        </p>
                                    </div>
                                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border border-zinc-700/80 bg-zinc-900 text-zinc-400">
                                        {ds.file_type || "file"}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </section>

                {/* Recent Models */}
                <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] overflow-hidden">
                    <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                            Recent Models
                        </h3>
                        <button onClick={() => onNavigate("models")} className="text-xs text-zinc-400 hover:text-zinc-100">
                            View all
                        </button>
                    </div>
                    {models.length === 0 ? (
                        <div className="px-4 py-8 text-center text-xs text-zinc-500">No models registered.</div>
                    ) : (
                        <div className="divide-y divide-zinc-800/70">
                            {models.slice(0, 5).map((m) => (
                                <div key={m.id} className="px-4 py-2.5 flex items-center justify-between gap-3 hover:bg-zinc-800/30">
                                    <div className="min-w-0">
                                        <p className="text-sm text-zinc-200 truncate">{m.name}</p>
                                        <p className="text-[11px] font-mono text-zinc-500">
                                            v{m.version || "1.0.0"} · {m.mlflow_run_id?.slice(0, 8)}
                                        </p>
                                    </div>
                                    <span
                                        className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                                            m.stage === "Production"
                                                ? "border-emerald-800/80 bg-emerald-950/60 text-emerald-300"
                                                : m.stage === "Archived"
                                                ? "border-zinc-800 bg-zinc-900 text-zinc-500"
                                                : "border-amber-800/70 bg-amber-950/50 text-amber-300"
                                        }`}
                                    >
                                        {m.stage || "Staging"}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </div>
        </div>
    );
}
