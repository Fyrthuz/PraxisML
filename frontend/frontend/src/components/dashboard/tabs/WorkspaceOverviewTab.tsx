"use client";
import React, { useEffect, useState } from "react";
import { ArrowUpRight, Check, CircleDashed } from "lucide-react";
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

interface WorkspaceOverviewTabProps {
    datasets: Dataset[];
    models: MLModel[];
    predictions: Prediction[];
    token: string | null;
    onNavigate: (view: View) => void;
    onOpenUploadDataset: () => void;
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

    useEffect(() => {
        if (!token) return;
        fetch(config.getFullApiUrl("/copilot/plugins"), {
            headers: { Authorization: `Bearer ${token}` },
        })
            .then((r) => (r.ok ? r.json() : null))
            .then((data) => {
                if (data?.plugins) setPlugins(data.plugins);
            })
            .catch(() => {});
    }, [token]);

    const prodModels = models.filter((m) => m.stage === "Production").length;
    const completedPredictions = predictions.filter((p) => p.status === "COMPLETED").length;

    const pipelineStages = [
        {
            idx: "01",
            title: "Ingest",
            summary: "Upload CSV, Parquet, Excel or ZIP archives.",
            done: datasets.length > 0,
            cta: datasets.length > 0 ? "Datasets" : "Upload",
            action: () => (datasets.length > 0 ? onNavigate("datasets") : onOpenUploadDataset()),
        },
        {
            idx: "02",
            title: "Preprocess",
            summary: "Profile columns, check drift and build pipelines.",
            done: datasets.some((d) => (d.version || 1) > 1),
            cta: "Pipelines",
            action: () => onNavigate("preprocessing"),
        },
        {
            idx: "03",
            title: "Train",
            summary: "Run Sklearn, PyTorch or NLP jobs via Celery + MLflow.",
            done: models.length > 0,
            cta: "Training",
            action: () => onNavigate("training"),
        },
        {
            idx: "04",
            title: "Serve & Monitor",
            summary: "Promote versions, run inference and inspect costs.",
            done: prodModels > 0 || completedPredictions > 0,
            cta: "Predictions",
            action: () => onNavigate("predictions"),
        },
    ];

    return (
        <div className="space-y-6">
            {/* Compact KPI row */}
            <div className="grid grid-cols-2 lg:grid-cols-4 border border-zinc-800/90 rounded-lg bg-[#14171e] divide-y sm:divide-y-0 sm:divide-x divide-zinc-800/90 overflow-hidden">
                <button
                    onClick={() => onNavigate("datasets")}
                    className="p-4 text-left hover:bg-zinc-800/30 transition-colors"
                >
                    <p className="text-[11px] font-medium text-zinc-500">Datasets</p>
                    <p className="text-2xl font-semibold tabular-nums text-zinc-100 mt-1">
                        {datasets.length}
                    </p>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono">object storage</p>
                </button>

                <button
                    onClick={() => onNavigate("models")}
                    className="p-4 text-left hover:bg-zinc-800/30 transition-colors"
                >
                    <p className="text-[11px] font-medium text-zinc-500">Models</p>
                    <div className="flex items-baseline gap-2 mt-1">
                        <p className="text-2xl font-semibold tabular-nums text-zinc-100">
                            {models.length}
                        </p>
                        <span className="text-[11px] font-mono text-emerald-400">
                            {prodModels} prod
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono">mlflow registry</p>
                </button>

                <button
                    onClick={() => onNavigate("predictions")}
                    className="p-4 text-left hover:bg-zinc-800/30 transition-colors"
                >
                    <p className="text-[11px] font-medium text-zinc-500">Predictions</p>
                    <div className="flex items-baseline gap-2 mt-1">
                        <p className="text-2xl font-semibold tabular-nums text-zinc-100">
                            {predictions.length}
                        </p>
                        <span className="text-[11px] font-mono text-zinc-400">
                            {completedPredictions} ok
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono">batch / single</p>
                </button>

                <button
                    onClick={() => onNavigate("copilot")}
                    className="p-4 text-left hover:bg-zinc-800/30 transition-colors"
                >
                    <p className="text-[11px] font-medium text-zinc-500">Active Plugins</p>
                    <p className="text-2xl font-semibold tabular-nums text-zinc-100 mt-1">
                        {plugins.length}
                    </p>
                    <p className="text-[11px] text-zinc-500 mt-0.5 font-mono truncate">
                        {plugins.length > 0 ? plugins.map((p) => p.name).join(", ") : "core only"}
                    </p>
                </button>
            </div>

            {/* Pipeline stages */}
            <section className="border border-zinc-800/90 rounded-lg bg-[#14171e]">
                <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                        ML Lifecycle
                    </h3>
                    <button
                        onClick={() => onNavigate("observability")}
                        className="text-xs text-zinc-400 hover:text-zinc-100 flex items-center gap-1"
                    >
                        Observability <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-zinc-800/80">
                    {pipelineStages.map((stage) => (
                        <div key={stage.idx} className="p-4 flex flex-col justify-between gap-3">
                            <div>
                                <div className="flex items-center justify-between mb-1.5">
                                    <span className="text-[11px] font-mono text-zinc-500">
                                        {stage.idx}
                                    </span>
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
                                <p className="text-xs text-zinc-500 mt-0.5 leading-relaxed">
                                    {stage.summary}
                                </p>
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

            {/* Recent tables */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] overflow-hidden">
                    <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                            Recent Datasets
                        </h3>
                        <button
                            onClick={() => onNavigate("datasets")}
                            className="text-xs text-zinc-400 hover:text-zinc-100"
                        >
                            View all
                        </button>
                    </div>
                    {datasets.length === 0 ? (
                        <div className="px-4 py-8 text-center text-xs text-zinc-500">
                            No datasets registered.
                        </div>
                    ) : (
                        <div className="divide-y divide-zinc-800/70">
                            {datasets.slice(0, 5).map((ds) => (
                                <div
                                    key={ds.id}
                                    className="px-4 py-2.5 flex items-center justify-between gap-3 hover:bg-zinc-800/30 transition-colors"
                                >
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

                <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] overflow-hidden">
                    <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                            Recent Models
                        </h3>
                        <button
                            onClick={() => onNavigate("models")}
                            className="text-xs text-zinc-400 hover:text-zinc-100"
                        >
                            View all
                        </button>
                    </div>
                    {models.length === 0 ? (
                        <div className="px-4 py-8 text-center text-xs text-zinc-500">
                            No models registered.
                        </div>
                    ) : (
                        <div className="divide-y divide-zinc-800/70">
                            {models.slice(0, 5).map((m) => (
                                <div
                                    key={m.id}
                                    className="px-4 py-2.5 flex items-center justify-between gap-3 hover:bg-zinc-800/30 transition-colors"
                                >
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
