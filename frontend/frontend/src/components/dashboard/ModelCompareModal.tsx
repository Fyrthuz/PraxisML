"use client";

import React, { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ArrowUpCircle, Loader2, Server, Trophy, X } from "lucide-react";
import { MLModel, api } from "@/lib/api";
import { getModelCompatibilityProfile } from "@/lib/modelCompatibility";

interface ModelCompareModalProps {
    models: MLModel[];
    onClose: () => void;
    onPromote?: (modelId: string) => Promise<void>;
    onDeployEndpoint?: (modelId: string) => void;
}

const LOWER_IS_BETTER = new Set(["rmse", "mse", "mae", "loss", "val_loss", "test_loss", "log_loss"]);

export default function ModelCompareModal({
    models,
    onClose,
    onPromote,
    onDeployEndpoint,
}: ModelCompareModalProps) {
    const [runData, setRunData] = useState<Record<string, any>>({});
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let active = true;
        const load = async () => {
            setLoading(true);
            const entries: Record<string, any> = {};
            await Promise.all(
                models.map(async (m) => {
                    if (!m.mlflow_run_id) return;
                    try {
                        const details = await api.getRunDetails(m.mlflow_run_id);
                        entries[m.id] = details;
                    } catch {
                        entries[m.id] = null;
                    }
                })
            );
            if (active) {
                setRunData(entries);
                setLoading(false);
            }
        };
        load();
        return () => {
            active = false;
        };
    }, [models]);

    const getModelMetrics = (m: MLModel): Record<string, number> => {
        const fromRun = runData[m.id]?.metrics || {};
        const fromMeta = m.metrics_metadata?.metrics || {};
        const merged: Record<string, number> = {};
        for (const [k, v] of Object.entries({ ...fromMeta, ...fromRun })) {
            if (typeof v === "number" && Number.isFinite(v)) merged[k] = v;
        }
        return merged;
    };

    const getModelParams = (m: MLModel): Record<string, any> => {
        const fromRun = runData[m.id]?.params || {};
        const fromMeta = m.metrics_metadata || {};
        const out: Record<string, any> = { ...fromRun };
        for (const [k, v] of Object.entries(fromMeta)) {
            if (!["metrics", "feature_names", "dataset_id", "dataset_name"].includes(k) && typeof v !== "object") {
                if (!(k in out)) out[k] = v;
            }
        }
        return out;
    };

    const allMetricKeys = useMemo(() => {
        const keys = new Set<string>();
        models.forEach((m) => {
            Object.keys(getModelMetrics(m)).forEach((k) => keys.add(k));
        });
        return Array.from(keys).sort();
    }, [models, runData]);

    const allParamKeys = useMemo(() => {
        const keys = new Set<string>();
        models.forEach((m) => {
            Object.keys(getModelParams(m)).forEach((k) => keys.add(k));
        });
        return Array.from(keys).sort();
    }, [models, runData]);

    const bestByMetric = useMemo(() => {
        const best: Record<string, number> = {};
        allMetricKeys.forEach((key) => {
            const vals = models
                .map((m) => getModelMetrics(m)[key])
                .filter((v): v is number => typeof v === "number");
            if (vals.length === 0) return;
            const lower = LOWER_IS_BETTER.has(key.toLowerCase());
            best[key] = lower ? Math.min(...vals) : Math.max(...vals);
        });
        return best;
    }, [allMetricKeys, models, runData]);

    const profiles = useMemo(() => models.map(getModelCompatibilityProfile), [models]);
    const uniqueScopes = useMemo(
        () => Array.from(new Set(profiles.map((p) => p.scopeLabel))),
        [profiles]
    );
    const uniqueDatasets = useMemo(
        () => Array.from(new Set(profiles.map((p) => p.datasetLabel).filter(Boolean))) as string[],
        [profiles]
    );

    return (
        <div
            className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4"
            onClick={onClose}
        >
            <div
                className="w-full max-w-5xl max-h-[88vh] flex flex-col rounded-lg border border-zinc-800 bg-[#101217] overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="px-5 py-3.5 border-b border-zinc-800 bg-[#0b0c0f] flex items-center justify-between">
                    <div>
                        <div className="flex items-center gap-2">
                            <h3 className="text-sm font-semibold text-zinc-100">
                                Side-by-Side Model Comparison
                            </h3>
                            {uniqueScopes.length === 1 && (
                                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-orange-400">
                                    {uniqueScopes[0]}
                                </span>
                            )}
                        </div>
                        <p className="text-xs text-zinc-500">
                            Comparing {models.length} compatible models across MLflow metrics and hyperparameters.
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>

                <div className="flex-1 overflow-auto p-5 space-y-5">
                    {uniqueDatasets.length > 1 && (
                        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3.5 py-2.5 flex items-start gap-2 text-xs text-amber-200">
                            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                            <span>
                                Different training datasets/versions detected (
                                <span className="font-mono">{uniqueDatasets.join(" vs ")}</span>). Metric comparisons are valid for benchmarking preprocessing iterations, but verify evaluation splits match.
                            </span>
                        </div>
                    )}
                    {loading && (
                        <div className="flex items-center gap-2 text-xs text-zinc-400">
                            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading MLflow run details...
                        </div>
                    )}

                    {/* Overview & Actions */}
                    <div className="overflow-x-auto rounded-lg border border-zinc-800">
                        <table className="w-full text-left text-xs">
                            <thead className="bg-[#0b0c0f] border-b border-zinc-800">
                                <tr>
                                    <th className="px-4 py-3 text-[11px] font-medium uppercase tracking-wider text-zinc-500 w-44">
                                        Property
                                    </th>
                                    {models.map((m, idx) => (
                                        <th key={m.id} className="px-4 py-3 text-zinc-100 font-semibold">
                                            <div className="truncate max-w-[200px]">{m.name}</div>
                                            <div className="text-[10px] font-mono font-normal text-zinc-500 mt-0.5">
                                                v{m.version || "1.0.0"} · {m.stage || "Staging"}
                                                {profiles[idx]?.datasetLabel ? ` · ${profiles[idx].datasetLabel}` : ""}
                                            </div>
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-zinc-800/80 font-mono tabular-nums">
                                {/* Metrics */}
                                {allMetricKeys.map((metric) => (
                                    <tr key={`m-${metric}`} className="hover:bg-zinc-800/20">
                                        <td className="px-4 py-2 text-zinc-400 font-sans text-xs">
                                            {metric}
                                        </td>
                                        {models.map((m) => {
                                            const val = getModelMetrics(m)[metric];
                                            const isBest =
                                                typeof val === "number" &&
                                                bestByMetric[metric] === val &&
                                                models.length > 1;
                                            return (
                                                <td
                                                    key={m.id}
                                                    className={`px-4 py-2 ${
                                                        isBest ? "text-emerald-400 font-semibold" : "text-zinc-200"
                                                    }`}
                                                >
                                                    <span className="inline-flex items-center gap-1.5">
                                                        {typeof val === "number" ? val.toFixed(4) : "—"}
                                                        {isBest && <Trophy className="w-3 h-3 text-orange-400" />}
                                                    </span>
                                                </td>
                                            );
                                        })}
                                    </tr>
                                ))}

                                {/* Params */}
                                {allParamKeys.slice(0, 14).map((param) => (
                                    <tr key={`p-${param}`} className="hover:bg-zinc-800/20">
                                        <td className="px-4 py-2 text-zinc-500 text-[11px]">
                                            param:{param}
                                        </td>
                                        {models.map((m) => (
                                            <td key={m.id} className="px-4 py-2 text-zinc-400 text-[11px]">
                                                {getModelParams(m)[param] !== undefined
                                                    ? String(getModelParams(m)[param])
                                                    : "—"}
                                            </td>
                                        ))}
                                    </tr>
                                ))}

                                {/* Direct actions */}
                                <tr className="bg-[#0b0c0f]">
                                    <td className="px-4 py-3 text-[11px] font-sans text-zinc-500">
                                        Actions
                                    </td>
                                    {models.map((m) => (
                                        <td key={m.id} className="px-4 py-3 font-sans">
                                            <div className="flex flex-wrap gap-1.5">
                                                {onDeployEndpoint && (
                                                    <button
                                                        onClick={() => {
                                                            onDeployEndpoint(m.id);
                                                            onClose();
                                                        }}
                                                        className="h-7 px-2.5 rounded bg-zinc-100 hover:bg-white text-zinc-950 text-[11px] font-medium inline-flex items-center gap-1"
                                                    >
                                                        <Server className="w-3 h-3" /> Deploy
                                                    </button>
                                                )}
                                                {onPromote && m.stage !== "Production" && (
                                                    <button
                                                        onClick={() => onPromote(m.id)}
                                                        className="h-7 px-2.5 rounded border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 text-[11px] inline-flex items-center gap-1"
                                                    >
                                                        <ArrowUpCircle className="w-3 h-3 text-emerald-400" /> Prod
                                                    </button>
                                                )}
                                            </div>
                                        </td>
                                    ))}
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}
