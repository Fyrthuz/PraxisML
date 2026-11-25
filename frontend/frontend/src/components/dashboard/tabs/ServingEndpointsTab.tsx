"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
    Activity,
    AlertTriangle,
    BarChart3,
    Check,
    Copy,
    GitBranch,
    Loader2,
    Pause,
    Play,
    Plus,
    RefreshCw,
    Send,
    Server,
    SlidersHorizontal,
    Trash2,
} from "lucide-react";
import toast from "react-hot-toast";
import { Dataset, MLModel } from "@/lib/api";
import { config } from "@/lib/config";
import { areModelsCompatible, getModelCompatibilityProfile } from "@/lib/modelCompatibility";

interface ModelEndpoint {
    id: string;
    name: string;
    slug: string;
    description?: string | null;
    model_id: string;
    model_name?: string | null;
    model_version?: string | null;
    model_stage?: string | null;
    feature_names: string[];
    task_type?: string | null;
    framework?: string | null;
    secondary_model_id?: string | null;
    secondary_model_name?: string | null;
    traffic_split_pct: number;
    preprocessing_pipeline_path?: string | null;
    status: "active" | "paused";
    auth_mode: string;
    invoke_path: string;
    api_invoke_path: string;
    request_count: number;
    error_count: number;
    error_rate_pct?: number;
    success_rate_pct?: number;
    avg_latency_ms: number;
    last_invoked_at?: string | null;
    created_at?: string | null;
}

interface EndpointMetrics {
    endpoint_id: string;
    slug: string;
    status: string;
    configured_canary_pct: number;
    total_requests: number;
    total_errors: number;
    success_count: number;
    success_rate_pct: number;
    error_rate_pct: number;
    avg_latency_ms: number;
    min_latency_ms: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    max_latency_ms: number;
    requests_last_1h: number;
    requests_last_24h: number;
    total_rows_scored: number;
    last_invoked_at: string | null;
    last_error_message: string | null;
    by_variant: Record<
        string,
        {
            count: number;
            pct: number;
            avg_latency_ms: number;
            error_count: number;
        }
    >;
    estimated_cost_usd: number;
    usage_events_count: number;
    latency_series: Array<{
        ts: string | null;
        latency_ms: number;
        variant: string;
        status: string;
        input_count: number;
    }>;
}

interface ServingEndpointsTabProps {
    token: string | null;
    models: MLModel[];
    datasets: Dataset[];
    userRole: string;
    preselectedModelId?: string | null;
    onClearPreselectedModel?: () => void;
}

const API = config.getFullApiUrl("");
const BASE_HOST = config.API_BASE_URL.replace(/\/$/, "");

function buildSamplePayload(ep: ModelEndpoint): string {
    const isNlp =
        ep.framework === "huggingface" ||
        ep.framework === "onnx_nlp" ||
        (ep.task_type || "").includes("text") ||
        (ep.task_type || "").includes("sentiment") ||
        (ep.task_type || "").includes("token");
    if (isNlp) {
        return JSON.stringify(
            { inputs: [{ text: "This MLOps platform is fast, reliable and easy to scale!" }] },
            null,
            2
        );
    }
    if (ep.feature_names && ep.feature_names.length > 0) {
        const sample: Record<string, number> = {};
        ep.feature_names.forEach((f) => {
            sample[f] = 0;
        });
        return JSON.stringify({ inputs: sample }, null, 2);
    }
    return JSON.stringify({ inputs: { feature_0: 0.0 } }, null, 2);
}

export default function ServingEndpointsTab({
    token,
    models,
    datasets,
    userRole,
    preselectedModelId,
    onClearPreselectedModel,
}: ServingEndpointsTabProps) {
    const canEdit = userRole === "admin" || userRole === "editor";
    const [endpoints, setEndpoints] = useState<ModelEndpoint[]>([]);
    const [loading, setLoading] = useState(true);
    const [showCreate, setShowCreate] = useState(false);
    const [selected, setSelected] = useState<ModelEndpoint | null>(null);
    const [copiedId, setCopiedId] = useState<string | null>(null);
    const [invocationLogs, setInvocationLogs] = useState<any[]>([]);
    const [loadingLogs, setLoadingLogs] = useState(false);
    const [logStatusFilter, setLogStatusFilter] = useState<string>("");
    const [logVariantFilter, setLogVariantFilter] = useState<string>("");
    const [epMetrics, setEpMetrics] = useState<EndpointMetrics | null>(null);
    const [updatingSplit, setUpdatingSplit] = useState(false);
    const [liveSplitDraft, setLiveSplitDraft] = useState<number>(0);

    // Form state
    const [name, setName] = useState("");
    const [slug, setSlug] = useState("");
    const [description, setDescription] = useState("");
    const [modelId, setModelId] = useState("");
    const [secondaryModelId, setSecondaryModelId] = useState("");
    const [trafficSplit, setTrafficSplit] = useState(10);
    const [pipelinePath, setPipelinePath] = useState("");
    const [submitting, setSubmitting] = useState(false);

    // Live invocation console
    const [payloadJson, setPayloadJson] = useState("");
    const [invoking, setInvoking] = useState(false);
    const [invokeResult, setInvokeResult] = useState<any>(null);

    const pipelineDatasets = datasets.filter((d) => Boolean(d.pipeline_path));

    const fetchEndpoints = useCallback(async () => {
        if (!token) return;
        setLoading(true);
        try {
            const res = await fetch(`${API}/serving/endpoints`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                const data = await res.json();
                const list: ModelEndpoint[] = data.items || [];
                setEndpoints(list);
                setSelected((prev) => {
                    if (!prev) return list[0] || null;
                    return list.find((e) => e.id === prev.id) || list[0] || null;
                });
            }
        } catch {
            toast.error("Failed to load model endpoints");
        } finally {
            setLoading(false);
        }
    }, [token]);

    const fetchEndpointMetrics = useCallback(
        async (endpointId: string) => {
            if (!token) return;
            try {
                const res = await fetch(`${API}/serving/endpoints/${endpointId}/metrics`, {
                    headers: { Authorization: `Bearer ${token}` },
                });
                if (res.ok) {
                    setEpMetrics(await res.json());
                }
            } catch {
                setEpMetrics(null);
            }
        },
        [token]
    );

    const fetchLogs = useCallback(
        async (endpointId: string, statusF?: string, variantF?: string) => {
            if (!token) return;
            setLoadingLogs(true);
            try {
                const params = new URLSearchParams({ limit: "40" });
                if (statusF) params.set("status", statusF);
                if (variantF) params.set("variant", variantF);
                const res = await fetch(
                    `${API}/serving/endpoints/${endpointId}/logs?${params.toString()}`,
                    { headers: { Authorization: `Bearer ${token}` } }
                );
                if (res.ok) {
                    const data = await res.json();
                    setInvocationLogs(data.items || []);
                }
            } catch {
                setInvocationLogs([]);
            } finally {
                setLoadingLogs(false);
            }
        },
        [token]
    );

    useEffect(() => {
        fetchEndpoints();
    }, [fetchEndpoints]);

    useEffect(() => {
        if (preselectedModelId) {
            const target = models.find((m) => m.id === preselectedModelId);
            setModelId(preselectedModelId);
            if (target && !name) {
                setName(`${target.name} endpoint`.slice(0, 60));
            }
            setShowCreate(true);
            onClearPreselectedModel?.();
        }
    }, [preselectedModelId, models, name, onClearPreselectedModel]);

    useEffect(() => {
        if (selected) {
            setPayloadJson(buildSamplePayload(selected));
            setInvokeResult(null);
            setLiveSplitDraft(selected.traffic_split_pct);
            setLogStatusFilter("");
            setLogVariantFilter("");
            fetchEndpointMetrics(selected.id);
            fetchLogs(selected.id);
        } else {
            setInvocationLogs([]);
            setEpMetrics(null);
        }
    }, [selected?.id, fetchEndpointMetrics, fetchLogs]);

    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!token || !modelId || !name.trim()) {
            toast.error("Provide an endpoint name and select a model.");
            return;
        }
        setSubmitting(true);
        try {
            const res = await fetch(`${API}/serving/endpoints`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    name: name.trim(),
                    slug: slug.trim() || undefined,
                    description: description.trim() || undefined,
                    model_id: modelId,
                    secondary_model_id: secondaryModelId || undefined,
                    traffic_split_pct: secondaryModelId ? trafficSplit : 0,
                    preprocessing_pipeline_path: pipelinePath || undefined,
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                toast.error(data.detail || "Failed to deploy endpoint");
                return;
            }
            toast.success(`Endpoint /serving/v1/${data.slug}/invocations deployed`);
            setName("");
            setSlug("");
            setDescription("");
            setSecondaryModelId("");
            setPipelinePath("");
            setShowCreate(false);
            await fetchEndpoints();
            setSelected(data);
        } catch {
            toast.error("Error deploying endpoint");
        } finally {
            setSubmitting(false);
        }
    };

    const handleToggleStatus = async (ep: ModelEndpoint) => {
        if (!token) return;
        const nextStatus = ep.status === "active" ? "paused" : "active";
        try {
            const res = await fetch(`${API}/serving/endpoints/${ep.id}`, {
                method: "PATCH",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ status: nextStatus }),
            });
            if (res.ok) {
                toast.success(`Endpoint ${nextStatus === "active" ? "resumed" : "paused"}`);
                await fetchEndpoints();
                await fetchEndpointMetrics(ep.id);
            }
        } catch {
            toast.error("Failed to update endpoint status");
        }
    };

    const handleUpdateCanarySplit = async () => {
        if (!token || !selected) return;
        setUpdatingSplit(true);
        try {
            const res = await fetch(`${API}/serving/endpoints/${selected.id}`, {
                method: "PATCH",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ traffic_split_pct: liveSplitDraft }),
            });
            if (res.ok) {
                toast.success(`Canary traffic updated to ${liveSplitDraft}%`);
                await fetchEndpoints();
                await fetchEndpointMetrics(selected.id);
            }
        } catch {
            toast.error("Failed to update canary split");
        } finally {
            setUpdatingSplit(false);
        }
    };

    const handleDelete = async (ep: ModelEndpoint) => {
        if (!token) return;
        if (!confirm(`Delete endpoint "${ep.name}" (/${ep.slug})?`)) return;
        try {
            const res = await fetch(`${API}/serving/endpoints/${ep.id}`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                toast.success("Endpoint deleted");
                setSelected(null);
                await fetchEndpoints();
            }
        } catch {
            toast.error("Failed to delete endpoint");
        }
    };

    const handleInvoke = async () => {
        if (!token || !selected) return;
        let parsed: any;
        try {
            parsed = JSON.parse(payloadJson);
        } catch {
            toast.error("Invalid JSON payload");
            return;
        }
        setInvoking(true);
        try {
            const res = await fetch(`${API}/serving/endpoints/${selected.slug}/invocations`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(parsed),
            });
            const data = await res.json();
            if (!res.ok) {
                toast.error(data.detail || "Invocation failed");
                setInvokeResult({ error: data.detail || "Invocation failed" });
            } else {
                setInvokeResult(data);
            }
            await fetchEndpoints();
            await fetchEndpointMetrics(selected.id);
            await fetchLogs(selected.id, logStatusFilter, logVariantFilter);
        } catch {
            toast.error("Error invoking endpoint");
        } finally {
            setInvoking(false);
        }
    };

    const copyText = (text: string, id: string) => {
        navigator.clipboard.writeText(text);
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 1500);
    };

    const curlSnippet = selected
        ? `curl -X POST "${BASE_HOST}${selected.invoke_path}" \\\n  -H "X-API-Key: px_YOUR_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '${payloadJson.replace(/\s+/g, " ")}'`
        : "";

    const totalFleetReqs = endpoints.reduce((acc, e) => acc + (e.request_count || 0), 0);
    const activeFleetCount = endpoints.filter((e) => e.status === "active").length;
    const maxSeriesLatency = Math.max(
        1,
        ...(epMetrics?.latency_series.map((p) => p.latency_ms) ?? [1])
    );

    return (
        <div className="space-y-5">
            {/* Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
                    <span className="font-mono tabular-nums text-zinc-100">
                        {activeFleetCount}/{endpoints.length}
                    </span>
                    <span>active endpoints</span>
                    <span className="text-zinc-700">|</span>
                    <span className="font-mono tabular-nums text-zinc-300">
                        {totalFleetReqs} total reqs
                    </span>
                    <span className="text-zinc-700">|</span>
                    <span className="font-mono text-zinc-500">
                        POST /serving/v1/&lt;slug&gt;/invocations
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => {
                            fetchEndpoints();
                            if (selected) {
                                fetchEndpointMetrics(selected.id);
                                fetchLogs(selected.id, logStatusFilter, logVariantFilter);
                            }
                        }}
                        className="h-8 px-3 text-xs rounded-md border border-zinc-800 bg-[#101217] text-zinc-300 hover:bg-zinc-800/80 flex items-center gap-1.5"
                    >
                        <RefreshCw className="w-3.5 h-3.5" /> Refresh
                    </button>
                    {canEdit && (
                        <button
                            onClick={() => setShowCreate((v) => !v)}
                            className="h-8 px-3 text-xs font-medium rounded-md bg-zinc-100 text-zinc-950 hover:bg-white flex items-center gap-1.5"
                        >
                            <Plus className="w-3.5 h-3.5" /> Deploy Endpoint
                        </button>
                    )}
                </div>
            </div>

            {/* Deploy Form */}
            {showCreate && (
                <form
                    onSubmit={handleCreate}
                    className="rounded-lg border border-zinc-800 bg-[#101217] p-5 space-y-4"
                >
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-zinc-100">
                            Deploy Model Endpoint
                        </h3>
                        <button
                            type="button"
                            onClick={() => setShowCreate(false)}
                            className="text-xs text-zinc-500 hover:text-zinc-300"
                        >
                            Cancel
                        </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-[11px] font-medium text-zinc-400 mb-1">
                                Endpoint Name
                            </label>
                            <input
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                placeholder="titanic-survival"
                                required
                                className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                            />
                        </div>
                        <div>
                            <label className="block text-[11px] font-medium text-zinc-400 mb-1">
                                URL Slug (optional)
                            </label>
                            <input
                                value={slug}
                                onChange={(e) => setSlug(e.target.value)}
                                placeholder="titanic-v1"
                                className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-600"
                            />
                        </div>
                        <div>
                            <label className="block text-[11px] font-medium text-zinc-400 mb-1">
                                Primary Model
                            </label>
                            <select
                                value={modelId}
                                onChange={(e) => {
                                    const nextId = e.target.value;
                                    setModelId(nextId);
                                    const nextPrimary = models.find((pm) => pm.id === nextId);
                                    if (nextPrimary && secondaryModelId) {
                                        const sec = models.find((sm) => sm.id === secondaryModelId);
                                        if (sec && !areModelsCompatible(nextPrimary, sec)) {
                                            setSecondaryModelId("");
                                        }
                                    }
                                    if (nextPrimary && getModelCompatibilityProfile(nextPrimary).domain !== "tabular") {
                                        setPipelinePath("");
                                    }
                                }}
                                required
                                className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                            >
                                <option value="">Select registered model...</option>
                                {models.map((m) => (
                                    <option key={m.id} value={m.id}>
                                        {m.name} (v{m.version || "1.0.0"} · {m.stage || "Staging"})
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-[11px] font-medium text-zinc-400 mb-1">
                                Canary / Secondary Model (optional)
                            </label>
                            <select
                                value={secondaryModelId}
                                onChange={(e) => setSecondaryModelId(e.target.value)}
                                className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                            >
                                <option value="">None (100% primary)</option>
                                {models
                                    .filter((m) => {
                                        if (m.id === modelId) return false;
                                        const primary = models.find((pm) => pm.id === modelId);
                                        return primary ? areModelsCompatible(primary, m) : true;
                                    })
                                    .map((m) => (
                                        <option key={m.id} value={m.id}>
                                            {m.name} (v{m.version || "1.0.0"})
                                        </option>
                                    ))}
                            </select>
                        </div>
                        <div>
                            <label className="block text-[11px] font-medium text-zinc-400 mb-1">
                                Canary Traffic Split ({secondaryModelId ? `${trafficSplit}%` : "0%"})
                            </label>
                            <input
                                type="range"
                                min={0}
                                max={100}
                                step={5}
                                disabled={!secondaryModelId}
                                value={secondaryModelId ? trafficSplit : 0}
                                onChange={(e) => setTrafficSplit(Number(e.target.value))}
                                className="w-full mt-2 accent-orange-500 disabled:opacity-40"
                            />
                        </div>
                        <div>
                            <label className="block text-[11px] font-medium text-zinc-400 mb-1">
                                Auto-Preprocessing Pipeline (optional)
                            </label>
                            {(() => {
                                const primaryModel = models.find((pm) => pm.id === modelId);
                                const isNonTabular = primaryModel
                                    ? getModelCompatibilityProfile(primaryModel).domain !== "tabular"
                                    : false;
                                return (
                                    <select
                                        value={isNonTabular ? "" : pipelinePath}
                                        disabled={isNonTabular}
                                        onChange={(e) => setPipelinePath(e.target.value)}
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600 disabled:opacity-45"
                                    >
                                        <option value="">
                                            {isNonTabular ? "Not applicable for NLP models (tokenizer embedded)" : "Inherit from model / none"}
                                        </option>
                                        {!isNonTabular && pipelineDatasets.map((d) => (
                                            <option key={d.id} value={d.pipeline_path || ""}>
                                                {d.name}
                                            </option>
                                        ))}
                                    </select>
                                );
                            })()}
                        </div>
                    </div>

                    <div className="flex items-center justify-between pt-2 border-t border-zinc-800/80">
                        <input
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="Description (optional)"
                            className="w-full max-w-md h-8 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                        />
                        <button
                            type="submit"
                            disabled={submitting}
                            className="h-8 px-4 text-xs font-medium rounded-md bg-zinc-100 text-zinc-950 hover:bg-white disabled:opacity-50 flex items-center gap-1.5"
                        >
                            {submitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                            Create Endpoint
                        </button>
                    </div>
                </form>
            )}

            {/* Main split layout */}
            {loading ? (
                <div className="h-48 flex items-center justify-center rounded-lg border border-zinc-800 bg-[#101217]">
                    <Loader2 className="w-5 h-5 animate-spin text-zinc-500" />
                </div>
            ) : endpoints.length === 0 ? (
                <div className="rounded-lg border border-dashed border-zinc-800 bg-[#101217] py-16 text-center space-y-3">
                    <Server className="w-7 h-7 text-zinc-600 mx-auto" />
                    <p className="text-sm font-medium text-zinc-200">No model endpoints deployed</p>
                    <p className="text-xs text-zinc-500 max-w-md mx-auto">
                        Publish any registered Tabular or NLP model to get a dedicated HTTP endpoint authenticated with API keys and optional Canary traffic splitting.
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                    {/* Endpoints List */}
                    <div className="lg:col-span-4 rounded-lg border border-zinc-800 bg-[#101217] overflow-hidden">
                        <div className="px-4 py-2.5 border-b border-zinc-800 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                            Deployed Endpoints ({endpoints.length})
                        </div>
                        <div className="divide-y divide-zinc-800/80">
                            {endpoints.map((ep) => {
                                const isSel = selected?.id === ep.id;
                                const errRate = ep.error_rate_pct ?? 0;
                                return (
                                    <button
                                        key={ep.id}
                                        onClick={() => setSelected(ep)}
                                        className={`w-full text-left px-4 py-3 transition-colors ${
                                            isSel ? "bg-zinc-800/70" : "hover:bg-zinc-800/30"
                                        }`}
                                    >
                                        <div className="flex items-center justify-between gap-2">
                                            <span className="text-sm font-medium text-zinc-100 truncate">
                                                {ep.name}
                                            </span>
                                            <span
                                                className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border ${
                                                    ep.status === "active"
                                                        ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                                                        : "border-zinc-700 bg-zinc-800 text-zinc-400"
                                                }`}
                                            >
                                                {ep.status}
                                            </span>
                                        </div>
                                        <p className="text-[11px] font-mono text-zinc-400 truncate mt-0.5">
                                            {ep.invoke_path}
                                        </p>
                                        <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500 font-mono tabular-nums">
                                            <span className="truncate max-w-[55%]">
                                                {ep.model_name || ep.model_id.slice(0, 8)}
                                                {ep.secondary_model_id && ep.traffic_split_pct > 0
                                                    ? ` · ${ep.traffic_split_pct}% canary`
                                                    : ""}
                                            </span>
                                            <span>
                                                {ep.request_count} req · {ep.avg_latency_ms}ms
                                                {errRate > 0 ? ` · ${errRate}% err` : ""}
                                            </span>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Inspector, Observability & Live Console */}
                    {selected && (
                        <div className="lg:col-span-8 rounded-lg border border-zinc-800 bg-[#101217] p-5 space-y-5">
                            {/* Header */}
                            <div className="flex flex-wrap items-start justify-between gap-3 pb-4 border-b border-zinc-800">
                                <div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <h3 className="text-base font-semibold text-zinc-100">
                                            {selected.name}
                                        </h3>
                                        <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">
                                            {selected.framework || "mlflow"}
                                        </span>
                                        {selected.preprocessing_pipeline_path && (
                                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                                                auto-preprocess
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-xs font-mono text-orange-400 mt-1">
                                        POST {BASE_HOST}
                                        {selected.invoke_path}
                                    </p>
                                </div>
                                {canEdit && (
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => handleToggleStatus(selected)}
                                            className="h-8 px-2.5 text-xs rounded-md border border-zinc-800 bg-[#0b0c0f] text-zinc-300 hover:bg-zinc-800 flex items-center gap-1.5"
                                        >
                                            {selected.status === "active" ? (
                                                <>
                                                    <Pause className="w-3.5 h-3.5" /> Pause
                                                </>
                                            ) : (
                                                <>
                                                    <Play className="w-3.5 h-3.5" /> Resume
                                                </>
                                            )}
                                        </button>
                                        <button
                                            onClick={() => handleDelete(selected)}
                                            className="h-8 px-2.5 text-xs rounded-md border border-zinc-800 bg-[#0b0c0f] text-red-400 hover:bg-red-500/10 flex items-center gap-1.5"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                )}
                            </div>

                            {/* Rich Endpoint Observability KPI Strip */}
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2.5">
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[10px] uppercase tracking-wider text-zinc-500">Requests</p>
                                    <p className="text-base font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                                        {epMetrics?.total_requests ?? selected.request_count}
                                    </p>
                                    <p className="text-[10px] font-mono text-zinc-500 mt-0.5">
                                        {epMetrics ? `${epMetrics.requests_last_1h} /1h · ${epMetrics.requests_last_24h} /24h` : "—"}
                                    </p>
                                </div>
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[10px] uppercase tracking-wider text-zinc-500">Success Rate</p>
                                    <p className="text-base font-mono font-semibold tabular-nums text-emerald-400 mt-0.5">
                                        {(epMetrics?.success_rate_pct ?? selected.success_rate_pct ?? 100).toFixed(1)}%
                                    </p>
                                    <p className="text-[10px] font-mono text-zinc-500 mt-0.5">
                                        {epMetrics?.total_errors ?? selected.error_count} errors ({(epMetrics?.error_rate_pct ?? 0).toFixed(1)}%)
                                    </p>
                                </div>
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[10px] uppercase tracking-wider text-zinc-500">Avg / P50</p>
                                    <p className="text-base font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                                        {epMetrics?.avg_latency_ms ?? selected.avg_latency_ms}ms
                                    </p>
                                    <p className="text-[10px] font-mono text-zinc-500 mt-0.5">
                                        p50: {epMetrics?.p50_latency_ms ?? 0}ms
                                    </p>
                                </div>
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[10px] uppercase tracking-wider text-zinc-500">P95 / P99</p>
                                    <p className="text-base font-mono font-semibold tabular-nums text-orange-400 mt-0.5">
                                        {epMetrics?.p95_latency_ms ?? 0}ms
                                    </p>
                                    <p className="text-[10px] font-mono text-zinc-500 mt-0.5">
                                        p99: {epMetrics?.p99_latency_ms ?? 0}ms
                                    </p>
                                </div>
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[10px] uppercase tracking-wider text-zinc-500">Rows Scored</p>
                                    <p className="text-base font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                                        {epMetrics?.total_rows_scored ?? 0}
                                    </p>
                                    <p className="text-[10px] font-mono text-zinc-500 mt-0.5">
                                        min {epMetrics?.min_latency_ms ?? 0} · max {epMetrics?.max_latency_ms ?? 0}ms
                                    </p>
                                </div>
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[10px] uppercase tracking-wider text-zinc-500">FinOps &amp; Telemetry</p>
                                    <p className="text-base font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                                        ${(epMetrics?.estimated_cost_usd ?? 0).toFixed(4)}
                                    </p>
                                    <p className="text-[10px] font-mono text-zinc-500 mt-0.5">
                                        {epMetrics?.usage_events_count ?? 0} usage events
                                    </p>
                                </div>
                            </div>

                            {/* Last Error Banner */}
                            {epMetrics?.last_error_message && (
                                <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 flex items-center gap-2 text-xs text-red-300 font-mono">
                                    <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-red-400" />
                                    <span className="truncate">Last error: {epMetrics.last_error_message}</span>
                                </div>
                            )}

                            {/* Latency Timeline + Traffic Routing Telemetry */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {/* Latency Sparkline Bars */}
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3.5 space-y-2.5">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[11px] font-medium text-zinc-300 flex items-center gap-1.5">
                                            <BarChart3 className="w-3.5 h-3.5 text-orange-400" />
                                            Recent Latency Timeline
                                        </span>
                                        <span className="text-[10px] font-mono text-zinc-500">
                                            last {epMetrics?.latency_series.length ?? 0} calls
                                        </span>
                                    </div>
                                    {epMetrics && epMetrics.latency_series.length > 0 ? (
                                        <div className="h-20 flex items-end gap-1 pt-2">
                                            {epMetrics.latency_series.map((pt, idx) => {
                                                const heightPct = Math.max(
                                                    10,
                                                    Math.round((pt.latency_ms / maxSeriesLatency) * 100)
                                                );
                                                const barCls =
                                                    pt.status !== "ok"
                                                        ? "bg-red-500/80"
                                                        : pt.variant === "canary"
                                                        ? "bg-orange-400/90"
                                                        : "bg-emerald-400/80";
                                                return (
                                                    <div
                                                        key={idx}
                                                        title={`${pt.variant} · ${pt.latency_ms}ms · ${pt.status}${pt.ts ? " @ " + new Date(pt.ts).toLocaleTimeString() : ""}`}
                                                        className={`flex-1 rounded-t-sm transition-all ${barCls}`}
                                                        style={{ height: `${heightPct}%` }}
                                                    />
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="h-20 flex items-center justify-center text-[11px] text-zinc-500">
                                            Invoke endpoint to plot real-time latency bars
                                        </div>
                                    )}
                                    <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500">
                                        <span className="flex items-center gap-2">
                                            <span className="inline-flex items-center gap-1">
                                                <span className="w-2 h-2 rounded-xs bg-emerald-400/80" /> primary
                                            </span>
                                            <span className="inline-flex items-center gap-1">
                                                <span className="w-2 h-2 rounded-xs bg-orange-400/90" /> canary
                                            </span>
                                            <span className="inline-flex items-center gap-1">
                                                <span className="w-2 h-2 rounded-xs bg-red-500/80" /> error
                                            </span>
                                        </span>
                                        <span>peak {maxSeriesLatency.toFixed(1)}ms</span>
                                    </div>
                                </div>

                                {/* Canary / Variant Split & Live Control */}
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3.5 space-y-2.5">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[11px] font-medium text-zinc-300 flex items-center gap-1.5">
                                            <GitBranch className="w-3.5 h-3.5 text-zinc-400" />
                                            Traffic Routing &amp; Variants
                                        </span>
                                        <span className="text-[10px] font-mono text-zinc-500">
                                            target {selected.traffic_split_pct}% canary
                                        </span>
                                    </div>
                                    <div className="space-y-1.5 text-[11px] font-mono">
                                        <div className="flex items-center justify-between">
                                            <span className="text-zinc-300 truncate max-w-[55%]">
                                                primary: {selected.model_name || selected.model_id.slice(0, 8)}
                                            </span>
                                            <span className="text-emerald-400 tabular-nums">
                                                {epMetrics?.by_variant?.primary?.count ?? 0} ({epMetrics?.by_variant?.primary?.pct ?? 0}%) ·{" "}
                                                {epMetrics?.by_variant?.primary?.avg_latency_ms ?? 0}ms
                                            </span>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-zinc-400 truncate max-w-[55%]">
                                                canary: {selected.secondary_model_name || "none"}
                                            </span>
                                            <span className="text-orange-400 tabular-nums">
                                                {epMetrics?.by_variant?.canary?.count ?? 0} ({epMetrics?.by_variant?.canary?.pct ?? 0}%) ·{" "}
                                                {epMetrics?.by_variant?.canary?.avg_latency_ms ?? 0}ms
                                            </span>
                                        </div>
                                    </div>
                                    {canEdit && selected.secondary_model_id && (
                                        <div className="pt-2 border-t border-zinc-800/80 flex items-center gap-2">
                                            <SlidersHorizontal className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
                                            <input
                                                type="range"
                                                min={0}
                                                max={100}
                                                step={5}
                                                value={liveSplitDraft}
                                                onChange={(e) => setLiveSplitDraft(Number(e.target.value))}
                                                className="flex-1 accent-orange-500"
                                            />
                                            <span className="text-[11px] font-mono text-zinc-300 w-8 text-right">
                                                {liveSplitDraft}%
                                            </span>
                                            <button
                                                type="button"
                                                disabled={updatingSplit || liveSplitDraft === selected.traffic_split_pct}
                                                onClick={handleUpdateCanarySplit}
                                                className="h-6 px-2 rounded bg-zinc-100 text-zinc-950 text-[10px] font-medium hover:bg-white disabled:opacity-40"
                                            >
                                                Apply
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Live Test Console */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <label className="text-xs font-medium text-zinc-300">
                                        Test Invocation Payload (JSON)
                                    </label>
                                    <button
                                        onClick={handleInvoke}
                                        disabled={invoking || selected.status !== "active"}
                                        className="h-7 px-3 text-xs font-medium rounded-md bg-zinc-100 text-zinc-950 hover:bg-white disabled:opacity-50 flex items-center gap-1.5"
                                    >
                                        {invoking ? (
                                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                        ) : (
                                            <Send className="w-3 h-3" />
                                        )}
                                        Send Request
                                    </button>
                                </div>
                                <textarea
                                    rows={5}
                                    value={payloadJson}
                                    onChange={(e) => setPayloadJson(e.target.value)}
                                    className="w-full p-3 rounded-md bg-[#0b0c0f] border border-zinc-800 font-mono text-xs text-zinc-200 focus:outline-none focus:border-zinc-600"
                                />
                                {invokeResult && (
                                    <pre className="p-3 rounded-md bg-[#0b0c0f] border border-zinc-800 font-mono text-xs text-emerald-300 overflow-x-auto max-h-48">
                                        {JSON.stringify(invokeResult, null, 2)}
                                    </pre>
                                )}
                            </div>

                            {/* cURL Snippet */}
                            <div className="space-y-1.5">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-medium text-zinc-400">
                                        cURL Integration (create a PAT in Governance &gt; API Keys)
                                    </span>
                                    <button
                                        onClick={() => copyText(curlSnippet, "curl")}
                                        className="text-xs text-zinc-400 hover:text-zinc-100 flex items-center gap-1"
                                    >
                                        {copiedId === "curl" ? (
                                            <>
                                                <Check className="w-3.5 h-3.5 text-emerald-400" /> Copied
                                            </>
                                        ) : (
                                            <>
                                                <Copy className="w-3.5 h-3.5" /> Copy
                                            </>
                                        )}
                                    </button>
                                </div>
                                <pre className="p-3 rounded-md bg-[#0b0c0f] border border-zinc-800 font-mono text-[11px] text-zinc-300 overflow-x-auto">
                                    {curlSnippet}
                                </pre>
                            </div>

                            {/* Recent Invocation Logs with filters */}
                            <div className="space-y-2.5 pt-3 border-t border-zinc-800">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className="text-xs font-semibold text-zinc-300 flex items-center gap-1.5">
                                            <Activity className="w-3.5 h-3.5 text-zinc-400" />
                                            Invocation Logs ({invocationLogs.length})
                                        </span>
                                        <div className="flex items-center gap-1">
                                            {[
                                                { label: "All", s: "", v: "" },
                                                { label: "OK", s: "ok", v: "" },
                                                { label: "Errors", s: "error", v: "" },
                                                { label: "Primary", s: "", v: "primary" },
                                                { label: "Canary", s: "", v: "canary" },
                                            ].map((f) => {
                                                const active =
                                                    logStatusFilter === f.s &&
                                                    logVariantFilter === f.v;
                                                return (
                                                    <button
                                                        key={f.label}
                                                        type="button"
                                                        onClick={() => {
                                                            setLogStatusFilter(f.s);
                                                            setLogVariantFilter(f.v);
                                                            fetchLogs(selected.id, f.s, f.v);
                                                        }}
                                                        className={`h-6 px-2 rounded text-[10px] font-mono border transition-colors ${
                                                            active
                                                                ? "bg-zinc-100 text-zinc-950 border-zinc-100"
                                                                : "bg-[#0b0c0f] text-zinc-400 border-zinc-800 hover:text-zinc-200"
                                                        }`}
                                                    >
                                                        {f.label}
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>
                                    <button
                                        onClick={() =>
                                            fetchLogs(selected.id, logStatusFilter, logVariantFilter)
                                        }
                                        className="text-[11px] text-zinc-400 hover:text-zinc-100 flex items-center gap-1"
                                    >
                                        <RefreshCw
                                            className={`w-3 h-3 ${loadingLogs ? "animate-spin" : ""}`}
                                        />{" "}
                                        Refresh
                                    </button>
                                </div>
                                {invocationLogs.length === 0 ? (
                                    <p className="text-xs text-zinc-500 py-3">
                                        No matching invocation logs captured yet.
                                    </p>
                                ) : (
                                    <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] divide-y divide-zinc-800/80 max-h-56 overflow-y-auto font-mono text-[11px]">
                                        {invocationLogs.map((log) => (
                                            <div key={log.id} className="px-3 py-2 space-y-1">
                                                <div className="flex items-center justify-between gap-2">
                                                    <div className="flex items-center gap-2">
                                                        <span
                                                            className={`px-1.5 py-0.5 rounded text-[10px] uppercase border ${
                                                                log.status === "ok"
                                                                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                                                                    : "border-red-500/30 bg-red-500/10 text-red-400"
                                                            }`}
                                                        >
                                                            {log.status}
                                                        </span>
                                                        <span
                                                            className={
                                                                log.variant === "canary"
                                                                    ? "text-orange-400"
                                                                    : "text-zinc-300"
                                                            }
                                                        >
                                                            {log.variant}
                                                        </span>
                                                        <span className="text-zinc-500">
                                                            {log.runtime}
                                                        </span>
                                                        <span className="text-zinc-300 tabular-nums">
                                                            {log.latency_ms}ms
                                                        </span>
                                                    </div>
                                                    <span className="text-[10px] text-zinc-500">
                                                        {log.created_at
                                                            ? new Date(
                                                                  log.created_at
                                                              ).toLocaleTimeString()
                                                            : ""}
                                                    </span>
                                                </div>
                                                <div className="text-[10px] text-zinc-500 truncate">
                                                    in: {JSON.stringify(log.sample_input)} →{" "}
                                                    {log.status === "ok"
                                                        ? `out: ${JSON.stringify(log.sample_output?.predictions)}`
                                                        : `err: ${log.error_message}`}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
