"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
    Check,
    Copy,
    Loader2,
    Pause,
    Play,
    Plus,
    RefreshCw,
    Send,
    Server,
    Trash2,
} from "lucide-react";
import toast from "react-hot-toast";
import { Dataset, MLModel } from "@/lib/api";
import { config } from "@/lib/config";

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
    avg_latency_ms: number;
    last_invoked_at?: string | null;
    created_at?: string | null;
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
    if (ep.framework === "huggingface" || ep.framework === "onnx_nlp") {
        return JSON.stringify({ inputs: [{ text: "Sample input text for inference" }] }, null, 2);
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

    const fetchLogs = useCallback(async (endpointId: string) => {
        if (!token) return;
        setLoadingLogs(true);
        try {
            const res = await fetch(`${API}/serving/endpoints/${endpointId}/logs?limit=25`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                const data = await res.json();
                setInvocationLogs(data.items || []);
            }
        } catch {
            setInvocationLogs([]);
        } finally {
            setLoadingLogs(false);
        }
    }, [token]);

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
            fetchLogs(selected.id);
        } else {
            setInvocationLogs([]);
        }
    }, [selected?.id, fetchLogs]);

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
            }
        } catch {
            toast.error("Failed to update endpoint status");
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
                await fetchEndpoints();
                await fetchLogs(selected.id);
            }
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

    return (
        <div className="space-y-6">
            {/* Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3 text-xs text-zinc-400">
                    <span className="font-mono tabular-nums text-zinc-200">{endpoints.length}</span>
                    <span>deployed endpoints</span>
                    <span className="text-zinc-700">|</span>
                    <span className="font-mono text-zinc-500">POST /serving/v1/&lt;slug&gt;/invocations</span>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={fetchEndpoints}
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
                        <h3 className="text-sm font-semibold text-zinc-100">Deploy Model Endpoint</h3>
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
                                onChange={(e) => setModelId(e.target.value)}
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
                                    .filter((m) => m.id !== modelId)
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
                            <select
                                value={pipelinePath}
                                onChange={(e) => setPipelinePath(e.target.value)}
                                className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                            >
                                <option value="">Inherit from model / none</option>
                                {pipelineDatasets.map((d) => (
                                    <option key={d.id} value={d.pipeline_path || ""}>
                                        {d.name}
                                    </option>
                                ))}
                            </select>
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
                        Publish any registered model to get a dedicated HTTP endpoint authenticated with API keys and optional Canary traffic splitting.
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                    {/* Endpoints Table */}
                    <div className="lg:col-span-5 rounded-lg border border-zinc-800 bg-[#101217] overflow-hidden">
                        <div className="px-4 py-2.5 border-b border-zinc-800 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                            Active & Paused Endpoints
                        </div>
                        <div className="divide-y divide-zinc-800/80">
                            {endpoints.map((ep) => {
                                const isSel = selected?.id === ep.id;
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
                                            </span>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Inspector & Live Console */}
                    {selected && (
                        <div className="lg:col-span-7 rounded-lg border border-zinc-800 bg-[#101217] p-5 space-y-5">
                            <div className="flex flex-wrap items-start justify-between gap-3 pb-4 border-b border-zinc-800">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <h3 className="text-base font-semibold text-zinc-100">
                                            {selected.name}
                                        </h3>
                                        <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">
                                            {selected.framework || "mlflow"}
                                        </span>
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

                            {/* Stats strip */}
                            <div className="grid grid-cols-3 gap-3">
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[11px] text-zinc-500">Requests</p>
                                    <p className="text-lg font-mono font-semibold tabular-nums text-zinc-100">
                                        {selected.request_count}
                                    </p>
                                </div>
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[11px] text-zinc-500">Avg Latency</p>
                                    <p className="text-lg font-mono font-semibold tabular-nums text-zinc-100">
                                        {selected.avg_latency_ms} ms
                                    </p>
                                </div>
                                <div className="rounded-md border border-zinc-800 bg-[#0b0c0f] p-3">
                                    <p className="text-[11px] text-zinc-500">Errors</p>
                                    <p className="text-lg font-mono font-semibold tabular-nums text-zinc-100">
                                        {selected.error_count}
                                    </p>
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
                                    rows={6}
                                    value={payloadJson}
                                    onChange={(e) => setPayloadJson(e.target.value)}
                                    className="w-full p-3 rounded-md bg-[#0b0c0f] border border-zinc-800 font-mono text-xs text-zinc-200 focus:outline-none focus:border-zinc-600"
                                />
                                {invokeResult && (
                                    <pre className="p-3 rounded-md bg-[#0b0c0f] border border-zinc-800 font-mono text-xs text-emerald-300 overflow-x-auto">
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

                            {/* Recent Invocation Logs */}
                            <div className="space-y-2 pt-3 border-t border-zinc-800">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-semibold text-zinc-300">
                                        Recent Invocation Logs ({invocationLogs.length})
                                    </span>
                                    <button
                                        onClick={() => fetchLogs(selected.id)}
                                        className="text-[11px] text-zinc-400 hover:text-zinc-100 flex items-center gap-1"
                                    >
                                        <RefreshCw className={`w-3 h-3 ${loadingLogs ? "animate-spin" : ""}`} /> Refresh
                                    </button>
                                </div>
                                {invocationLogs.length === 0 ? (
                                    <p className="text-xs text-zinc-500 py-3">
                                        No traffic logged yet. Send a test request above to capture live invocation logs.
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
                                                        <span className="text-zinc-300">{log.variant}</span>
                                                        <span className="text-zinc-500">{log.runtime}</span>
                                                        <span className="text-zinc-400 tabular-nums">{log.latency_ms}ms</span>
                                                    </div>
                                                    <span className="text-[10px] text-zinc-500">
                                                        {log.created_at ? new Date(log.created_at).toLocaleTimeString() : ""}
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
