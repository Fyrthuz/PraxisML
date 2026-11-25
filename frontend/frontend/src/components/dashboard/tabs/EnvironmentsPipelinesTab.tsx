"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
    Box,
    Code2,
    Cpu,
    Loader2,
    Play,
    Plus,
    RefreshCw,
    Trash2,
} from "lucide-react";
import toast from "react-hot-toast";
import { Dataset } from "@/lib/api";
import { config } from "@/lib/config";

interface EnvironmentsPipelinesTabProps {
    token: string | null;
    datasets: Dataset[];
    userRole: string;
    onDatasetCreated?: () => void;
}

const API = config.getFullApiUrl("");

const DEFAULT_PYTHON_TEMPLATE = `def transform(df):
    # Define your custom transformation using any package installed in the selected environment
    out = df.copy()
    return out
`;

export default function EnvironmentsPipelinesTab({
    token,
    datasets,
    userRole,
    onDatasetCreated,
}: EnvironmentsPipelinesTabProps) {
    const canEdit = userRole === "admin" || userRole === "editor";
    const [subTab, setSubTab] = useState<"environments" | "pipelines" | "capabilities">("environments");
    const [loading, setLoading] = useState(false);

    // Environments
    const [environments, setEnvironments] = useState<any[]>([]);
    const [envName, setEnvName] = useState("");
    const [envPython, setEnvPython] = useState("3.10");
    const [envBaseImage, setEnvBaseImage] = useState("python:3.10-slim");
    const [envPackages, setEnvPackages] = useState("pandas, numpy, scikit-learn");
    const [envProfile, setEnvProfile] = useState("cpu-heavy");

    // User Pipelines
    const [pipelines, setPipelines] = useState<any[]>([]);
    const [pipeName, setPipeName] = useState("");
    const [pipeEnvId, setPipeEnvId] = useState("");
    const [pipeCode, setPipeCode] = useState(DEFAULT_PYTHON_TEMPLATE);
    const [runDatasetId, setRunDatasetId] = useState("");
    const [runOutputName, setRunOutputName] = useState("");
    const [runningId, setRunningId] = useState<string | null>(null);
    const [lastRunOutput, setLastRunOutput] = useState<any>(null);

    // Pluggable capabilities
    const [capabilities, setCapabilities] = useState<any>(null);

    const tabularDatasets = datasets.filter(
        (d) => d.file_type && ["csv", "xlsx", "parquet"].includes(d.file_type)
    );

    const loadAll = useCallback(async () => {
        if (!token) return;
        setLoading(true);
        const headers = { Authorization: `Bearer ${token}` };
        try {
            const [envRes, pipeRes, capRes] = await Promise.all([
                fetch(`${API}/environments/`, { headers }),
                fetch(`${API}/pipelines/`, { headers }),
                fetch(`${API}/environments/capabilities`, { headers }),
            ]);
            if (envRes.ok) {
                const envData = await envRes.json();
                setEnvironments(envData.items || []);
                if (!pipeEnvId && envData.items?.[0]?.id) {
                    setPipeEnvId(envData.items[0].id);
                }
            }
            if (pipeRes.ok) setPipelines((await pipeRes.json()).items || []);
            if (capRes.ok) setCapabilities(await capRes.json());
        } catch {
            toast.error("Failed to load runtime environments");
        } finally {
            setLoading(false);
        }
    }, [token, pipeEnvId]);

    useEffect(() => {
        loadAll();
    }, [loadAll]);

    const handleCreateEnv = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!token || !envName.trim()) return;
        const packages = envPackages
            .split(/[,\n]+/)
            .map((p) => p.trim())
            .filter(Boolean);

        const res = await fetch(`${API}/environments/`, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${token}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                name: envName.trim(),
                python_version: envPython,
                base_image: envBaseImage.trim() || "python:3.10-slim",
                packages,
                compute_profile: envProfile,
                runner_type: "uv_isolated",
            }),
        });
        const data = await res.json();
        if (!res.ok) {
            toast.error(data.detail || "Failed to create environment");
            return;
        }
        setEnvName("");
        toast.success(`Environment "${data.name}" created`);
        await loadAll();
    };

    const handleDeleteEnv = async (id: string) => {
        if (!token) return;
        const res = await fetch(`${API}/environments/${id}`, {
            method: "DELETE",
            headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
            toast.success("Environment deleted");
            await loadAll();
        } else {
            const d = await res.json().catch(() => ({}));
            toast.error(d.detail || "Cannot delete environment");
        }
    };

    const handleCreatePipeline = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!token || !pipeName.trim()) return;
        const res = await fetch(`${API}/pipelines/`, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${token}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                name: pipeName.trim(),
                environment_id: pipeEnvId || undefined,
                engine: "python",
                custom_code: pipeCode,
            }),
        });
        const data = await res.json();
        if (!res.ok) {
            toast.error(data.detail || "Failed to create pipeline");
            return;
        }
        setPipeName("");
        toast.success(`Pipeline "${data.name}" saved`);
        await loadAll();
    };

    const handleRunPipeline = async (pipelineId: string) => {
        if (!token || !runDatasetId) {
            toast.error("Select a target dataset to run the pipeline on.");
            return;
        }
        setRunningId(pipelineId);
        setLastRunOutput(null);
        try {
            const res = await fetch(`${API}/pipelines/${pipelineId}/run`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    dataset_id: runDatasetId,
                    output_dataset_name: runOutputName.trim() || undefined,
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                toast.error(data.detail || "Pipeline execution failed");
                return;
            }
            setLastRunOutput(data);
            toast.success(`Dataset "${data.output_dataset_name}" created (${data.transformed_shape.join("×")})`);
            onDatasetCreated?.();
        } catch {
            toast.error("Error running pipeline");
        } finally {
            setRunningId(null);
        }
    };

    const handleDeletePipeline = async (id: string) => {
        if (!token) return;
        const res = await fetch(`${API}/pipelines/${id}`, {
            method: "DELETE",
            headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
            toast.success("Pipeline deleted");
            await loadAll();
        }
    };

    return (
        <div className="space-y-5">
            {/* Sub-tabs */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-3">
                <div className="flex items-center gap-1.5">
                    {[
                        { id: "environments", label: "Runtime Environments", icon: <Box className="w-3.5 h-3.5" /> },
                        { id: "pipelines", label: "Custom Pipelines", icon: <Code2 className="w-3.5 h-3.5" /> },
                        { id: "capabilities", label: "Pluggable Backends", icon: <Cpu className="w-3.5 h-3.5" /> },
                    ].map((t) => (
                        <button
                            key={t.id}
                            onClick={() => setSubTab(t.id as any)}
                            className={`h-8 px-3 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors ${
                                subTab === t.id
                                    ? "bg-zinc-800 text-zinc-100"
                                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                            }`}
                        >
                            {t.icon}
                            {t.label}
                        </button>
                    ))}
                </div>
                <button
                    onClick={loadAll}
                    className="h-8 px-3 text-xs rounded-md border border-zinc-800 bg-[#101217] text-zinc-300 hover:bg-zinc-800 flex items-center gap-1.5"
                >
                    {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                    Refresh
                </button>
            </div>

            {/* Runtime Environments */}
            {subTab === "environments" && (
                <div className="space-y-5">
                    {canEdit && (
                        <form
                            onSubmit={handleCreateEnv}
                            className="rounded-lg border border-zinc-800 bg-[#101217] p-5 space-y-4"
                        >
                            <h3 className="text-sm font-semibold text-zinc-100">
                                Define Runtime Environment
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                                <div>
                                    <label className="block text-[11px] text-zinc-400 mb-1">Environment Name</label>
                                    <input
                                        value={envName}
                                        onChange={(e) => setEnvName(e.target.value)}
                                        placeholder="xgboost-polars-311"
                                        required
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                                    />
                                </div>
                                <div>
                                    <label className="block text-[11px] text-zinc-400 mb-1">Python</label>
                                    <select
                                        value={envPython}
                                        onChange={(e) => setEnvPython(e.target.value)}
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs font-mono text-zinc-100"
                                    >
                                        <option value="3.10">3.10</option>
                                        <option value="3.11">3.11</option>
                                        <option value="3.12">3.12</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-[11px] text-zinc-400 mb-1">Base Container Image</label>
                                    <input
                                        value={envBaseImage}
                                        onChange={(e) => setEnvBaseImage(e.target.value)}
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs font-mono text-zinc-100"
                                    />
                                </div>
                                <div>
                                    <label className="block text-[11px] text-zinc-400 mb-1">Compute Profile</label>
                                    <select
                                        value={envProfile}
                                        onChange={(e) => setEnvProfile(e.target.value)}
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs font-mono text-zinc-100"
                                    >
                                        <option value="cpu-light">cpu-light</option>
                                        <option value="cpu-heavy">cpu-heavy</option>
                                        <option value="gpu-train">gpu-train</option>
                                    </select>
                                </div>
                            </div>
                            <div className="flex flex-wrap items-end gap-3">
                                <div className="flex-1 min-w-[260px]">
                                    <label className="block text-[11px] text-zinc-400 mb-1">
                                        PyPI Packages (comma or newline separated — resolved in isolation via uv)
                                    </label>
                                    <input
                                        value={envPackages}
                                        onChange={(e) => setEnvPackages(e.target.value)}
                                        placeholder="polars>=1.10, xgboost>=2.1, lightgbm, optuna"
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs font-mono text-zinc-100"
                                    />
                                </div>
                                <button
                                    type="submit"
                                    className="h-9 px-4 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium flex items-center gap-1.5"
                                >
                                    <Plus className="w-3.5 h-3.5" /> Create Environment
                                </button>
                            </div>
                        </form>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {environments.map((env) => (
                            <div
                                key={env.id}
                                className="rounded-lg border border-zinc-800 bg-[#101217] p-4 space-y-3"
                            >
                                <div className="flex items-start justify-between gap-2">
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <h4 className="text-sm font-semibold text-zinc-100">{env.name}</h4>
                                            {env.is_default && (
                                                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
                                                    default
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-[11px] font-mono text-zinc-500 mt-0.5">
                                            py{env.python_version} · {env.base_image} · {env.compute_profile} · runner:{env.runner_type}
                                        </p>
                                    </div>
                                    {!env.is_default && canEdit && (
                                        <button
                                            onClick={() => handleDeleteEnv(env.id)}
                                            className="p-1.5 text-zinc-500 hover:text-red-400 rounded"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                    )}
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                    {(env.packages || []).map((pkg: string) => (
                                        <span
                                            key={pkg}
                                            className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#0b0c0f] border border-zinc-800 text-zinc-300"
                                        >
                                            {pkg}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* User-Defined Reusable Pipelines */}
            {subTab === "pipelines" && (
                <div className="space-y-5">
                    {canEdit && (
                        <form
                            onSubmit={handleCreatePipeline}
                            className="rounded-lg border border-zinc-800 bg-[#101217] p-5 space-y-4"
                        >
                            <div className="flex flex-wrap items-center justify-between gap-2">
                                <div>
                                    <h3 className="text-sm font-semibold text-zinc-100">
                                        Define Custom Python Pipeline
                                    </h3>
                                    <p className="text-xs text-zinc-500">
                                        Write `def transform(df):` using any library declared in the selected Runtime Environment.
                                    </p>
                                </div>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                <div>
                                    <label className="block text-[11px] text-zinc-400 mb-1">Pipeline Name</label>
                                    <input
                                        value={pipeName}
                                        onChange={(e) => setPipeName(e.target.value)}
                                        placeholder="titanic-feature-engineering"
                                        required
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                                    />
                                </div>
                                <div>
                                    <label className="block text-[11px] text-zinc-400 mb-1">Runtime Environment</label>
                                    <select
                                        value={pipeEnvId}
                                        onChange={(e) => setPipeEnvId(e.target.value)}
                                        className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                                    >
                                        {environments.map((env) => (
                                            <option key={env.id} value={env.id}>
                                                {env.name} ({(env.packages || []).join(", ")})
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                            <div>
                                <label className="block text-[11px] text-zinc-400 mb-1">
                                    Transformation Code (`def transform(df) -&gt; pd.DataFrame`)
                                </label>
                                <textarea
                                    rows={7}
                                    value={pipeCode}
                                    onChange={(e) => setPipeCode(e.target.value)}
                                    className="w-full p-3 rounded-md bg-[#0b0c0f] border border-zinc-800 font-mono text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                                />
                            </div>
                            <div className="flex justify-end">
                                <button
                                    type="submit"
                                    className="h-9 px-4 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium flex items-center gap-1.5"
                                >
                                    <Plus className="w-3.5 h-3.5" /> Save Reusable Pipeline
                                </button>
                            </div>
                        </form>
                    )}

                    {/* Run target dataset selector */}
                    <div className="rounded-lg border border-zinc-800 bg-[#101217] p-4 flex flex-wrap items-end gap-3">
                        <div className="flex-1 min-w-[220px]">
                            <label className="block text-[11px] text-zinc-400 mb-1">
                                Target Dataset for Execution
                            </label>
                            <select
                                value={runDatasetId}
                                onChange={(e) => setRunDatasetId(e.target.value)}
                                className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                            >
                                <option value="">Select tabular dataset...</option>
                                {tabularDatasets.map((ds) => (
                                    <option key={ds.id} value={ds.id}>
                                        {ds.name} ({ds.num_rows}×{ds.num_columns})
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="flex-1 min-w-[200px]">
                            <label className="block text-[11px] text-zinc-400 mb-1">
                                Output Dataset Name (optional)
                            </label>
                            <input
                                value={runOutputName}
                                onChange={(e) => setRunOutputName(e.target.value)}
                                placeholder="e.g. titanic_custom_features"
                                className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                            />
                        </div>
                    </div>

                    {lastRunOutput && (
                        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-xs font-mono text-emerald-300 space-y-1">
                            <div>
                                Output dataset: <strong>{lastRunOutput.output_dataset_name}</strong> ({lastRunOutput.transformed_shape?.join("×")})
                            </div>
                            <div className="text-zinc-300">
                                Columns: {(lastRunOutput.transformed_columns || []).join(", ")}
                            </div>
                        </div>
                    )}

                    <div className="space-y-3">
                        {pipelines.map((p) => (
                            <div
                                key={p.id}
                                className="rounded-lg border border-zinc-800 bg-[#101217] p-4 flex flex-wrap items-center justify-between gap-4"
                            >
                                <div className="min-w-0">
                                    <div className="flex items-center gap-2">
                                        <h4 className="text-sm font-semibold text-zinc-100">{p.name}</h4>
                                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-orange-300">
                                            {p.engine}
                                        </span>
                                        <span className="text-[10px] font-mono text-zinc-500">v{p.version}</span>
                                    </div>
                                    <p className="text-xs font-mono text-zinc-500 mt-1">
                                        env: {p.environment_name || "default"}
                                    </p>
                                </div>
                                <div className="flex items-center gap-2">
                                    {canEdit && (
                                        <>
                                            <button
                                                onClick={() => handleRunPipeline(p.id)}
                                                disabled={!runDatasetId || runningId === p.id}
                                                className="h-8 px-3 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium disabled:opacity-40 flex items-center gap-1.5"
                                            >
                                                {runningId === p.id ? (
                                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                ) : (
                                                    <Play className="w-3.5 h-3.5" />
                                                )}
                                                Run on Dataset
                                            </button>
                                            <button
                                                onClick={() => handleDeletePipeline(p.id)}
                                                className="h-8 w-8 rounded-md border border-zinc-800 bg-[#0b0c0f] text-zinc-500 hover:text-red-400 flex items-center justify-center"
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        </>
                                    )}
                                </div>
                            </div>
                        ))}
                        {pipelines.length === 0 && (
                            <div className="rounded-lg border border-dashed border-zinc-800 bg-[#101217] py-12 text-center text-xs text-zinc-500">
                                No custom pipelines saved yet. Create one above to run reusable code across datasets.
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Pluggable Backends Introspection */}
            {subTab === "capabilities" && capabilities && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[
                        { title: "Storage Backends", items: capabilities.storage_backends },
                        { title: "Compute Backends", items: capabilities.compute_backends },
                        { title: "Environment Runners", items: capabilities.environment_runners },
                        { title: "Serving Runtimes", items: capabilities.serving_runtimes },
                        { title: "Event Bus Backends", items: capabilities.event_bus_backends },
                        { title: "Data Engines", items: capabilities.data_engines },
                        { title: "Experiment Trackers", items: capabilities.tracker_backends },
                    ].map((card) => (
                        <div
                            key={card.title}
                            className="rounded-lg border border-zinc-800 bg-[#101217] p-4 space-y-2.5"
                        >
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                                {card.title}
                            </h4>
                            <div className="flex flex-wrap gap-1.5">
                                {(card.items || []).map((name: string) => (
                                    <span
                                        key={name}
                                        className="text-xs font-mono px-2 py-0.5 rounded bg-[#0b0c0f] border border-zinc-800 text-zinc-200"
                                    >
                                        {name}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
