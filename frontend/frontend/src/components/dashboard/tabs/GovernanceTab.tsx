"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
    Check,
    Copy,
    FolderKanban,
    KeyRound,
    Loader2,
    Plus,
    RefreshCw,
    ScrollText,
    Trash2,
} from "lucide-react";
import toast from "react-hot-toast";
import { config } from "@/lib/config";

interface GovernanceTabProps {
    token: string | null;
    userRole: string;
}

const API = config.getFullApiUrl("");

export default function GovernanceTab({ token, userRole }: GovernanceTabProps) {
    const canEdit = userRole === "admin" || userRole === "editor";
    const [subTab, setSubTab] = useState<"api_keys" | "workspaces" | "audit">("api_keys");
    const [loading, setLoading] = useState(false);

    // API Keys
    const [keys, setKeys] = useState<any[]>([]);
    const [keyName, setKeyName] = useState("");
    const [keyScopes, setKeyScopes] = useState("serving:invoke,models:read");
    const [createdSecret, setCreatedSecret] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);

    // Workspaces
    const [workspaces, setWorkspaces] = useState<any[]>([]);
    const [wsName, setWsName] = useState("");
    const [wsEnv, setWsEnv] = useState("dev");
    const [wsCompute, setWsCompute] = useState("cpu-light");

    // Audit & Events
    const [auditRows, setAuditRows] = useState<any[]>([]);
    const [domainEvents, setDomainEvents] = useState<any[]>([]);
    const [showEvents, setShowEvents] = useState(false);

    const loadAll = useCallback(async () => {
        if (!token) return;
        setLoading(true);
        const headers = { Authorization: `Bearer ${token}` };
        try {
            const [keysRes, wsRes, auditRes, eventsRes] = await Promise.all([
                fetch(`${API}/api-keys/`, { headers }),
                fetch(`${API}/workspaces/`, { headers }),
                fetch(`${API}/audit/?limit=50`, { headers }),
                fetch(`${API}/audit/events?limit=30`, { headers }),
            ]);
            if (keysRes.ok) setKeys((await keysRes.json()).items || []);
            if (wsRes.ok) setWorkspaces((await wsRes.json()).items || []);
            if (auditRes.ok) setAuditRows((await auditRes.json()).items || []);
            if (eventsRes.ok) setDomainEvents((await eventsRes.json()).events || []);
        } catch {
            toast.error("Failed to load governance data");
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => {
        loadAll();
    }, [loadAll]);

    const handleCreateKey = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!token || !keyName.trim()) return;
        const scopes = keyScopes
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
        try {
            const res = await fetch(`${API}/api-keys/`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ name: keyName.trim(), scopes: scopes.length ? scopes : ["*"] }),
            });
            const data = await res.json();
            if (!res.ok) {
                toast.error(data.detail || "Failed to create API key");
                return;
            }
            setCreatedSecret(data.api_key);
            setKeyName("");
            toast.success("API key created. Copy it now — it will not be shown again.");
            await loadAll();
        } catch {
            toast.error("Error creating API key");
        }
    };

    const handleRevokeKey = async (id: string) => {
        if (!token) return;
        const res = await fetch(`${API}/api-keys/${id}`, {
            method: "DELETE",
            headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
            toast.success("API key revoked");
            await loadAll();
        }
    };

    const handleCreateWorkspace = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!token || !wsName.trim()) return;
        const res = await fetch(`${API}/workspaces/`, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${token}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                name: wsName.trim(),
                environment: wsEnv,
                default_compute_profile: wsCompute,
            }),
        });
        const data = await res.json();
        if (!res.ok) {
            toast.error(data.detail || "Failed to create workspace");
            return;
        }
        setWsName("");
        toast.success(`Workspace "${data.name}" created`);
        await loadAll();
    };

    return (
        <div className="space-y-5">
            {/* Sub-tab switcher */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-3">
                <div className="flex items-center gap-1.5">
                    {[
                        { id: "api_keys", label: "API Keys (PATs)", icon: <KeyRound className="w-3.5 h-3.5" /> },
                        { id: "workspaces", label: "Workspaces", icon: <FolderKanban className="w-3.5 h-3.5" /> },
                        { id: "audit", label: "Audit Log & Events", icon: <ScrollText className="w-3.5 h-3.5" /> },
                    ].map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setSubTab(tab.id as any)}
                            className={`h-8 px-3 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors ${
                                subTab === tab.id
                                    ? "bg-zinc-800 text-zinc-100"
                                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                            }`}
                        >
                            {tab.icon}
                            {tab.label}
                        </button>
                    ))}
                </div>
                <button
                    onClick={loadAll}
                    className="h-8 px-3 text-xs rounded-md border border-zinc-800 bg-[#101217] text-zinc-300 hover:bg-zinc-800/80 flex items-center gap-1.5"
                >
                    {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                    Refresh
                </button>
            </div>

            {/* API Keys */}
            {subTab === "api_keys" && (
                <div className="space-y-4">
                    {createdSecret && (
                        <div className="rounded-lg border border-orange-500/40 bg-orange-500/10 p-4 flex flex-wrap items-center justify-between gap-3">
                            <div>
                                <p className="text-xs font-semibold text-orange-300">
                                    Save this API key now — it is shown only once
                                </p>
                                <p className="text-xs font-mono text-zinc-100 mt-1 break-all">
                                    {createdSecret}
                                </p>
                            </div>
                            <button
                                onClick={() => {
                                    navigator.clipboard.writeText(createdSecret);
                                    setCopied(true);
                                    setTimeout(() => setCopied(false), 1500);
                                }}
                                className="h-8 px-3 rounded-md bg-zinc-100 text-zinc-950 text-xs font-medium flex items-center gap-1.5"
                            >
                                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                                Copy Key
                            </button>
                        </div>
                    )}

                    {canEdit && (
                        <form
                            onSubmit={handleCreateKey}
                            className="rounded-lg border border-zinc-800 bg-[#101217] p-4 flex flex-wrap items-end gap-3"
                        >
                            <div className="flex-1 min-w-[200px]">
                                <label className="block text-[11px] text-zinc-400 mb-1">Key Name</label>
                                <input
                                    value={keyName}
                                    onChange={(e) => setKeyName(e.target.value)}
                                    placeholder="prod-serving-client"
                                    required
                                    className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                                />
                            </div>
                            <div className="flex-1 min-w-[220px]">
                                <label className="block text-[11px] text-zinc-400 mb-1">
                                    Scopes (comma-separated, or *)
                                </label>
                                <input
                                    value={keyScopes}
                                    onChange={(e) => setKeyScopes(e.target.value)}
                                    className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs font-mono text-zinc-100"
                                />
                            </div>
                            <button
                                type="submit"
                                className="h-9 px-4 rounded-md bg-zinc-100 text-zinc-950 text-xs font-medium hover:bg-white flex items-center gap-1.5"
                            >
                                <Plus className="w-3.5 h-3.5" /> Generate Key
                            </button>
                        </form>
                    )}

                    <div className="rounded-lg border border-zinc-800 bg-[#101217] overflow-hidden">
                        <table className="w-full text-left text-xs">
                            <thead className="border-b border-zinc-800 text-[11px] uppercase tracking-wider text-zinc-500">
                                <tr>
                                    <th className="px-4 py-2.5">Name</th>
                                    <th className="px-4 py-2.5">Prefix</th>
                                    <th className="px-4 py-2.5">Scopes</th>
                                    <th className="px-4 py-2.5">Status</th>
                                    <th className="px-4 py-2.5 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-zinc-800/80">
                                {keys.map((k) => (
                                    <tr key={k.id} className="hover:bg-zinc-800/30">
                                        <td className="px-4 py-2.5 font-medium text-zinc-200">{k.name}</td>
                                        <td className="px-4 py-2.5 font-mono text-zinc-400">{k.key_prefix}...</td>
                                        <td className="px-4 py-2.5 font-mono text-zinc-400">
                                            {(k.scopes || []).join(", ")}
                                        </td>
                                        <td className="px-4 py-2.5">
                                            <span
                                                className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border ${
                                                    k.is_active
                                                        ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                                                        : "border-zinc-700 bg-zinc-800 text-zinc-500"
                                                }`}
                                            >
                                                {k.is_active ? "active" : "revoked"}
                                            </span>
                                        </td>
                                        <td className="px-4 py-2.5 text-right">
                                            {k.is_active && canEdit && (
                                                <button
                                                    onClick={() => handleRevokeKey(k.id)}
                                                    className="text-red-400 hover:text-red-300 inline-flex items-center gap-1"
                                                >
                                                    <Trash2 className="w-3.5 h-3.5" /> Revoke
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                                {keys.length === 0 && (
                                    <tr>
                                        <td colSpan={5} className="px-4 py-8 text-center text-zinc-500">
                                            No API keys created yet.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Workspaces */}
            {subTab === "workspaces" && (
                <div className="space-y-4">
                    {canEdit && (
                        <form
                            onSubmit={handleCreateWorkspace}
                            className="rounded-lg border border-zinc-800 bg-[#101217] p-4 flex flex-wrap items-end gap-3"
                        >
                            <div className="flex-1 min-w-[200px]">
                                <label className="block text-[11px] text-zinc-400 mb-1">Workspace Name</label>
                                <input
                                    value={wsName}
                                    onChange={(e) => setWsName(e.target.value)}
                                    placeholder="Fraud Detection Prod"
                                    required
                                    className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                                />
                            </div>
                            <div className="w-36">
                                <label className="block text-[11px] text-zinc-400 mb-1">Environment</label>
                                <select
                                    value={wsEnv}
                                    onChange={(e) => setWsEnv(e.target.value)}
                                    className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                                >
                                    <option value="dev">dev</option>
                                    <option value="staging">staging</option>
                                    <option value="prod">prod</option>
                                </select>
                            </div>
                            <div className="w-40">
                                <label className="block text-[11px] text-zinc-400 mb-1">Default Compute</label>
                                <select
                                    value={wsCompute}
                                    onChange={(e) => setWsCompute(e.target.value)}
                                    className="w-full h-9 px-3 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-100"
                                >
                                    <option value="cpu-light">cpu-light</option>
                                    <option value="cpu-heavy">cpu-heavy</option>
                                    <option value="gpu-train">gpu-train</option>
                                </select>
                            </div>
                            <button
                                type="submit"
                                className="h-9 px-4 rounded-md bg-zinc-100 text-zinc-950 text-xs font-medium hover:bg-white flex items-center gap-1.5"
                            >
                                <Plus className="w-3.5 h-3.5" /> Create Workspace
                            </button>
                        </form>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {workspaces.map((ws) => (
                            <div
                                key={ws.id}
                                className="rounded-lg border border-zinc-800 bg-[#101217] p-4 space-y-2"
                            >
                                <div className="flex items-center justify-between">
                                    <h4 className="text-sm font-medium text-zinc-100">{ws.name}</h4>
                                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-900 text-zinc-300">
                                        {ws.environment}
                                    </span>
                                </div>
                                <p className="text-xs font-mono text-zinc-500">slug: {ws.slug}</p>
                                <p className="text-[11px] font-mono text-zinc-400">
                                    compute: {ws.default_compute_profile}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Audit Log & Domain Events */}
            {subTab === "audit" && (
                <div className="space-y-3">
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setShowEvents(false)}
                            className={`h-7 px-3 rounded text-xs ${
                                !showEvents ? "bg-zinc-800 text-zinc-100" : "text-zinc-400"
                            }`}
                        >
                            Audit Log ({auditRows.length})
                        </button>
                        <button
                            onClick={() => setShowEvents(true)}
                            className={`h-7 px-3 rounded text-xs ${
                                showEvents ? "bg-zinc-800 text-zinc-100" : "text-zinc-400"
                            }`}
                        >
                            Redis Streams Events ({domainEvents.length})
                        </button>
                    </div>

                    <div className="rounded-lg border border-zinc-800 bg-[#101217] overflow-hidden">
                        {!showEvents ? (
                            <table className="w-full text-left text-xs">
                                <thead className="border-b border-zinc-800 text-[11px] uppercase tracking-wider text-zinc-500">
                                    <tr>
                                        <th className="px-4 py-2.5">Action</th>
                                        <th className="px-4 py-2.5">Resource</th>
                                        <th className="px-4 py-2.5">Actor</th>
                                        <th className="px-4 py-2.5">Timestamp</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-zinc-800/80 font-mono">
                                    {auditRows.map((row) => (
                                        <tr key={row.id} className="hover:bg-zinc-800/30">
                                            <td className="px-4 py-2 text-zinc-100">{row.action}</td>
                                            <td className="px-4 py-2 text-zinc-400">
                                                {row.resource_type}
                                                {row.resource_id ? `:${row.resource_id.slice(0, 8)}` : ""}
                                            </td>
                                            <td className="px-4 py-2 text-zinc-400">{row.actor_type}</td>
                                            <td className="px-4 py-2 text-zinc-500">
                                                {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
                                            </td>
                                        </tr>
                                    ))}
                                    {auditRows.length === 0 && (
                                        <tr>
                                            <td colSpan={4} className="px-4 py-8 text-center text-zinc-500 font-sans">
                                                No audit entries recorded yet.
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        ) : (
                            <div className="divide-y divide-zinc-800/80 font-mono text-xs">
                                {domainEvents.map((ev) => (
                                    <div key={ev.event_id} className="px-4 py-2.5 flex items-center justify-between gap-4">
                                        <div className="min-w-0">
                                            <span className="text-orange-400">{ev.event_type}</span>
                                            <span className="text-zinc-500 ml-2 truncate">
                                                {JSON.stringify(ev.payload)}
                                            </span>
                                        </div>
                                        <span className="text-[11px] text-zinc-500 shrink-0">
                                            {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : ""}
                                        </span>
                                    </div>
                                ))}
                                {domainEvents.length === 0 && (
                                    <p className="px-4 py-8 text-center text-zinc-500 font-sans">
                                        No domain events published in Redis Streams yet.
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
