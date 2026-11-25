"use client";
import React, { useEffect, useState } from "react";
import { RefreshCw, ArrowUpRight } from "lucide-react";
import { config } from "@/lib/config";
import type { View } from "@/components/layout/Sidebar";

interface PluginInfo {
    name: string;
    version: string;
    required_extra: string;
    default_queue: string;
    algorithms_count: number;
    copilot_tools_count: number;
}

interface ToolInfo {
    name: string;
    namespace: string;
    description: string;
    min_role: string;
    risk_tier: number;
    risk_tier_name: string;
    prod_requires_approval: boolean;
}

const TIER_META: Record<number, { label: string; cls: string }> = {
    0: { label: "T0 Read", cls: "border-zinc-700 bg-zinc-900 text-zinc-300" },
    1: { label: "T1 Safe", cls: "border-zinc-700 bg-zinc-900 text-zinc-200" },
    2: { label: "T2 Mutate", cls: "border-amber-800/70 bg-amber-950/50 text-amber-300" },
    3: { label: "T3 Approval", cls: "border-red-900/80 bg-red-950/50 text-red-300" },
};

export default function CopilotStudioTab({
    token,
    onNavigate,
}: {
    token: string | null;
    onNavigate: (view: View) => void;
}) {
    const [plugins, setPlugins] = useState<PluginInfo[]>([]);
    const [tools, setTools] = useState<ToolInfo[]>([]);
    const [role, setRole] = useState<string>("");
    const [selectedNs, setSelectedNs] = useState<string>("all");
    const [loading, setLoading] = useState(true);

    const loadData = async () => {
        if (!token) return;
        setLoading(true);
        try {
            const headers = { Authorization: `Bearer ${token}` };
            const [pRes, tRes] = await Promise.all([
                fetch(config.getFullApiUrl("/copilot/plugins"), { headers }),
                fetch(config.getFullApiUrl("/copilot/tools"), { headers }),
            ]);
            if (pRes.ok) {
                const pd = await pRes.json();
                setPlugins(pd.plugins || []);
            }
            if (tRes.ok) {
                const td = await tRes.json();
                setTools(td.tools || []);
                setRole(td.role || "");
            }
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [token]);

    const namespaces = ["all", ...Array.from(new Set(tools.map((t) => t.namespace)))];
    const filteredTools =
        selectedNs === "all" ? tools : tools.filter((t) => t.namespace === selectedNs);

    return (
        <div className="space-y-6">
            {/* Domain Plugins table */}
            <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] overflow-hidden">
                <div className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between">
                    <div>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                            Registered Domain Plugins
                        </h3>
                        <p className="text-xs text-zinc-500 mt-0.5">
                            Controlled via <code className="font-mono text-zinc-300">ENABLED_PLUGINS</code> and routed to typed compute queues.
                        </p>
                    </div>
                    <button
                        onClick={loadData}
                        disabled={loading}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-zinc-300 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-md transition-colors disabled:opacity-50"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
                        Refresh
                    </button>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                        <thead className="bg-[#101217] text-zinc-500 border-b border-zinc-800/80">
                            <tr>
                                <th className="px-4 py-2.5 font-medium">Domain</th>
                                <th className="px-4 py-2.5 font-medium">Version</th>
                                <th className="px-4 py-2.5 font-medium">Default Queue</th>
                                <th className="px-4 py-2.5 font-medium">Algorithms</th>
                                <th className="px-4 py-2.5 font-medium">Copilot Tools</th>
                                <th className="px-4 py-2.5 font-medium">Status</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/70">
                            {plugins.map((p) => (
                                <tr key={p.name} className="hover:bg-zinc-800/25">
                                    <td className="px-4 py-2.5 font-mono font-semibold text-zinc-100">
                                        {p.name}
                                    </td>
                                    <td className="px-4 py-2.5 font-mono text-zinc-400">
                                        {p.version}
                                    </td>
                                    <td className="px-4 py-2.5 font-mono text-orange-400">
                                        {p.default_queue}
                                    </td>
                                    <td className="px-4 py-2.5 font-mono tabular-nums text-zinc-300">
                                        {p.algorithms_count}
                                    </td>
                                    <td className="px-4 py-2.5 font-mono tabular-nums text-zinc-300">
                                        {p.copilot_tools_count}
                                    </td>
                                    <td className="px-4 py-2.5">
                                        <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-emerald-400">
                                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                                            active
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            {/* Copilot Tool Registry */}
            <section className="border border-zinc-800/90 rounded-lg bg-[#14171e] overflow-hidden">
                <div className="px-4 py-3 border-b border-zinc-800/80 flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                            Copilot Tool Registry
                        </h3>
                        <p className="text-xs text-zinc-500 mt-0.5">
                            Filtered for role <code className="font-mono text-zinc-200">{role || "viewer"}</code> · enforced by PolicyEngine.
                        </p>
                    </div>
                    <button
                        onClick={() => onNavigate("nlp")}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-zinc-950 bg-zinc-100 hover:bg-white rounded-md transition-colors"
                    >
                        Open Assistant Chat <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                </div>

                {/* Namespace filter */}
                <div className="px-4 py-2.5 bg-[#101217] border-b border-zinc-800/80 flex flex-wrap gap-1.5">
                    {namespaces.map((ns) => (
                        <button
                            key={ns}
                            onClick={() => setSelectedNs(ns)}
                            className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                                selectedNs === ns
                                    ? "bg-zinc-200 text-zinc-950 font-medium"
                                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
                            }`}
                        >
                            {ns}
                        </button>
                    ))}
                </div>

                <div className="divide-y divide-zinc-800/70">
                    {filteredTools.map((tool) => {
                        const tier = TIER_META[tool.risk_tier] || TIER_META[0];
                        return (
                            <div
                                key={tool.name}
                                className="px-4 py-3 flex flex-wrap items-start justify-between gap-3 hover:bg-zinc-800/25"
                            >
                                <div className="space-y-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <code className="text-xs font-mono font-semibold text-zinc-100">
                                            {tool.name}
                                        </code>
                                        <span className="text-[10px] font-mono text-zinc-500">
                                            min:{tool.min_role}
                                        </span>
                                    </div>
                                    <p className="text-xs text-zinc-400">{tool.description}</p>
                                </div>
                                <span
                                    className={`text-[10px] font-mono px-2 py-0.5 rounded border shrink-0 ${tier.cls}`}
                                >
                                    {tier.label}
                                </span>
                            </div>
                        );
                    })}
                </div>
            </section>
        </div>
    );
}
