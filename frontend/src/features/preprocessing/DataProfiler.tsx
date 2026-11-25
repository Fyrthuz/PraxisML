"use client";

import React, { useEffect, useState } from "react";
import { api, DatasetProfile } from "@/lib/api";
import { BarChart, Bar, Tooltip, ResponsiveContainer } from "recharts";
import { AlertCircle, Loader2 } from "lucide-react";

interface DataProfilerProps {
    datasetId: string;
    tenantId: string;
}

export default function DataProfiler({ datasetId, tenantId }: DataProfilerProps) {
    const [profile, setProfile] = useState<DatasetProfile | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let isMounted = true;
        const fetchProfile = async () => {
            try {
                setLoading(true);
                const data = await api.getDatasetProfile(datasetId, tenantId);
                if (isMounted) {
                    setProfile(data);
                    setError(null);
                }
            } catch (err: any) {
                if (isMounted) setError(err.message || "Failed to load dataset profile");
            } finally {
                if (isMounted) setLoading(false);
            }
        };

        if (datasetId && tenantId) {
            fetchProfile();
        }
    }, [datasetId, tenantId]);

    if (loading) {
        return (
            <div className="flex justify-center items-center h-40 bg-[#101217] rounded-lg border border-zinc-800">
                <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
            </div>
        );
    }

    if (error || !profile) {
        return (
            <div className="p-4 bg-red-500/10 border border-red-500/30 text-red-300 rounded-lg text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error || "Profile not available"}</span>
            </div>
        );
    }

    return (
        <div className="space-y-4 bg-[#101217] p-5 rounded-lg border border-zinc-800">
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-100">{profile.dataset_name}</h3>
                    <p className="text-xs font-mono tabular-nums text-zinc-500">
                        {profile.num_rows.toLocaleString()} rows × {profile.num_columns} columns
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3.5">
                {Object.entries(profile.profile).map(([colName, stats]) => (
                    <div
                        key={colName}
                        className="bg-[#0b0c0f] border border-zinc-800 rounded-lg p-4 hover:border-zinc-700 transition-colors"
                    >
                        <div className="flex justify-between items-center mb-3">
                            <h4
                                className="text-xs font-mono font-medium text-zinc-100 truncate pr-2"
                                title={colName}
                            >
                                {colName}
                            </h4>
                            <span
                                className={`px-1.5 py-0.5 text-[10px] font-mono uppercase rounded border ${
                                    stats.type === "numeric"
                                        ? "bg-zinc-900 text-zinc-300 border-zinc-700"
                                        : stats.type === "categorical"
                                        ? "bg-orange-500/10 text-orange-300 border-orange-500/30"
                                        : "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                                }`}
                            >
                                {stats.type}
                            </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2.5 mb-3 text-xs font-mono tabular-nums">
                            <div className="bg-[#101217] px-2.5 py-2 rounded border border-zinc-800/80">
                                <p className="text-zinc-500 text-[10px] font-sans">Missing</p>
                                <p
                                    className={`font-medium ${
                                        stats.null_pct > 0 ? "text-amber-400" : "text-zinc-300"
                                    }`}
                                >
                                    {stats.null_count}{" "}
                                    <span className="text-[10px] opacity-70">({stats.null_pct}%)</span>
                                </p>
                            </div>
                            <div className="bg-[#101217] px-2.5 py-2 rounded border border-zinc-800/80">
                                <p className="text-zinc-500 text-[10px] font-sans">Distinct</p>
                                <p className="font-medium text-zinc-300">{stats.distinct_count}</p>
                            </div>
                        </div>

                        {stats.type === "numeric" && (
                            <>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 mb-3 text-[11px] font-mono tabular-nums">
                                    <div className="flex justify-between border-b border-zinc-800/70 pb-1">
                                        <span className="text-zinc-500 font-sans">Min</span>
                                        <span className="text-zinc-300">{stats.min?.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between border-b border-zinc-800/70 pb-1">
                                        <span className="text-zinc-500 font-sans">Max</span>
                                        <span className="text-zinc-300">
                                            {stats.max?.toString().slice(0, 8)}
                                        </span>
                                    </div>
                                    <div className="flex justify-between border-b border-zinc-800/70 pb-1">
                                        <span className="text-zinc-500 font-sans">Mean</span>
                                        <span className="text-zinc-300">{stats.mean?.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between border-b border-zinc-800/70 pb-1">
                                        <span className="text-zinc-500 font-sans">Zeros</span>
                                        <span className="text-zinc-300">{stats.zeros}</span>
                                    </div>
                                </div>

                                {stats.histogram && (
                                    <div className="h-20 bg-[#101217] p-1.5 rounded border border-zinc-800/70">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart
                                                data={stats.histogram.counts.map((c, i) => ({
                                                    name: (
                                                        (stats.histogram!.bins[i] +
                                                            stats.histogram!.bins[i + 1]) /
                                                        2
                                                    ).toFixed(1),
                                                    count: c,
                                                }))}
                                            >
                                                <Tooltip
                                                    cursor={{ fill: "#27272a", opacity: 0.5 }}
                                                    contentStyle={{
                                                        backgroundColor: "#101217",
                                                        borderColor: "#27272a",
                                                        color: "#f4f4f5",
                                                        fontSize: "11px",
                                                        padding: "4px 8px",
                                                        borderRadius: "4px",
                                                    }}
                                                    itemStyle={{ color: "#fb923c" }}
                                                />
                                                <Bar dataKey="count" fill="#f97316" radius={[2, 2, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}
                            </>
                        )}

                        {stats.type === "categorical" && stats.top_values && (
                            <div className="bg-[#101217] p-2.5 rounded border border-zinc-800/70 space-y-1.5">
                                <p className="text-[10px] text-zinc-500 uppercase tracking-wider">
                                    Top Values
                                </p>
                                {stats.top_values.slice(0, 4).map((v, i) => (
                                    <div
                                        key={i}
                                        className="flex justify-between items-center text-[11px] font-mono"
                                    >
                                        <span className="truncate w-3/4 text-zinc-300" title={v.value}>
                                            {v.value}
                                        </span>
                                        <span className="text-zinc-400 tabular-nums">{v.count}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
