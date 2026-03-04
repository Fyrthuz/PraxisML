"use client";

import React, { useEffect, useState } from "react";
import { api, DatasetProfile } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Loader2 } from "lucide-react";

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
            <div className="flex justify-center items-center h-48 bg-slate-950 rounded-xl border border-slate-800/50">
                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            </div>
        );
    }

    if (error || !profile) {
        return (
            <div className="p-4 bg-red-950/30 border border-red-900/50 text-red-400 rounded-xl">
                ⚠️ {error || "Profile not available"}
            </div>
        );
    }

    return (
        <div className="space-y-6 bg-slate-950 p-6 rounded-xl border border-slate-800/50">
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-xl font-semibold text-slate-100">{profile.dataset_name} Profile</h3>
                    <p className="text-sm text-slate-400">
                        {profile.num_rows} rows × {profile.num_columns} columns
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {Object.entries(profile.profile).map(([colName, stats]) => (
                    <div key={colName} className="bg-slate-900/50 border border-slate-800/80 rounded-xl shadow-sm p-5 hover:shadow-md hover:border-slate-700/80 transition-all">
                        <div className="flex justify-between items-start mb-5">
                            <h4 className="font-semibold text-slate-200 truncate pr-2" title={colName}>
                                {colName}
                            </h4>
                            <span className={`px-2.5 py-1 text-xs font-medium rounded-md border ${stats.type === 'numeric' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                                stats.type === 'categorical' ? 'bg-purple-500/10 text-purple-400 border-purple-500/20' :
                                    'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                }`}>
                                {stats.type}
                            </span>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-5 text-sm">
                            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800/50">
                                <p className="text-slate-400 text-xs mb-1">Missing</p>
                                <p className={`font-semibold ${stats.null_pct > 0 ? 'text-amber-400' : 'text-slate-300'}`}>
                                    {stats.null_count} <span className="text-xs font-normal opacity-70">({stats.null_pct}%)</span>
                                </p>
                            </div>
                            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800/50">
                                <p className="text-slate-400 text-xs mb-1">Distinct</p>
                                <p className="font-semibold text-slate-300">{stats.distinct_count}</p>
                            </div>
                        </div>

                        {stats.type === 'numeric' && (
                            <>
                                <div className="grid grid-cols-2 gap-x-6 gap-y-2.5 mb-5 text-xs">
                                    <div className="flex justify-between border-b border-slate-800/60 pb-1.5">
                                        <span className="text-slate-400">Min</span>
                                        <span className="font-medium text-slate-300">{stats.min?.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between border-b border-slate-800/60 pb-1.5">
                                        <span className="text-slate-400">Max</span>
                                        <span className="font-medium text-slate-300">{stats.max?.toString().slice(0, 8)}</span>
                                    </div>
                                    <div className="flex justify-between border-b border-slate-800/60 pb-1.5">
                                        <span className="text-slate-400">Mean</span>
                                        <span className="font-medium text-slate-300">{stats.mean?.toFixed(2)}</span>
                                    </div>
                                    <div className="flex justify-between border-b border-slate-800/60 pb-1.5">
                                        <span className="text-slate-400">Zeros</span>
                                        <span className="font-medium text-slate-300">{stats.zeros}</span>
                                    </div>
                                </div>

                                {stats.histogram && (
                                    <div className="h-28 mt-2 bg-slate-900/40 p-2 rounded-lg border border-slate-800/40">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={stats.histogram.counts.map((c, i) => ({
                                                // Approx bin center
                                                name: ((stats.histogram!.bins[i] + stats.histogram!.bins[i + 1]) / 2).toFixed(1),
                                                count: c
                                            }))}>
                                                <Tooltip
                                                    cursor={{ fill: '#334155', opacity: 0.4 }}
                                                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', fontSize: '12px', padding: '6px 10px', borderRadius: '6px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)' }}
                                                    itemStyle={{ color: '#60a5fa' }}
                                                />
                                                <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}
                            </>
                        )}

                        {stats.type === 'categorical' && stats.top_values && (
                            <div className="mt-5 bg-slate-900/40 p-3 rounded-lg border border-slate-800/40">
                                <p className="text-xs text-slate-400 mb-3 font-medium uppercase tracking-wider">Top Values</p>
                                <div className="space-y-2">
                                    {stats.top_values.slice(0, 5).map((v, i) => (
                                        <div key={i} className="flex justify-between items-center text-xs">
                                            <span className="truncate w-3/4 text-slate-300" title={v.value}>{v.value}</span>
                                            <span className="font-medium text-slate-300 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700/50">{v.count}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {stats.type === 'datetime' && (
                            <div className="grid grid-cols-1 gap-y-3 mb-2 text-xs mt-5 bg-slate-900/40 p-3 rounded-lg border border-slate-800/40">
                                <div className="flex justify-between items-center border-b border-slate-800/60 pb-2">
                                    <span className="text-slate-400">Min</span>
                                    <span className="font-medium text-slate-300 bg-slate-800/50 px-2 py-1 rounded">{stats.min?.toString()}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-slate-400">Max</span>
                                    <span className="font-medium text-slate-300 bg-slate-800/50 px-2 py-1 rounded">{stats.max?.toString()}</span>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
