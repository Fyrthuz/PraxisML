"use client";

import React, { useState } from "react";
import { Dataset, DatasetProfile, PreprocessingConfig, api } from "@/lib/api";
import DataProfiler from "../preprocessing/DataProfiler";
import PipelineBuilder from "../preprocessing/PipelineBuilder";
import { Database, Filter } from "lucide-react";
import toast from "react-hot-toast";

interface PreprocessingTabProps {
    datasets: Dataset[];
    tenantId: string;
    onPreprocessingApplied: () => void;
}

export default function PreprocessingTab({ datasets, tenantId, onPreprocessingApplied }: PreprocessingTabProps) {
    const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
    const [profile, setProfile] = useState<DatasetProfile | null>(null);

    const handleDatasetSelect = async (id: string) => {
        setSelectedDatasetId(id);
        if (id) {
            try {
                const data = await api.getDatasetProfile(id, tenantId);
                setProfile(data);
            } catch (error) {
                console.error("Failed to load profile", error);
                setProfile(null);
            }
        } else {
            setProfile(null);
        }
    };

    const handleApplyPipeline = async (config: PreprocessingConfig) => {
        try {
            const result = await api.applyPreprocessing(selectedDatasetId, config, tenantId);
            toast.success(`Pipeline applied! New dataset ${result.new_dataset_name} created.`);
            onPreprocessingApplied();
        } catch (error: any) {
            toast.error(error.message || "Failed to apply pipeline");
        }
    };

    const tabularDatasets = datasets.filter(d => ['csv', 'excel', 'parquet', 'xlsx'].includes(d.file_type || ''));

    return (
        <div className="space-y-8 animate-in fade-in zoom-in-95 duration-500 relative z-10 w-full">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-md">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-indigo-500/20 rounded-xl relative group">
                        <div className="absolute inset-0 bg-indigo-500/20 blur-xl group-hover:bg-indigo-500/40 transition-colors"></div>
                        <Filter className="w-6 h-6 text-indigo-400 relative z-10" />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-white mb-1">Data Preprocessing</h2>
                        <p className="text-sm text-indigo-200/60 max-w-xl leading-relaxed">
                            Profile tabular data, build transformation pipelines, and create clean datasets.
                        </p>
                    </div>
                </div>

                <div className="flex items-center space-x-3 w-full md:w-auto">
                    <div className="relative w-full md:w-64">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Database className="h-4 w-4 text-indigo-400" />
                        </div>
                        <select
                            value={selectedDatasetId}
                            onChange={(e) => handleDatasetSelect(e.target.value)}
                            className="block w-full pl-10 pr-4 py-2 bg-neutral-900 border border-neutral-800 rounded-xl shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-sm text-white appearance-none"
                        >
                            <option value="">-- Select Tabular Dataset --</option>
                            {tabularDatasets.map((ds) => (
                                <option key={ds.id} value={ds.id}>
                                    {ds.name} (v{ds.version || 1})
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            {selectedDatasetId ? (
                <div className="space-y-6">
                    {/* Profiler Section */}
                    {profile ? (
                        <>
                            <div className="overflow-hidden">
                                <DataProfiler datasetId={selectedDatasetId} tenantId={tenantId} />
                            </div>

                            {/* Pipeline Section */}
                            <div className="overflow-hidden">
                                <PipelineBuilder
                                    datasetId={selectedDatasetId}
                                    tenantId={tenantId}
                                    profile={profile}
                                    onApply={handleApplyPipeline}
                                />
                            </div>
                        </>
                    ) : (
                        <div className="animate-pulse bg-white/5 h-64 rounded-2xl border border-white/10 backdrop-blur-md"></div>
                    )}
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center p-12 text-center bg-white/5 rounded-2xl border border-white/10 border-dashed backdrop-blur-md">
                    <div className="w-16 h-16 rounded-full bg-indigo-500/10 flex items-center justify-center mb-4">
                        <Filter className="w-8 h-8 text-indigo-400" />
                    </div>
                    <h3 className="text-lg font-medium text-white mb-2">No Dataset Selected</h3>
                    <p className="text-neutral-400 max-w-sm">
                        Select a tabular dataset from the dropdown above to view its profile and apply preprocessing transformations.
                    </p>
                </div>
            )}
        </div>
    );
}
