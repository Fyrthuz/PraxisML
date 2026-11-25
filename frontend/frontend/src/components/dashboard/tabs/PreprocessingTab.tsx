"use client";

import React, { useState } from "react";
import { Dataset, DatasetProfile, PreprocessingConfig, api } from "@/lib/api";
import DataProfiler from "../preprocessing/DataProfiler";
import PipelineBuilder from "../preprocessing/PipelineBuilder";
import { Database, Filter, Plus, ArrowLeft, Play, History } from "lucide-react";
import toast from "react-hot-toast";

interface PreprocessingTabProps {
    datasets: Dataset[];
    tenantId: string;
    onPreprocessingApplied: () => void;
}

export default function PreprocessingTab({ datasets, tenantId, onPreprocessingApplied }: PreprocessingTabProps) {
    const [view, setView] = useState<"list" | "create">("list");
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
            setView("list");
            setSelectedDatasetId("");
            setProfile(null);
        } catch (error: any) {
            toast.error(error.message || "Failed to apply pipeline");
        }
    };

    const tabularDatasets = datasets.filter(d => ['csv', 'excel', 'parquet', 'xlsx'].includes(d.file_type || ''));
    const preprocessedDatasets = datasets.filter(d => d.pipeline_path);

    if (view === "create") {
        return (
            <div className="space-y-6 w-full">
                <div className="flex flex-wrap items-center justify-between gap-4 bg-[#101217] p-4 rounded-lg border border-zinc-800">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => { setView("list"); setSelectedDatasetId(""); setProfile(null); }}
                            className="h-8 w-8 rounded-md border border-zinc-800 bg-[#0b0c0f] flex items-center justify-center text-zinc-400 hover:text-zinc-100"
                        >
                            <ArrowLeft className="w-4 h-4" />
                        </button>
                        <div>
                            <h3 className="text-sm font-semibold text-zinc-100">Create Preprocessing Pipeline</h3>
                            <p className="text-xs text-zinc-500">Configure ColumnTransformer steps for a tabular dataset.</p>
                        </div>
                    </div>

                    <div className="relative w-64">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <Database className="h-3.5 w-3.5 text-zinc-500" />
                        </div>
                        <select
                            value={selectedDatasetId}
                            onChange={(e) => handleDatasetSelect(e.target.value)}
                            className="block w-full h-9 pl-9 pr-4 bg-[#0b0c0f] border border-zinc-800 rounded-md text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                        >
                            <option value="">Select tabular dataset...</option>
                            {tabularDatasets.map((ds) => (
                                <option key={ds.id} value={ds.id}>
                                    {ds.name} (v{ds.version || 1})
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {selectedDatasetId ? (
                    <div className="space-y-6">
                        {profile ? (
                            <>
                                <DataProfiler datasetId={selectedDatasetId} tenantId={tenantId} />
                                <PipelineBuilder
                                    datasetId={selectedDatasetId}
                                    tenantId={tenantId}
                                    profile={profile}
                                    onApply={handleApplyPipeline}
                                />
                            </>
                        ) : (
                            <div className="animate-pulse bg-[#101217] h-48 rounded-lg border border-zinc-800" />
                        )}
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center p-16 text-center bg-[#101217] rounded-lg border border-zinc-800 border-dashed">
                        <Database className="w-7 h-7 text-zinc-600 mb-3" />
                        <h3 className="text-sm font-medium text-zinc-200 mb-1">Select a Dataset</h3>
                        <p className="text-xs text-zinc-500 max-w-sm">
                            Pick a tabular dataset above to inspect column distributions and build a transformation pipeline.
                        </p>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="space-y-6 w-full">
            <div className="flex justify-between items-center bg-[#101217] p-4 rounded-lg border border-zinc-800">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-zinc-900 border border-zinc-800 rounded-md">
                        <Filter className="w-4 h-4 text-zinc-300" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-zinc-100">Saved Pipelines</h3>
                        <p className="text-xs text-zinc-500">
                            <span className="font-mono tabular-nums text-zinc-300">{preprocessedDatasets.length}</span> preprocessed dataset versions linked to MLflow pipelines.
                        </p>
                    </div>
                </div>
                <button
                    onClick={() => setView("create")}
                    className="h-8 px-3 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium flex items-center gap-1.5"
                >
                    <Plus className="w-3.5 h-3.5" /> New Pipeline
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {preprocessedDatasets.length === 0 ? (
                    <div className="col-span-full flex flex-col items-center justify-center p-16 text-center bg-[#101217] rounded-lg border border-zinc-800 border-dashed">
                        <History className="w-7 h-7 text-zinc-600 mb-3" />
                        <h3 className="text-sm font-medium text-zinc-200 mb-1">No Pipelines Found</h3>
                        <p className="text-xs text-zinc-500 max-w-sm">
                            Create a preprocessing pipeline to impute missing values, encode categoricals and scale numeric features.
                        </p>
                    </div>
                ) : (
                    preprocessedDatasets.map((ds) => (
                        <button
                            key={ds.id}
                            onClick={() => { setSelectedDatasetId(ds.id); handleDatasetSelect(ds.id); }}
                            className={`text-left bg-[#101217] border p-4 rounded-lg transition-colors ${
                                selectedDatasetId === ds.id
                                    ? "border-orange-500/70 bg-zinc-900/70"
                                    : "border-zinc-800 hover:border-zinc-700"
                            }`}
                        >
                            <div className="flex justify-between items-start mb-3">
                                <div className="p-1.5 bg-zinc-900 border border-zinc-800 rounded text-zinc-300">
                                    <Play className="w-3.5 h-3.5" />
                                </div>
                                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded border border-emerald-500/30">
                                    saved
                                </span>
                            </div>
                            <h4 className="text-sm font-medium text-zinc-100 truncate">{ds.name}</h4>
                            <p className="text-xs text-zinc-500 mt-1.5 line-clamp-2 min-h-[2rem]">
                                {ds.description || "Preprocessed tabular dataset."}
                            </p>
                            <div className="mt-3 pt-2.5 border-t border-zinc-800/80 flex items-center justify-between text-[11px] font-mono tabular-nums text-zinc-500">
                                <span className="flex items-center gap-1">
                                    <Database className="w-3 h-3" /> v{ds.version || 1}
                                </span>
                                <span>{new Date(ds.created_at).toLocaleDateString()}</span>
                            </div>
                        </button>
                    ))
                )}
            </div>

            {selectedDatasetId && view === "list" && (
                <div className="pt-2">
                    {profile ? (
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 text-zinc-300">
                                <History className="w-4 h-4 text-zinc-500" />
                                <h3 className="text-sm font-semibold">Pipeline Details</h3>
                            </div>
                            <PipelineBuilder
                                datasetId={selectedDatasetId}
                                tenantId={tenantId}
                                profile={profile}
                                onApply={handleApplyPipeline}
                            />
                        </div>
                    ) : (
                        <div className="animate-pulse bg-[#101217] h-48 rounded-lg border border-zinc-800" />
                    )}
                </div>
            )}
        </div>
    );
}
