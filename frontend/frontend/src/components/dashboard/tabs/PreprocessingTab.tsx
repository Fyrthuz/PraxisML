"use client";

import React, { useState } from "react";
import { Dataset, DatasetProfile, PreprocessingConfig, api } from "@/lib/api";
import DataProfiler from "../preprocessing/DataProfiler";
import PipelineBuilder from "../preprocessing/PipelineBuilder";
import { Database, Filter, Plus, ArrowLeft, Play, History } from "lucide-react";
import { Button } from "@/components/ui/button";
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
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 w-full">
                <div className="flex items-center justify-between bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-md">
                    <div className="flex items-center gap-4">
                        <Button 
                            variant="ghost" 
                            size="icon" 
                            onClick={() => { setView("list"); setSelectedDatasetId(""); setProfile(null); }}
                            className="text-neutral-400 hover:text-white"
                        >
                            <ArrowLeft className="w-5 h-5" />
                        </Button>
                        <div>
                            <h2 className="text-xl font-bold text-white mb-1">Create New Pipeline</h2>
                            <p className="text-sm text-indigo-200/60">Configure transformations for your tabular data.</p>
                        </div>
                    </div>

                    <div className="flex items-center space-x-3">
                        <div className="relative w-64">
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
                            <div className="animate-pulse bg-white/5 h-64 rounded-2xl border border-white/10 backdrop-blur-md"></div>
                        )}
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center p-20 text-center bg-white/5 rounded-2xl border border-white/10 border-dashed backdrop-blur-md">
                        <div className="w-16 h-16 rounded-full bg-indigo-500/10 flex items-center justify-center mb-4">
                            <Database className="w-8 h-8 text-indigo-400" />
                        </div>
                        <h3 className="text-lg font-medium text-white mb-2">Select a Dataset</h3>
                        <p className="text-neutral-400 max-w-sm">
                            Pick a dataset to start building your preprocessing pipeline.
                        </p>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-500 w-full">
            <div className="flex justify-between items-center bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-md">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-indigo-500/20 rounded-xl">
                        <Filter className="w-6 h-6 text-indigo-400" />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-white mb-1">Preprocessing Pipelines</h2>
                        <p className="text-sm text-indigo-200/60">View and manage existing data transformation pipelines.</p>
                    </div>
                </div>
                <Button className="bg-indigo-600 hover:bg-indigo-700" onClick={() => setView("create")}>
                    <Plus className="w-4 h-4 mr-2" /> New Pipeline
                </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {preprocessedDatasets.length === 0 ? (
                    <div className="col-span-full flex flex-col items-center justify-center p-20 text-center bg-white/5 rounded-2xl border border-white/10 border-dashed backdrop-blur-md">
                        <div className="w-16 h-16 rounded-full bg-indigo-500/10 flex items-center justify-center mb-4">
                            <History className="w-8 h-8 text-indigo-400" />
                        </div>
                        <h3 className="text-lg font-medium text-white mb-2">No Pipelines Found</h3>
                        <p className="text-neutral-400 max-w-sm">
                            You haven't created any preprocessing pipelines yet. Click "New Pipeline" to get started.
                        </p>
                    </div>
                ) : (
                    preprocessedDatasets.map((ds) => (
                        <div 
                            key={ds.id} 
                            onClick={() => { setSelectedDatasetId(ds.id); handleDatasetSelect(ds.id); }}
                            className={`group bg-neutral-800/40 border p-5 rounded-2xl cursor-pointer transition-all hover:scale-[1.02] ${
                                selectedDatasetId === ds.id ? "border-indigo-500 ring-1 ring-indigo-500" : "border-neutral-700 hover:border-indigo-500/50"
                            }`}
                        >
                            <div className="flex justify-between items-start mb-4">
                                <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400 group-hover:bg-indigo-500/20 transition-colors">
                                    <Play className="w-5 h-5" />
                                </div>
                                <span className="text-[10px] px-2 py-0.5 bg-indigo-500/20 text-indigo-400 rounded-full border border-indigo-500/30">
                                    ACTIVE
                                </span>
                            </div>
                            <h4 className="font-bold text-white group-hover:text-indigo-300 transition-colors truncate">{ds.name}</h4>
                            <p className="text-xs text-neutral-400 mt-2 line-clamp-2 min-h-[2.5rem]">{ds.description || "No description provided."}</p>
                            
                            <div className="mt-4 flex items-center justify-between text-[10px] text-neutral-500">
                                <span className="flex items-center gap-1">
                                    <Database className="w-3 h-3" />
                                    v{ds.version || 1}
                                </span>
                                <span>{new Date(ds.created_at).toLocaleDateString()}</span>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {selectedDatasetId && view === "list" && (
                <div className="animate-in fade-in slide-in-from-top-4 duration-500">
                    {profile ? (
                        <div className="space-y-6">
                            <div className="flex items-center gap-2 text-indigo-400 mb-2">
                                <History className="w-5 h-5" />
                                <h3 className="text-lg font-bold">Pipeline Details</h3>
                            </div>
                            <PipelineBuilder
                                datasetId={selectedDatasetId}
                                tenantId={tenantId}
                                profile={profile}
                                onApply={handleApplyPipeline}
                            />
                        </div>
                    ) : (
                        <div className="animate-pulse bg-white/5 h-64 rounded-2xl border border-white/10 backdrop-blur-md"></div>
                    )}
                </div>
            )}
        </div>
    );
}
