import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Plus, Cpu, Trash2, Loader2, RefreshCw, ChevronRight, Info, ExternalLink, Activity, Database, Tag, Download, Server } from 'lucide-react';
import { api, MLModel } from '@/lib/api';
import toast from 'react-hot-toast';

interface RegisteredModel {
    name: string;
    display_name?: string;
    description: string;
    latest_versions: { version: number; stage: string }[];
}

interface RegistryTabProps {
    token?: string | null;
    onRefresh?: () => void;
    models?: MLModel[];
    onDeployEndpoint?: (modelId: string) => void;
}

export default function RegistryTab({ token, onRefresh, models = [], onDeployEndpoint }: RegistryTabProps) {
    const [registeredModels, setRegisteredModels] = useState<RegisteredModel[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isCreating, setIsCreating] = useState(false);
    const [newModelName, setNewModelName] = useState('');
    const [newModelDesc, setNewModelDesc] = useState('');

    // Selection state
    const [selectedModel, setSelectedModel] = useState<RegisteredModel | null>(null);
    const [versions, setVersions] = useState<any[]>([]);
    const [isLoadingVersions, setIsLoadingVersions] = useState(false);
    const [selectedRun, setSelectedRun] = useState<any | null>(null);
    const [isLoadingRun, setIsLoadingRun] = useState(false);

    const fetchRegistry = async () => {
        if (!token) return;
        setIsLoading(true);
        try {
            const data = await api.getRegisteredModels();
            setRegisteredModels(data.models || []);
        } catch (err) {
            console.error('Failed to fetch registered models:', err);
            toast.error('Failed to load registered models');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchRegistry();
    }, [token]);

    const handleSelectModel = async (model: RegisteredModel) => {
        setSelectedModel(model);
        setSelectedRun(null);
        setIsLoadingVersions(true);
        try {
            const data = await api.getRegistryVersions(model.name);
            setVersions(data.versions || []);
        } catch (err) {
            toast.error('Failed to fetch model versions');
        } finally {
            setIsLoadingVersions(false);
        }
    };

    const handleSelectVersion = async (runId: string) => {
        setIsLoadingRun(true);
        try {
            const data = await api.getRunDetails(runId);
            setSelectedRun(data);
        } catch (err) {
            toast.error('Failed to fetch run details');
        } finally {
            setIsLoadingRun(false);
        }
    };

    const handleCreateModel = async () => {
        if (!token || !newModelName.trim()) return;
        
        setIsCreating(true);
        try {
            await api.createRegisteredModel(newModelName.trim(), newModelDesc);
            toast.success('Registered model created successfully');
            setNewModelName('');
            setNewModelDesc('');
            fetchRegistry();
        } catch (err: any) {
            toast.error(err.message || 'Failed to create registered model');
        } finally {
            setIsCreating(false);
        }
    };

    const handleDeleteModel = async (e: React.MouseEvent, name: string) => {
        e.stopPropagation();
        if (!token || !confirm(`Delete registered model "${name}"? This will not delete the model versions.`)) return;
        
        try {
            await api.deleteRegisteredModel(name);
            toast.success('Registered model deleted');
            if (selectedModel?.name === name) {
                setSelectedModel(null);
                setVersions([]);
                setSelectedRun(null);
            }
            fetchRegistry();
        } catch (err: any) {
            toast.error(err.message || 'Failed to delete registered model');
        }
    };

    const handleDownloadVersion = async () => {
        if (!selectedRun) return;
        try {
            toast.loading('Preparing download...', { id: 'download-model' });
            await api.downloadModelByRunId(selectedRun.run_id, selectedModel?.name || 'model');
            toast.success('Model downloaded successfully', { id: 'download-model' });
        } catch (err: any) {
            toast.error(err.message || 'Failed to download model', { id: 'download-model' });
        }
    };

    const getStageBadge = (stage: string) => {
        const colors: Record<string, string> = {
            Staging: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
            Production: 'bg-green-500/20 text-green-400 border-green-500/30',
            Archived: 'bg-neutral-500/20 text-neutral-400 border-neutral-500/30',
            None: 'bg-neutral-500/20 text-neutral-400 border-neutral-500/30',
        };
        return colors[stage] || colors.None;
    };

    const formatTimestamp = (ts: number) => {
        return new Date(ts).toLocaleString();
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <h3 className="text-xl font-medium">MLflow Model Registry</h3>
                    {selectedModel && (
                        <div className="flex items-center gap-2 text-neutral-500">
                            <ChevronRight className="w-4 h-4" />
                            <span className="text-sm font-medium text-white">{selectedModel.display_name || selectedModel.name}</span>
                        </div>
                    )}
                </div>
                <div className="flex gap-2">
                    {selectedModel && (
                        <Button variant="ghost" size="sm" onClick={() => { setSelectedModel(null); setSelectedRun(null); }}>
                            Back to Registry
                        </Button>
                    )}
                    <Button variant="outline" size="sm" onClick={fetchRegistry} disabled={isLoading}>
                        <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                        Refresh
                    </Button>
                </div>
            </div>

            {!selectedModel && (
                <div className="bg-[#101217] border border-zinc-800 p-4 rounded-lg">
                    <h4 className="text-xs font-semibold text-zinc-300 mb-3">Create Registered Model</h4>
                    <div className="flex flex-wrap gap-2.5">
                        <input
                            type="text"
                            placeholder="Model name (e.g., titanic-classifier)"
                            value={newModelName}
                            onChange={(e) => setNewModelName(e.target.value)}
                            className="flex-1 min-w-[200px] h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-zinc-100 text-xs focus:outline-none focus:border-zinc-600"
                        />
                        <input
                            type="text"
                            placeholder="Description (optional)"
                            value={newModelDesc}
                            onChange={(e) => setNewModelDesc(e.target.value)}
                            className="flex-1 min-w-[200px] h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-zinc-100 text-xs focus:outline-none focus:border-zinc-600"
                        />
                        <button
                            type="button"
                            onClick={handleCreateModel}
                            disabled={isCreating || !newModelName.trim()}
                            className="h-9 px-4 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium disabled:opacity-40 flex items-center gap-1.5"
                        >
                            {isCreating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
                            Create
                        </button>
                    </div>
                </div>
            )}

            {isLoading ? (
                <div className="text-center py-12 text-zinc-500">
                    <Loader2 className="w-6 h-6 animate-spin mx-auto mb-3" />
                    <p className="text-xs">Fetching models from MLflow...</p>
                </div>
            ) : !selectedModel ? (
                registeredModels.length === 0 ? (
                    <div className="border border-dashed border-zinc-800 bg-[#101217] rounded-lg p-12 text-center text-xs text-zinc-500">
                        No registered models yet. Create one above to get started.
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {registeredModels.map((model) => (
                            <div
                                key={model.name}
                                onClick={() => handleSelectModel(model)}
                                className="bg-[#101217] border border-zinc-800 p-4 rounded-lg hover:border-zinc-700 cursor-pointer transition-colors"
                            >
                                <div className="flex justify-between items-start">
                                    <div className="flex items-start gap-3">
                                        <div className="p-2 bg-zinc-900 border border-zinc-800 rounded-md text-zinc-300">
                                            <Cpu className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h4 className="font-semibold text-zinc-100 text-sm">
                                                {model.display_name || model.name}
                                            </h4>
                                            <p className="text-xs text-zinc-500 mt-0.5 line-clamp-1">{model.description || "No description provided"}</p>
                                        </div>
                                    </div>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={(e) => handleDeleteModel(e, model.name)}
                                        className="h-7 w-7 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 rounded-md"
                                    >
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </Button>
                                </div>
                                
                                {model.latest_versions && model.latest_versions.length > 0 && (
                                    <div className="mt-3.5 pt-3 border-t border-zinc-800/80 flex flex-wrap gap-1.5">
                                        {model.latest_versions.map((v) => (
                                            <span 
                                                key={v.version}
                                                className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border ${getStageBadge(v.stage)}`}
                                            >
                                                v{v.version} · {v.stage}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                    {/* Versions List */}
                    <div className="lg:col-span-1 space-y-3">
                        <h4 className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Versions History</h4>
                        {isLoadingVersions ? (
                            <div className="py-8 text-center"><Loader2 className="w-5 h-5 animate-spin mx-auto text-zinc-500" /></div>
                        ) : versions.length === 0 ? (
                            <p className="text-xs text-zinc-500 p-4 bg-[#101217] rounded-lg border border-dashed border-zinc-800">No versions found.</p>
                        ) : (
                            <div className="space-y-2">
                                {versions.sort((a, b) => b.version - a.version).map((v) => (
                                    <div
                                        key={v.version}
                                        onClick={() => handleSelectVersion(v.run_id)}
                                        className={`p-3.5 rounded-lg border cursor-pointer transition-colors ${
                                            selectedRun?.run_id === v.run_id
                                                ? 'bg-zinc-800/70 border-orange-500/60'
                                                : 'bg-[#101217] border-zinc-800 hover:border-zinc-700'
                                        }`}
                                    >
                                        <div className="flex justify-between items-center">
                                            <div>
                                                <span className="text-sm font-mono font-semibold text-zinc-100">v{v.version}</span>
                                                <div className={`mt-1 text-[10px] font-mono w-fit px-1.5 py-0.5 rounded border ${getStageBadge(v.stage)}`}>
                                                    {v.stage}
                                                </div>
                                            </div>
                                            <ChevronRight className={`w-4 h-4 ${selectedRun?.run_id === v.run_id ? 'text-orange-400' : 'text-zinc-600'}`} />
                                        </div>
                                        <p className="text-[10px] font-mono text-zinc-500 mt-2">
                                            {formatTimestamp(v.creation_timestamp)}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Run Details */}
                    <div className="lg:col-span-2">
                        {isLoadingRun ? (
                            <div className="h-full flex flex-col items-center justify-center p-12 bg-[#101217] rounded-lg border border-zinc-800">
                                <Loader2 className="w-6 h-6 animate-spin text-zinc-400 mb-3" />
                                <p className="text-xs text-zinc-400">Loading experiment data from MLflow...</p>
                            </div>
                        ) : selectedRun ? (
                            <div className="bg-[#101217] border border-zinc-800 rounded-lg overflow-hidden">
                                <div className="p-4 bg-[#0b0c0f] border-b border-zinc-800 flex justify-between items-center">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-md">
                                            <Activity className="w-4 h-4" />
                                        </div>
                                        <div>
                                            <h4 className="text-sm font-semibold text-zinc-100">Experiment Details</h4>
                                            <p className="text-[11px] text-zinc-500 font-mono">run: {selectedRun.run_id}</p>
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        {onDeployEndpoint && (() => {
                                            const matched = models.find((m) => m.mlflow_run_id === selectedRun.run_id);
                                            if (!matched) return null;
                                            return (
                                                <button
                                                    type="button"
                                                    onClick={() => onDeployEndpoint(matched.id)}
                                                    className="h-8 px-3 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium inline-flex items-center gap-1.5"
                                                >
                                                    <Server className="w-3.5 h-3.5" />
                                                    Deploy Endpoint
                                                </button>
                                            );
                                        })()}
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={handleDownloadVersion}
                                            className="h-8 text-xs border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800"
                                            title="Download this version"
                                        >
                                            <Download className="w-3.5 h-3.5 mr-1.5" />
                                            Download
                                        </Button>
                                        <a 
                                            href={`http://localhost:5000/#/experiments/${selectedRun.experiment_id}/runs/${selectedRun.run_id}`} 
                                            target="_blank" 
                                            rel="noopener noreferrer"
                                            className="h-8 w-8 flex items-center justify-center hover:bg-zinc-800 rounded-md text-zinc-400 border border-zinc-800"
                                            title="View in MLflow UI"
                                        >
                                            <ExternalLink className="w-4 h-4" />
                                        </a>
                                    </div>
                                </div>

                                <div className="p-6 space-y-8">
                                    {/* Metrics */}
                                    <div className="space-y-4">
                                        <div className="flex items-center gap-2">
                                            <Activity className="w-4 h-4 text-emerald-400" />
                                            <span className="text-sm font-bold text-white uppercase tracking-wider">Metrics</span>
                                        </div>
                                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                                            {Object.keys(selectedRun.metrics).length > 0 ? (
                                                Object.entries(selectedRun.metrics).map(([key, val]: [string, any]) => (
                                                    <div key={key} className="bg-neutral-900/50 border border-neutral-800 p-3 rounded-xl">
                                                        <p className="text-[10px] text-neutral-500 uppercase font-bold mb-1 line-clamp-1">{key}</p>
                                                        <p className="text-lg font-mono text-emerald-400 font-bold">
                                                            {typeof val === 'number' ? val.toFixed(4) : String(val)}
                                                        </p>
                                                    </div>
                                                ))
                                            ) : (
                                                <p className="text-xs text-neutral-500 col-span-full">No metrics logged for this run.</p>
                                            )}
                                        </div>
                                    </div>

                                    {/* Parameters */}
                                    <div className="space-y-4">
                                        <div className="flex items-center gap-2">
                                            <Database className="w-4 h-4 text-blue-400" />
                                            <span className="text-sm font-bold text-white uppercase tracking-wider">Parameters</span>
                                        </div>
                                        <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
                                            <table className="w-full text-xs text-left">
                                                <thead>
                                                    <tr className="border-b border-neutral-800 bg-neutral-800/50">
                                                        <th className="px-4 py-2 text-neutral-500 font-bold">KEY</th>
                                                        <th className="px-4 py-2 text-neutral-500 font-bold">VALUE</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {Object.entries(selectedRun.params).map(([key, val]: [string, any]) => (
                                                        <tr key={key} className="border-b border-neutral-800 hover:bg-neutral-700/20 transition-colors">
                                                            <td className="px-4 py-2 font-mono text-blue-400">{key}</td>
                                                            <td className="px-4 py-2 text-neutral-300 break-all">{String(val)}</td>
                                                        </tr>
                                                    ))}
                                                    {Object.keys(selectedRun.params).length === 0 && (
                                                        <tr>
                                                            <td colSpan={2} className="px-4 py-4 text-center text-neutral-500 italic">No parameters found</td>
                                                        </tr>
                                                    )}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>

                                    {/* Tags */}
                                    <div className="space-y-4">
                                        <div className="flex items-center gap-2">
                                            <Tag className="w-4 h-4 text-orange-400" />
                                            <span className="text-sm font-bold text-white uppercase tracking-wider">Tags</span>
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                            {Object.entries(selectedRun.tags)
                                                .filter(([key]) => !key.startsWith('mlflow.'))
                                                .map(([key, val]) => (
                                                    <div key={key} className="bg-neutral-800 border border-neutral-700 px-3 py-1.5 rounded-lg flex gap-2 items-center">
                                                        <span className="text-[10px] font-bold text-neutral-500 uppercase">{key}:</span>
                                                        <span className="text-xs text-orange-400">{val as string}</span>
                                                    </div>
                                                ))}
                                            {Object.entries(selectedRun.tags).filter(([key]) => !key.startsWith('mlflow.')).length === 0 && (
                                                <p className="text-xs text-neutral-500 italic">No custom tags.</p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center p-12 bg-neutral-800/10 rounded-2xl border border-neutral-800 border-dashed">
                                <div className="p-4 bg-neutral-800 rounded-full text-neutral-600 mb-4">
                                    <Info className="w-8 h-8" />
                                </div>
                                <h4 className="text-white font-medium mb-1">No version selected</h4>
                                <p className="text-sm text-neutral-500">Select a version from the list to view its experiment details.</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
