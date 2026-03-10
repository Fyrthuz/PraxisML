import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Plus, Cpu, Trash2, Loader2, RefreshCw, ChevronRight, Info, ExternalLink, Activity, Database, Tag, Download } from 'lucide-react';
import { api } from '../../../lib/api';
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
}

export default function RegistryTab({ token, onRefresh }: RegistryTabProps) {
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
                <div className="bg-neutral-800/40 border border-neutral-700 p-4 rounded-xl">
                    <h4 className="text-sm font-medium text-neutral-300 mb-3">Create New Registered Model</h4>
                    <div className="flex gap-3">
                        <input
                            type="text"
                            placeholder="Model name (e.g., unet-segmentation)"
                            value={newModelName}
                            onChange={(e) => setNewModelName(e.target.value)}
                            className="flex-1 bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
                        />
                        <input
                            type="text"
                            placeholder="Description (optional)"
                            value={newModelDesc}
                            onChange={(e) => setNewModelDesc(e.target.value)}
                            className="flex-1 bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
                        />
                        <Button 
                            onClick={handleCreateModel} 
                            disabled={isCreating || !newModelName.trim()}
                            className="bg-indigo-600 hover:bg-indigo-700"
                        >
                            {isCreating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4 mr-2" />}
                            Create
                        </Button>
                    </div>
                </div>
            )}

            {isLoading ? (
                <div className="text-center py-12 text-neutral-500">
                    <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
                    <p>Fetching models from MLflow...</p>
                </div>
            ) : !selectedModel ? (
                registeredModels.length === 0 ? (
                    <div className="border-2 border-dashed border-neutral-700 rounded-xl p-12 text-center text-neutral-500">
                        No registered models yet. Create one above to get started.
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {registeredModels.map((model) => (
                            <div
                                key={model.name}
                                onClick={() => handleSelectModel(model)}
                                className="bg-neutral-800/40 border border-neutral-700 p-5 rounded-xl hover:border-indigo-500/50 cursor-pointer transition-all group"
                            >
                                <div className="flex justify-between items-start">
                                    <div className="flex items-start gap-4">
                                        <div className="p-3 bg-indigo-500/10 rounded-xl text-indigo-400 group-hover:scale-110 transition-transform">
                                            <Cpu className="w-7 h-7" />
                                        </div>
                                        <div>
                                            <h4 className="font-bold text-white text-lg">
                                                {model.display_name || model.name}
                                            </h4>
                                            <p className="text-sm text-neutral-400 mt-1 line-clamp-1">{model.description || "No description provided"}</p>
                                        </div>
                                    </div>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={(e) => handleDeleteModel(e, model.name)}
                                        className="text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </Button>
                                </div>
                                
                                {model.latest_versions && model.latest_versions.length > 0 && (
                                    <div className="mt-5 pt-4 border-t border-neutral-700/50 flex flex-wrap gap-2">
                                        {model.latest_versions.map((v) => (
                                            <span 
                                                key={v.version}
                                                className={`text-[10px] uppercase font-bold px-2.5 py-1 rounded-md border ${getStageBadge(v.stage)}`}
                                            >
                                                v{v.version} • {v.stage}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in slide-in-from-right-4 duration-300">
                    {/* Versions List */}
                    <div className="lg:col-span-1 space-y-4">
                        <h4 className="text-sm font-bold text-neutral-500 uppercase tracking-wider ml-1">Versions History</h4>
                        {isLoadingVersions ? (
                            <div className="py-8 text-center"><Loader2 className="w-6 h-6 animate-spin mx-auto text-neutral-500" /></div>
                        ) : versions.length === 0 ? (
                            <p className="text-sm text-neutral-500 italic p-4 bg-neutral-800/20 rounded-xl border border-dashed border-neutral-700">No versions found.</p>
                        ) : (
                            <div className="space-y-2">
                                {versions.sort((a, b) => b.version - a.version).map((v) => (
                                    <div
                                        key={v.version}
                                        onClick={() => handleSelectVersion(v.run_id)}
                                        className={`p-4 rounded-xl border cursor-pointer transition-all ${
                                            selectedRun?.run_id === v.run_id
                                                ? 'bg-indigo-500/10 border-indigo-500 shadow-lg shadow-indigo-500/10'
                                                : 'bg-neutral-800/40 border-neutral-700 hover:border-neutral-500'
                                        }`}
                                    >
                                        <div className="flex justify-between items-center">
                                            <div>
                                                <span className="text-lg font-bold text-white">Version {v.version}</span>
                                                <div className={`mt-1 text-[10px] w-fit px-1.5 py-0.5 rounded border ${getStageBadge(v.stage)}`}>
                                                    {v.stage}
                                                </div>
                                            </div>
                                            <ChevronRight className={`w-4 h-4 transition-transform ${selectedRun?.run_id === v.run_id ? 'translate-x-1 text-indigo-400' : 'text-neutral-600'}`} />
                                        </div>
                                        <p className="text-[10px] text-neutral-500 mt-3 flex items-center gap-1">
                                            Created: {formatTimestamp(v.creation_timestamp)}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Run Details */}
                    <div className="lg:col-span-2">
                        {isLoadingRun ? (
                            <div className="h-full flex flex-col items-center justify-center p-12 bg-neutral-800/20 rounded-2xl border border-neutral-800">
                                <Loader2 className="w-8 h-8 animate-spin text-indigo-500 mb-4" />
                                <p className="text-neutral-400">Loading experiment data from MLflow...</p>
                            </div>
                        ) : selectedRun ? (
                            <div className="bg-neutral-800/40 border border-neutral-700 rounded-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                                <div className="p-6 bg-indigo-500/5 border-b border-neutral-700 flex justify-between items-center">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-indigo-500 text-white rounded-lg">
                                            <Activity className="w-5 h-5" />
                                        </div>
                                        <div>
                                            <h4 className="font-bold text-white">Experiment Details</h4>
                                            <p className="text-xs text-neutral-400 font-mono">Run ID: {selectedRun.run_id}</p>
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={handleDownloadVersion}
                                            className="text-indigo-400 border-indigo-500/30 hover:bg-indigo-500/10"
                                            title="Download this version"
                                        >
                                            <Download className="w-4 h-4 mr-2" />
                                            Download
                                        </Button>
                                        <a 
                                            href={`http://localhost:5000/#/experiments/${selectedRun.experiment_id}/runs/${selectedRun.run_id}`} 
                                            target="_blank" 
                                            rel="noopener noreferrer"
                                            className="p-2 hover:bg-neutral-700 rounded-lg text-neutral-400 transition-colors"
                                            title="View in MLflow UI"
                                        >
                                            <ExternalLink className="w-5 h-5" />
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
