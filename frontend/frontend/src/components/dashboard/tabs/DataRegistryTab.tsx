import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Loader2, RefreshCw, GitBranch, Upload, Download, ExternalLink, FlaskConical } from 'lucide-react';
import { api } from '../../../lib/api';
import toast from 'react-hot-toast';

interface DatasetRegistry {
    name: string;
    display_name?: string;
    versions: number;
    datasets: {
        id: string;
        name: string;
        version: number;
        dvc_hash: string;
        dvc_version: number;
    }[];
}

interface DataRegistryTabProps {
    tenantId?: string;
    token?: string | null;
    onRefresh?: () => void;
    onOpenDataset?: (datasetId: string) => void;
    onTrainDataset?: (datasetId: string) => void;
}

export default function DataRegistryTab({
    tenantId,
    token,
    onRefresh,
    onOpenDataset,
    onTrainDataset,
}: DataRegistryTabProps) {
    const [registries, setRegistries] = useState<DatasetRegistry[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedRegistry, setSelectedRegistry] = useState<string | null>(null);
    const [versions, setVersions] = useState<any[]>([]);
    const [isLoadingVersions, setIsLoadingVersions] = useState(false);

    const fetchRegistries = async () => {
        if (!token || !tenantId) return;
        setIsLoading(true);
        try {
            const data = await api.getDatasetRegistries(tenantId);
            const list = data || [];
            setRegistries(list);
            if (!selectedRegistry && list.length > 0) {
                setSelectedRegistry(list[0].name);
                fetchVersions(list[0].name);
            }
        } catch (err) {
            console.error('Failed to fetch dataset registries:', err);
            toast.error('Failed to load data registries');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchRegistries();
    }, [token, tenantId]);

    const fetchVersions = async (registryName: string) => {
        if (!token || !tenantId) return;
        setIsLoadingVersions(true);
        try {
            const data = await api.getDatasetVersions(registryName, tenantId);
            setVersions(data.versions || []);
        } catch (err) {
            console.error('Failed to fetch versions:', err);
            toast.error('Failed to load versions');
        } finally {
            setIsLoadingVersions(false);
        }
    };

    const handleSelectRegistry = (name: string) => {
        setSelectedRegistry(name);
        fetchVersions(name);
    };

    const handlePromote = async (datasetId: string) => {
        if (!token || !tenantId) return;
        try {
            await api.promoteDataset(datasetId, tenantId);
            toast.success('Dataset promoted to production');
            fetchRegistries();
            if (selectedRegistry) fetchVersions(selectedRegistry);
            if (onRefresh) onRefresh();
        } catch (err: any) {
            toast.error(err.message || 'Failed to promote dataset');
        }
    };

    const handlePush = async (datasetId: string) => {
        if (!token || !tenantId) return;
        try {
            await api.pushDatasetToRemote(datasetId, tenantId);
            toast.success('Dataset pushed to DVC remote');
        } catch (err: any) {
            toast.error(err.message || 'Failed to push dataset');
        }
    };

    const handleDownload = async (dataset: any) => {
        if (!token || !tenantId) return;
        const toastId = toast.loading('Downloading dataset...');
        try {
            await api.downloadDataset(dataset.id, dataset.name);
            toast.success('Download started', { id: toastId });
        } catch (err: any) {
            toast.error(err.message || 'Failed to download dataset', { id: toastId });
        }
    };

    const getFileSize = (bytes: number) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    };

    return (
        <div className="space-y-5">
            <div className="flex justify-between items-center">
                <h3 className="text-sm font-semibold text-zinc-200">Data Registry (DVC)</h3>
                <Button
                    variant="outline"
                    size="sm"
                    onClick={fetchRegistries}
                    disabled={isLoading}
                    className="h-8 border-zinc-800 bg-[#14171e] text-zinc-300 hover:bg-zinc-800 text-xs rounded-md"
                >
                    <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${isLoading ? 'animate-spin' : ''}`} />
                    Refresh
                </Button>
            </div>

            {isLoading ? (
                <div className="text-center py-8 text-zinc-500">
                    <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                    <span className="text-xs">Loading registries...</span>
                </div>
            ) : registries.length === 0 ? (
                <div className="border border-dashed border-zinc-800 rounded-lg p-10 text-center text-xs text-zinc-500 bg-[#14171e]">
                    No DVC-tracked datasets yet. Enable DVC tracking when uploading a dataset.
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Registry List */}
                    <div className="space-y-2.5">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                            Registries
                        </h4>
                        {registries.map((reg) => {
                            const isSelected = selectedRegistry === reg.name;
                            return (
                                <div
                                    key={reg.name}
                                    onClick={() => handleSelectRegistry(reg.name)}
                                    className={`p-3.5 rounded-lg border cursor-pointer transition-colors ${
                                        isSelected
                                            ? 'bg-zinc-800/70 border-zinc-600'
                                            : 'bg-[#14171e] border-zinc-800/90 hover:border-zinc-700'
                                    }`}
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-zinc-900 border border-zinc-800 rounded-md text-orange-400">
                                            <GitBranch className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <h5 className="text-sm font-medium text-zinc-100 truncate">
                                                {reg.display_name || reg.name.replace(/^tenant_[^_]+_/, '')}
                                            </h5>
                                            <p className="text-[11px] font-mono text-zinc-500">
                                                {reg.datasets.length} dataset(s) · {reg.versions} version(s)
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Version Details */}
                    <div className="space-y-2.5">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                            {selectedRegistry
                                ? `Versions · ${registries.find(r => r.name === selectedRegistry)?.display_name || selectedRegistry.replace(/^tenant_[^_]+_/, '')}`
                                : 'Select a registry'}
                        </h4>

                        {!selectedRegistry ? (
                            <div className="border border-zinc-800/90 rounded-lg bg-[#14171e] text-center py-8 text-zinc-500 text-xs">
                                Select a registry to inspect versions.
                            </div>
                        ) : isLoadingVersions ? (
                            <div className="text-center py-8 text-zinc-500">
                                <Loader2 className="w-5 h-5 animate-spin mx-auto" />
                            </div>
                        ) : versions.length === 0 ? (
                            <div className="border border-zinc-800/90 rounded-lg bg-[#14171e] text-center py-8 text-zinc-500 text-xs">
                                No versions found.
                            </div>
                        ) : (
                            <div className="space-y-2.5">
                                {versions.map((v) => (
                                    <div
                                        key={v.id}
                                        className="p-4 bg-[#14171e] border border-zinc-800/90 rounded-lg space-y-3"
                                    >
                                        <div className="flex justify-between items-start gap-3">
                                            <div className="min-w-0">
                                                <div className="flex items-center gap-2">
                                                    <h5 className="text-sm font-medium text-zinc-100 truncate">
                                                        {v.name}
                                                    </h5>
                                                    {v.is_active && (
                                                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-emerald-800/80 bg-emerald-950/60 text-emerald-300">
                                                            production
                                                        </span>
                                                    )}
                                                </div>
                                                <p className="text-xs font-mono text-zinc-400 mt-0.5">
                                                    v{v.version} · {getFileSize(v.file_size_bytes || 0)}
                                                </p>
                                                <p
                                                    className="text-[11px] text-zinc-500 font-mono mt-0.5 truncate"
                                                    title={v.dvc_hash}
                                                >
                                                    md5: {v.dvc_hash?.substring(0, 12)}...
                                                </p>
                                            </div>
                                            <span className="text-[11px] text-zinc-500 font-mono shrink-0">
                                                {new Date(v.created_at).toLocaleDateString()}
                                            </span>
                                        </div>

                                        <div className="pt-2.5 border-t border-zinc-800/80 flex flex-wrap gap-1.5">
                                            {onOpenDataset && (
                                                <Button
                                                    size="sm"
                                                    onClick={() => onOpenDataset(v.id)}
                                                    className="h-7 px-2.5 text-xs bg-zinc-100 hover:bg-white text-zinc-950 font-medium rounded-md"
                                                >
                                                    <ExternalLink className="w-3 h-3 mr-1" />
                                                    View Dataset
                                                </Button>
                                            )}
                                            {onTrainDataset && (
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    onClick={() => onTrainDataset(v.id)}
                                                    className="h-7 px-2.5 text-xs border-zinc-800 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 rounded-md"
                                                >
                                                    <FlaskConical className="w-3 h-3 mr-1" />
                                                    Train
                                                </Button>
                                            )}
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                onClick={() => handlePush(v.id)}
                                                className="h-7 px-2.5 text-xs border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 rounded-md"
                                            >
                                                <Upload className="w-3 h-3 mr-1" />
                                                Push
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                onClick={() => handleDownload(v)}
                                                className="h-7 px-2.5 text-xs border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 rounded-md"
                                            >
                                                <Download className="w-3 h-3 mr-1" />
                                                Download
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                onClick={() => handlePromote(v.id)}
                                                className="h-7 px-2.5 text-xs border-zinc-800 bg-zinc-900 text-emerald-300 hover:bg-zinc-800 rounded-md"
                                            >
                                                Promote
                                            </Button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
