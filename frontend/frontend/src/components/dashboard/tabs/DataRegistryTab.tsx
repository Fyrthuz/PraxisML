import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Database, Trash2, Loader2, RefreshCw, GitBranch, Upload, Download } from 'lucide-react';
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
}

export default function DataRegistryTab({ tenantId, token, onRefresh }: DataRegistryTabProps) {
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
            setRegistries(data || []);
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
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h3 className="text-xl font-medium">Data Registry (DVC)</h3>
                <Button variant="outline" size="sm" onClick={fetchRegistries} disabled={isLoading}>
                    <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                    Refresh
                </Button>
            </div>

            {isLoading ? (
                <div className="text-center py-8 text-neutral-500">
                    <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                    Loading...
                </div>
            ) : registries.length === 0 ? (
                <div className="border-2 border-dashed border-neutral-700 rounded-xl p-12 text-center text-neutral-500">
                    No DVC-tracked datasets yet. Enable DVC tracking when uploading a dataset.
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Registry List */}
                    <div className="space-y-3">
                        <h4 className="text-sm font-medium text-neutral-400">Data Registries</h4>
                        {registries.map((reg) => (
                            <div
                                key={reg.name}
                                onClick={() => handleSelectRegistry(reg.name)}
                                className={`p-4 rounded-xl border cursor-pointer transition-colors ${
                                    selectedRegistry === reg.name
                                        ? 'bg-indigo-500/10 border-indigo-500/30'
                                        : 'bg-neutral-800/40 border-neutral-700 hover:border-neutral-600'
                                }`}
                            >
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-orange-500/10 rounded-lg text-orange-400">
                                        <GitBranch className="w-5 h-5" />
                                    </div>
                                    <div className="flex-1">
                                        <h5 className="font-medium text-white">
                                            {reg.display_name || reg.name.replace(/^tenant_[^_]+_/, '')}
                                        </h5>
                                        <p className="text-xs text-neutral-400">
                                            {reg.datasets.length} dataset(s) • {reg.versions} version(s)
                                        </p>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Version Details */}
                    <div className="space-y-3">
                        <h4 className="text-sm font-medium text-neutral-400">
                            {selectedRegistry ? `Versions: ${registries.find(r => r.name === selectedRegistry)?.display_name || selectedRegistry.replace(/^tenant_[^_]+_/, '')}` : 'Select a registry'}
                        </h4>
                        
                        {!selectedRegistry ? (
                            <div className="text-center py-8 text-neutral-500 text-sm">
                                Select a registry to view versions
                            </div>
                        ) : isLoadingVersions ? (
                            <div className="text-center py-8 text-neutral-500">
                                <Loader2 className="w-6 h-6 animate-spin mx-auto" />
                            </div>
                        ) : versions.length === 0 ? (
                            <div className="text-center py-8 text-neutral-500 text-sm">
                                No versions found
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {versions.map((v) => (
                                    <div
                                        key={v.id}
                                        className="p-4 bg-neutral-800/40 border border-neutral-700 rounded-xl"
                                    >
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <h5 className="font-medium text-white">{v.name}</h5>
                                                <p className="text-xs text-neutral-400 mt-1">
                                                    Version {v.version} • {getFileSize(v.file_size_bytes || 0)}
                                                </p>
                                                <p className="text-[10px] text-neutral-500 font-mono mt-1 truncate" title={v.dvc_hash}>
                                                    Hash: {v.dvc_hash?.substring(0, 12)}...
                                                </p>
                                            </div>
                                            <div className="flex gap-2">
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    onClick={() => handlePush(v.id)}
                                                    className="text-xs"
                                                >
                                                    <Upload className="w-3 h-3 mr-1" />
                                                    Push
                                                </Button>
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    onClick={() => handleDownload(v)}
                                                    className="text-xs text-emerald-400 hover:text-emerald-300"
                                                >
                                                    <Download className="w-3 h-3 mr-1" />
                                                    Download
                                                </Button>
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    onClick={() => handlePromote(v.id)}
                                                    className="text-xs"
                                                >
                                                    Promote
                                                </Button>
                                            </div>
                                        </div>
                                        <p className="text-xs text-neutral-500 mt-2">
                                            Created: {new Date(v.created_at).toLocaleString()}
                                        </p>
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
