import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Plus, Table2, Database, Eye, Trash2, X, ArrowRightCircle, Activity } from 'lucide-react';
import { Dataset, DatasetPreview, PreprocessingStep, api } from '../../../lib/api';
import DriftPanel from './DriftPanel';

interface DatasetsTabProps {
    datasets: Dataset[];
    previewData: DatasetPreview | null;
    deletingDatasetId: string | null;
    setIsDatasetModalOpen: (val: boolean) => void;
    setDeletingDatasetId: (id: string | null) => void;
    handleDeleteDataset: (id: string) => void;
    handlePreviewDataset: (id: string) => void;
    setPreviewData: (data: DatasetPreview | null) => void;
    fileTypeBadgeColor: (ft?: string) => string;
}

export default function DatasetsTab({
    datasets,
    previewData,
    deletingDatasetId,
    setIsDatasetModalOpen,
    setDeletingDatasetId,
    handleDeleteDataset,
    handlePreviewDataset,
    setPreviewData,
    fileTypeBadgeColor,
    tenantId,
    token
}: DatasetsTabProps & { tenantId: string; token: string | null }) {
    const [pipelineSteps, setPipelineSteps] = React.useState<PreprocessingStep[] | null>(null);
    const [targetCol, setTargetCol] = React.useState<string | null>(null);
    const [selectedDatasetForDrift, setSelectedDatasetForDrift] = useState<Dataset | null>(null);

    React.useEffect(() => {
        if (!previewData) {
            setPipelineSteps(null);
            setTargetCol(null);
            return;
        }

        const fetchPipeline = async () => {
            try {
                const data = await api.getDatasetPipeline(previewData.dataset_id, tenantId);
                if (data.steps && data.steps.length > 0) {
                    setPipelineSteps(data.steps);
                    setTargetCol(data.target_column || null);
                }
            } catch (err) {
                console.error("Failed to fetch dataset pipeline", err);
            }
        };
        fetchPipeline();
    }, [previewData, tenantId]);

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h3 className="text-xl font-medium">Available Datasets</h3>
                <Button className="bg-indigo-600 hover:bg-indigo-700" onClick={() => setIsDatasetModalOpen(true)}>
                    <Plus className="w-4 h-4 mr-2" /> Upload Dataset
                </Button>
            </div>
            {datasets.length === 0 ? (
                <div className="border-2 border-dashed border-neutral-700 rounded-xl p-12 text-center text-neutral-500">
                    No datasets found. Upload your first dataset to get started.
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {datasets.map(ds => (
                        <div key={ds.id} className="bg-neutral-800/40 border border-neutral-700 p-5 rounded-xl flex items-start gap-4 hover:border-indigo-500/50 transition-colors group">
                            <div className="p-3 bg-indigo-500/10 rounded-lg text-indigo-400">
                                {ds.file_type && ['csv', 'xlsx', 'parquet'].includes(ds.file_type)
                                    ? <Table2 className="w-6 h-6" />
                                    : <Database className="w-6 h-6" />}
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                    <h4 className="font-medium text-white truncate">{ds.name}</h4>
                                    {ds.file_type && (
                                        <span className={`text-[10px] px-2 py-0.5 rounded-full border font-semibold uppercase ${fileTypeBadgeColor(ds.file_type)}`}>
                                            {ds.file_type}
                                        </span>
                                    )}
                                    {ds.version && ds.version > 1 && (
                                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                                            v{ds.version}
                                        </span>
                                    )}
                                    {ds.is_dvc_tracked && (
                                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30" title={`DVC Hash: ${ds.dvc_hash?.substring(0, 12)}...`}>
                                            DVC
                                        </span>
                                    )}
                                </div>
                                <p className="text-sm text-neutral-400 mt-1 line-clamp-2">{ds.description || "No description provided."}</p>
                                <div className="mt-3 flex items-center gap-3 text-xs text-neutral-500">
                                    <span>{(ds.file_size_bytes / 1024).toFixed(1)} KB</span>
                                    <span>•</span>
                                    <span>{new Date(ds.created_at).toLocaleDateString()}</span>
                                    {ds.num_rows != null && (
                                        <>
                                            <span>•</span>
                                            <span className="text-indigo-400">{ds.num_rows.toLocaleString()} rows × {ds.num_columns} cols</span>
                                        </>
                                    )}
                                </div>
                                {/* Action buttons */}
                                <div className="mt-3 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    {ds.file_type && ['csv', 'xlsx', 'parquet'].includes(ds.file_type) && (
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            className="h-7 text-xs text-neutral-400 hover:text-indigo-400 px-2"
                                            onClick={() => handlePreviewDataset(ds.id)}
                                        >
                                            <Eye className="w-3 h-3 mr-1" /> Preview
                                        </Button>
                                    )}
                                    {ds.file_type && ['csv', 'xlsx', 'parquet'].includes(ds.file_type) && (
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            className="h-7 text-xs text-neutral-400 hover:text-amber-400 px-2"
                                            onClick={() => setSelectedDatasetForDrift(ds)}
                                        >
                                            <Activity className="w-3 h-3 mr-1" /> Drift
                                        </Button>
                                    )}
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="h-7 text-xs text-neutral-400 hover:text-red-400 px-2"
                                        onClick={() => setDeletingDatasetId(ds.id)}
                                    >
                                        <Trash2 className="w-3 h-3 mr-1" /> Delete
                                    </Button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Delete confirmation */}
            {deletingDatasetId && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                    <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-8 max-w-md w-full">
                        <h3 className="text-lg font-bold mb-2">Delete Dataset?</h3>
                        <p className="text-neutral-400 text-sm mb-6">This will permanently delete the file from disk. Predictions linked to this dataset will NOT be deleted (lineage preserved).</p>
                        <div className="flex gap-3 justify-end">
                            <Button variant="ghost" onClick={() => setDeletingDatasetId(null)}>Cancel</Button>
                            <Button className="bg-red-600 hover:bg-red-700" onClick={() => handleDeleteDataset(deletingDatasetId)}>
                                <Trash2 className="w-4 h-4 mr-2" /> Delete
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {/* Preview Modal */}
            {previewData && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
                    <div className="w-full max-w-5xl max-h-[85vh] overflow-auto bg-neutral-900 border border-neutral-800 rounded-3xl p-8 relative">
                        <button onClick={() => setPreviewData(null)} className="absolute top-6 right-6 text-neutral-400 hover:text-white">
                            <X className="w-5 h-5" />
                        </button>
                        <div className="mb-6">
                            <h2 className="text-2xl font-bold">Data Preview</h2>
                            <p className="text-neutral-400 text-sm mt-1">
                                {previewData.num_rows.toLocaleString()} rows × {previewData.num_columns} columns •
                                <span className="uppercase ml-1 text-xs font-semibold text-indigo-400">{previewData.file_type}</span>
                            </p>
                        </div>
                        <div className="overflow-x-auto rounded-xl border border-neutral-800">
                            <table className="w-full text-sm border-collapse">
                                <thead>
                                    <tr className="bg-black/60">
                                        {previewData.column_names.map(col => (
                                            <th key={col} className="px-4 py-3 text-left text-xs font-semibold text-neutral-400 uppercase tracking-wider whitespace-nowrap">
                                                {col}
                                                <span className="block text-[10px] text-neutral-600 font-normal normal-case">{previewData.column_dtypes[col]}</span>
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-neutral-800">
                                    {previewData.preview_rows.map((row, i) => (
                                        <tr key={i} className="hover:bg-white/5 transition-colors">
                                            {previewData.column_names.map(col => (
                                                <td key={col} className="px-4 py-2 text-neutral-300 whitespace-nowrap max-w-[200px] truncate">
                                                    {row[col] != null ? String(row[col]) : <span className="text-neutral-600 italic">null</span>}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Pipeline info inside preview modal */}
                        {pipelineSteps && pipelineSteps.length > 0 && (
                            <div className="mt-8 pt-6 border-t border-neutral-800">
                                <div className="flex items-center justify-between mb-4">
                                    <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-2">
                                        <ArrowRightCircle className="w-4 h-4 text-indigo-400" />
                                        Applied Preprocessing Pipeline
                                    </h4>
                                    <span className="text-[10px] text-indigo-400 font-mono">v{datasets.find(d => d.id === previewData.dataset_id)?.version || 1} Transformation</span>
                                </div>
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                                    {pipelineSteps.map((step, i) => (
                                        <div key={i} className="bg-black/40 border border-neutral-800 rounded-xl p-4">
                                            <div className="flex justify-between items-center mb-2">
                                                <span className="text-[9px] font-bold text-neutral-600 uppercase">Step {i + 1}</span>
                                                <span className="px-1.5 py-0.5 bg-indigo-500/10 text-indigo-400 rounded text-[8px] font-bold uppercase">{step.type}</span>
                                            </div>
                                            <p className="text-xs text-white font-medium capitalize mb-1">{step.method || step.strategy || 'Default'}</p>
                                            <p className="text-[10px] text-neutral-500 truncate">{step.columns?.length || 0} cols</p>
                                        </div>
                                    ))}
                                </div>
                                {targetCol && (
                                    <div className="mt-4 text-[10px] text-neutral-500 bg-indigo-500/5 px-3 py-2 rounded-lg inline-flex items-center gap-2">
                                        <span>Target variable:</span>
                                        <span className="text-indigo-400 font-bold uppercase">{targetCol}</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Drift Panel Modal */}
            {selectedDatasetForDrift && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
                    <div className="w-full max-w-2xl max-h-[85vh] overflow-auto bg-neutral-900 border border-neutral-800 rounded-3xl p-8 relative">
                        <button 
                            onClick={() => setSelectedDatasetForDrift(null)} 
                            className="absolute top-6 right-6 text-neutral-400 hover:text-white"
                        >
                            <X className="w-5 h-5" />
                        </button>
                        <div className="mb-6">
                            <h2 className="text-2xl font-bold flex items-center gap-2">
                                <Activity className="w-6 h-6 text-amber-400" />
                                Data Drift Monitor
                            </h2>
                            <p className="text-neutral-400 text-sm mt-1">
                                Análisis de estabilidad de distribuciones para: {selectedDatasetForDrift.name}
                            </p>
                        </div>
                        <DriftPanel dataset={selectedDatasetForDrift} token={token} />
                    </div>
                </div>
            )}
        </div>
    );
}