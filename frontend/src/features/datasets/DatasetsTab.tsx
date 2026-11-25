import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Plus, Table2, Database, Eye, Trash2, X, ArrowRightCircle, Activity } from 'lucide-react';
import { Dataset, DatasetPreview, PreprocessingStep, api } from '@/lib/api';
import DriftPanel from '@/features/datasets/DriftPanel';

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
                <h3 className="text-sm font-semibold text-zinc-200">Registered Datasets</h3>
                <Button
                    size="sm"
                    className="h-8 bg-zinc-100 hover:bg-white text-zinc-950 font-medium text-xs rounded-md"
                    onClick={() => setIsDatasetModalOpen(true)}
                >
                    <Plus className="w-3.5 h-3.5 mr-1.5" /> Upload dataset
                </Button>
            </div>
            {datasets.length === 0 ? (
                <div className="border border-dashed border-zinc-800 rounded-lg p-10 text-center text-xs text-zinc-500 bg-[#14171e]">
                    No datasets found. Upload a CSV, Parquet, Excel or ZIP file to begin.
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {datasets.map(ds => (
                        <div key={ds.id} className="bg-[#14171e] border border-zinc-800/90 p-4 rounded-lg flex items-start gap-3.5 hover:border-zinc-700 transition-colors group">
                            <div className="p-2 bg-zinc-900 border border-zinc-800 rounded-md text-zinc-400">
                                {ds.file_type && ['csv', 'xlsx', 'parquet'].includes(ds.file_type)
                                    ? <Table2 className="w-4 h-4" />
                                    : <Database className="w-4 h-4" />}
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                    <h4 className="text-sm font-medium text-zinc-100 truncate">{ds.name}</h4>
                                    {ds.file_type && (
                                        <span className={`text-[10px] px-1.5 py-0.5 rounded border font-mono uppercase ${fileTypeBadgeColor(ds.file_type)}`}>
                                            {ds.file_type}
                                        </span>
                                    )}
                                    {ds.version && ds.version > 1 && (
                                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 text-amber-300 border border-amber-800/60">
                                            v{ds.version}
                                        </span>
                                    )}
                                    {ds.is_dvc_tracked && (
                                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 text-orange-400 border border-orange-800/60" title={`DVC Hash: ${ds.dvc_hash?.substring(0, 12)}...`}>
                                            DVC
                                        </span>
                                    )}
                                </div>
                                <p className="text-xs text-zinc-400 mt-1 line-clamp-2">{ds.description || "No description."}</p>
                                <div className="mt-2.5 flex items-center gap-2.5 text-[11px] font-mono tabular-nums text-zinc-500">
                                    <span>{(ds.file_size_bytes / 1024).toFixed(1)} KB</span>
                                    <span>·</span>
                                    <span>{new Date(ds.created_at).toLocaleDateString()}</span>
                                    {ds.num_rows != null && (
                                        <>
                                            <span>·</span>
                                            <span className="text-zinc-300">{ds.num_rows.toLocaleString()} × {ds.num_columns}</span>
                                        </>
                                    )}
                                </div>
                                {/* Actions */}
                                <div className="mt-3 pt-2.5 border-t border-zinc-800/80 flex flex-wrap items-center gap-1.5">
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
                    <div className="bg-[#101217] border border-zinc-800 rounded-xl p-6 max-w-md w-full">
                        <h3 className="text-base font-semibold text-zinc-100 mb-2">Delete Dataset?</h3>
                        <p className="text-zinc-400 text-xs mb-6">This will permanently delete the file from disk. Predictions linked to this dataset will NOT be deleted (lineage preserved).</p>
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
                    <div className="w-full max-w-5xl max-h-[85vh] overflow-auto bg-[#101217] border border-zinc-800 rounded-xl p-6 relative">
                        <button onClick={() => setPreviewData(null)} className="absolute top-5 right-5 text-zinc-400 hover:text-white">
                            <X className="w-5 h-5" />
                        </button>
                        <div className="mb-5">
                            <h2 className="text-lg font-semibold text-zinc-100">Data Preview</h2>
                            <p className="text-zinc-400 text-xs font-mono mt-1">
                                {previewData.num_rows.toLocaleString()} rows × {previewData.num_columns} columns •
                                <span className="uppercase ml-1 font-semibold text-orange-400">{previewData.file_type}</span>
                            </p>
                        </div>
                        <div className="overflow-x-auto rounded-lg border border-zinc-800">
                            <table className="w-full text-xs border-collapse">
                                <thead>
                                    <tr className="bg-[#0b0c0f]">
                                        {previewData.column_names.map(col => (
                                            <th key={col} className="px-3 py-2.5 text-left text-[11px] font-semibold text-zinc-400 uppercase tracking-wider whitespace-nowrap">
                                                {col}
                                                <span className="block text-[10px] text-zinc-600 font-mono font-normal normal-case">{previewData.column_dtypes[col]}</span>
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-zinc-800/80">
                                    {previewData.preview_rows.map((row, i) => (
                                        <tr key={i} className="hover:bg-zinc-800/30 transition-colors">
                                            {previewData.column_names.map(col => (
                                                <td key={col} className="px-3 py-2 text-zinc-300 font-mono whitespace-nowrap max-w-[200px] truncate">
                                                    {row[col] != null ? String(row[col]) : <span className="text-zinc-600 italic">null</span>}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Pipeline info inside preview modal */}
                        {pipelineSteps && pipelineSteps.length > 0 && (
                            <div className="mt-6 pt-5 border-t border-zinc-800">
                                <div className="flex items-center justify-between mb-3">
                                    <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                                        <ArrowRightCircle className="w-4 h-4 text-orange-400" />
                                        Applied Preprocessing Pipeline
                                    </h4>
                                    <span className="text-[11px] text-orange-400 font-mono">v{datasets.find(d => d.id === previewData.dataset_id)?.version || 1} Transformation</span>
                                </div>
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                                    {pipelineSteps.map((step, i) => (
                                        <div key={i} className="bg-[#0b0c0f] border border-zinc-800 rounded-lg p-3.5">
                                            <div className="flex justify-between items-center mb-1.5">
                                                <span className="text-[10px] font-mono text-zinc-500 uppercase">Step {i + 1}</span>
                                                <span className="px-1.5 py-0.5 bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded text-[9px] font-mono uppercase">{step.type}</span>
                                            </div>
                                            <p className="text-xs text-zinc-100 font-medium capitalize mb-1">{step.method || step.strategy || 'Default'}</p>
                                            <p className="text-[10px] font-mono text-zinc-500 truncate">{step.columns?.length || 0} cols</p>
                                        </div>
                                    ))}
                                </div>
                                {targetCol && (
                                    <div className="mt-3 text-[11px] font-mono text-zinc-400 bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-md inline-flex items-center gap-2">
                                        <span>Target variable:</span>
                                        <span className="text-orange-400 font-semibold">{targetCol}</span>
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
                    <div className="w-full max-w-2xl max-h-[85vh] overflow-auto bg-[#101217] border border-zinc-800 rounded-xl p-6 relative">
                        <button 
                            onClick={() => setSelectedDatasetForDrift(null)} 
                            className="absolute top-5 right-5 text-zinc-400 hover:text-white"
                        >
                            <X className="w-5 h-5" />
                        </button>
                        <div className="mb-5">
                            <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
                                <Activity className="w-5 h-5 text-amber-400" />
                                Data Drift Monitor
                            </h2>
                            <p className="text-zinc-400 text-xs mt-1">
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