import React from 'react';
import { Button } from '@/components/ui/button';
import { CheckCircle2, XCircle, Clock, Activity, Loader2, ExternalLink, UploadCloud } from 'lucide-react';
import { MLModel, Prediction, Dataset } from '../../../lib/api';
import SingleTabularInference from '../SingleTabularInference';
import { config } from '@/lib/config';

interface PredictionsTabProps {
    predictions: Prediction[];
    models: MLModel[];
    datasets: Dataset[];
    token: string | null;
    setPredictions: React.Dispatch<React.SetStateAction<Prediction[]>>;
    setSelectedPredictionView: (pred: Prediction | null) => void;
    // Lifted Quick Inference props
    selectedDataset: string;
    setSelectedDataset: (val: string) => void;
    selectedModel: string;
    setSelectedModel: (val: string) => void;
    uncertaintyMethod: string;
    setUncertaintyMethod: (val: string) => void;
    isActionLoading: boolean;
    handleRunInference: (batchFile?: File) => void;
}

export default function PredictionsTab({
    predictions,
    models,
    datasets,
    token,
    setPredictions,
    setSelectedPredictionView,
    selectedDataset,
    setSelectedDataset,
    selectedModel,
    setSelectedModel,
    uncertaintyMethod,
    setUncertaintyMethod,
    isActionLoading,
    handleRunInference
}: PredictionsTabProps) {
    const [batchFile, setBatchFile] = React.useState<File | null>(null);

    const renderStatusIcon = (status: string) => {
        switch (status) {
            case 'COMPLETED': return <CheckCircle2 className="w-4 h-4 text-green-400" />;
            case 'FAILED': return <XCircle className="w-4 h-4 text-red-400" />;
            case 'PENDING': return <Clock className="w-4 h-4 text-yellow-500" />;
            case 'IN_PROGRESS':
            case 'RUNNING':
            case 'STARTED': return <Activity className="w-4 h-4 text-blue-400 animate-pulse" />;
            default: return <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />;
        }
    };

    const renderStatusClass = (status: string) => {
        switch (status) {
            case 'COMPLETED': return 'text-green-400';
            case 'FAILED': return 'text-red-400';
            case 'PENDING': return 'text-yellow-500';
            case 'IN_PROGRESS':
            case 'RUNNING':
            case 'STARTED': return 'text-blue-400';
            default: return 'text-indigo-400';
        }
    };

    return (
        <div className="space-y-6">
            <h3 className="text-xl font-medium">Inference History</h3>
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-black/40 text-neutral-400 text-xs uppercase tracking-wider">
                            <th className="px-6 py-4 font-semibold">Status</th>
                            <th className="px-6 py-4 font-semibold">Model / Method</th>
                            <th className="px-6 py-4 font-semibold">Dataset</th>
                            <th className="px-6 py-4 font-semibold">Created</th>
                            <th className="px-6 py-4 font-semibold">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-800">
                        {predictions.map(pred => {
                            return (
                                <tr key={pred.id} className="hover:bg-white/5 transition-colors">
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-2">
                                            {renderStatusIcon(pred.status)}
                                            <span className={`text-sm ${renderStatusClass(pred.status)}`}>
                                                {pred.status}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="text-sm font-medium text-white">
                                            {models.find(m => m.id === pred.model_id)?.name || pred.model_id.substring(0, 8)}
                                        </div>
                                        <div className="text-xs text-neutral-500 mt-0.5">{pred.method}</div>
                                    </td>
                                    <td className="px-6 py-4 text-sm text-neutral-400">
                                        {pred.dataset_id ? (datasets.find(d => d.id === pred.dataset_id)?.name || pred.dataset_id.substring(0, 8)) : "Tabular Data"}
                                    </td>
                                    <td className="px-6 py-4 text-xs text-neutral-500">
                                        {new Date(pred.created_at).toLocaleString()}
                                    </td>
                                    <td className="px-6 py-4">
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            className="text-neutral-400 hover:text-white"
                                            disabled={pred.status !== 'COMPLETED'}
                                            onClick={() => setSelectedPredictionView(pred)}
                                        >
                                            <ExternalLink className="w-4 h-4 mr-2" /> View
                                        </Button>
                                    </td>
                                </tr>
                            )
                        })}
                        {predictions.length === 0 && (
                            <tr>
                                <td colSpan={5} className="px-6 py-12 text-center text-neutral-500 italic">
                                    No predictions yet. Head to Workspace to run your first inference.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {/* Quick Inference Panel (Moved from DashboardMVP) */}
            <div className="mt-16 p-8 bg-neutral-900/40 border border-indigo-500/20 rounded-3xl backdrop-blur-md">
                <div className="flex items-center gap-4 mb-8">
                    <div className="p-3 bg-indigo-500/20 rounded-2xl text-indigo-400">
                        <Activity className="w-6 h-6" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold">Batch Tabular Inference</h3>
                        <p className="text-sm text-neutral-400">Upload a CSV/Excel file to run predictions on multiple rows.</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                    {/* File Upload Dropzone */}
                    <div className="md:col-span-1">
                        <label className="text-xs font-bold text-neutral-500 uppercase ml-1 mb-2 block">Batch Data File (.csv, .xlsx)</label>
                        <div
                            className="bg-neutral-950/50 border-2 border-dashed border-neutral-800 rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer hover:bg-neutral-900/50 transition-all group"
                            onClick={() => document.getElementById('batch-file-input')?.click()}
                        >
                            <input
                                id="batch-file-input"
                                type="file"
                                className="hidden"
                                accept=".csv,.xlsx,.parquet"
                                onChange={(e) => {
                                    if (e.target.files?.[0]) {
                                        // Store file in a local state if needed, or just trigger the parent handler
                                        // For simplicity, we'll use a local state here to show the filename
                                        setBatchFile(e.target.files[0]);
                                    }
                                }}
                            />
                            {batchFile ? (
                                <div className="flex flex-col items-center">
                                    <CheckCircle2 className="w-8 h-8 text-green-500 mb-2" />
                                    <p className="text-sm font-medium text-white">{batchFile.name}</p>
                                    <p className="text-[10px] text-neutral-500 mt-1">Ready to process</p>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center">
                                    <UploadCloud className="w-8 h-8 text-neutral-600 group-hover:text-indigo-400 mb-2 transition-colors" />
                                    <p className="text-sm text-neutral-500">Click to upload CSV</p>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-neutral-500 uppercase ml-1">ML Model</label>
                            <select
                                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none text-white"
                                value={selectedModel}
                                onChange={(e) => {
                                    setSelectedModel(e.target.value);
                                    setUncertaintyMethod('none');
                                }}
                            >
                                <option value="" disabled>Select a model</option>
                                {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                            </select>
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-neutral-500 uppercase ml-1">Method</label>
                            <select
                                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none text-white"
                                value={uncertaintyMethod}
                                onChange={(e) => setUncertaintyMethod(e.target.value)}
                            >
                                <option value="none">None (Standard)</option>
                                <option value="entropy">Predictive Entropy</option>
                                <option value="tree_variance">Tree Variance (Ensembles)</option>
                                <option value="conformal">Conformal Prediction</option>
                            </select>
                        </div>
                    </div>
                </div>

                <Button
                    className="w-full bg-indigo-600 hover:bg-indigo-700 h-14 rounded-2xl text-lg font-bold shadow-2xl shadow-indigo-600/20"
                    disabled={(!selectedDataset && !batchFile) || !selectedModel || isActionLoading}
                    onClick={() => handleRunInference(batchFile || undefined)}
                >
                    {isActionLoading ? (
                        <>
                            <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                            Processing Batch...
                        </>
                    ) : (
                        <>
                            <Activity className="w-5 h-5 mr-2" />
                            Start Batch Inference
                        </>
                    )}
                </Button>
            </div>

            <div className="mt-12">
                <SingleTabularInference
                    models={models}
                    onPredictionStarted={() => {
                        // Refresh the predictions list when a single file starts
                        if (token) {
                            fetch(config.getFullApiUrl('/predictions'), {
                                headers: { Authorization: `Bearer ${token}` }
                            }).then(r => r.json()).then(preds => {
                                setPredictions(Array.isArray(preds) ? preds : []);
                            });
                        }
                    }}
                />
            </div>
        </div >
    );
}
