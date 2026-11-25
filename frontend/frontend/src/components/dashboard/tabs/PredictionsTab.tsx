import React from 'react';
import { Button } from '@/components/ui/button';
import { CheckCircle2, XCircle, Clock, Activity, Loader2, ExternalLink, UploadCloud } from 'lucide-react';
import { MLModel, Prediction, Dataset } from '../../../lib/api';
import { getModelCompatibilityProfile } from '../../../lib/modelCompatibility';
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
    handleRunInference: (batchFile?: File, explain?: boolean) => void;
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
    const [batchExplain, setBatchExplain] = React.useState<boolean>(true);
    const tabularModels = React.useMemo(
        () => models.filter((m) => getModelCompatibilityProfile(m).domain === 'tabular'),
        [models]
    );
    const hiddenNonTabularCount = models.length - tabularModels.length;

    const renderStatusIcon = (status: string) => {
        switch (status) {
            case 'COMPLETED': return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
            case 'FAILED': return <XCircle className="w-4 h-4 text-red-400" />;
            case 'PENDING': return <Clock className="w-4 h-4 text-amber-400" />;
            case 'IN_PROGRESS':
            case 'RUNNING':
            case 'STARTED': return <Activity className="w-4 h-4 text-orange-400 animate-pulse" />;
            default: return <Loader2 className="w-4 h-4 text-zinc-400 animate-spin" />;
        }
    };

    const renderStatusClass = (status: string) => {
        switch (status) {
            case 'COMPLETED': return 'text-emerald-400';
            case 'FAILED': return 'text-red-400';
            case 'PENDING': return 'text-amber-400';
            case 'IN_PROGRESS':
            case 'RUNNING':
            case 'STARTED': return 'text-orange-400';
            default: return 'text-zinc-400';
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-[#101217] border border-zinc-800 rounded-lg overflow-hidden">
                <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-200">Inference History</span>
                    <span className="text-[11px] font-mono tabular-nums text-zinc-500">{predictions.length} runs</span>
                </div>
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-[#0b0c0f] text-zinc-500 text-[11px] uppercase tracking-wider border-b border-zinc-800">
                            <th className="px-5 py-3 font-medium">Status</th>
                            <th className="px-5 py-3 font-medium">Model / Method</th>
                            <th className="px-5 py-3 font-medium">Dataset</th>
                            <th className="px-5 py-3 font-medium">Created</th>
                            <th className="px-5 py-3 font-medium text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/80">
                        {predictions.map(pred => {
                            return (
                                <tr key={pred.id} className="hover:bg-zinc-800/30 transition-colors">
                                    <td className="px-5 py-3">
                                        <div className="flex items-center gap-2">
                                            {renderStatusIcon(pred.status)}
                                            <span className={`text-xs font-mono ${renderStatusClass(pred.status)}`}>
                                                {pred.status}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-5 py-3">
                                        <div className="text-xs font-medium text-zinc-100">
                                            {models.find(m => m.id === pred.model_id)?.name || pred.model_id.substring(0, 8)}
                                        </div>
                                        <div className="text-[11px] font-mono text-zinc-500 mt-0.5">{pred.method}</div>
                                    </td>
                                    <td className="px-5 py-3 text-xs text-zinc-400">
                                        {pred.dataset_id ? (datasets.find(d => d.id === pred.dataset_id)?.name || pred.dataset_id.substring(0, 8)) : "Tabular Data"}
                                    </td>
                                    <td className="px-5 py-3 text-xs font-mono text-zinc-500">
                                        {new Date(pred.created_at).toLocaleString()}
                                    </td>
                                    <td className="px-5 py-3 text-right">
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            className="h-7 text-xs text-zinc-400 hover:text-zinc-100"
                                            disabled={pred.status !== 'COMPLETED'}
                                            onClick={() => setSelectedPredictionView(pred)}
                                        >
                                            <ExternalLink className="w-3.5 h-3.5 mr-1.5" /> View
                                        </Button>
                                    </td>
                                </tr>
                            )
                        })}
                        {predictions.length === 0 && (
                            <tr>
                                <td colSpan={5} className="px-5 py-10 text-center text-xs text-zinc-500">
                                    No predictions yet. Run a batch or single-row prediction below.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {/* Batch Tabular Inference */}
            <div className="p-5 bg-[#101217] border border-zinc-800 rounded-lg space-y-5">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-zinc-900 border border-zinc-800 rounded-md text-zinc-300">
                        <Activity className="w-4 h-4" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-zinc-100">Batch Tabular Inference</h3>
                        <p className="text-xs text-zinc-500">Upload a CSV, Excel or Parquet file to score multiple rows asynchronously.</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div>
                        <label className="text-[11px] font-medium text-zinc-400 mb-1.5 block">Batch Data File (.csv, .xlsx, .parquet)</label>
                        <div
                            className="bg-[#0b0c0f] border border-dashed border-zinc-700 rounded-lg p-6 flex flex-col items-center justify-center text-center cursor-pointer hover:border-zinc-500 transition-colors"
                            onClick={() => document.getElementById('batch-file-input')?.click()}
                        >
                            <input
                                id="batch-file-input"
                                type="file"
                                className="hidden"
                                accept=".csv,.xlsx,.parquet"
                                onChange={(e) => {
                                    if (e.target.files?.[0]) {
                                        setBatchFile(e.target.files[0]);
                                    }
                                }}
                            />
                            {batchFile ? (
                                <div className="flex flex-col items-center">
                                    <CheckCircle2 className="w-6 h-6 text-emerald-400 mb-1.5" />
                                    <p className="text-xs font-mono text-zinc-100">{batchFile.name}</p>
                                    <p className="text-[10px] text-zinc-500 mt-0.5">Ready to process</p>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center">
                                    <UploadCloud className="w-6 h-6 text-zinc-500 mb-1.5" />
                                    <p className="text-xs text-zinc-400">Click to select file</p>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="space-y-4">
                        <div className="space-y-1.5">
                            <label className="text-[11px] font-medium text-zinc-400">Tabular ML Model</label>
                            <select
                                className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                                value={tabularModels.some((m) => m.id === selectedModel) ? selectedModel : ''}
                                onChange={(e) => {
                                    setSelectedModel(e.target.value);
                                    setUncertaintyMethod('none');
                                }}
                            >
                                <option value="" disabled>Select a tabular model</option>
                                {tabularModels.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                            </select>
                            {hiddenNonTabularCount > 0 && (
                                <p className="text-[10px] text-zinc-500">
                                    {hiddenNonTabularCount} non-tabular model{hiddenNonTabularCount > 1 ? 's' : ''} hidden (use NLP Engine or Model Serving).
                                </p>
                            )}
                        </div>
                        <div className="space-y-1.5">
                            <label className="text-[11px] font-medium text-zinc-400">Uncertainty Method</label>
                            <select
                                className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                                value={uncertaintyMethod}
                                onChange={(e) => setUncertaintyMethod(e.target.value)}
                            >
                                <option value="none">None (Standard)</option>
                                <option value="entropy">Predictive Entropy</option>
                                <option value="tree_variance">Tree Variance (Ensembles)</option>
                                <option value="conformal">Conformal Prediction</option>
                            </select>
                        </div>
                        <label className="flex items-center gap-2.5 pt-1 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={batchExplain}
                                onChange={(e) => setBatchExplain(e.target.checked)}
                                className="w-3.5 h-3.5 rounded border-zinc-700 bg-[#0b0c0f] accent-orange-500"
                            />
                            <span className="text-xs text-zinc-300">
                                Include SHAP Explainability
                                <span className="text-[11px] text-zinc-500 ml-1">(feature attributions)</span>
                            </span>
                        </label>
                    </div>
                </div>

                <button
                    type="button"
                    className="w-full h-10 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-semibold disabled:opacity-40 flex items-center justify-center gap-2 transition-colors"
                    disabled={(!selectedDataset && !batchFile) || !selectedModel || isActionLoading}
                    onClick={() => handleRunInference(batchFile || undefined, batchExplain)}
                >
                    {isActionLoading ? (
                        <><Loader2 className="w-4 h-4 animate-spin" /> Processing Batch...</>
                    ) : (
                        <><Activity className="w-4 h-4" /> Start Batch Inference</>
                    )}
                </button>
            </div>

            <div className="mt-12">
                <SingleTabularInference
                    models={tabularModels}
                    onPredictionStarted={() => {
                        // Refresh the predictions list when a single file starts
                        if (token) {
                            fetch(config.getFullApiUrl('/tabular/predictions'), {
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
