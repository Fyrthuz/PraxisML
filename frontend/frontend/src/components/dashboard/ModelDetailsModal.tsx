import React, { useState } from 'react';
import { X, Cpu, Tag, Calendar, Database, Target, Activity, Copy, Check, Download, Trash2, Loader2, ArrowRightCircle } from 'lucide-react';
import { MLModel, api, PreprocessingStep } from '../../lib/api';
import { Button } from '@/components/ui/button';

interface ModelDetailsModalProps {
    model: MLModel | null;
    onClose: () => void;
    handleDeleteModel: (modelId: string) => Promise<boolean>;
}

export default function ModelDetailsModal({ model, onClose, handleDeleteModel }: ModelDetailsModalProps) {
    if (!model) return null;

    const metrics = model.metrics_metadata?.metrics || {};
    const hyperparams = Object.entries(model.metrics_metadata || {}).filter(
        ([key]) => !['framework', 'algorithm', 'task_type', 'target_column', 'metrics', 'feature_names', 'dataset_id', 'dataset_name'].includes(key)
    );

    const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
    const [isDeleting, setIsDeleting] = useState(false);
    const [pipelineSteps, setPipelineSteps] = useState<PreprocessingStep[] | null>(null);
    const [targetCol, setTargetCol] = useState<string | null>(null);
    const [isLoadingPipeline, setIsLoadingPipeline] = useState(false);

    React.useEffect(() => {
        const fetchPipeline = async () => {
            const datasetId = model.metrics_metadata?.dataset_id;
            if (!datasetId) return;

            setIsLoadingPipeline(true);
            try {
                const data = await api.getDatasetPipeline(datasetId, model.tenant_id);
                if (data.steps && data.steps.length > 0) {
                    setPipelineSteps(data.steps);
                    setTargetCol(data.target_column || null);
                }
            } catch (err) {
                console.error("Failed to fetch model dataset pipeline", err);
            } finally {
                setIsLoadingPipeline(false);
            }
        };
        fetchPipeline();
    }, [model]);

    const handleCopy = (text: string, id: string) => {
        navigator.clipboard.writeText(text);
        setCopiedStates(prev => ({ ...prev, [id]: true }));
        setTimeout(() => {
            setCopiedStates(prev => ({ ...prev, [id]: false }));
        }, 2000);
    };

    const handleDownload = () => {
        window.open(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1"}/models/${model.id}/download?tenant_id=${model.tenant_id}`, '_blank');
    };

    const onDeleteClick = async () => {
        if (confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
            setIsDeleting(true);
            const success = await handleDeleteModel(model.id);
            if (success) {
                onClose();
            }
            setIsDeleting(false);
        }
    };

    const CopyButton = ({ text, id }: { text: string, id: string }) => (
        <button
            onClick={() => handleCopy(text, id)}
            className="ml-2 mt-1 p-1 text-neutral-500 hover:text-white hover:bg-neutral-800 rounded transition-colors"
            title="Copy to clipboard"
        >
            {copiedStates[id] ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
    );

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-neutral-900 border border-neutral-800 w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-3xl shadow-2xl scale-in-center">
                <div className="sticky top-0 z-10 p-6 border-b border-neutral-800 flex justify-between items-center bg-neutral-900/90 backdrop-blur">
                    <div className="flex items-center gap-3">
                        <div className="p-3 bg-indigo-500/10 rounded-xl text-indigo-400">
                            <Cpu className="w-6 h-6" />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold text-white">{model.name}</h3>
                            <p className="text-sm text-neutral-400 mt-1">{model.description || "No description provided."}</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button
                            variant="destructive"
                            size="sm"
                            className="bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20"
                            onClick={onDeleteClick}
                            disabled={isDeleting}
                        >
                            {isDeleting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Trash2 className="w-4 h-4 mr-2" />}
                            Delete Model
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            className="bg-neutral-800 border-neutral-700 hover:bg-neutral-700 hover:text-white"
                            onClick={handleDownload}
                        >
                            <Download className="w-4 h-4 mr-2" />
                            Download Model
                        </Button>
                        <button onClick={onClose} className="p-2 hover:bg-neutral-800 rounded-full text-neutral-400 transition-colors" disabled={isDeleting}>
                            <X className="w-6 h-6" />
                        </button>
                    </div>
                </div>

                <div className="p-8 space-y-8">
                    {/* General Information */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div className="bg-neutral-950 p-4 rounded-2xl border border-neutral-800/50 flex justify-between items-start">
                            <div className="flex-1 min-w-0">
                                <div className="text-neutral-500 text-xs font-bold uppercase mb-1 flex items-center gap-2">
                                    <Tag className="w-3 h-3 flex-shrink-0" /> Framework
                                </div>
                                <div className="text-white capitalize truncate">{model.metrics_metadata?.framework || 'Unknown'}</div>
                            </div>
                            <CopyButton text={model.metrics_metadata?.framework || 'Unknown'} id="framework" />
                        </div>
                        <div className="bg-neutral-950 p-4 rounded-2xl border border-neutral-800/50 flex justify-between items-start">
                            <div className="flex-1 min-w-0">
                                <div className="text-neutral-500 text-xs font-bold uppercase mb-1 flex items-center gap-2">
                                    <Cpu className="w-3 h-3 flex-shrink-0" /> Algorithm
                                </div>
                                <div className="text-white capitalize truncate">{model.metrics_metadata?.algorithm || 'Unknown'}</div>
                            </div>
                            <CopyButton text={model.metrics_metadata?.algorithm || 'Unknown'} id="algorithm" />
                        </div>
                        <div className="bg-neutral-950 p-4 rounded-2xl border border-neutral-800/50 flex justify-between items-start">
                            <div className="flex-1 min-w-0">
                                <div className="text-neutral-500 text-xs font-bold uppercase mb-1 flex items-center gap-2">
                                    <Target className="w-3 h-3 flex-shrink-0" /> Task Type
                                </div>
                                <div className="text-white capitalize truncate">{model.metrics_metadata?.task_type || 'Unknown'}</div>
                            </div>
                            <CopyButton text={model.metrics_metadata?.task_type || 'Unknown'} id="task_type" />
                        </div>
                        <div className="bg-neutral-950 p-4 rounded-2xl border border-neutral-800/50 flex justify-between items-start">
                            <div className="flex-1 min-w-0">
                                <div className="text-neutral-500 text-xs font-bold uppercase mb-1 flex items-center gap-2">
                                    <Database className="w-3 h-3 flex-shrink-0" /> Dataset
                                </div>
                                <div className="text-white truncate">{model.metrics_metadata?.dataset_name || 'Unknown'}</div>
                            </div>
                            <CopyButton text={model.metrics_metadata?.dataset_name || 'Unknown'} id="dataset" />
                        </div>
                        <div className="bg-neutral-950 p-4 rounded-2xl border border-neutral-800/50 flex justify-between items-start">
                            <div className="flex-1 min-w-0">
                                <div className="text-neutral-500 text-xs font-bold uppercase mb-1 flex items-center gap-2">
                                    <Activity className="w-3 h-3 flex-shrink-0" /> Target Column
                                </div>
                                <div className="text-white truncate">{model.metrics_metadata?.target_column || 'Unknown'}</div>
                            </div>
                            <CopyButton text={model.metrics_metadata?.target_column || 'Unknown'} id="target_column" />
                        </div>
                        <div className="bg-neutral-950 p-4 rounded-2xl border border-neutral-800/50 flex justify-between items-start">
                            <div className="flex-1 min-w-0">
                                <div className="text-neutral-500 text-xs font-bold uppercase mb-1 flex items-center gap-2">
                                    <Calendar className="w-3 h-3 flex-shrink-0" /> MLFlow Run ID
                                </div>
                                <div className="text-white text-sm font-mono truncate">{model.mlflow_run_id}</div>
                            </div>
                            <CopyButton text={model.mlflow_run_id} id="mlflow_run_id" />
                        </div>
                    </div>

                    {/* Metrics Section */}
                    {Object.keys(metrics).length > 0 && (
                        <div>
                            <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider mb-4 pb-2 border-b border-neutral-800">
                                Evaluation Metrics
                            </h4>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {Object.entries(metrics).map(([key, value]) => (
                                    <div key={key} className="bg-indigo-500/5 border border-indigo-500/10 p-4 rounded-2xl">
                                        <div className="text-indigo-300 text-xs font-bold uppercase mb-2 truncate" title={key}>
                                            {key.replace(/cv_mean_|test_/g, '').replace(/_/g, ' ')}
                                        </div>
                                        <div className="text-white text-xl font-mono">
                                            {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Hyperparameters Section */}
                    {hyperparams.length > 0 && (
                        <div>
                            <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider mb-4 pb-2 border-b border-neutral-800">
                                Hyperparameters
                            </h4>
                            <div className="bg-neutral-950 rounded-2xl border border-neutral-800 overflow-hidden">
                                <table className="w-full text-sm text-left">
                                    <thead className="text-xs text-neutral-500 uppercase bg-neutral-900 border-b border-neutral-800">
                                        <tr>
                                            <th className="px-6 py-3 font-medium">Parameter</th>
                                            <th className="px-6 py-3 font-medium">Value</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-neutral-800">
                                        {hyperparams.map(([key, value]) => (
                                            <tr key={key} className="hover:bg-neutral-900/50 transition-colors">
                                                <td className="px-6 py-3 font-mono text-neutral-400">{key}</td>
                                                <td className="px-6 py-3 text-white">
                                                    <div className="flex items-center justify-between gap-2">
                                                        <span className="font-mono truncate">
                                                            {typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)}
                                                        </span>
                                                        <CopyButton text={typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)} id={`hp_${key}`} />
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Features Section */}
                    {model.metrics_metadata?.feature_names && (
                        <div>
                            <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider mb-4 pb-2 border-b border-neutral-800">
                                Input Features ({model.metrics_metadata.feature_names.length})
                            </h4>
                            <div className="flex flex-wrap gap-2">
                                {model.metrics_metadata.feature_names.map((feat: string) => (
                                    <span key={feat} className="px-3 py-1 bg-neutral-800 text-neutral-300 rounded-lg text-sm border border-neutral-700">
                                        {feat}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Preprocessing Pipeline Section */}
                    {pipelineSteps && pipelineSteps.length > 0 && (
                        <div>
                            <div className="flex items-center justify-between mb-4 pb-2 border-b border-neutral-800">
                                <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider">
                                    Preprocessing Pipeline
                                </h4>
                                <span className="text-[10px] text-indigo-400 font-mono">Lineage: Dataset Transformation</span>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                                {pipelineSteps.map((step, i) => (
                                    <div key={i} className="bg-neutral-950 border border-neutral-800 rounded-2xl p-4">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">Step {i + 1}</span>
                                            <span className="px-2 py-0.5 bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-md text-[9px] font-bold uppercase">{step.type}</span>
                                        </div>
                                        <div className="flex items-center gap-2 mb-2">
                                            <div className="p-1 bg-indigo-500/20 rounded-md">
                                                <ArrowRightCircle className="w-3 h-3 text-indigo-400" />
                                            </div>
                                            <p className="text-xs text-white font-medium capitalize">{step.method || step.strategy || 'Default'}</p>
                                        </div>
                                        <p className="text-[10px] text-neutral-400 line-clamp-2" title={step.columns?.join(", ")}>
                                            {step.columns?.length || 0} columns
                                        </p>
                                    </div>
                                ))}
                            </div>

                            {targetCol && (
                                <div className="mt-4 p-3 bg-indigo-500/5 border border-indigo-500/10 rounded-xl flex items-center justify-between">
                                    <span className="text-xs text-neutral-400">Target Column used in preprocessing:</span>
                                    <span className="text-xs font-bold text-indigo-400 px-2 py-1 bg-indigo-500/10 rounded-lg">{targetCol}</span>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
