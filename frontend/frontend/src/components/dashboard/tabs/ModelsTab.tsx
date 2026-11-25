import React, { useState } from 'react';
import { Cpu, ArrowUpCircle, Archive, Server, GitCompare, CheckSquare, Square } from 'lucide-react';
import { MLModel, api } from '../../../lib/api';
import ModelDetailsModal from '../ModelDetailsModal';
import ModelCompareModal from '../ModelCompareModal';

interface ModelsTabProps {
    models: MLModel[];
    setIsModelModalOpen: (val: boolean) => void;
    handleDeleteModel: (modelId: string) => Promise<boolean>;
    token?: string | null;
    onRefresh?: () => void;
    onDeployEndpoint?: (modelId: string) => void;
}

const getStageBadge = (stage?: string) => {
    const colors: Record<string, string> = {
        Staging: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
        Production: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
        Archived: 'bg-zinc-800 text-zinc-400 border-zinc-700',
    };
    return colors[stage || 'Staging'] || colors.Staging;
};

export default function ModelsTab({
    models,
    handleDeleteModel,
    token,
    onRefresh,
    onDeployEndpoint,
}: ModelsTabProps) {
    const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
    const [promotingId, setPromotingId] = useState<string | null>(null);
    const [compareIds, setCompareIds] = useState<string[]>([]);
    const [showCompare, setShowCompare] = useState(false);

    const toggleCompare = (modelId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setCompareIds((prev) => {
            if (prev.includes(modelId)) return prev.filter((id) => id !== modelId);
            if (prev.length >= 4) return prev;
            return [...prev, modelId];
        });
    };

    const handlePromote = async (model: MLModel, e?: React.MouseEvent) => {
        e?.stopPropagation();
        if (!token) return;
        setPromotingId(model.id);
        try {
            await api.promoteModel(model.id, 'Production');
            if (onRefresh) onRefresh();
        } catch (err) {
            console.error('Failed to promote model:', err);
        } finally {
            setPromotingId(null);
        }
    };

    const handleArchive = async (model: MLModel, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!token) return;
        setPromotingId(model.id);
        try {
            await api.archiveModel(model.id);
            if (onRefresh) onRefresh();
        } catch (err) {
            console.error('Failed to archive model:', err);
        } finally {
            setPromotingId(null);
        }
    };

    const compareModels = models.filter((m) => compareIds.includes(m.id));

    return (
        <div className="space-y-4">
            {/* Toolbar with Compare trigger */}
            {models.length > 0 && (
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <span className="text-xs text-zinc-500">
                        Select 2–4 models to compare metrics & hyperparameters side by side.
                    </span>
                    <div className="flex items-center gap-2">
                        {compareIds.length > 0 && (
                            <button
                                type="button"
                                onClick={() => setCompareIds([])}
                                className="h-8 px-2.5 text-xs text-zinc-400 hover:text-zinc-200"
                            >
                                Clear ({compareIds.length})
                            </button>
                        )}
                        <button
                            type="button"
                            disabled={compareIds.length < 2}
                            onClick={() => setShowCompare(true)}
                            className="h-8 px-3 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium disabled:opacity-40 flex items-center gap-1.5"
                        >
                            <GitCompare className="w-3.5 h-3.5" />
                            Compare Selected ({compareIds.length})
                        </button>
                    </div>
                </div>
            )}

            {models.length === 0 ? (
                <div className="border border-dashed border-zinc-800 bg-[#101217] rounded-lg p-14 text-center space-y-2">
                    <Cpu className="w-7 h-7 text-zinc-600 mx-auto" />
                    <p className="text-sm font-medium text-zinc-200">No models registered yet</p>
                    <p className="text-xs text-zinc-500">Train a tabular or NLP model to register it automatically in MLflow.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {models.map(model => {
                        const isChecked = compareIds.includes(model.id);
                        return (
                            <div
                                key={model.id}
                                onClick={() => setSelectedModel(model)}
                                className={`bg-[#101217] border p-4 rounded-lg flex items-start gap-3.5 cursor-pointer transition-colors ${
                                    isChecked ? 'border-orange-500/60 bg-zinc-900/60' : 'border-zinc-800 hover:border-zinc-700'
                                }`}
                            >
                                <button
                                    type="button"
                                    onClick={(e) => toggleCompare(model.id, e)}
                                    title="Select for comparison"
                                    className="mt-0.5 text-zinc-500 hover:text-zinc-200 shrink-0"
                                >
                                    {isChecked ? (
                                        <CheckSquare className="w-4 h-4 text-orange-400" />
                                    ) : (
                                        <Square className="w-4 h-4" />
                                    )}
                                </button>
                                <div className="flex-1 min-w-0">
                                    <div className="flex justify-between items-start gap-2">
                                        <h4 className="text-sm font-semibold text-zinc-100 truncate">{model.name}</h4>
                                        <span className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border shrink-0 ${getStageBadge(model.stage)}`}>
                                            {model.stage || 'Staging'}
                                        </span>
                                    </div>
                                    <p className="text-xs text-zinc-500 mt-1 line-clamp-1">
                                        {model.description || `${model.metrics_metadata?.framework || 'MLflow'} model`}
                                    </p>
                                    <div className="mt-3 flex flex-wrap items-center gap-1.5 font-mono text-[10px]">
                                        <span className="px-1.5 py-0.5 bg-zinc-900 border border-zinc-800 text-zinc-300 rounded">
                                            v{model.version || '1.0.0'}
                                        </span>
                                        <span className="px-1.5 py-0.5 bg-zinc-900 border border-zinc-800 text-zinc-400 rounded">
                                            run:{(model.mlflow_run_id || '').substring(0, 8)}
                                        </span>
                                        {model.metrics_metadata?.domain && (
                                            <span className="px-1.5 py-0.5 bg-zinc-900 border border-zinc-800 text-zinc-300 rounded uppercase">
                                                {model.metrics_metadata.domain}
                                            </span>
                                        )}
                                        {model.metrics_metadata?.algorithm && (
                                            <span className="px-1.5 py-0.5 bg-zinc-900 border border-zinc-800 text-orange-300 rounded">
                                                {model.metrics_metadata.algorithm}
                                            </span>
                                        )}
                                        {((model as any).preprocessing_pipeline_path ||
                                            model.metrics_metadata?.preprocessing_pipeline_path) && (
                                            <span className="px-1.5 py-0.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded">
                                                auto-preprocess
                                            </span>
                                        )}
                                    </div>
                                    <div className="mt-3 pt-2.5 border-t border-zinc-800/80 flex flex-wrap items-center gap-2">
                                        {onDeployEndpoint && (
                                            <button
                                                type="button"
                                                className="h-7 px-2.5 text-xs font-medium rounded bg-zinc-100 hover:bg-white text-zinc-950 inline-flex items-center gap-1.5"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onDeployEndpoint(model.id);
                                                }}
                                            >
                                                <Server className="w-3 h-3" />
                                                Deploy Endpoint
                                            </button>
                                        )}
                                        {model.stage !== 'Production' && model.stage !== 'Archived' && (
                                            <>
                                                <button
                                                    type="button"
                                                    className="h-7 px-2.5 text-xs rounded border border-zinc-700 bg-[#0b0c0f] hover:bg-zinc-800 text-zinc-200 inline-flex items-center gap-1.5"
                                                    onClick={(e) => handlePromote(model, e)}
                                                    disabled={promotingId === model.id}
                                                >
                                                    <ArrowUpCircle className="w-3.5 h-3.5 text-emerald-400" />
                                                    {promotingId === model.id ? 'Promoting...' : 'Promote'}
                                                </button>
                                                <button
                                                    type="button"
                                                    className="h-7 px-2.5 text-xs rounded border border-zinc-800 bg-[#0b0c0f] hover:bg-zinc-800 text-zinc-400 inline-flex items-center gap-1.5"
                                                    onClick={(e) => handleArchive(model, e)}
                                                    disabled={promotingId === model.id}
                                                >
                                                    <Archive className="w-3.5 h-3.5" />
                                                    Archive
                                                </button>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            <ModelDetailsModal
                model={selectedModel}
                onClose={() => setSelectedModel(null)}
                handleDeleteModel={handleDeleteModel}
                token={token}
                onPromote={onRefresh}
            />

            {showCompare && compareModels.length >= 2 && (
                <ModelCompareModal
                    models={compareModels}
                    onClose={() => setShowCompare(false)}
                    onPromote={async (modelId) => {
                        const target = models.find((m) => m.id === modelId);
                        if (target) await handlePromote(target);
                    }}
                    onDeployEndpoint={onDeployEndpoint}
                />
            )}
        </div>
    );
}
