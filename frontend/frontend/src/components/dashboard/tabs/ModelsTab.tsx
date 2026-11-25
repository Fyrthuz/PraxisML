import React, { useMemo, useState } from 'react';
import { Cpu, ArrowUpCircle, Archive, Server, GitCompare, CheckSquare, Square, Lock } from 'lucide-react';
import { MLModel, api } from '../../../lib/api';
import { areModelsCompatible, getModelCompatibilityProfile } from '../../../lib/modelCompatibility';
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
    const [domainFilter, setDomainFilter] = useState<string>('all');

    const anchorModel = useMemo(
        () => (compareIds.length > 0 ? models.find((m) => m.id === compareIds[0]) || null : null),
        [compareIds, models]
    );
    const anchorProfile = useMemo(
        () => (anchorModel ? getModelCompatibilityProfile(anchorModel) : null),
        [anchorModel]
    );

    const availableDomains = useMemo(() => {
        const set = new Set<string>();
        models.forEach((m) => set.add(getModelCompatibilityProfile(m).domain));
        return ['all', ...Array.from(set).sort()];
    }, [models]);

    const visibleModels = useMemo(() => {
        if (domainFilter === 'all') return models;
        return models.filter((m) => getModelCompatibilityProfile(m).domain === domainFilter);
    }, [models, domainFilter]);

    const toggleCompare = (model: MLModel, e: React.MouseEvent) => {
        e.stopPropagation();
        if (anchorModel && !compareIds.includes(model.id) && !areModelsCompatible(anchorModel, model)) {
            return;
        }
        setCompareIds((prev) => {
            if (prev.includes(model.id)) return prev.filter((id) => id !== model.id);
            if (prev.length >= 4) return prev;
            return [...prev, model.id];
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
            {/* Toolbar with Domain Filter & Compare trigger */}
            {models.length > 0 && (
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                        <div className="flex items-center gap-1">
                            {availableDomains.map((dom) => (
                                <button
                                    key={dom}
                                    type="button"
                                    onClick={() => setDomainFilter(dom)}
                                    className={`h-7 px-2.5 rounded-md text-xs font-medium border transition-colors capitalize ${
                                        domainFilter === dom
                                            ? 'bg-zinc-100 text-zinc-950 border-zinc-100'
                                            : 'bg-[#101217] text-zinc-400 border-zinc-800 hover:text-zinc-200'
                                    }`}
                                >
                                    {dom === 'all' ? `All (${models.length})` : dom}
                                </button>
                            ))}
                        </div>
                        {anchorProfile ? (
                            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-orange-500/10 border border-orange-500/30 text-orange-300">
                                Comparing scope: {anchorProfile.scopeLabel}
                            </span>
                        ) : (
                            <span className="text-xs text-zinc-500">
                                Select 2–4 compatible models (same domain &amp; task) to compare side by side.
                            </span>
                        )}
                    </div>
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
                    {visibleModels.map(model => {
                        const isChecked = compareIds.includes(model.id);
                        const profile = getModelCompatibilityProfile(model);
                        const isIncompatible = Boolean(
                            anchorModel && !isChecked && !areModelsCompatible(anchorModel, model)
                        );
                        return (
                            <div
                                key={model.id}
                                onClick={() => setSelectedModel(model)}
                                className={`bg-[#101217] border p-4 rounded-lg flex items-start gap-3.5 cursor-pointer transition-colors ${
                                    isChecked
                                        ? 'border-orange-500/60 bg-zinc-900/60'
                                        : isIncompatible
                                        ? 'border-zinc-800/60 opacity-55 hover:opacity-80'
                                        : 'border-zinc-800 hover:border-zinc-700'
                                }`}
                            >
                                <button
                                    type="button"
                                    disabled={isIncompatible}
                                    onClick={(e) => toggleCompare(model, e)}
                                    title={
                                        isIncompatible && anchorProfile
                                            ? `Incompatible (${profile.scopeLabel}) — current comparison scope is ${anchorProfile.scopeLabel}`
                                            : 'Select for comparison'
                                    }
                                    className={`mt-0.5 shrink-0 ${
                                        isIncompatible
                                            ? 'text-zinc-700 cursor-not-allowed'
                                            : 'text-zinc-500 hover:text-zinc-200'
                                    }`}
                                >
                                    {isChecked ? (
                                        <CheckSquare className="w-4 h-4 text-orange-400" />
                                    ) : isIncompatible ? (
                                        <Lock className="w-4 h-4 text-zinc-600" />
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
