import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Plus, Cpu, ArrowUpCircle, Archive } from 'lucide-react';
import { MLModel, api } from '../../../lib/api';
import ModelDetailsModal from '../ModelDetailsModal';

interface ModelsTabProps {
    models: MLModel[];
    setIsModelModalOpen: (val: boolean) => void;
    handleDeleteModel: (modelId: string) => Promise<boolean>;
    token?: string | null;
    onRefresh?: () => void;
}

const getStageBadge = (stage?: string) => {
    const colors: Record<string, string> = {
        Staging: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
        Production: 'bg-green-500/20 text-green-400 border-green-500/30',
        Archived: 'bg-neutral-500/20 text-neutral-400 border-neutral-500/30',
    };
    return colors[stage || 'Staging'] || colors.Staging;
};

export default function ModelsTab({ models, setIsModelModalOpen, handleDeleteModel, token, onRefresh }: ModelsTabProps) {
    const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
    const [promotingId, setPromotingId] = useState<string | null>(null);

    const handlePromote = async (model: MLModel, e: React.MouseEvent) => {
        e.stopPropagation();
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

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h3 className="text-xl font-medium">ML Models</h3>
            </div>
            {models.length === 0 ? (
                <div className="border-2 border-dashed border-neutral-700 rounded-xl p-12 text-center text-neutral-500">
                    No models registered yet.
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {models.map(model => (
                        <div
                            key={model.id}
                            onClick={() => setSelectedModel(model)}
                            className="bg-neutral-800/40 border border-neutral-700 p-5 rounded-xl flex items-start gap-4 hover:border-indigo-500/50 cursor-pointer transition-colors"
                        >
                            <div className="p-3 bg-purple-500/10 rounded-lg text-purple-400">
                                <Cpu className="w-6 h-6" />
                            </div>
                            <div className="flex-1">
                                <div className="flex justify-between items-start">
                                    <h4 className="font-medium text-white">{model.name}</h4>
                                    {model.is_public && <span className="text-[10px] px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full border border-green-500/30">PUBLIC</span>}
                                </div>
                                <p className="text-sm text-neutral-400 mt-1">{model.description || "PyTorch Model"}</p>
                                <div className="mt-3 flex flex-wrap gap-2">
                                    <span className={`text-[10px] px-2 py-0.5 rounded-full border ${getStageBadge(model.stage)}`}>
                                        {model.stage || 'Staging'}
                                    </span>
                                    <span className="text-[10px] px-2 py-0.5 bg-indigo-500/20 text-indigo-300 rounded-full">
                                        v{model.version || '1.0.0'}
                                    </span>
                                    <span className="text-[10px] px-2 py-0.5 bg-neutral-700 text-neutral-300 rounded-full">
                                        ID: {(model.mlflow_run_id || '').substring(0, 8)}...
                                    </span>
                                    {model.metrics_metadata?.architecture && (
                                        <span className="text-[10px] px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded-full">
                                            {model.metrics_metadata.architecture}
                                        </span>
                                    )}
                                </div>
                                {model.stage !== 'Production' && model.stage !== 'Archived' && (
                                    <div className="mt-3 flex gap-2">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            className="text-xs"
                                            onClick={(e) => handlePromote(model, e)}
                                            disabled={promotingId === model.id}
                                        >
                                            <ArrowUpCircle className="w-3 h-3 mr-1" />
                                            {promotingId === model.id ? 'Promoting...' : 'Promote'}
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            className="text-xs"
                                            onClick={(e) => handleArchive(model, e)}
                                            disabled={promotingId === model.id}
                                        >
                                            <Archive className="w-3 h-3 mr-1" />
                                            Archive
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <ModelDetailsModal
                model={selectedModel}
                onClose={() => setSelectedModel(null)}
                handleDeleteModel={handleDeleteModel}
                token={token}
                onPromote={onRefresh}
            />
        </div>
    );
}
