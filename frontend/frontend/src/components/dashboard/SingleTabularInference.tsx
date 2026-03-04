import React, { useState, useMemo } from 'react';
import { CheckCircle2, XCircle, Loader2, PlayCircle, Settings2 } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { Button } from '@/components/ui/button';
import { MLModel } from '@/lib/api';

interface SingleTabularInferenceProps {
    models: MLModel[];
    onPredictionStarted?: () => void;
}

export default function SingleTabularInference({ models, onPredictionStarted }: SingleTabularInferenceProps) {
    const { token } = useAuth();
    const [isUploading, setIsUploading] = useState(false);
    const [selectedModel, setSelectedModel] = useState<string>(models.length > 0 ? models[0].id : '');
    const [uncertaintyMethod, setUncertaintyMethod] = useState('none');
    const [error, setError] = useState<string | null>(null);
    const [successMsg, setSuccessMsg] = useState<string | null>(null);

    // Tabular features state
    const [featureValues, setFeatureValues] = useState<Record<string, string>>({});

    const selectedModelObj = useMemo(() => models.find(m => m.id === selectedModel), [models, selectedModel]);
    const featureNames = selectedModelObj?.metrics_metadata?.feature_names || [];

    const handleFeatureChange = (featureName: string, value: string) => {
        setFeatureValues(prev => ({
            ...prev,
            [featureName]: value
        }));
    };

    const handleSubmit = async () => {
        if (!selectedModel || !token) return;

        setIsUploading(true);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('model_id', selectedModel);
            formData.append('uncertainty_method', uncertaintyMethod);
            formData.append('features', JSON.stringify(featureValues));

            const res = await fetch('http://localhost:8000/api/v1/predictions/predict/single', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: formData,
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || 'Prediction request failed');
            }

            setSuccessMsg('Single sample inference started successfully!');
            setFeatureValues({});
            if (onPredictionStarted) onPredictionStarted();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-6 lg:p-10 shadow-2xl">
            <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-indigo-500/20 rounded-2xl text-indigo-400">
                    <Settings2 className="w-6 h-6" />
                </div>
                <div>
                    <h3 className="text-2xl font-bold text-white">Single Sample Tabular Inference</h3>
                    <p className="text-sm text-neutral-400">Enter feature values manually to run a quick prediction.</p>
                </div>
            </div>

            {error && (
                <div className="mb-6 p-4 bg-red-500/10 text-red-500 border border-red-500/20 rounded-xl text-sm flex items-center animate-in fade-in slide-in-from-top-2">
                    <XCircle className="w-4 h-4 mr-2" /> {error}
                </div>
            )}
            {successMsg && (
                <div className="mb-6 p-4 bg-green-500/10 text-green-500 border border-green-500/20 rounded-xl text-sm flex items-center animate-in fade-in slide-in-from-top-2">
                    <CheckCircle2 className="w-4 h-4 mr-2" /> {successMsg}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
                {/* Left Side: Model Selector */}
                <div className="space-y-6">
                    <div className="space-y-2">
                        <label className="text-xs font-bold text-neutral-500 uppercase tracking-widest ml-1">Configuration</label>
                        <div className="p-1 bg-black/20 rounded-2xl border border-neutral-800">
                            <select
                                className="w-full bg-transparent border-none rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-0 text-white cursor-pointer"
                                value={selectedModel}
                                onChange={(e) => {
                                    setSelectedModel(e.target.value);
                                    setUncertaintyMethod('none');
                                }}
                            >
                                <option value="" disabled>Select a model</option>
                                {models.map(m => (
                                    <option key={m.id} value={m.id} className="bg-neutral-900">{m.name}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-xs font-bold text-neutral-500 uppercase tracking-widest ml-1">Uncertainty Method</label>
                        <div className="p-1 bg-black/20 rounded-2xl border border-neutral-800">
                            <select
                                className="w-full bg-transparent border-none rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-0 text-white cursor-pointer"
                                value={uncertaintyMethod}
                                onChange={(e) => setUncertaintyMethod(e.target.value)}
                            >
                                <option value="none" className="bg-neutral-900">None (Standard)</option>
                                <option value="entropy" className="bg-neutral-900">Predictive Entropy</option>
                                <option value="tree_variance" className="bg-neutral-900">Tree Variance (Ensembles)</option>
                                <option value="conformal" className="bg-neutral-900">Conformal Prediction</option>
                            </select>
                        </div>
                    </div>

                    <Button
                        className="w-full h-14 bg-indigo-600 hover:bg-indigo-700 rounded-2xl font-bold mt-4 shadow-lg shadow-indigo-600/20 transition-all hover:scale-[1.02] active:scale-[0.98]"
                        disabled={!selectedModel || isUploading}
                        onClick={handleSubmit}
                    >
                        {isUploading ? (
                            <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Processing...</>
                        ) : (
                            <><PlayCircle className="w-5 h-5 mr-2" /> Run Prediction</>
                        )}
                    </Button>
                </div>

                {/* Right Side: Feature Form */}
                <div className="lg:col-span-2">
                    <div className="bg-black/30 border border-neutral-800/50 rounded-2xl p-6 h-full max-h-[500px] overflow-y-auto custom-scrollbar shadow-inner">
                        <div className="flex items-center justify-between mb-6 border-b border-neutral-800 pb-4">
                            <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-widest">Input Features</h4>
                            <span className="px-3 py-1 bg-indigo-500/10 text-indigo-400 rounded-full text-[10px] font-bold uppercase tracking-tighter">
                                {featureNames.length} Total
                            </span>
                        </div>

                        {featureNames.length > 0 ? (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
                                {featureNames.map((featureName: string) => (
                                    <div key={featureName} className="group flex flex-col space-y-1.5">
                                        <label className="text-xs font-semibold text-neutral-500 transition-colors group-focus-within:text-indigo-400 ml-1">
                                            {featureName}
                                        </label>
                                        <input
                                            type="text"
                                            className="w-full bg-neutral-950/50 border border-neutral-800 hover:border-neutral-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/30 text-white placeholder-neutral-700 transition-all"
                                            placeholder="Enter value..."
                                            value={featureValues[featureName] || ''}
                                            onChange={(e) => handleFeatureChange(featureName, e.target.value)}
                                        />
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-48 text-neutral-500 space-y-2">
                                <Settings2 className="w-8 h-8 opacity-20" />
                                <p className="text-sm italic">Select a model to view its required features.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
