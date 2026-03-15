import React, { useState, useRef, useMemo } from 'react';
import { UploadCloud, CheckCircle2, XCircle, Loader2, PlayCircle } from 'lucide-react';
import { useAuth } from '../AuthContext';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { MLModel } from '@/lib/api';
import { config } from '@/lib/config';

interface SingleImageUploadProps {
    models: MLModel[];
    onPredictionStarted?: () => void;
}

export default function SingleImageUpload({ models, onPredictionStarted }: SingleImageUploadProps) {
    const { token } = useAuth();
    const [file, setFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [selectedModel, setSelectedModel] = useState<string>(models.length > 0 ? models[0].id : '');
    const [uncertaintyMethod, setUncertaintyMethod] = useState('none');
    const [error, setError] = useState<string | null>(null);
    const [successMsg, setSuccessMsg] = useState<string | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Tabular features state
    const [featureValues, setFeatureValues] = useState<Record<string, string>>({});

    const selectedModelObj = useMemo(() => models.find(m => m.id === selectedModel), [models, selectedModel]);
    const isTabular = selectedModelObj?.metrics_metadata?.framework === 'sklearn';
    const featureNames = selectedModelObj?.metrics_metadata?.feature_names || [];

    const handleFeatureChange = (featureName: string, value: string) => {
        setFeatureValues(prev => ({
            ...prev,
            [featureName]: value
        }));
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => setIsDragging(false);

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const uploadedFile = e.dataTransfer.files[0];
            setFile(uploadedFile);
            setError(null);
            setSuccessMsg(null);
            if (uploadedFile.type.startsWith('image/')) {
                setImagePreview(URL.createObjectURL(uploadedFile));
            } else {
                setImagePreview(null);
            }
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            const uploadedFile = e.target.files[0];
            setFile(uploadedFile);
            setError(null);
            setSuccessMsg(null);

            if (uploadedFile.type.startsWith('image/')) {
                setImagePreview(URL.createObjectURL(uploadedFile));
            } else {
                setImagePreview(null);
            }
        }
    };

    const handleSubmit = async () => {
        if (!selectedModel || !token) return;
        if (!isTabular && !file) return;

        setIsUploading(true);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('model_id', selectedModel);
            formData.append('uncertainty_method', uncertaintyMethod);

            if (isTabular) {
                formData.append('features', JSON.stringify(featureValues));
            } else if (file) {
                formData.append('file', file);
            }

            const res = await fetch(config.getFullApiUrl('/predictions/predict/single'), {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: formData,
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || 'Prediction request failed');
            }

            setSuccessMsg('Inference started successfully!');
            setFile(null);
            setImagePreview(null);
            setFeatureValues({});
            if (onPredictionStarted) onPredictionStarted();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="bg-neutral-900 border border-neutral-800 rounded-3xl p-6 lg:p-8">
            <h3 className="text-xl font-bold mb-2">Single File Inference</h3>
            <p className="text-sm text-neutral-400 mb-6">Drop a single image or tabular file to immediately run an inference task bypassing datasets.</p>

            {error && <div className="mb-6 p-4 bg-red-500/10 text-red-500 rounded-xl text-sm flex items-center"><XCircle className="w-4 h-4 mr-2" /> {error}</div>}
            {successMsg && <div className="mb-6 p-4 bg-green-500/10 text-green-500 rounded-xl text-sm flex items-center"><CheckCircle2 className="w-4 h-4 mr-2" /> {successMsg}</div>}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Left Side: Configuration */}
                <div className="space-y-6">
                    <div className="space-y-2">
                        <label className="text-xs font-bold text-neutral-500 uppercase ml-1">Target Model</label>
                        <select
                            className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            value={selectedModel}
                            onChange={(e) => {
                                setSelectedModel(e.target.value);
                                setUncertaintyMethod('none');
                            }}
                        >
                            <option value="" disabled>Select a model</option>
                            {models.map(m => (
                                <option key={m.id} value={m.id}>{m.name} {m.is_public ? '(Public)' : ''}</option>
                            ))}
                        </select>
                    </div>

                    <div className="space-y-2">
                        <label className="text-xs font-bold text-neutral-500 uppercase ml-1">Uncertainty Method</label>
                        <select
                            className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                            value={uncertaintyMethod}
                            onChange={(e) => setUncertaintyMethod(e.target.value)}
                        >
                            <option value="none">None (Standard Inference)</option>
                            {(() => {
                                const selectedModelObj = models.find(m => m.id === selectedModel);
                                const framework = selectedModelObj?.metrics_metadata?.framework || 'pytorch';
                                if (framework === 'sklearn') {
                                    return (
                                        <>
                                            <option value="entropy">Predictive Entropy</option>
                                            <option value="tree_variance">Tree Variance (Ensembles)</option>
                                            <option value="conformal">Conformal Prediction</option>
                                        </>
                                    );
                                } else {
                                    return (
                                        <>
                                            <option value="mc_dropout">MC Dropout</option>
                                            <option value="tta">Test Time Augmentation</option>
                                            <option value="noisy_inference">Noisy Data Inference</option>
                                            <option value="ensemble">Ensemble Average</option>
                                        </>
                                    );
                                }
                            })()}
                        </select>
                    </div>

                    <Button
                        className="w-full h-14 bg-indigo-600 hover:bg-indigo-700 rounded-2xl font-bold mt-4"
                        disabled={(!isTabular && !file) || !selectedModel || isUploading}
                        onClick={handleSubmit}
                    >
                        {isUploading ? (
                            <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Uploading...</>
                        ) : (
                            <><PlayCircle className="w-5 h-5 mr-2" /> Start Inference</>
                        )}
                    </Button>
                </div>

                {/* Right Side: Dropzone or Form */}
                <div>
                    {isTabular ? (
                        <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 h-full max-h-[350px] overflow-y-auto custom-scrollbar">
                            <h4 className="text-sm font-bold text-neutral-300 mb-4 px-1">Feature Inputs</h4>
                            {featureNames.length > 0 ? (
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                    {featureNames.map((featureName: string) => (
                                        <div key={featureName} className="space-y-1">
                                            <label className="text-xs font-medium text-neutral-400 ml-1">{featureName}</label>
                                            <input
                                                type="text"
                                                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 text-white placeholder-neutral-600"
                                                placeholder="0.0"
                                                value={featureValues[featureName] || ''}
                                                onChange={(e) => handleFeatureChange(featureName, e.target.value)}
                                            />
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-sm text-neutral-500 italic p-4 text-center">No features found in model metadata.</div>
                            )}
                        </div>
                    ) : (
                        <div
                            className={cn(
                                "h-full min-h-[250px] relative rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-8 text-center transition-all duration-200 cursor-pointer overflow-hidden",
                                isDragging ? "border-indigo-500 bg-indigo-500/10" : "border-neutral-700 bg-neutral-900/50 hover:bg-neutral-800",
                                file ? "border-indigo-500/50 bg-indigo-500/5" : ""
                            )}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleFileChange}
                                accept="image/*,.npy,.nii,.nii.gz,.csv,.xlsx,.parquet"
                                className="hidden"
                            />

                            {file ? (
                                <div className="space-y-4">
                                    {imagePreview ? (
                                        <img src={imagePreview} alt="Preview" className="mx-auto max-h-32 object-contain rounded-lg shadow-md" />
                                    ) : (
                                        <div className="mx-auto w-16 h-16 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
                                            <CheckCircle2 className="w-8 h-8" />
                                        </div>
                                    )}
                                    <div>
                                        <p className="font-bold text-white truncate max-w-[200px]">{file.name}</p>
                                        <p className="text-xs text-neutral-400 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    </div>
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-neutral-500 hover:text-white"
                                        onClick={(e) => { e.stopPropagation(); setFile(null); setImagePreview(null); }}
                                    >
                                        Clear Selection
                                    </Button>
                                </div>
                            ) : (
                                <div className="space-y-4 pointer-events-none">
                                    <div className="mx-auto w-16 h-16 rounded-full bg-neutral-800 text-neutral-400 flex items-center justify-center">
                                        <UploadCloud className="w-8 h-8" />
                                    </div>
                                    <div>
                                        <p className="font-medium text-white">Click or Drop File Here</p>
                                        <p className="text-xs text-neutral-500 mt-2 max-w-[200px]">Supports PNG, JPG, NIfTI, Numpy (.npy) slices, CSV, Excel, and Parquet.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
