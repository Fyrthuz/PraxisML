"use client";

import React, { useState, useEffect } from "react";
import { DatasetProfile, PreprocessingConfig, PreprocessingStep, api } from "@/lib/api";
import { PlusCircle, Trash2, ArrowRightCircle, Save } from "lucide-react";

interface PipelineBuilderProps {
    datasetId: string;
    tenantId: string;
    profile: DatasetProfile | null;
    onApply: (config: PreprocessingConfig) => void;
}

export default function PipelineBuilder({ datasetId, tenantId, profile, onApply }: PipelineBuilderProps) {
    const [steps, setSteps] = useState<PreprocessingStep[]>([]);
    const [targetColumn, setTargetColumn] = useState<string>("");
    const [preview, setPreview] = useState<any>(null);
    const [isPreviewLoad, setIsPreviewLoad] = useState(false);
    const [existingSteps, setExistingSteps] = useState<PreprocessingStep[] | null>(null);
    const [isExistingLoad, setIsExistingLoad] = useState(false);

    useEffect(() => {
        const fetchPipeline = async () => {
            setIsExistingLoad(true);
            try {
                const data = await api.getDatasetPipeline(datasetId, tenantId);
                if (data.steps && data.steps.length > 0) {
                    setExistingSteps(data.steps);
                    if (data.target_column) setTargetColumn(data.target_column);
                } else {
                    setExistingSteps(null);
                }
            } catch (err) {
                console.error("Failed to fetch pipeline", err);
            } finally {
                setIsExistingLoad(false);
            }
        };
        if (datasetId) fetchPipeline();
    }, [datasetId, tenantId]);

    const stepTypes = [
        { id: "impute", label: "Imputation" },
        { id: "scale", label: "Scaling" },
        { id: "encode", label: "Encoding" },
        { id: "feature_eng", label: "Feature Engineering" },
        { id: "drop", label: "Drop Columns" },
    ];

    const strategies: Record<string, string[]> = {
        impute: ["mean", "median", "most_frequent", "constant"],
        scale: ["standard", "minmax", "robust"],
        encode: ["onehot", "ordinal"],
        feature_eng: ["log_transform", "polynomial", "binning"],
        drop: ["drop"],
    };

    const addStep = (type: any) => {
        const newStep: PreprocessingStep = {
            type,
            columns: [],
            method: type !== 'impute' ? strategies[type][0] : undefined,
            strategy: type === 'impute' ? strategies[type][0] : undefined
        };
        setSteps([...steps, newStep]);
    };

    const removeStep = (index: number) => {
        setSteps(steps.filter((_, i) => i !== index));
    };

    const updateStep = (index: number, updates: Partial<PreprocessingStep>) => {
        const newSteps = [...steps];
        newSteps[index] = { ...newSteps[index], ...updates };
        setSteps(newSteps);
    };

    const handlePreview = async () => {
        if (steps.length === 0) return;
        setIsPreviewLoad(true);
        try {
            const config: PreprocessingConfig = {
                dataset_id: datasetId,
                target_column: targetColumn || undefined,
                steps
            };
            const result = await api.previewPreprocessing(config, tenantId);
            setPreview(result);
        } catch (err) {
            alert("Preview failed. Check steps configuration.");
            console.error(err);
        } finally {
            setIsPreviewLoad(false);
        }
    };

    const handleApply = () => {
        if (steps.length === 0) return;
        const config: PreprocessingConfig = {
            dataset_id: datasetId,
            target_column: targetColumn || undefined,
            steps
        };
        onApply(config);
    };

    const allColumns = profile ? Object.keys(profile.profile) : [];

    return (
        <div className="bg-slate-950 border border-slate-800/50 rounded-xl shadow-sm p-6 space-y-6">
            {/* Existing Pipeline Section */}
            {existingSteps && (
                <div className="bg-blue-500/5 border border-blue-500/20 rounded-xl p-6 mb-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <div className="p-1.5 bg-blue-500/20 rounded-lg">
                                <ArrowRightCircle className="w-4 h-4 text-blue-400" />
                            </div>
                            <h4 className="text-sm font-bold text-blue-400 uppercase tracking-widest">Active Pipeline Configuration</h4>
                        </div>
                        <span className="text-[10px] text-blue-500/60 font-mono">Retrieved from MLFlow Artifacts</span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                        {existingSteps.map((step, i) => (
                            <div key={i} className="bg-slate-900/50 border border-slate-800 rounded-lg p-3">
                                <div className="flex justify-between items-center mb-1">
                                    <span className="text-[10px] font-bold text-slate-500 uppercase">Step {i + 1}</span>
                                    <span className="px-1.5 py-0.5 bg-blue-500/10 text-blue-400 rounded text-[9px] font-bold uppercase">{step.type}</span>
                                </div>
                                <p className="text-xs text-slate-300 font-medium">{step.method || step.strategy}</p>
                                <p className="text-[9px] text-slate-500 mt-1 truncate">
                                    {step.columns.length} columns: {step.columns.join(", ")}
                                </p>
                            </div>
                        ))}
                    </div>
                    {targetColumn && (
                        <div className="mt-4 pt-3 border-t border-blue-500/10 flex items-center gap-2 text-xs">
                            <span className="text-slate-500">Target Column:</span>
                            <span className="text-emerald-400 font-bold">{targetColumn}</span>
                        </div>
                    )}
                </div>
            )}

            <div className="flex justify-between items-center">
                <h3 className="text-xl font-semibold text-slate-100">
                    {existingSteps ? "Create New Version Pipeline" : "Pipeline Builder"}
                </h3>
                <div className="flex space-x-2">
                    <div className="flex items-center space-x-2">
                        <label className="text-sm text-slate-300">Target Column:</label>
                        <select
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            className="text-sm border-slate-700 bg-slate-900 text-slate-200 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        >
                            <option value="">-- None --</option>
                            {allColumns.map(col => (
                                <option key={col} value={col}>{col}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            <div className="space-y-4">
                {steps.map((step, idx) => (
                    <div key={idx} className="flex flex-col md:flex-row gap-4 p-4 border border-slate-800/80 rounded-lg bg-slate-900/50 items-start md:items-center hover:border-slate-700/80 transition-all">
                        <div className="w-full md:w-32 font-medium text-slate-300 uppercase text-sm tracking-wider">
                            {idx + 1}. {step.type}
                        </div>

                        <div className="flex-1 w-full space-y-3 md:space-y-0 md:flex md:space-x-4">
                            {/* Strategy / Method Selector */}
                            <div className="w-full md:w-48">
                                <label className="block text-xs font-medium text-slate-400 mb-1">Method Strategy</label>
                                <select
                                    value={step.type === 'impute' ? step.strategy : step.method}
                                    onChange={(e) => updateStep(idx, step.type === 'impute' ? { strategy: e.target.value } : { method: e.target.value })}
                                    disabled={step.type === 'drop'}
                                    className="w-full text-sm border-slate-700 bg-slate-900 text-slate-200 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50"
                                >
                                    {strategies[step.type].map(s => <option key={s} value={s}>{s === 'drop' ? 'Remove Column(s)' : s}</option>)}
                                </select>
                            </div>

                            {/* Columns Selector (Multi) */}
                            <div className="w-full flex-1">
                                <label className="block text-xs font-medium text-slate-400 mb-1">Target Columns</label>
                                <select
                                    multiple
                                    value={step.columns}
                                    onChange={(e) => {
                                        const values = Array.from(e.target.selectedOptions, option => option.value);
                                        updateStep(idx, { columns: values });
                                    }}
                                    className="w-full text-sm border-slate-700 bg-slate-900 text-slate-200 rounded-md shadow-sm focus:border-blue-500 focus:ring-blue-500 h-20"
                                >
                                    {allColumns.filter(c => c !== targetColumn).map(col => (
                                        <option key={col} value={col}>
                                            {col} {profile?.profile[col].type === 'numeric' ? '(N)' : '(C)'}
                                        </option>
                                    ))}
                                </select>
                                <p className="text-xs text-slate-500 mt-1">Hold Ctrl/Cmd to select multiple</p>
                            </div>
                        </div>

                        <button onClick={() => removeStep(idx)} className="p-2 text-red-500 hover:bg-red-950/30 rounded-full transition-colors self-end md:self-center">
                            <Trash2 className="w-5 h-5" />
                        </button>
                    </div>
                ))}
            </div>

            <div className="flex gap-2 pt-2 border-t border-slate-800/60 flex-wrap">
                {stepTypes.map(type => (
                    <button
                        key={type.id}
                        onClick={() => addStep(type.id)}
                        className="flex items-center px-3 py-1.5 text-sm bg-slate-800/50 hover:bg-slate-800 text-slate-300 rounded-lg transition-colors border border-slate-700"
                    >
                        <PlusCircle className="w-4 h-4 mr-2 text-blue-500" />
                        Add {type.label}
                    </button>
                ))}
            </div>

            {(steps.length > 0) && (
                <div className="flex justify-end pt-6 space-x-3">
                    <button
                        onClick={handlePreview}
                        disabled={isPreviewLoad || steps.length === 0}
                        className="flex items-center px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg shadow-sm text-sm font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-50"
                    >
                        {isPreviewLoad ? "Previewing..." : "Preview Row Data"}
                    </button>
                    <button
                        onClick={handleApply}
                        disabled={steps.length === 0}
                        className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg shadow-sm text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
                    >
                        <Save className="w-4 h-4 mr-2" />
                        Apply & Save Dataset
                    </button>
                </div>
            )}

            {/* Preview Section */}
            {preview && (
                <div className="mt-8 border-t border-slate-800/60 pt-6">
                    <h4 className="text-lg font-medium text-slate-100 mb-4 flex items-center">
                        <ArrowRightCircle className="inline w-5 h-5 mr-2 text-blue-500" />
                        Transformation Preview
                        <span className="text-sm font-normal text-slate-400 ml-4">
                            ({preview.original_shape[1]} ➔ {preview.transformed_shape[1]} columns)
                        </span>
                    </h4>
                    <div className="overflow-x-auto rounded-lg border border-slate-800/60 shadow-sm">
                        <table className="min-w-full divide-y divide-slate-800/60">
                            <thead className="bg-slate-900">
                                <tr>
                                    {preview.transformed_columns.map((col: string) => (
                                        <th key={col} className="px-3 py-2 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="bg-slate-950 divide-y divide-slate-800/60">
                                {preview.preview_rows.map((row: any, i: number) => (
                                    <tr key={i} className="hover:bg-slate-900/50">
                                        {preview.transformed_columns.map((col: string) => (
                                            <td key={col} className="px-3 py-2 whitespace-nowrap text-sm text-slate-300">
                                                {typeof row[col] === 'number' ? row[col].toFixed(4).replace(/\.0000$/, '') : String(row[col])}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
