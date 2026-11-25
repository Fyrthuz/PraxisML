"use client";

import React, { useState, useEffect } from "react";
import { DatasetProfile, PreprocessingConfig, PreprocessingStep, api } from "@/lib/api";
import {
    Plus,
    Trash2,
    ArrowRight,
    Save,
    Eye,
    Sparkles,
    Edit3,
    Loader2,
    RotateCcw,
} from "lucide-react";
import toast from "react-hot-toast";

interface PipelineBuilderProps {
    datasetId: string;
    tenantId: string;
    profile: DatasetProfile | null;
    onApply: (config: PreprocessingConfig) => void;
}

const STEP_TYPES: { id: PreprocessingStep["type"]; label: string; hint: string }[] = [
    { id: "drop", label: "Drop", hint: "Remove unused or ID columns" },
    { id: "impute", label: "Impute", hint: "Fill missing values" },
    { id: "scale", label: "Scale", hint: "Normalize numeric ranges" },
    { id: "encode", label: "Encode", hint: "Convert categories to numbers" },
    { id: "feature_eng", label: "Feature Eng", hint: "Log, polynomial or binning" },
    { id: "rename", label: "Rename", hint: "Assign custom variable names" },
];

const STRATEGIES: Record<string, string[]> = {
    impute: ["median", "mean", "most_frequent", "constant"],
    scale: ["standard", "minmax", "robust"],
    encode: ["onehot", "ordinal"],
    feature_eng: ["log_transform", "polynomial", "binning"],
    drop: ["drop"],
    rename: ["rename"],
};

export default function PipelineBuilder({
    datasetId,
    tenantId,
    profile,
    onApply,
}: PipelineBuilderProps) {
    const [steps, setSteps] = useState<PreprocessingStep[]>([]);
    const [targetColumn, setTargetColumn] = useState<string>("");
    const [outputDatasetName, setOutputDatasetName] = useState<string>("");
    const [columnRenames, setColumnRenames] = useState<Record<string, string>>({});
    const [basePreviewCols, setBasePreviewCols] = useState<string[]>([]);
    const [preview, setPreview] = useState<any>(null);
    const [isPreviewLoad, setIsPreviewLoad] = useState(false);
    const [existingSteps, setExistingSteps] = useState<PreprocessingStep[] | null>(null);
    const [existingRenames, setExistingRenames] = useState<Record<string, string>>({});

    useEffect(() => {
        const fetchPipeline = async () => {
            try {
                const data = await api.getDatasetPipeline(datasetId, tenantId);
                if (data.steps && data.steps.length > 0) {
                    setExistingSteps(data.steps);
                    if (data.target_column) setTargetColumn(data.target_column);
                    if (data.column_renames) setExistingRenames(data.column_renames);
                } else {
                    setExistingSteps(null);
                    setExistingRenames({});
                }
            } catch (err) {
                console.error("Failed to fetch pipeline", err);
            }
        };
        if (datasetId) fetchPipeline();
    }, [datasetId, tenantId]);

    const allColumns = profile ? Object.keys(profile.profile) : [];
    const featureColumns = allColumns.filter((c) => c !== targetColumn);

    const addStep = (type: PreprocessingStep["type"]) => {
        const newStep: PreprocessingStep = {
            type,
            columns: [],
            method: type !== "impute" && type !== "drop" && type !== "rename" ? STRATEGIES[type][0] : undefined,
            strategy: type === "impute" ? STRATEGIES[type][0] : undefined,
            output_names: type === "rename" ? {} : undefined,
        };
        setSteps((prev) => [...prev, newStep]);
    };

    const removeStep = (index: number) => {
        setSteps((prev) => prev.filter((_, i) => i !== index));
    };

    const updateStep = (index: number, updates: Partial<PreprocessingStep>) => {
        setSteps((prev) => {
            const next = [...prev];
            next[index] = { ...next[index], ...updates };
            return next;
        });
    };

    const toggleStepColumn = (index: number, col: string) => {
        const current = steps[index].columns;
        const next = current.includes(col)
            ? current.filter((c) => c !== col)
            : [...current, col];
        updateStep(index, { columns: next });
    };

    const handleAutoSuggest = () => {
        if (!profile) return;
        const dropCols: string[] = [];
        const imputeNumCols: string[] = [];
        const scaleCols: string[] = [];
        const encodeCols: string[] = [];

        for (const col of featureColumns) {
            const stats = profile.profile[col];
            if (!stats) continue;
            const isIdLike = /id$/i.test(col) && stats.distinct_count >= profile.num_rows * 0.9;
            const isHighCardCat = stats.type !== "numeric" && stats.distinct_count > 20;
            const isMostlyNull = stats.null_pct >= 60;

            if (isIdLike || isHighCardCat || isMostlyNull) {
                dropCols.push(col);
            } else if (stats.type === "numeric") {
                if (stats.null_count > 0) imputeNumCols.push(col);
                scaleCols.push(col);
            } else {
                encodeCols.push(col);
            }
        }

        const suggested: PreprocessingStep[] = [];
        if (dropCols.length) suggested.push({ type: "drop", columns: dropCols });
        if (imputeNumCols.length) suggested.push({ type: "impute", strategy: "median", columns: imputeNumCols });
        if (scaleCols.length) suggested.push({ type: "scale", method: "standard", columns: scaleCols });
        if (encodeCols.length) suggested.push({ type: "encode", method: "onehot", columns: encodeCols });

        if (suggested.length === 0) {
            toast("Dataset already looks clean — add steps manually if needed.");
            return;
        }
        setSteps(suggested);
        setPreview(null);
        setColumnRenames({});
        toast.success(`Suggested ${suggested.length} steps from dataset profile`);
    };

    const buildCleanRenameMap = (): Record<string, string> => {
        const clean: Record<string, string> = {};
        for (const [orig, renamed] of Object.entries(columnRenames)) {
            const trimmed = renamed.trim();
            if (trimmed && trimmed !== orig) {
                clean[orig] = trimmed;
            }
        }
        return clean;
    };

    const handlePreview = async (includeRenames = true) => {
        if (steps.length === 0) {
            toast.error("Add at least one transformation step first.");
            return;
        }
        setIsPreviewLoad(true);
        try {
            const renames = includeRenames ? buildCleanRenameMap() : {};
            const config: PreprocessingConfig = {
                dataset_id: datasetId,
                target_column: targetColumn || undefined,
                steps,
                column_renames: Object.keys(renames).length ? renames : undefined,
            };
            const result = await api.previewPreprocessing(config, tenantId);
            setPreview(result);
            if (!includeRenames || basePreviewCols.length === 0) {
                setBasePreviewCols(result.transformed_columns || []);
            }
        } catch (err: any) {
            toast.error(err.message || "Preview failed. Check your step configuration.");
        } finally {
            setIsPreviewLoad(false);
        }
    };

    const handleApply = () => {
        if (steps.length === 0) return;
        const renames = buildCleanRenameMap();
        const config: PreprocessingConfig = {
            dataset_id: datasetId,
            target_column: targetColumn || undefined,
            steps,
            column_renames: Object.keys(renames).length ? renames : undefined,
            output_dataset_name: outputDatasetName.trim() || undefined,
        };
        onApply(config);
    };

    return (
        <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-6">
            {/* Existing Saved Pipeline Banner */}
            {existingSteps && (
                <div className="bg-[#0b0c0f] border border-zinc-800 rounded-lg p-4 space-y-3">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-zinc-200 flex items-center gap-2">
                            <ArrowRight className="w-3.5 h-3.5 text-orange-400" />
                            Saved Pipeline Steps
                        </span>
                        <span className="text-[11px] font-mono text-zinc-500">MLflow Artifact</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                        {existingSteps.map((step, i) => (
                            <div
                                key={i}
                                className="bg-[#101217] border border-zinc-800 rounded-md p-3"
                            >
                                <div className="flex justify-between items-center mb-1">
                                    <span className="text-[10px] font-mono text-zinc-500">
                                        #{i + 1}
                                    </span>
                                    <span className="px-1.5 py-0.5 bg-zinc-800 text-zinc-200 rounded text-[10px] font-mono uppercase">
                                        {step.type}
                                    </span>
                                </div>
                                <p className="text-xs font-medium text-zinc-200">
                                    {step.method || step.strategy || step.type}
                                </p>
                                <p className="text-[11px] font-mono text-zinc-500 mt-1 truncate">
                                    {step.columns.join(", ")}
                                </p>
                            </div>
                        ))}
                    </div>
                    <div className="flex flex-wrap items-center gap-4 text-xs text-zinc-400 pt-2 border-t border-zinc-800/80">
                        {targetColumn && (
                            <span>
                                Target: <strong className="font-mono text-emerald-400">{targetColumn}</strong>
                            </span>
                        )}
                        {Object.keys(existingRenames).length > 0 && (
                            <span>
                                Custom names:{" "}
                                <span className="font-mono text-zinc-300">
                                    {Object.entries(existingRenames)
                                        .map(([k, v]) => `${k} → ${v}`)
                                        .join(", ")}
                                </span>
                            </span>
                        )}
                    </div>
                </div>
            )}

            {/* Header & Global Controls */}
            <div className="flex flex-wrap items-end justify-between gap-4 pb-4 border-b border-zinc-800">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-100">
                        {existingSteps ? "Build New Pipeline Version" : "Pipeline Builder"}
                    </h3>
                    <p className="text-xs text-zinc-500 mt-0.5">
                        Steps on the same column are chained in order. Column names stay clean and can be renamed freely.
                    </p>
                </div>

                <div className="flex flex-wrap items-end gap-3">
                    <div>
                        <label className="block text-[11px] text-zinc-400 mb-1">Target Column</label>
                        <select
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            className="h-8 px-2.5 text-xs bg-[#0b0c0f] border border-zinc-800 text-zinc-100 rounded-md focus:outline-none focus:border-zinc-600"
                        >
                            <option value="">None (unsupervised)</option>
                            {allColumns.map((col) => (
                                <option key={col} value={col}>
                                    {col}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-[11px] text-zinc-400 mb-1">
                            Output Dataset Name (optional)
                        </label>
                        <input
                            value={outputDatasetName}
                            onChange={(e) => setOutputDatasetName(e.target.value)}
                            placeholder="e.g. titanic_clean"
                            className="h-8 px-2.5 text-xs bg-[#0b0c0f] border border-zinc-800 text-zinc-100 rounded-md focus:outline-none focus:border-zinc-600 w-44"
                        />
                    </div>
                    <button
                        type="button"
                        onClick={handleAutoSuggest}
                        className="h-8 px-3 rounded-md border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-200 flex items-center gap-1.5"
                    >
                        <Sparkles className="w-3.5 h-3.5 text-orange-400" />
                        Auto-suggest
                    </button>
                </div>
            </div>

            {/* Steps List */}
            <div className="space-y-3">
                {steps.length === 0 && (
                    <div className="rounded-lg border border-dashed border-zinc-800 bg-[#0b0c0f] py-8 text-center">
                        <p className="text-xs text-zinc-400">
                            No steps added yet. Click <strong>Auto-suggest</strong> or add a transformation below.
                        </p>
                    </div>
                )}

                {steps.map((step, idx) => (
                    <div
                        key={idx}
                        className="p-4 border border-zinc-800 rounded-lg bg-[#0b0c0f] space-y-3"
                    >
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <div className="flex items-center gap-2.5">
                                <span className="w-5 h-5 rounded bg-zinc-800 text-[11px] font-mono text-zinc-300 flex items-center justify-center">
                                    {idx + 1}
                                </span>
                                <span className="text-xs font-semibold uppercase tracking-wider text-zinc-100">
                                    {step.type}
                                </span>
                                {step.type !== "drop" && step.type !== "rename" && (
                                    <select
                                        value={step.type === "impute" ? step.strategy : step.method}
                                        onChange={(e) =>
                                            updateStep(
                                                idx,
                                                step.type === "impute"
                                                    ? { strategy: e.target.value }
                                                    : { method: e.target.value }
                                            )
                                        }
                                        className="h-7 px-2 text-xs font-mono bg-[#101217] border border-zinc-800 text-zinc-200 rounded"
                                    >
                                        {STRATEGIES[step.type].map((s) => (
                                            <option key={s} value={s}>
                                                {s}
                                            </option>
                                        ))}
                                    </select>
                                )}
                            </div>

                            <div className="flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={() =>
                                        updateStep(idx, {
                                            columns: featureColumns.filter(
                                                (c) => profile?.profile[c]?.type === "numeric"
                                            ),
                                        })
                                    }
                                    className="text-[11px] text-zinc-500 hover:text-zinc-200 px-2 py-0.5 rounded border border-zinc-800"
                                >
                                    All numeric
                                </button>
                                <button
                                    type="button"
                                    onClick={() =>
                                        updateStep(idx, {
                                            columns: featureColumns.filter(
                                                (c) => profile?.profile[c]?.type !== "numeric"
                                            ),
                                        })
                                    }
                                    className="text-[11px] text-zinc-500 hover:text-zinc-200 px-2 py-0.5 rounded border border-zinc-800"
                                >
                                    All categorical
                                </button>
                                <button
                                    type="button"
                                    onClick={() => removeStep(idx)}
                                    className="p-1 text-zinc-500 hover:text-red-400 rounded"
                                    title="Remove step"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        {/* Clickable column chips */}
                        <div className="flex flex-wrap gap-1.5">
                            {featureColumns.map((col) => {
                                const stats = profile?.profile[col];
                                const active = step.columns.includes(col);
                                return (
                                    <button
                                        key={col}
                                        type="button"
                                        onClick={() => toggleStepColumn(idx, col)}
                                        className={`px-2.5 py-1 rounded-md text-xs font-mono border transition-colors flex items-center gap-1.5 ${
                                            active
                                                ? "bg-zinc-100 text-zinc-950 border-zinc-100 font-medium"
                                                : "bg-[#101217] text-zinc-400 border-zinc-800 hover:border-zinc-700 hover:text-zinc-200"
                                        }`}
                                    >
                                        <span>{col}</span>
                                        <span
                                            className={`text-[10px] ${
                                                active ? "text-zinc-600" : "text-zinc-600"
                                            }`}
                                        >
                                            {stats?.type === "numeric" ? "num" : "cat"}
                                        </span>
                                        {stats && stats.null_count > 0 && (
                                            <span
                                                className={`text-[10px] ${
                                                    active ? "text-orange-700" : "text-amber-400"
                                                }`}
                                            >
                                                {stats.null_count}∅
                                            </span>
                                        )}
                                    </button>
                                );
                            })}
                        </div>

                        {/* Inline custom output names for rename or 1-to-1 steps */}
                        {step.columns.length > 0 &&
                            (step.type === "rename" ||
                                (step.type !== "drop" && step.method !== "onehot" && step.method !== "polynomial")) && (
                                <div className="pt-2 border-t border-zinc-800/70">
                                    <p className="text-[11px] text-zinc-500 mb-2 flex items-center gap-1.5">
                                        <Edit3 className="w-3 h-3" />
                                        Custom output variable names (optional — leave blank to keep original name)
                                    </p>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                                        {step.columns.map((col) => (
                                            <div
                                                key={col}
                                                className="flex items-center gap-1.5 bg-[#101217] border border-zinc-800 rounded px-2 py-1"
                                            >
                                                <span className="text-[11px] font-mono text-zinc-500 truncate max-w-[45%]">
                                                    {col}
                                                </span>
                                                <ArrowRight className="w-3 h-3 text-zinc-600 shrink-0" />
                                                <input
                                                    value={step.output_names?.[col] || ""}
                                                    onChange={(e) => {
                                                        const nextMap = {
                                                            ...(step.output_names || {}),
                                                            [col]: e.target.value,
                                                        };
                                                        if (!e.target.value.trim()) delete nextMap[col];
                                                        updateStep(idx, { output_names: nextMap });
                                                    }}
                                                    placeholder={col}
                                                    className="flex-1 min-w-0 bg-transparent text-xs font-mono text-zinc-100 focus:outline-none"
                                                />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                    </div>
                ))}
            </div>

            {/* Add Step Buttons */}
            <div className="flex flex-wrap gap-2 pt-3 border-t border-zinc-800">
                {STEP_TYPES.map((type) => (
                    <button
                        key={type.id}
                        type="button"
                        onClick={() => addStep(type.id)}
                        title={type.hint}
                        className="h-8 px-3 text-xs bg-[#0b0c0f] hover:bg-zinc-800 text-zinc-300 rounded-md border border-zinc-800 flex items-center gap-1.5 transition-colors"
                    >
                        <Plus className="w-3.5 h-3.5 text-orange-400" />
                        {type.label}
                    </button>
                ))}
            </div>

            {/* Action Bar */}
            {steps.length > 0 && (
                <div className="flex flex-wrap items-center justify-end gap-2.5 pt-2">
                    <button
                        type="button"
                        onClick={() => handlePreview(false)}
                        disabled={isPreviewLoad}
                        className="h-9 px-4 rounded-md border border-zinc-700 bg-[#0b0c0f] hover:bg-zinc-800 text-xs font-medium text-zinc-200 flex items-center gap-1.5 disabled:opacity-50"
                    >
                        {isPreviewLoad ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                            <Eye className="w-3.5 h-3.5" />
                        )}
                        Preview Output & Rename Variables
                    </button>
                    <button
                        type="button"
                        onClick={handleApply}
                        className="h-9 px-4 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-medium flex items-center gap-1.5"
                    >
                        <Save className="w-3.5 h-3.5" />
                        Apply & Save Dataset
                    </button>
                </div>
            )}

            {/* Output Column Renaming + Preview Table */}
            {preview && (
                <div className="pt-5 border-t border-zinc-800 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                            <h4 className="text-sm font-semibold text-zinc-100">
                                Output Variables & Preview
                            </h4>
                            <p className="text-xs text-zinc-500">
                                Shape:{" "}
                                <span className="font-mono text-zinc-300">
                                    {preview.original_shape[0]}×{preview.original_shape[1]} →{" "}
                                    {preview.transformed_shape[0]}×{preview.transformed_shape[1]}
                                </span>
                                . Edit any output variable name below before saving.
                            </p>
                        </div>
                        <div className="flex items-center gap-2">
                            {Object.keys(columnRenames).length > 0 && (
                                <button
                                    type="button"
                                    onClick={() => {
                                        setColumnRenames({});
                                        handlePreview(false);
                                    }}
                                    className="h-7 px-2.5 text-xs text-zinc-400 hover:text-zinc-100 border border-zinc-800 rounded flex items-center gap-1"
                                >
                                    <RotateCcw className="w-3 h-3" /> Reset names
                                </button>
                            )}
                            <button
                                type="button"
                                onClick={() => handlePreview(true)}
                                disabled={isPreviewLoad}
                                className="h-7 px-3 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded border border-zinc-700"
                            >
                                Refresh preview with names
                            </button>
                        </div>
                    </div>

                    {/* Editable rename grid for ALL output variables (including One-Hot outputs) */}
                    {basePreviewCols.length > 0 && (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 p-3 rounded-lg bg-[#0b0c0f] border border-zinc-800">
                            {basePreviewCols.map((origCol) => (
                                <div
                                    key={origCol}
                                    className="flex items-center gap-1.5 px-2 py-1.5 rounded bg-[#101217] border border-zinc-800"
                                >
                                    <span
                                        className="text-[11px] font-mono text-zinc-500 truncate max-w-[44%]"
                                        title={origCol}
                                    >
                                        {origCol}
                                    </span>
                                    <ArrowRight className="w-3 h-3 text-zinc-600 shrink-0" />
                                    <input
                                        value={columnRenames[origCol] ?? origCol}
                                        onChange={(e) =>
                                            setColumnRenames((prev) => ({
                                                ...prev,
                                                [origCol]: e.target.value,
                                            }))
                                        }
                                        className="flex-1 min-w-0 bg-transparent text-xs font-mono text-zinc-100 focus:outline-none focus:text-orange-300"
                                    />
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Preview Table */}
                    <div className="overflow-x-auto rounded-lg border border-zinc-800">
                        <table className="min-w-full divide-y divide-zinc-800 text-xs">
                            <thead className="bg-[#0b0c0f]">
                                <tr>
                                    {preview.transformed_columns.map((col: string) => (
                                        <th
                                            key={col}
                                            className="px-3 py-2 text-left font-mono text-[11px] text-zinc-400 whitespace-nowrap"
                                        >
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="bg-[#101217] divide-y divide-zinc-800/70 font-mono tabular-nums">
                                {preview.preview_rows.map((row: any, i: number) => (
                                    <tr key={i} className="hover:bg-zinc-800/30">
                                        {preview.transformed_columns.map((col: string) => (
                                            <td
                                                key={col}
                                                className="px-3 py-2 whitespace-nowrap text-zinc-300"
                                            >
                                                {typeof row[col] === "number"
                                                    ? Number(row[col]).toFixed(4).replace(/\.?0+$/, "")
                                                    : String(row[col] ?? "—")}
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
