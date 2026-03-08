"use client";
import React, { useState, useEffect } from "react";
import { Loader2, Table2, X, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dataset, MLModel, Prediction } from "@/lib/api";

interface PredictionResultsModalProps {
    prediction: Prediction;
    datasets: Dataset[];
    models: MLModel[];
    token: string | null;
    onClose: () => void;
}

export default function PredictionResultsModal({
    prediction,
    datasets,
    models,
    token,
    onClose,
}: PredictionResultsModalProps) {
    const [data, setData] = useState<{
        prediction?: any;
        uncertainty?: any;
        input_data?: any;
    } | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!token) return;
        setLoading(true);
        setError(null);
        fetch(`http://localhost:8000/api/v1/predictions/${prediction.id}/data`, {
            headers: { Authorization: `Bearer ${token}` },
        })
            .then((res) => {
                if (!res.ok)
                    throw new Error(`Server returned ${res.status}: ${res.statusText}`);
                return res.json();
            })
            .then((json) => {
                setData(json);
                setLoading(false);
            })
            .catch((err) => {
                console.error("Failed to fetch prediction data:", err);
                setError(err.message || "Failed to load prediction results");
                setLoading(false);
            });
    }, [prediction.id, token]);

    const renderTable = () => {
        if (!data) return null;

        const normalizeArray = (arr: any, primaryLength?: number) => {
            if (arr === undefined || arr === null) return [];
            if (!Array.isArray(arr)) return [[arr]];
            if (arr.length === 0) return [];
            if (!Array.isArray(arr[0])) {
                if (primaryLength === 1 && arr.length > 1) return [arr];
                return arr.map((item: any) => [item]);
            }
            return arr;
        };

        const rawPreds = data.prediction;
        const normalizedPreds = normalizeArray(rawPreds);
        const primaryLen = normalizedPreds.length;

        const inputs = normalizeArray(data.input_data, primaryLen);
        const uncs = normalizeArray(data.uncertainty, primaryLen);

        const numRows = Math.max(primaryLen, inputs.length, uncs.length);

        const associatedDataset = datasets.find((d) => d.id === prediction.dataset_id);
        const associatedModel = models.find((m) => m.id === prediction.model_id);

        let featureNames: string[] = [];
        if (associatedDataset?.column_names) {
            featureNames = associatedDataset.column_names;
        } else if (associatedModel?.metrics_metadata?.feature_names) {
            featureNames = associatedModel.metrics_metadata.feature_names;
        }

        const numFeatures = inputs[0]?.length || 0;

        const featureHeaders: string[] = [];
        for (let j = 0; j < numFeatures; j++) {
            featureHeaders.push(featureNames[j] || `Feature ${j + 1}`);
        }

        if (numRows === 0) {
            return (
                <div className="flex flex-col items-center justify-center py-20 bg-neutral-950/30 rounded-3xl border border-dashed border-neutral-800">
                    <Table2 className="w-12 h-12 text-neutral-700 mb-4" />
                    <p className="text-neutral-500 font-medium">
                        No results data found for this prediction.
                    </p>
                </div>
            );
        }

        return (
            <div className="relative group/table">
                <div className="overflow-x-auto bg-neutral-900 border border-neutral-800 rounded-2xl shadow-2xl max-h-[500px] scrollbar-thin scrollbar-thumb-neutral-700 scrollbar-track-transparent">
                    <table className="w-full text-left border-collapse text-sm">
                        <thead className="sticky top-0 bg-neutral-950/95 backdrop-blur-md z-20">
                            <tr>
                                <th className="px-4 py-3 text-neutral-500 font-bold uppercase tracking-wider text-[10px] border-b border-neutral-800 w-12 text-center">
                                    #
                                </th>
                                {featureHeaders.map((name, idx) => (
                                    <th
                                        key={idx}
                                        className="px-4 py-3 text-neutral-300 font-bold uppercase tracking-wider text-[10px] border-b border-neutral-800 whitespace-nowrap min-w-[100px]"
                                    >
                                        {name}
                                    </th>
                                ))}
                                <th className="px-4 py-3 text-emerald-400 font-bold uppercase tracking-wider text-[10px] border-b border-emerald-500/20 bg-emerald-500/5 whitespace-nowrap">
                                    Prediction
                                </th>
                                <th className="px-4 py-3 text-amber-400 font-bold uppercase tracking-wider text-[10px] border-b border-amber-500/20 bg-amber-500/5 whitespace-nowrap">
                                    Uncertainty
                                </th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-800/50">
                            {Array.from({ length: numRows }).map((_, i) => (
                                <tr
                                    key={i}
                                    className="hover:bg-white/[0.02] transition-colors group/row"
                                >
                                    <td className="px-4 py-3 text-neutral-600 font-mono text-center text-xs">
                                        {i + 1}
                                    </td>
                                    {Array.from({ length: numFeatures }).map((_, j) => (
                                        <td
                                            key={j}
                                            className="px-4 py-3 font-mono text-xs text-neutral-400 border-r border-neutral-800/30"
                                        >
                                            {inputs[i] && inputs[i][j] !== undefined
                                                ? typeof inputs[i][j] === "number"
                                                    ? inputs[i][j].toFixed(4)
                                                    : String(inputs[i][j])
                                                : "—"}
                                        </td>
                                    ))}
                                    <td className="px-4 py-3 text-emerald-400 font-mono font-bold bg-emerald-400/5 border-l border-emerald-500/10">
                                        {normalizedPreds[i]
                                            ? Array.isArray(normalizedPreds[i]) &&
                                                normalizedPreds[i].length === 1
                                                ? typeof normalizedPreds[i][0] === "number"
                                                    ? normalizedPreds[i][0].toFixed(6)
                                                    : JSON.stringify(normalizedPreds[i][0])
                                                : typeof normalizedPreds[i] === "number"
                                                    ? normalizedPreds[i].toFixed(6)
                                                    : JSON.stringify(normalizedPreds[i])
                                            : "—"}
                                    </td>
                                    <td className="px-4 py-3 text-amber-500 font-mono font-bold bg-amber-500/5 border-l border-amber-500/10">
                                        {uncs[i]
                                            ? Array.isArray(uncs[i]) && uncs[i].length === 1
                                                ? typeof uncs[i][0] === "number"
                                                    ? uncs[i][0].toFixed(6)
                                                    : JSON.stringify(uncs[i][0])
                                                : typeof uncs[i] === "number"
                                                    ? uncs[i].toFixed(6)
                                                    : JSON.stringify(uncs[i])
                                            : "—"}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div className="mt-4 flex justify-between items-center text-[10px] text-neutral-500 px-2">
                    <p>
                        Showing {numRows} samples × {numFeatures} features
                    </p>
                    <p>Dataset Source: {associatedDataset?.name || "Unknown"}</p>
                </div>
            </div>
        );
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm animate-in fade-in duration-300">
            <div className="w-full max-w-6xl max-h-[90vh] flex flex-col bg-neutral-900 border border-neutral-800 rounded-[2rem] overflow-hidden shadow-[0_0_50px_-12px_rgba(79,70,229,0.2)]">
                {/* Header */}
                <div className="p-8 pb-6 border-b border-neutral-800 flex items-start justify-between bg-gradient-to-b from-white/[0.02] to-transparent">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <div className="p-2.5 bg-indigo-500/10 rounded-xl">
                                <Table2 className="w-6 h-6 text-indigo-400" />
                            </div>
                            <h2 className="text-2xl font-bold tracking-tight">
                                Tabular Inference Results
                            </h2>
                        </div>
                        <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs mt-4">
                            <div className="flex items-center gap-2">
                                <span className="text-neutral-500 uppercase font-bold tracking-widest text-[9px]">
                                    ID
                                </span>
                                <span className="font-mono text-indigo-300 bg-indigo-500/10 px-2 py-0.5 rounded-md">
                                    {prediction.id}
                                </span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-neutral-500 uppercase font-bold tracking-widest text-[9px]">
                                    Method
                                </span>
                                <span className="px-2 py-0.5 bg-neutral-800 rounded-md text-indigo-100 font-bold border border-neutral-700">
                                    {prediction.method.toUpperCase()}
                                </span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-neutral-500 uppercase font-bold tracking-widest text-[9px]">
                                    Status
                                </span>
                                <div className="flex items-center gap-1.5 px-2 py-0.5 bg-emerald-500/10 rounded-md border border-emerald-500/20">
                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                    <span className="text-emerald-400 font-bold lowercase">
                                        {prediction.status}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-3 text-neutral-500 hover:text-white hover:bg-neutral-800 rounded-2xl transition-all"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {/* Body */}
                <div className="p-8 overflow-y-auto flex-1 bg-black/20">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-32 gap-6">
                            <div className="relative">
                                <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full" />
                                <Loader2 className="w-12 h-12 animate-spin text-indigo-500 relative z-10" />
                            </div>
                            <div className="text-center">
                                <p className="text-white font-medium text-lg">
                                    Retrieving tabular data
                                </p>
                                <p className="text-neutral-500 text-sm mt-1 animate-pulse">
                                    Synchronizing with cloud storage...
                                </p>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center py-20 gap-6 bg-red-500/5 rounded-3xl border border-red-500/10">
                            <XCircle className="w-16 h-16 text-red-500/50" />
                            <div className="text-center">
                                <p className="text-red-400 font-bold text-lg">Error Loading Data</p>
                                <p className="text-neutral-400 text-sm mt-1 max-w-sm mx-auto">
                                    {error}
                                </p>
                            </div>
                            <Button
                                onClick={onClose}
                                variant="ghost"
                                className="text-neutral-400 hover:text-white"
                            >
                                Dismiss and Close
                            </Button>
                        </div>
                    ) : (
                        renderTable()
                    )}
                </div>

                {/* Footer */}
                <div className="p-8 border-t border-neutral-800 bg-neutral-900/50 flex flex-col md:flex-row gap-4 items-center justify-between">
                    <p className="text-xs text-neutral-500 max-w-md">
                        This view shows tabular model outputs combined with their input features.
                        Uncertainty scores indicate the model's confidence per individual sample.
                    </p>
                    <div className="flex gap-3">
                        <Button
                            variant="outline"
                            onClick={onClose}
                            className="bg-transparent border-neutral-700 text-neutral-300 hover:bg-neutral-800 rounded-xl px-6"
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={onClose}
                            className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl px-10 shadow-xl shadow-indigo-600/20 font-bold"
                        >
                            Finish Review
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}
