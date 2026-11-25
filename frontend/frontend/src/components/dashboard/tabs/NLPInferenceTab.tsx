"use client";
import React, { useEffect } from "react";
import { Loader2, Cpu, Send, FileSpreadsheet } from "lucide-react";
import { MLModel, NLPPrediction } from "@/lib/api";

interface Props {
    nlpModels: MLModel[];
    isPredicting: boolean;
    prediction: NLPPrediction | null;
    onPredict: (modelId: string, text: string) => void;
    onBatch: (modelId: string, textColumn: string, file: File) => void;
}

const SAMPLE_TEXTS = [
    "Excellent model serving latency and rock-solid MLflow tracking!",
    "The deployment crashed with timeout errors and missing metrics.",
    "Standard release with minor configuration updates for staging.",
];

export default function NLPInferenceTab({
    nlpModels,
    isPredicting,
    prediction,
    onPredict,
    onBatch,
}: Props) {
    const [modelId, setModelId] = React.useState("");
    const [text, setText] = React.useState("");
    const [textColumn, setTextColumn] = React.useState("text");
    const [file, setFile] = React.useState<File | null>(null);

    useEffect(() => {
        if (!modelId && nlpModels.length > 0) {
            setModelId(nlpModels[0].id);
        }
    }, [modelId, nlpModels]);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                    <div className="flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-orange-400" />
                        <h4 className="text-xs font-semibold text-zinc-100 uppercase tracking-wider">
                            NLP Real-Time &amp; Batch Inference
                        </h4>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-500">
                        {nlpModels.length} NLP model{nlpModels.length !== 1 ? "s" : ""}
                    </span>
                </div>

                <div>
                    <label className="text-[11px] font-medium text-zinc-400">NLP Model</label>
                    <select
                        className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 mt-1 focus:outline-none focus:border-zinc-600"
                        value={modelId}
                        onChange={(e) => setModelId(e.target.value)}
                    >
                        <option value="" disabled>
                            {nlpModels.length === 0 ? "No NLP models registered yet" : "Select NLP model..."}
                        </option>
                        {nlpModels.map((m) => (
                            <option key={m.id} value={m.id}>
                                {m.name} ({m.stage || "Staging"})
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <div className="flex items-center justify-between">
                        <label className="text-[11px] font-medium text-zinc-400">Input Text</label>
                        <div className="flex items-center gap-1">
                            {SAMPLE_TEXTS.map((sample, idx) => (
                                <button
                                    key={idx}
                                    type="button"
                                    onClick={() => setText(sample)}
                                    className="px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-[10px] font-mono text-zinc-400 hover:text-zinc-200"
                                >
                                    sample #{idx + 1}
                                </button>
                            ))}
                        </div>
                    </div>
                    <textarea
                        className="w-full bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 py-2.5 text-xs text-zinc-100 mt-1 min-h-24 focus:outline-none focus:border-zinc-600"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Enter text to classify or extract entities..."
                    />
                </div>

                <button
                    onClick={() => onPredict(modelId, text)}
                    disabled={isPredicting || !modelId || !text.trim()}
                    className="w-full h-9 flex items-center justify-center gap-2 px-4 bg-zinc-100 hover:bg-white disabled:opacity-40 text-zinc-950 rounded-md text-xs font-medium transition-colors"
                >
                    {isPredicting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                    {isPredicting ? "Running inference..." : "Run NLP Inference"}
                </button>

                <div className="border-t border-zinc-800 pt-4 space-y-2.5">
                    <div className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-400">
                        <FileSpreadsheet className="w-3.5 h-3.5" />
                        Async Batch CSV Scoring (Celery)
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        <input
                            className="w-full bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 py-2 text-xs text-zinc-300"
                            type="file"
                            accept=".csv"
                            onChange={(e) => setFile(e.target.files?.[0] || null)}
                        />
                        <input
                            className="w-full bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 py-2 text-xs font-mono text-zinc-200"
                            value={textColumn}
                            onChange={(e) => setTextColumn(e.target.value)}
                            placeholder="text column (default: text)"
                        />
                    </div>
                    <button
                        onClick={() => file && onBatch(modelId, textColumn, file)}
                        disabled={!modelId || !file}
                        className="w-full h-8 px-4 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 disabled:opacity-40 text-zinc-200 rounded-md text-xs font-medium"
                    >
                        Enqueue Batch CSV Job
                    </button>
                </div>
            </div>

            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                    <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                        Inference Output &amp; Telemetry
                    </h4>
                    <span className="text-[10px] font-mono text-zinc-500">domain: nlp</span>
                </div>
                {prediction ? (
                    <div className="space-y-4">
                        <div className="grid grid-cols-3 gap-2.5">
                            <div className="rounded-md bg-[#0b0c0f] border border-zinc-800 p-3">
                                <span className="text-[10px] uppercase text-zinc-500 block">Predicted Label</span>
                                <span className="text-base font-mono font-semibold text-zinc-100 mt-0.5 block">
                                    {prediction.label ?? "—"}
                                </span>
                            </div>
                            <div className="rounded-md bg-[#0b0c0f] border border-zinc-800 p-3">
                                <span className="text-[10px] uppercase text-zinc-500 block">Confidence</span>
                                <span className="text-base font-mono font-semibold text-emerald-400 tabular-nums mt-0.5 block">
                                    {prediction.confidence != null
                                        ? `${(prediction.confidence * 100).toFixed(1)}%`
                                        : "—"}
                                </span>
                            </div>
                            <div className="rounded-md bg-[#0b0c0f] border border-zinc-800 p-3">
                                <span className="text-[10px] uppercase text-zinc-500 block">Latency</span>
                                <span className="text-base font-mono font-semibold text-orange-400 tabular-nums mt-0.5 block">
                                    {prediction.latency_ms.toFixed(1)} ms
                                </span>
                            </div>
                        </div>

                        {prediction.confidence != null && (
                            <div className="space-y-1">
                                <div className="flex justify-between text-[11px] font-mono text-zinc-500">
                                    <span>Confidence score</span>
                                    <span>{(prediction.confidence * 100).toFixed(2)}%</span>
                                </div>
                                <div className="h-1.5 rounded-full bg-zinc-900 overflow-hidden">
                                    <div
                                        className="h-full bg-emerald-400"
                                        style={{ width: `${Math.min(100, prediction.confidence * 100)}%` }}
                                    />
                                </div>
                            </div>
                        )}

                        {prediction.tokens && prediction.tokens.length > 0 && (
                            <div className="space-y-1.5">
                                <span className="text-[11px] font-mono text-zinc-500">
                                    Tokens ({prediction.tokens.length})
                                </span>
                                <div className="flex flex-wrap gap-1 bg-[#0b0c0f] border border-zinc-800 rounded-md p-2.5 max-h-28 overflow-y-auto">
                                    {prediction.tokens.slice(0, 40).map((tok, i) => (
                                        <span
                                            key={i}
                                            className="px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-[10px] font-mono text-zinc-300"
                                        >
                                            {tok}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {prediction.entities.length > 0 && (
                            <div className="space-y-1.5">
                                <span className="text-[11px] font-mono text-zinc-500">
                                    Extracted Entities ({prediction.entities.length})
                                </span>
                                <div className="space-y-1">
                                    {prediction.entities.map((e, i) => (
                                        <div
                                            key={i}
                                            className="flex justify-between bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 py-2 text-xs font-mono"
                                        >
                                            <span className="text-zinc-200">{e.entity}</span>
                                            <span className="text-orange-400">{e.label}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <p className="text-xs text-zinc-500 py-6">
                        Select an NLP model and run inference to inspect label, confidence, tokenization, and latency (automatically tracked in Observability → Usage).
                    </p>
                )}
            </div>
        </div>
    );
}
