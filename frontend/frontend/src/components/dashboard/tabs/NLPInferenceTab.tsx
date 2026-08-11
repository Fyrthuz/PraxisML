"use client";
import React from "react";
import { Loader2, Cpu } from "lucide-react";
import { MLModel, NLPPrediction } from "@/lib/api";

interface Props {
    nlpModels: MLModel[];
    isPredicting: boolean;
    prediction: NLPPrediction | null;
    onPredict: (modelId: string, text: string) => void;
    onBatch: (modelId: string, textColumn: string, file: File) => void;
}

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

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-sky-500/20 rounded-xl text-sky-400">
                        <Cpu className="w-5 h-5" />
                    </div>
                    <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider">Inferencia</h4>
                </div>

                <div>
                    <label className="text-xs font-semibold text-neutral-500 ml-1">Modelo NLP</label>
                    <select className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                        <option value="" disabled>Selecciona modelo</option>
                        {nlpModels.map((m) => (
                            <option key={m.id} value={m.id}>{m.name}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="text-xs font-semibold text-neutral-500 ml-1">Texto</label>
                    <textarea
                        className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1 min-h-24"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Escribe un texto para clasificar..."
                    />
                </div>

                <button
                    onClick={() => onPredict(modelId, text)}
                    disabled={isPredicting || !modelId || !text}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white rounded-xl font-semibold transition-colors"
                >
                    {isPredicting ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                    {isPredicting ? "Prediciendo..." : "Predecir"}
                </button>

                <div className="border-t border-neutral-800 pt-4 mt-4">
                    <label className="text-xs font-semibold text-neutral-500 ml-1">Batch CSV</label>
                    <input className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
                    <input className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-2" value={textColumn} onChange={(e) => setTextColumn(e.target.value)} placeholder="Columna de texto (default: text)" />
                    <button
                        onClick={() => file && onBatch(modelId, textColumn, file)}
                        disabled={!modelId || !file}
                        className="w-full mt-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 text-neutral-200 rounded-xl text-sm"
                    >
                        Encolar Batch
                    </button>
                </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">Resultado</h4>
                {prediction ? (
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-neutral-500">Label</span>
                            <span className="text-lg font-bold text-white">{prediction.label ?? "—"}</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-neutral-500">Confianza</span>
                            <span className="text-sm text-emerald-400">
                                {prediction.confidence != null ? (prediction.confidence * 100).toFixed(1) + "%" : "—"}
                            </span>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-neutral-500">Latencia</span>
                            <span className="text-sm text-neutral-300">{prediction.latency_ms.toFixed(1)} ms</span>
                        </div>
                        {prediction.entities.length > 0 && (
                            <div className="space-y-1">
                                <span className="text-xs text-neutral-500">Entidades</span>
                                {prediction.entities.map((e, i) => (
                                    <div key={i} className="flex justify-between bg-neutral-950 rounded-lg px-3 py-2 text-sm">
                                        <span className="text-neutral-300">{e.entity}</span>
                                        <span className="text-sky-400">{e.label}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ) : (
                    <p className="text-sm text-neutral-500">La predicción aparecerá aquí.</p>
                )}
            </div>
        </div>
    );
}
