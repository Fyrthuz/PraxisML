"use client";
import React from "react";
import { Loader2, FlaskConical } from "lucide-react";
import { Dataset, NLPAlgorithm } from "@/lib/api";

interface Props {
    datasets: Dataset[];
    algorithms: NLPAlgorithm[];
    isTraining: boolean;
    trainingStatus: any;
    onTrain: (body: Record<string, any>) => void;
}

export default function NLPTrainingTab({
    datasets,
    algorithms,
    isTraining,
    trainingStatus,
    onTrain,
}: Props) {
    const [dataset, setDataset] = React.useState("");
    const [textCol, setTextCol] = React.useState("");
    const [labelCol, setLabelCol] = React.useState("");
    const [algorithm, setAlgorithm] = React.useState("distilbert");
    const [taskType, setTaskType] = React.useState("text_classification");
    const [hyperparams, setHyperparams] = React.useState<Record<string, any>>({});

    const selected = algorithms.find((a) => a.id === algorithm);
    const selectedDataset = datasets.find((d) => d.id === dataset);
    const cols = selectedDataset?.column_names || [];

    const handleTrain = () => {
        onTrain({
            dataset_id: dataset,
            text_column: textCol,
            label_column: labelCol || null,
            algorithm,
            task_type: taskType,
            hyperparams,
        });
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-emerald-500/20 rounded-xl text-emerald-400">
                        <FlaskConical className="w-5 h-5" />
                    </div>
                    <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider">Entrenar Modelo NLP</h4>
                </div>

                <div>
                    <label className="text-xs font-semibold text-neutral-500 ml-1">Dataset</label>
                    <select
                        className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1"
                        value={dataset}
                        onChange={(e) => { setDataset(e.target.value); setTextCol(""); setLabelCol(""); }}
                    >
                        <option value="" disabled>Selecciona dataset</option>
                        {datasets.map((d) => (
                            <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                    </select>
                </div>

                {dataset && (
                    <>
                        <div>
                            <label className="text-xs font-semibold text-neutral-500 ml-1">Columna de texto</label>
                            <select className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1" value={textCol} onChange={(e) => setTextCol(e.target.value)}>
                                <option value="" disabled>Selecciona columna</option>
                                {cols.map((c) => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="text-xs font-semibold text-neutral-500 ml-1">Columna etiqueta (opcional)</label>
                            <select className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1" value={labelCol} onChange={(e) => setLabelCol(e.target.value)}>
                                <option value="">— Ninguna —</option>
                                {cols.filter((c) => c !== textCol).map((c) => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                    </>
                )}

                <div>
                    <label className="text-xs font-semibold text-neutral-500 ml-1">Arquitectura</label>
                    <select className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1" value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
                        {algorithms.map((a) => <option key={a.id} value={a.id}>{a.display_name}</option>)}
                    </select>
                </div>

                <div>
                    <label className="text-xs font-semibold text-neutral-500 ml-1">Tarea</label>
                    <select className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm mt-1" value={taskType} onChange={(e) => setTaskType(e.target.value)}>
                        {(selected?.task_types || []).map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}
                    </select>
                </div>

                <button
                    onClick={handleTrain}
                    disabled={isTraining || !dataset || !textCol}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded-xl font-semibold transition-colors"
                >
                    {isTraining ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                    {isTraining ? "Entrenando..." : "Entrenar"}
                </button>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">Estado</h4>
                {trainingStatus ? (
                    <pre className="bg-neutral-950 rounded-xl p-4 text-xs text-emerald-300 overflow-auto whitespace-pre-wrap">
                        {JSON.stringify(trainingStatus, null, 2)}
                    </pre>
                ) : (
                    <p className="text-sm text-neutral-500">Lanza un entrenamiento para ver el estado aquí.</p>
                )}
            </div>
        </div>
    );
}
