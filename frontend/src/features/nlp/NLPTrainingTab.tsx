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
    const [epochs, setEpochs] = React.useState(2);

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
            hyperparams: { epochs },
        });
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                    <div className="flex items-center gap-2">
                        <FlaskConical className="w-4 h-4 text-orange-400" />
                        <h4 className="text-xs font-semibold text-zinc-100 uppercase tracking-wider">
                            Fine-Tune NLP Model (HuggingFace + ONNX)
                        </h4>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-500">queue: gpu-train</span>
                </div>

                <div>
                    <label className="text-[11px] font-medium text-zinc-400">Dataset</label>
                    <select
                        className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 mt-1 focus:outline-none focus:border-zinc-600"
                        value={dataset}
                        onChange={(e) => {
                            setDataset(e.target.value);
                            setTextCol("");
                            setLabelCol("");
                        }}
                    >
                        <option value="" disabled>
                            Select tabular text dataset...
                        </option>
                        {datasets.map((d) => (
                            <option key={d.id} value={d.id}>
                                {d.name} ({d.num_rows ?? "?"} rows)
                            </option>
                        ))}
                    </select>
                </div>

                {dataset && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div>
                            <label className="text-[11px] font-medium text-zinc-400">Text Column</label>
                            <select
                                className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 mt-1 focus:outline-none focus:border-zinc-600"
                                value={textCol}
                                onChange={(e) => setTextCol(e.target.value)}
                            >
                                <option value="" disabled>
                                    Select column...
                                </option>
                                {cols.map((c) => (
                                    <option key={c} value={c}>
                                        {c}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <label className="text-[11px] font-medium text-zinc-400">
                                Label Column
                            </label>
                            <select
                                className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 mt-1 focus:outline-none focus:border-zinc-600"
                                value={labelCol}
                                onChange={(e) => setLabelCol(e.target.value)}
                            >
                                <option value="">— None —</option>
                                {cols
                                    .filter((c) => c !== textCol)
                                    .map((c) => (
                                        <option key={c} value={c}>
                                            {c}
                                        </option>
                                    ))}
                            </select>
                        </div>
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div>
                        <label className="text-[11px] font-medium text-zinc-400">Architecture</label>
                        <select
                            className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 mt-1 focus:outline-none focus:border-zinc-600"
                            value={algorithm}
                            onChange={(e) => setAlgorithm(e.target.value)}
                        >
                            {algorithms.map((a) => (
                                <option key={a.id} value={a.id}>
                                    {a.display_name}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="text-[11px] font-medium text-zinc-400">Task</label>
                        <select
                            className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 mt-1 focus:outline-none focus:border-zinc-600"
                            value={taskType}
                            onChange={(e) => setTaskType(e.target.value)}
                        >
                            {(selected?.task_types || ["text_classification"]).map((t) => (
                                <option key={t} value={t}>
                                    {t.replace(/_/g, " ")}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="text-[11px] font-medium text-zinc-400">Epochs</label>
                        <input
                            type="number"
                            min={1}
                            max={20}
                            value={epochs}
                            onChange={(e) => setEpochs(Number(e.target.value))}
                            className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs font-mono text-zinc-100 mt-1 focus:outline-none focus:border-zinc-600"
                        />
                    </div>
                </div>

                <button
                    onClick={handleTrain}
                    disabled={isTraining || !dataset || !textCol}
                    className="w-full h-9 flex items-center justify-center gap-2 px-4 bg-zinc-100 hover:bg-white disabled:opacity-40 text-zinc-950 rounded-md text-xs font-medium transition-colors"
                >
                    {isTraining ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
                    {isTraining ? "Training in Celery..." : "Launch NLP Training"}
                </button>
            </div>

            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-3">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                    <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                        Job Telemetry &amp; MLflow Status
                    </h4>
                    <span className="text-[10px] font-mono text-zinc-500">
                        {trainingStatus?.status || "IDLE"}
                    </span>
                </div>
                {trainingStatus ? (
                    <pre className="bg-[#0b0c0f] border border-zinc-800 rounded-md p-3.5 text-xs font-mono text-emerald-400 overflow-auto whitespace-pre-wrap max-h-72">
                        {JSON.stringify(trainingStatus, null, 2)}
                    </pre>
                ) : (
                    <div className="text-xs text-zinc-500 space-y-2 py-4">
                        <p>
                            Launch a HuggingFace fine-tuning job to log parameters, metrics, model artifacts, and ONNX runtime export automatically into MLflow.
                        </p>
                        <p className="font-mono text-[11px] text-zinc-600">
                            Supported architectures: {algorithms.map((a) => a.id).join(", ") || "distilbert, bert-base, roberta"}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
