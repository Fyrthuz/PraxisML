import React from 'react';
import { FlaskConical, Loader2, CheckCircle2 } from 'lucide-react';
import { Dataset, AlgorithmInfo, TrainingStatus } from '../../../lib/api';

interface TrainingTabProps {
    datasets: Dataset[];
    algorithms: AlgorithmInfo[];
    trainDataset: string;
    setTrainDataset: (val: string) => void;
    trainTarget: string;
    setTrainTarget: (val: string) => void;
    trainTaskType: string;
    setTrainTaskType: (val: string) => void;
    trainAlgorithm: string;
    setTrainAlgorithm: (val: string) => void;
    trainHyperparams: Record<string, any>;
    setTrainHyperparams: React.Dispatch<React.SetStateAction<Record<string, any>>>;
    trainModelName: string;
    setTrainModelName: (val: string) => void;
    trainRegistryName: string;
    setTrainRegistryName: (val: string) => void;
    validationStrategy: 'holdout' | 'cross_validation';
    setValidationStrategy: (val: 'holdout' | 'cross_validation') => void;
    testSize: number;
    setTestSize: (val: number) => void;
    nFolds: number;
    setNFolds: (val: number) => void;
    isTraining: boolean;
    trainingStatus: TrainingStatus | null;
    handleStartTraining: () => void;
}

export default function TrainingTab({
    datasets,
    algorithms,
    trainDataset,
    setTrainDataset,
    trainTarget,
    setTrainTarget,
    trainTaskType,
    setTrainTaskType,
    trainAlgorithm,
    setTrainAlgorithm,
    trainHyperparams,
    setTrainHyperparams,
    trainModelName,
    setTrainModelName,
    trainRegistryName,
    setTrainRegistryName,
    validationStrategy,
    setValidationStrategy,
    testSize,
    setTestSize,
    nFolds,
    setNFolds,
    isTraining,
    trainingStatus,
    handleStartTraining
}: TrainingTabProps) {
    const selectedAlgo = algorithms.find(a => a.id === trainAlgorithm);
    const selectedDatasetObj = datasets.find(d => d.id === trainDataset);
    const datasetCols = selectedDatasetObj?.column_names || [];
    const tabularDatasets = datasets.filter(ds => ds.file_type && ['csv', 'xlsx', 'parquet'].includes(ds.file_type));

    return (
        <div className="space-y-6">
            {tabularDatasets.length === 0 ? (
                <div className="border border-dashed border-zinc-800 bg-[#101217] rounded-lg p-14 text-center text-xs text-zinc-500">
                    No tabular datasets found. Upload a .csv, .xlsx, or .parquet file first.
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                    {/* Left: Dataset, Target, Algorithm */}
                    <div className="lg:col-span-6 space-y-5">
                        <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                            <div className="flex items-center justify-between">
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                                    1. Dataset & Target
                                </h4>
                                {selectedDatasetObj?.pipeline_path && (
                                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
                                        preprocessed
                                    </span>
                                )}
                            </div>
                            <div className="space-y-1.5">
                                <label className="text-[11px] font-medium text-zinc-400">Dataset</label>
                                <select
                                    className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                                    value={trainDataset}
                                    onChange={e => { setTrainDataset(e.target.value); setTrainTarget(''); }}
                                >
                                    <option value="" disabled>Select tabular dataset...</option>
                                    {tabularDatasets.map(ds => (
                                        <option key={ds.id} value={ds.id}>
                                            {ds.name} ({ds.num_rows}×{ds.num_columns}){ds.pipeline_path ? ' [clean]' : ''}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            {trainDataset && (
                                <div className="space-y-1.5">
                                    <label className="text-[11px] font-medium text-zinc-400">Target Column</label>
                                    <select
                                        className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-600"
                                        value={trainTarget}
                                        onChange={e => setTrainTarget(e.target.value)}
                                    >
                                        <option value="" disabled>Select target column...</option>
                                        {datasetCols.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                            )}
                            <div className="space-y-1.5">
                                <label className="text-[11px] font-medium text-zinc-400">Task Type</label>
                                <div className="grid grid-cols-2 gap-2">
                                    {['classification', 'regression'].map(tt => (
                                        <button
                                            key={tt}
                                            type="button"
                                            onClick={() => setTrainTaskType(tt)}
                                            className={`h-8 rounded-md text-xs font-medium border transition-colors ${
                                                trainTaskType === tt
                                                    ? 'bg-zinc-100 text-zinc-950 border-zinc-100'
                                                    : 'bg-[#0b0c0f] text-zinc-400 border-zinc-800 hover:text-zinc-200'
                                            }`}
                                        >
                                            {tt.charAt(0).toUpperCase() + tt.slice(1)}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                                2. Algorithm
                            </h4>
                            <div className="grid grid-cols-2 gap-2">
                                {algorithms.filter(a => a.task_types.includes(trainTaskType)).map(algo => (
                                    <button
                                        key={algo.id}
                                        type="button"
                                        onClick={() => {
                                            setTrainAlgorithm(algo.id);
                                            const defaults: Record<string, any> = {};
                                            algo.hyperparams.forEach(hp => { defaults[hp.name] = hp.default; });
                                            setTrainHyperparams(defaults);
                                        }}
                                        className={`p-3 rounded-md text-left text-xs font-medium border transition-colors ${
                                            trainAlgorithm === algo.id
                                                ? 'bg-zinc-800 text-zinc-100 border-orange-500/70'
                                                : 'bg-[#0b0c0f] text-zinc-300 border-zinc-800 hover:border-zinc-700'
                                        }`}
                                    >
                                        <div className="flex items-center justify-between gap-2">
                                            <span className="truncate">{algo.display_name}</span>
                                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">
                                                {algo.framework === 'pytorch' ? 'PT' : 'SK'}
                                            </span>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div className="space-y-1.5">
                                <label className="text-[11px] font-medium text-zinc-400">Model Name (optional)</label>
                                <input
                                    type="text"
                                    className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                                    placeholder="e.g. titanic-rf-v1"
                                    value={trainModelName}
                                    onChange={e => setTrainModelName(e.target.value)}
                                />
                            </div>
                            <div className="space-y-1.5">
                                <label className="text-[11px] font-medium text-zinc-400">Registry Name (optional)</label>
                                <input
                                    type="text"
                                    className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs font-mono text-zinc-100 focus:outline-none focus:border-zinc-600"
                                    placeholder="e.g. titanic_classifier"
                                    value={trainRegistryName}
                                    onChange={e => setTrainRegistryName(e.target.value)}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Right: Hyperparameters, Validation & Launch */}
                    <div className="lg:col-span-6 space-y-5">
                        {selectedAlgo && (
                            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                                <div className="flex items-center justify-between">
                                    <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                                        3. {selectedAlgo.display_name} Hyperparameters
                                    </h4>
                                    <div className="flex gap-1.5">
                                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">entropy</span>
                                        {selectedAlgo.supports_tree_variance && (
                                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">tree_var</span>
                                        )}
                                        {selectedAlgo.supports_proba && (
                                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">conformal</span>
                                        )}
                                    </div>
                                </div>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                                    {selectedAlgo.hyperparams.map(hp => (
                                        <div key={hp.name} className="space-y-1">
                                            <label className="text-[11px] font-medium text-zinc-400">{hp.label}</label>
                                            {hp.type === 'select' ? (
                                                <select
                                                    className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100"
                                                    value={trainHyperparams[hp.name] ?? hp.default ?? ''}
                                                    onChange={e => setTrainHyperparams(prev => ({ ...prev, [hp.name]: e.target.value === 'null' ? null : e.target.value }))}
                                                >
                                                    {hp.options?.map(opt => (
                                                        <option key={String(opt.value)} value={String(opt.value)}>{opt.label}</option>
                                                    ))}
                                                </select>
                                            ) : (
                                                <input
                                                    type="number"
                                                    className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs font-mono tabular-nums text-zinc-100"
                                                    value={trainHyperparams[hp.name] ?? hp.default ?? ''}
                                                    min={hp.min}
                                                    max={hp.max}
                                                    step={hp.type === 'float' ? 0.01 : 1}
                                                    onChange={e => setTrainHyperparams(prev => ({ ...prev, [hp.name]: hp.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value) }))}
                                                />
                                            )}
                                            {(hp.min != null || hp.max != null) && (
                                                <p className="text-[10px] font-mono text-zinc-600">Range: {hp.min ?? '—'} .. {hp.max ?? '—'}</p>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                                4. Validation Strategy
                            </h4>
                            <div className="grid grid-cols-2 gap-2">
                                <button
                                    type="button"
                                    onClick={() => setValidationStrategy('holdout')}
                                    className={`h-8 rounded-md text-xs font-medium border transition-colors ${
                                        validationStrategy === 'holdout'
                                            ? 'bg-zinc-100 text-zinc-950 border-zinc-100'
                                            : 'bg-[#0b0c0f] text-zinc-400 border-zinc-800 hover:text-zinc-200'
                                    }`}
                                >
                                    Holdout Split
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setValidationStrategy('cross_validation')}
                                    className={`h-8 rounded-md text-xs font-medium border transition-colors ${
                                        validationStrategy === 'cross_validation'
                                            ? 'bg-zinc-100 text-zinc-950 border-zinc-100'
                                            : 'bg-[#0b0c0f] text-zinc-400 border-zinc-800 hover:text-zinc-200'
                                    }`}
                                >
                                    Cross-Validation
                                </button>
                            </div>
                            {validationStrategy === 'holdout' ? (
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-[11px] font-mono text-zinc-400">
                                        <span>Test Size</span>
                                        <span className="text-zinc-200">{(testSize * 100).toFixed(0)}%</span>
                                    </div>
                                    <input
                                        type="range"
                                        className="w-full accent-orange-500"
                                        min={0.1}
                                        max={0.5}
                                        step={0.05}
                                        value={testSize}
                                        onChange={e => setTestSize(parseFloat(e.target.value))}
                                    />
                                </div>
                            ) : (
                                <div className="space-y-1.5">
                                    <label className="text-[11px] text-zinc-400">Folds</label>
                                    <div className="grid grid-cols-3 gap-2">
                                        {[3, 5, 10].map(k => (
                                            <button
                                                key={k}
                                                type="button"
                                                onClick={() => setNFolds(k)}
                                                className={`h-8 rounded-md text-xs font-mono border ${
                                                    nFolds === k
                                                        ? 'bg-zinc-800 text-zinc-100 border-orange-500/70'
                                                        : 'bg-[#0b0c0f] text-zinc-400 border-zinc-800'
                                                }`}
                                            >
                                                {k}-fold
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        <button
                            type="button"
                            className="w-full h-11 rounded-md bg-zinc-100 hover:bg-white text-zinc-950 text-sm font-semibold disabled:opacity-40 flex items-center justify-center gap-2 transition-colors"
                            disabled={!trainDataset || !trainTarget || !trainAlgorithm || isTraining}
                            onClick={handleStartTraining}
                        >
                            {isTraining ? (
                                <><Loader2 className="w-4 h-4 animate-spin" /> Training ({trainingStatus?.status || 'queued'})</>
                            ) : (
                                <><FlaskConical className="w-4 h-4" /> Start Training Job</>
                            )}
                        </button>

                        {trainingStatus?.status === 'SUCCESS' && trainingStatus.result && (
                            <div className="bg-[#101217] border border-emerald-500/40 rounded-lg p-5 space-y-3">
                                <h4 className="text-sm font-semibold text-emerald-400 flex items-center gap-2">
                                    <CheckCircle2 className="w-4 h-4" /> Training Completed
                                </h4>
                                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                                    {Object.entries(trainingStatus.result.metrics).map(([key, val]) => (
                                        <div key={key} className="bg-[#0b0c0f] border border-zinc-800 rounded-md p-3">
                                            <div className="text-[10px] text-zinc-500 uppercase">{key}</div>
                                            <div className="text-base font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                                                {typeof val === 'number' ? val.toFixed(4) : String(val)}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                <p className="text-[11px] font-mono text-zinc-500">
                                    MLflow run: <span className="text-zinc-300">{trainingStatus.result.mlflow_run_id}</span>
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
