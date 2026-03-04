import React from 'react';
import { Button } from '@/components/ui/button';
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
    const getSelectedAlgo = () => algorithms.find(a => a.id === trainAlgorithm);
    const getSelectedDatasetCols = () => {
        const ds = datasets.find(d => d.id === trainDataset);
        return ds?.column_names || [];
    };

    const selectedAlgo = getSelectedAlgo();
    const tabularDatasets = datasets.filter(ds => ds.file_type && ['csv', 'xlsx', 'parquet'].includes(ds.file_type));

    return (
        <div className="space-y-8">
            <div className="flex items-center gap-4 mb-2">
                <div className="p-3 bg-emerald-500/20 rounded-2xl text-emerald-400">
                    <FlaskConical className="w-6 h-6" />
                </div>
                <div>
                    <h3 className="text-xl font-bold">Train a Model</h3>
                    <p className="text-sm text-neutral-400">Select a tabular dataset, choose an algorithm, configure hyperparameters, and train.</p>
                </div>
            </div>

            {tabularDatasets.length === 0 ? (
                <div className="border-2 border-dashed border-neutral-700 rounded-xl p-12 text-center text-neutral-500">
                    No tabular datasets found. Upload a .csv, .xlsx, or .parquet file first.
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left: Configuration */}
                    <div className="space-y-6">
                        <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                            <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">Data</h4>
                            <div className="space-y-2">
                                <label className="text-xs font-semibold text-neutral-500 ml-1">Dataset</label>
                                <select
                                    className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-emerald-500/50 appearance-none"
                                    value={trainDataset}
                                    onChange={e => { setTrainDataset(e.target.value); setTrainTarget(''); }}
                                >
                                    <option value="" disabled>Select dataset</option>
                                    {tabularDatasets.map(ds => (
                                        <option key={ds.id} value={ds.id}>{ds.name} ({ds.num_rows} rows × {ds.num_columns} cols)</option>
                                    ))}
                                </select>
                            </div>
                            {trainDataset && (
                                <div className="space-y-2">
                                    <label className="text-xs font-semibold text-neutral-500 ml-1">Target Column</label>
                                    <select
                                        className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-emerald-500/50 appearance-none"
                                        value={trainTarget}
                                        onChange={e => setTrainTarget(e.target.value)}
                                    >
                                        <option value="" disabled>Select target</option>
                                        {getSelectedDatasetCols().map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                            )}
                            <div className="space-y-2">
                                <label className="text-xs font-semibold text-neutral-500 ml-1">Task Type</label>
                                <div className="flex gap-2">
                                    {['classification', 'regression'].map(tt => (
                                        <button
                                            key={tt}
                                            onClick={() => setTrainTaskType(tt)}
                                            className={`flex-1 py-2 rounded-xl text-sm font-medium transition-colors ${trainTaskType === tt ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-neutral-800 text-neutral-400 border border-neutral-700 hover:bg-neutral-700'}`}
                                        >
                                            {tt.charAt(0).toUpperCase() + tt.slice(1)}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                            <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">Algorithm</h4>
                            <div className="grid grid-cols-2 gap-2">
                                {algorithms.filter(a => a.task_types.includes(trainTaskType)).map(algo => (
                                    <button
                                        key={algo.id}
                                        onClick={() => {
                                            setTrainAlgorithm(algo.id);
                                            const defaults: Record<string, any> = {};
                                            algo.hyperparams.forEach(hp => { defaults[hp.name] = hp.default; });
                                            setTrainHyperparams(defaults);
                                        }}
                                        className={`p-3 rounded-xl text-left text-sm font-medium transition-all ${trainAlgorithm === algo.id ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-lg shadow-emerald-500/10' : 'bg-neutral-800 text-neutral-300 border border-neutral-700 hover:border-neutral-600'}`}
                                    >
                                        <div className="flex items-center justify-between">
                                            <span>{algo.display_name}</span>
                                            <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-bold ${algo.framework === 'pytorch' ? 'bg-orange-500/20 text-orange-400' : 'bg-blue-500/20 text-blue-400'}`}>
                                                {algo.framework === 'pytorch' ? 'PT' : 'SK'}
                                            </span>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-neutral-500 ml-1">Model Name (optional)</label>
                            <input
                                type="text"
                                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-emerald-500/50"
                                placeholder="e.g. My Classifier v1"
                                value={trainModelName}
                                onChange={e => setTrainModelName(e.target.value)}
                            />
                        </div>
                    </div>

                    {/* Right: Hyperparameters + Launch */}
                    <div className="space-y-6">
                        {selectedAlgo && (
                            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                                <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">
                                    {selectedAlgo.display_name} — Hyperparameters
                                </h4>
                                {selectedAlgo.hyperparams.map(hp => (
                                    <div key={hp.name} className="space-y-1">
                                        <label className="text-xs font-semibold text-neutral-500 ml-1">{hp.label}</label>
                                        {hp.type === 'select' ? (
                                            <select
                                                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-emerald-500/50 appearance-none"
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
                                                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-emerald-500/50"
                                                value={trainHyperparams[hp.name] ?? hp.default ?? ''}
                                                min={hp.min}
                                                max={hp.max}
                                                step={hp.type === 'float' ? 0.01 : 1}
                                                onChange={e => setTrainHyperparams(prev => ({ ...prev, [hp.name]: hp.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value) }))}
                                            />
                                        )}
                                        {(hp.min != null || hp.max != null) && (
                                            <p className="text-[10px] text-neutral-600 ml-1">Range: {hp.min ?? '—'} to {hp.max ?? '—'}</p>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {selectedAlgo && (
                            <div className="bg-neutral-900/30 border border-neutral-800 rounded-xl p-4 text-xs text-neutral-500 space-y-1">
                                <span className="font-bold text-neutral-400">Supported uncertainty methods:</span>
                                <div className="flex gap-2 mt-1 flex-wrap">
                                    <span className="px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">entropy</span>
                                    {selectedAlgo.supports_tree_variance && (
                                        <span className="px-2 py-0.5 rounded-full bg-orange-500/10 text-orange-400 border border-orange-500/20">tree_variance</span>
                                    )}
                                    {selectedAlgo.supports_proba && (
                                        <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">conformal</span>
                                    )}
                                    <span className="px-2 py-0.5 rounded-full bg-neutral-700 text-neutral-400">none</span>
                                </div>
                            </div>
                        )}

                        {/* Validation Strategy */}
                        <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                            <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">Validation Strategy</h4>
                            <div className="grid grid-cols-2 gap-3">
                                <button
                                    onClick={() => setValidationStrategy('holdout')}
                                    className={`p-3 rounded-xl text-sm font-medium transition-all text-center ${validationStrategy === 'holdout'
                                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 shadow-lg shadow-cyan-500/10'
                                        : 'bg-neutral-800 text-neutral-300 border border-neutral-700 hover:border-neutral-600'
                                        }`}
                                >
                                    Holdout Split
                                </button>
                                <button
                                    onClick={() => setValidationStrategy('cross_validation')}
                                    className={`p-3 rounded-xl text-sm font-medium transition-all text-center ${validationStrategy === 'cross_validation'
                                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 shadow-lg shadow-cyan-500/10'
                                        : 'bg-neutral-800 text-neutral-300 border border-neutral-700 hover:border-neutral-600'
                                        }`}
                                >
                                    Cross-Validation
                                </button>
                            </div>
                            {validationStrategy === 'holdout' ? (
                                <div className="space-y-1">
                                    <label className="text-xs font-semibold text-neutral-500 ml-1">Test Size ({(testSize * 100).toFixed(0)}%)</label>
                                    <input
                                        type="range"
                                        className="w-full accent-cyan-500"
                                        min={0.1}
                                        max={0.5}
                                        step={0.05}
                                        value={testSize}
                                        onChange={e => setTestSize(parseFloat(e.target.value))}
                                    />
                                    <div className="flex justify-between text-[10px] text-neutral-600">
                                        <span>10%</span><span>50%</span>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-1">
                                    <label className="text-xs font-semibold text-neutral-500 ml-1">Number of Folds</label>
                                    <div className="flex gap-2">
                                        {[3, 5, 10].map(k => (
                                            <button
                                                key={k}
                                                onClick={() => setNFolds(k)}
                                                className={`flex-1 py-2 rounded-xl text-sm font-medium transition-all ${nFolds === k
                                                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                                                    : 'bg-neutral-800 text-neutral-300 border border-neutral-700 hover:border-neutral-600'
                                                    }`}
                                            >
                                                {k}-fold
                                            </button>
                                        ))}
                                    </div>
                                    <p className="text-[10px] text-neutral-600 ml-1">Model will be trained {nFolds}× and metrics averaged. Final model trains on all data.</p>
                                </div>
                            )}
                        </div>

                        <Button
                            className="w-full bg-emerald-600 hover:bg-emerald-700 h-14 rounded-2xl text-lg font-bold shadow-2xl shadow-emerald-600/20"
                            disabled={!trainDataset || !trainTarget || !trainAlgorithm || isTraining}
                            onClick={handleStartTraining}
                        >
                            {isTraining ? (
                                <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Training {trainingStatus?.status || '...'}</>
                            ) : (
                                <><FlaskConical className="w-5 h-5 mr-2" /> Train Model</>
                            )}
                        </Button>

                        {trainingStatus?.status === 'SUCCESS' && trainingStatus.result && (
                            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-6 space-y-3">
                                <h4 className="font-bold text-emerald-400 flex items-center gap-2">
                                    <CheckCircle2 className="w-5 h-5" /> Training Complete
                                </h4>
                                <div className="grid grid-cols-2 gap-3 text-sm">
                                    {Object.entries(trainingStatus.result.metrics).map(([key, val]) => (
                                        <div key={key} className="bg-neutral-900/60 rounded-xl p-3 text-center">
                                            <div className="text-lg font-bold text-white">{typeof val === 'number' ? val.toFixed(4) : String(val)}</div>
                                            <div className="text-[10px] text-neutral-500 uppercase tracking-wider mt-1">{key}</div>
                                        </div>
                                    ))}
                                </div>
                                <p className="text-xs text-neutral-500 mt-2">
                                    MLFlow run: <span className="font-mono text-indigo-400">{trainingStatus.result.mlflow_run_id?.substring(0, 12)}</span>
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
