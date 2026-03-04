"use client"
import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import {
    UploadCloud,
    FileText,
    Settings,
    PlayCircle,
    Loader2,
    Database,
    Cpu,
    BarChart3,
    Plus,
    CheckCircle2,
    XCircle,
    Clock,
    Activity,
    ExternalLink,
    Trash2,
    Eye,
    Table2,
    X,
    GraduationCap,
    FlaskConical,
    Filter
} from 'lucide-react';
import { api, Dataset, MLModel, Prediction, DatasetPreview, AlgorithmInfo, TrainingStatus } from '@/lib/api';

import UploadModal from './UploadModal';
import { useAuth } from '../AuthContext';
import SingleTabularInference from './SingleTabularInference';
import toast from 'react-hot-toast';
import DatasetsTab from './tabs/DatasetsTab';
import ModelsTab from './tabs/ModelsTab';
import PredictionsTab from './tabs/PredictionsTab';
import TrainingTab from './tabs/TrainingTab';
import PreprocessingTab from './tabs/PreprocessingTab';

type View = 'datasets' | 'preprocessing' | 'models' | 'predictions' | 'training';

/* ─── Authenticated Image Component ─── */
function AuthImage({ url, alt, token }: { url: string; alt: string; token: string | null }) {
    const [blobUrl, setBlobUrl] = useState<string | null>(null);
    const [error, setError] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!token) return;
        let cancelled = false;

        fetch(url, { headers: { Authorization: `Bearer ${token}` } })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.blob();
            })
            .then(blob => {
                if (!cancelled) {
                    setBlobUrl(URL.createObjectURL(blob));
                    setLoading(false);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setError(true);
                    setLoading(false);
                }
            });

        return () => {
            cancelled = true;
            if (blobUrl) URL.revokeObjectURL(blobUrl);
        };
    }, [url, token]);

    if (loading) return <Loader2 className="w-8 h-8 animate-spin text-neutral-500" />;
    if (error || !blobUrl) return <p className="text-neutral-600 text-sm italic">Not available</p>;

    return <img src={blobUrl} alt={alt} className="max-w-full max-h-full object-contain" />;
}

/* ─── Prediction Results Modal ─── */
function PredictionResultsModal({ prediction, datasets, models, token, onClose }: { prediction: Prediction; datasets: Dataset[]; models: MLModel[]; token: string | null; onClose: () => void }) {
    const [data, setData] = useState<{ prediction?: any; uncertainty?: any; input_data?: any } | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!token) return;
        setLoading(true);
        setError(null);
        fetch(`http://localhost:8000/api/v1/predictions/${prediction.id}/data`, {
            headers: { Authorization: `Bearer ${token}` }
        })
            .then(res => {
                if (!res.ok) throw new Error(`Server returned ${res.status}: ${res.statusText}`);
                return res.json();
            })
            .then(json => {
                setData(json);
                setLoading(false);
            })
            .catch(err => {
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
        // Determine the number of inference samples by looking at prediction array
        const rawPreds = data.prediction;
        const normalizedPreds = normalizeArray(rawPreds);
        const primaryLen = normalizedPreds.length;

        const inputs = normalizeArray(data.input_data, primaryLen);
        const uncs = normalizeArray(data.uncertainty, primaryLen);

        const numRows = Math.max(primaryLen, inputs.length, uncs.length);

        // Get column names if possible
        const associatedDataset = datasets.find(d => d.id === prediction.dataset_id);
        const associatedModel = models.find(m => m.id === prediction.model_id);

        // We need to identify which columns are features vs target
        // Usually, the input_data sent to inference excludes the target_column
        let featureNames: string[] = [];
        if (associatedDataset?.column_names) {
            featureNames = associatedDataset.column_names;
        } else if (associatedModel?.metrics_metadata?.feature_names) {
            // Fallback to model metadata if dataset is missing or doesn't have names
            featureNames = associatedModel.metrics_metadata.feature_names;
        }

        const numFeatures = inputs[0]?.length || 0;

        // Generate header labels for features
        const featureHeaders = [];
        for (let j = 0; j < numFeatures; j++) {
            featureHeaders.push(featureNames[j] || `Feature ${j + 1}`);
        }

        if (numRows === 0) {
            return (
                <div className="flex flex-col items-center justify-center py-20 bg-neutral-950/30 rounded-3xl border border-dashed border-neutral-800">
                    <Table2 className="w-12 h-12 text-neutral-700 mb-4" />
                    <p className="text-neutral-500 font-medium">No results data found for this prediction.</p>
                </div>
            );
        }

        return (
            <div className="relative group/table">
                <div className="overflow-x-auto bg-neutral-900 border border-neutral-800 rounded-2xl shadow-2xl max-h-[500px] scrollbar-thin scrollbar-thumb-neutral-700 scrollbar-track-transparent">
                    <table className="w-full text-left border-collapse text-sm">
                        <thead className="sticky top-0 bg-neutral-950/95 backdrop-blur-md z-20">
                            <tr>
                                <th className="px-4 py-3 text-neutral-500 font-bold uppercase tracking-wider text-[10px] border-b border-neutral-800 w-12 text-center">#</th>
                                {featureHeaders.map((name, idx) => (
                                    <th key={idx} className="px-4 py-3 text-neutral-300 font-bold uppercase tracking-wider text-[10px] border-b border-neutral-800 whitespace-nowrap min-w-[100px]">
                                        {name}
                                    </th>
                                ))}
                                <th className="px-4 py-3 text-emerald-400 font-bold uppercase tracking-wider text-[10px] border-b border-emerald-500/20 bg-emerald-500/5 whitespace-nowrap">Prediction</th>
                                <th className="px-4 py-3 text-amber-400 font-bold uppercase tracking-wider text-[10px] border-b border-amber-500/20 bg-amber-500/5 whitespace-nowrap">Uncertainty</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-800/50">
                            {Array.from({ length: numRows }).map((_, i) => (
                                <tr key={i} className="hover:bg-white/[0.02] transition-colors group/row">
                                    <td className="px-4 py-3 text-neutral-600 font-mono text-center text-xs">{i + 1}</td>

                                    {/* Individual Feature Columns */}
                                    {Array.from({ length: numFeatures }).map((_, j) => (
                                        <td key={j} className="px-4 py-3 font-mono text-xs text-neutral-400 border-r border-neutral-800/30">
                                            {inputs[i] && inputs[i][j] !== undefined ? (
                                                typeof inputs[i][j] === 'number' ? inputs[i][j].toFixed(4) : String(inputs[i][j])
                                            ) : "—"}
                                        </td>
                                    ))}

                                    <td className="px-4 py-3 text-emerald-400 font-mono font-bold bg-emerald-400/5 border-l border-emerald-500/10">
                                        {normalizedPreds[i] ? (
                                            Array.isArray(normalizedPreds[i]) && normalizedPreds[i].length === 1 ? (
                                                typeof normalizedPreds[i][0] === 'number' ? normalizedPreds[i][0].toFixed(6) : JSON.stringify(normalizedPreds[i][0])
                                            ) : (
                                                typeof normalizedPreds[i] === 'number' ? normalizedPreds[i].toFixed(6) : JSON.stringify(normalizedPreds[i])
                                            )
                                        ) : "—"}
                                    </td>
                                    <td className="px-4 py-3 text-amber-500 font-mono font-bold bg-amber-500/5 border-l border-amber-500/10">
                                        {uncs[i] ? (
                                            Array.isArray(uncs[i]) && uncs[i].length === 1 ? (
                                                typeof uncs[i][0] === 'number' ? uncs[i][0].toFixed(6) : JSON.stringify(uncs[i][0])
                                            ) : (
                                                typeof uncs[i] === 'number' ? uncs[i].toFixed(6) : JSON.stringify(uncs[i])
                                            )
                                        ) : "—"}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div className="mt-4 flex justify-between items-center text-[10px] text-neutral-500 px-2">
                    <p>Showing {numRows} samples × {numFeatures} features</p>
                    <p>Dataset Source: {associatedDataset?.name || "Unknown"}</p>
                </div>
            </div>
        );
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm animate-in fade-in duration-300">
            <div className="w-full max-w-6xl max-h-[90vh] flex flex-col bg-neutral-900 border border-neutral-800 rounded-[2rem] overflow-hidden shadow-[0_0_50px_-12px_rgba(79,70,229,0.2)]">
                <div className="p-8 pb-6 border-b border-neutral-800 flex items-start justify-between bg-gradient-to-b from-white/[0.02] to-transparent">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <div className="p-2.5 bg-indigo-500/10 rounded-xl">
                                <Table2 className="w-6 h-6 text-indigo-400" />
                            </div>
                            <h2 className="text-2xl font-bold tracking-tight">Tabular Inference Results</h2>
                        </div>
                        <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs mt-4">
                            <div className="flex items-center gap-2">
                                <span className="text-neutral-500 uppercase font-bold tracking-widest text-[9px]">ID</span>
                                <span className="font-mono text-indigo-300 bg-indigo-500/10 px-2 py-0.5 rounded-md">{prediction.id}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-neutral-500 uppercase font-bold tracking-widest text-[9px]">Method</span>
                                <span className="px-2 py-0.5 bg-neutral-800 rounded-md text-indigo-100 font-bold border border-neutral-700">{prediction.method.toUpperCase()}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-neutral-500 uppercase font-bold tracking-widest text-[9px]">Status</span>
                                <div className="flex items-center gap-1.5 px-2 py-0.5 bg-emerald-500/10 rounded-md border border-emerald-500/20">
                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                                    <span className="text-emerald-400 font-bold lowercase">{prediction.status}</span>
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

                <div className="p-8 overflow-y-auto flex-1 bg-black/20">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-32 gap-6">
                            <div className="relative">
                                <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full"></div>
                                <Loader2 className="w-12 h-12 animate-spin text-indigo-500 relative z-10" />
                            </div>
                            <div className="text-center">
                                <p className="text-white font-medium text-lg">Retrieving tabular data</p>
                                <p className="text-neutral-500 text-sm mt-1 animate-pulse">Synchronizing with cloud storage...</p>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center py-20 gap-6 bg-red-500/5 rounded-3xl border border-red-500/10">
                            <XCircle className="w-16 h-16 text-red-500/50" />
                            <div className="text-center">
                                <p className="text-red-400 font-bold text-lg">Error Loading Data</p>
                                <p className="text-neutral-400 text-sm mt-1 max-w-sm mx-auto">{error}</p>
                            </div>
                            <Button onClick={onClose} variant="ghost" className="text-neutral-400 hover:text-white">
                                Dismiss and Close
                            </Button>
                        </div>
                    ) : (
                        renderTable()
                    )}
                </div>

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


export default function DashboardMVP() {
    const { token, user, tenant, setTenant, logout } = useAuth();

    const [activeView, setActiveView] = useState<View>('datasets');
    const [tenantName, setTenantName] = useState("Loading...");

    // Data states
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [models, setModels] = useState<MLModel[]>([]);
    const [predictions, setPredictions] = useState<Prediction[]>([]);

    // Modals
    const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);
    const [isModelModalOpen, setIsModelModalOpen] = useState(false);
    const [isTenantModalOpen, setIsTenantModalOpen] = useState(false);
    const [selectedPredictionView, setSelectedPredictionView] = useState<Prediction | null>(null);

    // Preview / Delete state (Fase 1)
    const [previewData, setPreviewData] = useState<DatasetPreview | null>(null);
    const [deletingDatasetId, setDeletingDatasetId] = useState<string | null>(null);

    // Tenant Switcher State
    const [userTenants, setUserTenants] = useState<any[]>([]);
    const [isSwitchingTenant, setIsSwitchingTenant] = useState(false);

    // Loading states
    const [isLoading, setIsLoading] = useState(true);
    const [isActionLoading, setIsActionLoading] = useState(false);

    // Selection for Inference
    const [selectedDataset, setSelectedDataset] = useState<string>('');
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [uncertaintyMethod, setUncertaintyMethod] = useState('none');

    // Training state (Fase 2)
    const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
    const [trainAlgorithm, setTrainAlgorithm] = useState<string>('');
    const [trainDataset, setTrainDataset] = useState<string>('');
    const [trainTarget, setTrainTarget] = useState<string>('');
    const [trainTaskType, setTrainTaskType] = useState<string>('classification');
    const [trainHyperparams, setTrainHyperparams] = useState<Record<string, any>>({});
    const [trainModelName, setTrainModelName] = useState<string>('');
    const [trainingTaskId, setTrainingTaskId] = useState<string | null>(null);
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
    const [isTraining, setIsTraining] = useState(false);
    // Validation strategy
    const [validationStrategy, setValidationStrategy] = useState<'holdout' | 'cross_validation'>('holdout');
    const [testSize, setTestSize] = useState<number>(0.2);
    const [nFolds, setNFolds] = useState<number>(5);

    useEffect(() => {
        loadInitialData();
        // Load algorithms for training tab
        fetch('http://localhost:8000/api/v1/training/algorithms')
            .then(r => r.json())
            .then(algos => { if (Array.isArray(algos)) setAlgorithms(algos); })
            .catch(() => { });
    }, []);

    const loadInitialData = async () => {
        setIsLoading(true);
        try {
            if (!token) return;
            const headers = { Authorization: `Bearer ${token}` };

            // Fetch tenants
            const tenantsRes = await fetch('http://localhost:8000/api/v1/tenants/my_tenants', { headers });
            if (tenantsRes.ok) {
                const fetchedTenants = await tenantsRes.json();
                setUserTenants(fetchedTenants);
                if (fetchedTenants.length === 0) {
                    setIsTenantModalOpen(true);
                }
            }

            const [ds, mds, preds] = await Promise.all([
                fetch('http://localhost:8000/api/v1/datasets/', { headers }).then(r => r.json()),
                fetch('http://localhost:8000/api/v1/models/', { headers }).then(r => r.json()),
                fetch('http://localhost:8000/api/v1/predictions', { headers }).then(r => r.json())
            ]);

            setDatasets(Array.isArray(ds) ? ds : []);
            setModels(Array.isArray(mds) ? mds : []);
            setPredictions(Array.isArray(preds) ? preds : []);

            if (ds.length > 0) setSelectedDataset(ds[0].id);
            if (mds.length > 0) setSelectedModel(mds[0].id);
        } catch (error) {
            console.error("Failed to load data", error);
        } finally {
            setIsLoading(false);
        }
    };

    // Poll for pending predictions
    useEffect(() => {
        if (!token || !predictions.some(p => ['PENDING', 'STARTED', 'IN_PROGRESS', 'RUNNING'].includes(p.status))) return;

        const interval = setInterval(async () => {
            try {
                const updatedPreds = await fetch('http://localhost:8000/api/v1/predictions', {
                    headers: { Authorization: `Bearer ${token}` }
                }).then(r => r.json());

                if (Array.isArray(updatedPreds)) {
                    setPredictions(prevPreds => {
                        // Check for newly failed or succeeded predictions
                        updatedPreds.forEach(updated => {
                            const prev = prevPreds.find(p => p.id === updated.id);
                            if (prev && prev.status !== updated.status) {
                                if (updated.status === 'COMPLETED') {
                                    toast.success(`Inference ${updated.id.substring(0, 8)} completed successfully`);
                                    // Automatically open the modal with the results
                                    setSelectedPredictionView(updated);
                                } else if (updated.status === 'FAILED') {
                                    toast.error(`Inference ${updated.id.substring(0, 8)} failed! Check backend logs.`);
                                }
                            }
                        });
                        return updatedPreds;
                    });
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 3000);

        return () => clearInterval(interval);
    }, [predictions, token]);

    const handleDatasetUpload = async (formData: FormData) => {
        if (!token) return;
        await fetch('http://localhost:8000/api/v1/datasets/', {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            body: formData,
        });
        loadInitialData();
    };

    const handleDeleteDataset = async (datasetId: string) => {
        if (!token) return;
        try {
            await fetch(`http://localhost:8000/api/v1/datasets/${datasetId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` },
            });
            toast.success('Dataset deleted successfully');
            setDeletingDatasetId(null);
            loadInitialData();
        } catch (err) {
            toast.error('Failed to delete dataset');
        }
    };

    const handleDeleteModel = async (modelId: string) => {
        if (!token || !tenant) return false;
        try {
            const res = await fetch(`http://localhost:8000/api/v1/models/${modelId}?tenant_id=${tenant.id}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!res.ok) throw new Error("Failed to delete model");
            toast.success('Model deleted successfully');
            loadInitialData();
            return true;
        } catch (err) {
            toast.error('Failed to delete model');
            return false;
        }
    };

    const handlePreviewDataset = async (datasetId: string) => {
        if (!token) return;
        try {
            const res = await fetch(`http://localhost:8000/api/v1/datasets/${datasetId}/preview`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!res.ok) throw new Error('Preview failed');
            const data = await res.json();
            setPreviewData(data);
        } catch (err: any) {
            toast.error(err.message || 'Failed to preview dataset');
        }
    };

    const handleTenantCreate = async (formData: FormData) => {
        if (!token) return;
        try {
            const res = await fetch('http://localhost:8000/api/v1/tenants/', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name: formData.get('name') }),
            });
            if (res.ok) {
                const newTenant = await res.json();
                toast.success("Organization created successfully! Switch to it using the sidebar.");
                if (userTenants.length === 0) {
                    setTenant(newTenant);
                }
                loadInitialData();
                setIsTenantModalOpen(false);
            } else {
                const data = await res.json();
                toast.error(data.detail || "Failed to create organization");
            }
        } catch (error: any) {
            toast.error("An error occurred creating the tenant.");
        }
    };

    const handleModelUpload = async (formData: FormData) => {
        if (!token) return;
        await fetch('http://localhost:8000/api/v1/models/upload', {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            body: formData,
        });
        loadInitialData();
    };

    const handleRunInference = async (batchFile?: File) => {
        if ((!selectedDataset && !batchFile) || !selectedModel || !token) return;
        setIsActionLoading(true);
        try {
            let res;
            if (batchFile) {
                const formData = new FormData();
                formData.append('file', batchFile);
                formData.append('model_id', selectedModel);
                formData.append('uncertainty_method', uncertaintyMethod);

                res = await fetch('http://localhost:8000/api/v1/predictions/predict/batch', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`
                    },
                    body: formData
                });
            } else {
                res = await fetch('http://localhost:8000/api/v1/predictions/predict', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        dataset_id: selectedDataset,
                        model_id: selectedModel,
                        uncertainty_method: uncertaintyMethod
                    })
                });
            }

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || "Error triggering inference.");
            }

            toast.success("Batch inference task enqueued!");
            setActiveView('predictions');
            // Refresh predictions list
            const updatedPreds = await fetch('http://localhost:8000/api/v1/predictions', {
                headers: { Authorization: `Bearer ${token}` }
            }).then(r => r.json());

            setPredictions(Array.isArray(updatedPreds) ? updatedPreds : []);
        } catch (error: any) {
            toast.error(error.message || "Error triggering inference.");
            console.error(error);
        } finally {
            setIsActionLoading(false);
        }
    };

    const fileTypeBadgeColor = (ft?: string) => {
        switch (ft) {
            case 'csv': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
            case 'xlsx': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
            case 'parquet': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
            case 'zip': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
            default: return 'bg-neutral-700 text-neutral-300';
        }
    };

    // ── Training polling ────────────────────────────────────────────────────
    useEffect(() => {
        if (!trainingTaskId) return;
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`http://localhost:8000/api/v1/training/status/${trainingTaskId}`);
                const data = await res.json();
                setTrainingStatus(data);
                if (data.status === 'SUCCESS') {
                    toast.success('Model training completed!');
                    setIsTraining(false);
                    setTrainingTaskId(null);
                    loadInitialData();
                } else if (data.status === 'FAILURE') {
                    toast.error(`Training failed: ${data.error || 'Unknown error'}`);
                    setIsTraining(false);
                    setTrainingTaskId(null);
                }
            } catch (err) { console.error('Training poll error', err); }
        }, 3000);
        return () => clearInterval(interval);
    }, [trainingTaskId]);

    const handleStartTraining = async () => {
        if (!trainDataset || !trainTarget || !trainAlgorithm || !token) return;
        setIsTraining(true);
        setTrainingStatus(null);
        try {
            const res = await fetch('http://localhost:8000/api/v1/training/train', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: trainDataset,
                    target_column: trainTarget,
                    algorithm: trainAlgorithm,
                    task_type: trainTaskType,
                    hyperparams: trainHyperparams,
                    validation: {
                        strategy: validationStrategy,
                        test_size: testSize,
                        n_folds: nFolds,
                        shuffle: true,
                        random_state: 42,
                    },
                    model_name: trainModelName || undefined,
                }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Training failed');
            }
            const data = await res.json();
            setTrainingTaskId(data.task_id);
            toast.success('Training task launched!');
        } catch (err: any) {
            toast.error(err.message || 'Failed to start training');
            setIsTraining(false);
        }
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-screen bg-neutral-900 text-white">
                <Loader2 className="w-10 h-10 animate-spin text-indigo-500" />
            </div>
        );
    }

    return (
        <div className="flex h-screen bg-neutral-950 text-white font-sans selection:bg-indigo-500/30">
            {/* Sidebar */}
            <aside className="w-72 border-r border-neutral-900 bg-neutral-950 p-8 flex flex-col">
                <div className="flex items-center gap-3 mb-12">
                    <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
                        <BarChart3 className="w-5 h-5" />
                    </div>
                    <h1 className="text-xl font-bold tracking-tight">MegSeg</h1>
                </div>

                <nav className="space-y-1.5 flex-1">
                    <button
                        onClick={() => setActiveView('datasets')}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 ${activeView === 'datasets' ? 'bg-indigo-600 text-white' : 'text-neutral-400 hover:text-white hover:bg-neutral-900'}`}
                    >
                        <Database className="w-5 h-5" /> Datasets
                    </button>
                    <button
                        onClick={() => setActiveView('preprocessing')}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 ${activeView === 'preprocessing' ? 'bg-indigo-600 text-white' : 'text-neutral-400 hover:text-white hover:bg-neutral-900'}`}
                    >
                        <Filter className="w-5 h-5" /> Preprocessing
                    </button>
                    <button
                        onClick={() => setActiveView('models')}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 ${activeView === 'models' ? 'bg-indigo-600 text-white' : 'text-neutral-400 hover:text-white hover:bg-neutral-900'}`}
                    >
                        <Cpu className="w-5 h-5" /> Models
                    </button>
                    <button
                        onClick={() => setActiveView('predictions')}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 ${activeView === 'predictions' ? 'bg-indigo-600 text-white' : 'text-neutral-400 hover:text-white hover:bg-neutral-900'}`}
                    >
                        <Clock className="w-5 h-5" /> Predictions
                    </button>
                    <button
                        onClick={() => setActiveView('training')}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 ${activeView === 'training' ? 'bg-indigo-600 text-white' : 'text-neutral-400 hover:text-white hover:bg-neutral-900'}`}
                    >
                        <FlaskConical className="w-5 h-5" /> Training
                    </button>
                </nav>

                <div className="mt-auto pt-6 border-t border-neutral-900">
                    <div className="flex flex-col gap-2 p-3 bg-neutral-900/50 rounded-2xl relative">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-sm">
                                {tenant?.name ? tenant.name.substring(0, 2).toUpperCase() : tenantName.substring(0, 2).toUpperCase()}
                            </div>
                            <div className="flex-1 overflow-hidden">
                                <p className="text-sm font-medium truncate">{tenant?.name || tenantName}</p>
                                <p className="text-[10px] text-neutral-500">Free Tier</p>
                            </div>
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 w-8 p-0 hover:bg-neutral-800 text-neutral-400"
                                onClick={() => setIsSwitchingTenant(!isSwitchingTenant)}
                            >
                                <Settings className="w-4 h-4" />
                            </Button>
                        </div>

                        {/* Tenant Switcher Dropdown UI integrated into the sidebar */}
                        {isSwitchingTenant && (
                            <div className="mt-2 pt-2 border-t border-neutral-800 space-y-1">
                                <span className="text-[10px] font-bold text-neutral-500 uppercase ml-1">Switch Organization</span>
                                {userTenants.map(t => (
                                    <button
                                        key={t.id}
                                        className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors ${t.id === tenant?.id ? 'bg-indigo-500/10 text-indigo-400' : 'text-neutral-400 hover:bg-neutral-800 hover:text-white'}`}
                                        onClick={() => {
                                            toast.error("To switch tenant, you must re-login and select it!"); // TODO: Implement token reissue for new tenant
                                        }}
                                    >
                                        {t.name}
                                    </button>
                                ))}
                                <Button
                                    className="w-full mt-2 bg-neutral-800 hover:bg-neutral-700 text-xs"
                                    size="sm"
                                    onClick={() => {
                                        setIsSwitchingTenant(false);
                                        setIsTenantModalOpen(true);
                                    }}
                                >
                                    <Plus className="w-3 h-3 mr-1" /> New Organization
                                </Button>
                            </div>
                        )}

                        <Button
                            variant="destructive"
                            size="sm"
                            className="w-full mt-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20"
                            onClick={logout}
                        >
                            Log Out
                        </Button>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 p-12 overflow-auto bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-indigo-500/5 via-transparent to-transparent">
                <header className="mb-12 flex justify-between items-end">
                    <div>
                        <div className="flex items-center gap-2 text-indigo-400 mb-1">
                            <div className="w-1 h-1 rounded-full bg-indigo-400"></div>
                            <span className="text-[10px] font-bold uppercase tracking-widest">Active Workspace</span>
                        </div>
                        <h2 className="text-4xl font-bold tracking-tight capitalize">{activeView}</h2>
                    </div>
                </header>

                <div className="max-w-6xl">
                    {activeView === 'datasets' && (
                        <DatasetsTab
                            datasets={datasets}
                            previewData={previewData}
                            deletingDatasetId={deletingDatasetId}
                            setIsDatasetModalOpen={setIsDatasetModalOpen}
                            setDeletingDatasetId={setDeletingDatasetId}
                            handleDeleteDataset={handleDeleteDataset}
                            handlePreviewDataset={handlePreviewDataset}
                            setPreviewData={setPreviewData}
                            fileTypeBadgeColor={fileTypeBadgeColor}
                            tenantId={tenant?.id || ''}
                        />
                    )}
                    {activeView === 'preprocessing' && (
                        <PreprocessingTab
                            datasets={datasets}
                            tenantId={tenant?.id || ''}
                            onPreprocessingApplied={() => {
                                loadInitialData(); // Refresh datasets to show the new preprocessed version
                                setActiveView('datasets');
                            }}
                        />
                    )}
                    {activeView === 'models' && (
                        <ModelsTab
                            models={models}
                            setIsModelModalOpen={setIsModelModalOpen}
                            handleDeleteModel={handleDeleteModel}
                        />
                    )}
                    {activeView === 'predictions' && (
                        <PredictionsTab
                            predictions={predictions}
                            models={models}
                            datasets={datasets}
                            token={token}
                            setPredictions={setPredictions}
                            setSelectedPredictionView={setSelectedPredictionView}
                            selectedDataset={selectedDataset}
                            setSelectedDataset={setSelectedDataset}
                            selectedModel={selectedModel}
                            setSelectedModel={setSelectedModel}
                            uncertaintyMethod={uncertaintyMethod}
                            setUncertaintyMethod={setUncertaintyMethod}
                            isActionLoading={isActionLoading}
                            handleRunInference={handleRunInference}
                        />
                    )}
                    {activeView === 'training' && (
                        <TrainingTab
                            datasets={datasets}
                            algorithms={algorithms}
                            trainDataset={trainDataset}
                            setTrainDataset={setTrainDataset}
                            trainTarget={trainTarget}
                            setTrainTarget={setTrainTarget}
                            trainTaskType={trainTaskType}
                            setTrainTaskType={setTrainTaskType}
                            trainAlgorithm={trainAlgorithm}
                            setTrainAlgorithm={setTrainAlgorithm}
                            trainHyperparams={trainHyperparams}
                            setTrainHyperparams={setTrainHyperparams}
                            trainModelName={trainModelName}
                            setTrainModelName={setTrainModelName}
                            validationStrategy={validationStrategy}
                            setValidationStrategy={setValidationStrategy}
                            testSize={testSize}
                            setTestSize={setTestSize}
                            nFolds={nFolds}
                            setNFolds={setNFolds}
                            isTraining={isTraining}
                            trainingStatus={trainingStatus}
                            handleStartTraining={handleStartTraining}
                        />
                    )}


                </div>
            </main>

            {/* Modals */}
            <UploadModal
                isOpen={isDatasetModalOpen}
                onClose={() => setIsDatasetModalOpen(false)}
                title="Upload Dataset"
                description="Upload tabular data (.csv, .xlsx, .parquet) or image archives (.zip)."
                fileAccept=".csv,.xlsx,.parquet,.zip"
                onUpload={handleDatasetUpload}
                fields={[
                    { name: 'name', label: 'Dataset Name', type: 'text', placeholder: 'e.g. Brain MRI 2024', required: true },
                    { name: 'description', label: 'Description', type: 'textarea', placeholder: 'Describe the contents of the dataset' }
                ]}
            />

            <UploadModal
                isOpen={isModelModalOpen}
                onClose={() => setIsModelModalOpen(false)}
                title="Register ML Model"
                description="Upload a PyTorch (.pth) file to register it in MLFlow."
                fileAccept=".pth"
                onUpload={handleModelUpload}
                fields={[
                    { name: 'name', label: 'Model Name', type: 'text', placeholder: 'e.g. UNet Segmentation v1', required: true },
                    { name: 'description', label: 'Description', type: 'textarea' },
                    { name: 'architecture', label: 'Architecture', type: 'text', defaultValue: 'UNet' },
                    { name: 'num_classes', label: 'Classes', type: 'number', defaultValue: 2 },
                    {
                        name: 'is_public',
                        label: 'Visibility',
                        type: 'select',
                        options: [{ label: 'Private (Only you)', value: 'false' }, { label: 'Public (Everyone)', value: 'true' }],
                        defaultValue: 'false'
                    }
                ]}
            />

            <UploadModal
                isOpen={isTenantModalOpen}
                onClose={() => {
                    if (userTenants.length > 0) setIsTenantModalOpen(false);
                    else toast.error("You must create an organization to continue.");
                }}
                title={userTenants.length === 0 ? "Welcome! Let's get started." : "Create New Organization"}
                description="Create an isolated Workspace/Organization to manage datasets and models."
                onUpload={handleTenantCreate}
                requireFile={false}
                hideCloseButton={userTenants.length === 0}
                fields={[
                    { name: 'name', label: 'Organization Name', type: 'text', placeholder: 'e.g. Acme MedCorp', required: true }
                ]}
            />

            {/* Prediction Results Modal */}
            {selectedPredictionView && (
                <PredictionResultsModal
                    prediction={selectedPredictionView}
                    datasets={datasets}
                    models={models}
                    token={token}
                    onClose={() => setSelectedPredictionView(null)}
                />
            )}
        </div>
    );
}
