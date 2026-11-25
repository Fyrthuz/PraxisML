"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Loader2, Plus, RefreshCw, Search } from "lucide-react";
import toast from "react-hot-toast";

import { useAuth } from "@/components/AuthContext";
import Sidebar, { View } from "./Sidebar";
import UploadModal from "./UploadModal";
import CommandPalette from "./CommandPalette";
import { Button } from "@/components/ui/button";
import WorkspaceOverviewTab from "./tabs/WorkspaceOverviewTab";
import CopilotStudioTab from "./tabs/CopilotStudioTab";
import DatasetsTab from "./tabs/DatasetsTab";
import ModelsTab from "./tabs/ModelsTab";
import RegistryTab from "./tabs/RegistryTab";
import DataRegistryTab from "./tabs/DataRegistryTab";
import PredictionsTab from "./tabs/PredictionsTab";
import TrainingTab from "./tabs/TrainingTab";
import PreprocessingTab from "./tabs/PreprocessingTab";
import UsersTab from "./tabs/UsersTab";
import PredictionResultsModal from "@/components/modals/PredictionResultsModal";
import StreamingInference from "./StreamingInference";
import ObservabilityTab from "./tabs/ObservabilityTab";
import NLPTab from "./tabs/NLPTab";
import ServingEndpointsTab from "./tabs/ServingEndpointsTab";
import GovernanceTab from "./tabs/GovernanceTab";
import EnvironmentsPipelinesTab from "./tabs/EnvironmentsPipelinesTab";

import { useDatasets } from "@/hooks/useDatasets";
import { useModels } from "@/hooks/useModels";
import { usePredictions } from "@/hooks/usePredictions";
import { useTraining } from "@/hooks/useTraining";

import { Prediction } from "@/lib/api";
import { config } from '@/lib/config';

const VIEW_META: Record<View, { title: string; subtitle: string; section: string }> = {
    overview: {
        title: "Workspace Overview",
        subtitle: "End-to-end overview of your datasets, models, predictions and active domain plugins.",
        section: "Workspace",
    },
    datasets: {
        title: "Datasets",
        subtitle: "Ingest, preview, profile and monitor data drift across CSV, Parquet, Excel and ZIP files.",
        section: "Data & Catalog",
    },
    data_registry: {
        title: "Data Registry",
        subtitle: "Track dataset versions, DVC hashes and production data promotions.",
        section: "Data & Catalog",
    },
    preprocessing: {
        title: "Preprocessing Pipelines",
        subtitle: "Design declarative ColumnTransformer pipelines and generate cleaned dataset versions.",
        section: "Data & Catalog",
    },
    environments: {
        title: "Runtime Environments & Custom Pipelines",
        subtitle: "Define isolated Python dependency environments and run custom user pipelines across datasets.",
        section: "Data & Catalog",
    },
    training: {
        title: "Tabular Training",
        subtitle: "Train Scikit-learn and PyTorch MLP models asynchronously with MLflow tracking.",
        section: "ML & AI Studio",
    },
    nlp: {
        title: "NLP Studio",
        subtitle: "Fine-tune Transformers (ONNX), run semantic search (pgvector) and chat with the RAG assistant.",
        section: "ML & AI Studio",
    },
    models: {
        title: "Models",
        subtitle: "Manage trained and uploaded models, download artifacts and inspect run metrics.",
        section: "ML & AI Studio",
    },
    registry: {
        title: "Model Registry",
        subtitle: "Promote model versions across Staging, Production and Archived stages in MLflow.",
        section: "ML & AI Studio",
    },
    endpoints: {
        title: "Model Endpoints",
        subtitle: "Publish any registered model as a dedicated REST endpoint with API keys and Canary routing.",
        section: "Serving & Inference",
    },
    predictions: {
        title: "Predictions",
        subtitle: "Run batch or single-row inference with uncertainty quantification and SHAP explainability.",
        section: "Serving & Inference",
    },
    streaming: {
        title: "Real-Time Streaming",
        subtitle: "Low-latency continuous inference over WebSockets.",
        section: "Serving & Inference",
    },
    observability: {
        title: "Observability & FinOps",
        subtitle: "Monitor usage latency, dynamic resource pricing, monthly cost forecasts and alerts.",
        section: "Operations & Governance",
    },
    copilot: {
        title: "Copilot & Plugins",
        subtitle: "Inspect active modular domain plugins and RBAC-governed agent tools.",
        section: "Operations & Governance",
    },
    governance: {
        title: "API Keys, Workspaces & Audit",
        subtitle: "Manage scoped Personal Access Tokens, environment workspaces, audit trail and Redis Streams events.",
        section: "Operations & Governance",
    },
    users: {
        title: "Users & RBAC",
        subtitle: "Manage organization members and admin / editor / viewer access permissions.",
        section: "Operations & Governance",
    },
};

const API = config.getFullApiUrl("");

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fileTypeBadgeColor(ft?: string) {
    switch (ft) {
        case "csv": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
        case "xlsx": return "bg-blue-500/20 text-blue-400 border-blue-500/30";
        case "parquet": return "bg-orange-500/20 text-orange-400 border-orange-500/30";
        case "zip": return "bg-purple-500/20 text-purple-400 border-purple-500/30";
        default: return "bg-neutral-700 text-neutral-300";
    }
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export default function DashboardMVP() {
    const { token, tenant, setTenant, logout, userRole } = useAuth();

    const [activeView, setActiveView] = useState<View>("overview");
    const [isLoading, setIsLoading] = useState(true);
    const [isCommandOpen, setIsCommandOpen] = useState(false);
    const [deployTargetModelId, setDeployTargetModelId] = useState<string | null>(null);
    const [isTenantModalOpen, setIsTenantModalOpen] = useState(false);
    const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);
    const [isModelModalOpen, setIsModelModalOpen] = useState(false);
    const [isSwitchingTenant, setIsSwitchingTenant] = useState(false);
    const [userTenants, setUserTenants] = useState<any[]>([]);
    const [selectedPredictionView, setSelectedPredictionView] = useState<Prediction | null>(null);

    const navigateTo = useCallback((view: View) => {
        setActiveView(view);
        if (typeof window !== "undefined") {
            const url = new URL(window.location.href);
            url.searchParams.set("view", view);
            window.history.replaceState({}, "", url.toString());
        }
    }, []);

    // Sync initial view from URL (?view=...) and handle Cmd+K / Ctrl+K
    useEffect(() => {
        if (typeof window === "undefined") return;
        const params = new URLSearchParams(window.location.search);
        const initial = params.get("view") as View | null;
        if (initial && initial in VIEW_META) {
            setActiveView(initial);
        }
        const onPopState = () => {
            const next = new URLSearchParams(window.location.search).get("view") as View | null;
            if (next && next in VIEW_META) setActiveView(next);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
                e.preventDefault();
                setIsCommandOpen((v) => !v);
            }
        };
        window.addEventListener("popstate", onPopState);
        window.addEventListener("keydown", onKeyDown);
        return () => {
            window.removeEventListener("popstate", onPopState);
            window.removeEventListener("keydown", onKeyDown);
        };
    }, []);

    const handleDeployModel = useCallback(
        (modelId: string) => {
            setDeployTargetModelId(modelId);
            navigateTo("endpoints");
        },
        [navigateTo]
    );

    // ── Domain hooks ─────────────────────────────────────────────────────────
    const datasetHook = useDatasets(token);
    const modelHook = useModels(token);

    const handlePredictionComplete = useCallback(
        (prediction: Prediction) => setSelectedPredictionView(prediction),
        []
    );

    const predHook = usePredictions(token, handlePredictionComplete);

    const handleTrainingComplete = useCallback(async () => {
        await modelHook.fetchModels();
    }, [modelHook]);

    const trainingHook = useTraining(token, handleTrainingComplete);

    // ── Bootstrap ─────────────────────────────────────────────────────────────
    useEffect(() => {
        if (!token) { setIsLoading(false); return; }
        loadInitialData();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [token]);

    const loadInitialData = async () => {
        setIsLoading(true);
        try {
            const headers = { Authorization: `Bearer ${token}` };

            // Tenants
            const tenantsRes = await fetch(`${API}/tenants/my_tenants`, { headers });
            if (tenantsRes.ok) {
                const fetched = await tenantsRes.json();
                setUserTenants(fetched);
                if (fetched.length === 0) setIsTenantModalOpen(true);
            }

            // Parallel data fetch
            await Promise.all([
                datasetHook.fetchDatasets(),
                modelHook.fetchModels(),
                predHook.fetchPredictions(),
            ]);
        } catch (err) {
            console.error("Failed to load data", err);
        } finally {
            setIsLoading(false);
        }
    };

    // ── Tenant handlers ───────────────────────────────────────────────────────
    const handleTenantCreate = async (formData: FormData) => {
        if (!token) return;
        try {
            const res = await fetch(`${API}/tenants/`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ name: formData.get("name") }),
            });
            if (res.ok) {
                const newTenant = await res.json();
                toast.success("Organization created successfully!");
                if (userTenants.length === 0) setTenant(newTenant);
                await loadInitialData();
                setIsTenantModalOpen(false);
            } else {
                const data = await res.json();
                toast.error(data.detail || "Failed to create organization");
            }
        } catch {
            toast.error("An error occurred creating the tenant.");
        }
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-screen bg-[#0b0c0f] text-zinc-200">
                <Loader2 className="w-6 h-6 animate-spin text-zinc-400" />
            </div>
        );
    }

    const tenantName = tenant?.name || "My Organization";
    const tenantInitials = tenantName.substring(0, 2).toUpperCase();

    return (
        <div className="flex h-screen bg-[#0b0c0f] text-zinc-100 font-sans selection:bg-orange-500/25">
            {/* Sidebar */}
            <Sidebar
                activeView={activeView}
                onNavigate={navigateTo}
                tenantName={tenantName}
                tenantInitials={tenantInitials}
                userTenants={userTenants}
                isSwitchingTenant={isSwitchingTenant}
                onToggleSwitchTenant={() => setIsSwitchingTenant((v) => !v)}
                onCreateTenant={() => {
                    setIsSwitchingTenant(false);
                    setIsTenantModalOpen(true);
                }}
                currentTenantId={tenant?.id}
                onLogout={logout}
                userRole={userRole}
                counts={{
                    datasets: datasetHook.datasets.length,
                    models: modelHook.models.length,
                    predictions: predHook.predictions.length,
                }}
            />

            {/* Main workspace */}
            <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
                <header className="h-14 shrink-0 px-6 border-b border-zinc-800/90 bg-[#0b0c0f] flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2 min-w-0">
                        <span className="text-xs text-zinc-500 hidden sm:inline">
                            {VIEW_META[activeView].section}
                        </span>
                        <span className="text-xs text-zinc-700 hidden sm:inline">/</span>
                        <h2 className="text-sm font-semibold text-zinc-100 truncate">
                            {VIEW_META[activeView].title}
                        </h2>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                        <button
                            type="button"
                            onClick={() => setIsCommandOpen(true)}
                            className="h-8 px-2.5 rounded-md border border-zinc-800 bg-[#14171e] text-zinc-400 hover:text-zinc-200 hover:border-zinc-700 text-xs flex items-center gap-2"
                        >
                            <Search className="w-3.5 h-3.5" />
                            <span className="hidden md:inline">Jump to...</span>
                            <kbd className="text-[10px] font-mono px-1 py-0.5 rounded bg-[#0b0c0f] border border-zinc-800 text-zinc-500">
                                ⌘K
                            </kbd>
                        </button>
                        <Button
                            variant="outline"
                            size="sm"
                            className="h-8 border-zinc-800 bg-[#14171e] text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 text-xs rounded-md"
                            onClick={loadInitialData}
                        >
                            <RefreshCw className="w-3.5 h-3.5 mr-1.5" /> Refresh
                        </Button>
                        {userRole !== "viewer" && (
                            <Button
                                size="sm"
                                className="h-8 bg-zinc-100 hover:bg-white text-zinc-950 font-medium text-xs rounded-md"
                                onClick={() => setIsDatasetModalOpen(true)}
                            >
                                <Plus className="w-3.5 h-3.5 mr-1.5" /> Upload dataset
                            </Button>
                        )}
                    </div>
                </header>

                <main className="flex-1 px-6 py-6 overflow-auto">
                    <p className="text-xs text-zinc-500 mb-5">
                        {VIEW_META[activeView].subtitle}
                    </p>
                    <div className="max-w-6xl">
                    {activeView === "overview" && (
                        <WorkspaceOverviewTab
                            datasets={datasetHook.datasets}
                            models={modelHook.models}
                            predictions={predHook.predictions}
                            token={token}
                            onNavigate={navigateTo}
                            onOpenUploadDataset={() => setIsDatasetModalOpen(true)}
                        />
                    )}

                    {activeView === "copilot" && (
                        <CopilotStudioTab token={token} onNavigate={navigateTo} />
                    )}
                    {activeView === "datasets" && (
                        <DatasetsTab
                            datasets={datasetHook.datasets}
                            previewData={datasetHook.previewData}
                            deletingDatasetId={datasetHook.deletingDatasetId}
                            setIsDatasetModalOpen={setIsDatasetModalOpen}
                            setDeletingDatasetId={datasetHook.setDeletingDatasetId}
                            handleDeleteDataset={datasetHook.deleteDataset}
                            handlePreviewDataset={datasetHook.previewDataset}
                            setPreviewData={datasetHook.setPreviewData}
                            fileTypeBadgeColor={fileTypeBadgeColor}
                            tenantId={tenant?.id || ""}
                            token={token}
                        />
                    )}

                    {activeView === "preprocessing" && (
                        <PreprocessingTab
                            datasets={datasetHook.datasets}
                            tenantId={tenant?.id || ""}
                            onPreprocessingApplied={async () => {
                                await datasetHook.fetchDatasets();
                                setActiveView("datasets");
                            }}
                        />
                    )}

                    {activeView === "models" && (
                        <ModelsTab
                            models={modelHook.models}
                            setIsModelModalOpen={setIsModelModalOpen}
                            handleDeleteModel={(id) =>
                                modelHook.deleteModel(id, tenant?.id || "")
                            }
                            token={token}
                            onRefresh={() => modelHook.fetchModels()}
                            onDeployEndpoint={handleDeployModel}
                        />
                    )}

                    {activeView === "environments" && (
                        <EnvironmentsPipelinesTab
                            token={token}
                            datasets={datasetHook.datasets}
                            userRole={userRole}
                            onDatasetCreated={() => datasetHook.fetchDatasets()}
                        />
                    )}

                    {activeView === "endpoints" && (
                        <ServingEndpointsTab
                            token={token}
                            models={modelHook.models}
                            datasets={datasetHook.datasets}
                            userRole={userRole}
                            preselectedModelId={deployTargetModelId}
                            onClearPreselectedModel={() => setDeployTargetModelId(null)}
                        />
                    )}

                    {activeView === "governance" && (
                        <GovernanceTab token={token} userRole={userRole} />
                    )}

                    {activeView === "predictions" && (
                        <PredictionsTab
                            predictions={predHook.predictions}
                            models={modelHook.models}
                            datasets={datasetHook.datasets}
                            token={token}
                            setPredictions={predHook.setPredictions}
                            setSelectedPredictionView={setSelectedPredictionView}
                            selectedDataset={predHook.selectedDataset}
                            setSelectedDataset={predHook.setSelectedDataset}
                            selectedModel={predHook.selectedModel}
                            setSelectedModel={predHook.setSelectedModel}
                            uncertaintyMethod={predHook.uncertaintyMethod}
                            setUncertaintyMethod={predHook.setUncertaintyMethod}
                            isActionLoading={predHook.isActionLoading}
                            handleRunInference={predHook.runInference}
                        />
                    )}

                    {activeView === "training" && (
                        <TrainingTab
                            datasets={datasetHook.datasets}
                            algorithms={trainingHook.algorithms}
                            trainDataset={trainingHook.trainDataset}
                            setTrainDataset={trainingHook.setTrainDataset}
                            trainTarget={trainingHook.trainTarget}
                            setTrainTarget={trainingHook.setTrainTarget}
                            trainTaskType={trainingHook.trainTaskType}
                            setTrainTaskType={trainingHook.setTrainTaskType}
                            trainAlgorithm={trainingHook.trainAlgorithm}
                            setTrainAlgorithm={trainingHook.setTrainAlgorithm}
                            trainHyperparams={trainingHook.trainHyperparams}
                            setTrainHyperparams={trainingHook.setTrainHyperparams}
                            trainModelName={trainingHook.trainModelName}
                            setTrainModelName={trainingHook.setTrainModelName}
                            trainRegistryName={trainingHook.trainRegistryName}
                            setTrainRegistryName={trainingHook.setTrainRegistryName}
                            validationStrategy={trainingHook.validationStrategy}
                            setValidationStrategy={trainingHook.setValidationStrategy}
                            testSize={trainingHook.testSize}
                            setTestSize={trainingHook.setTestSize}
                            nFolds={trainingHook.nFolds}
                            setNFolds={trainingHook.setNFolds}
                            isTraining={trainingHook.isTraining}
                            trainingStatus={trainingHook.trainingStatus}
                            handleStartTraining={trainingHook.startTraining}
                        />
                    )}

                    {activeView === "registry" && (
                        <RegistryTab
                            token={token}
                            onRefresh={() => modelHook.fetchModels()}
                            models={modelHook.models}
                            onDeployEndpoint={handleDeployModel}
                        />
                    )}

                    {activeView === "data_registry" && (
                        <DataRegistryTab
                            tenantId={tenant?.id}
                            token={token}
                            onRefresh={() => datasetHook.fetchDatasets()}
                            onOpenDataset={(datasetId) => {
                                navigateTo("datasets");
                                datasetHook.previewDataset(datasetId);
                            }}
                            onTrainDataset={(datasetId) => {
                                trainingHook.setTrainDataset(datasetId);
                                navigateTo("training");
                            }}
                        />
                    )}

                    {activeView === "streaming" && (
                        <StreamingInference
                            models={modelHook.models}
                            token={token}
                        />
                    )}

                    {activeView === "users" && (
                        <UsersTab
                            token={token}
                            currentUserRole={userRole}
                            tenantId={tenant?.id || ""}
                            onRefreshUsers={() => {}}
                        />
                    )}

                    {activeView === "observability" && (
                        <ObservabilityTab token={token} />
                    )}

                    {activeView === "nlp" && (
                        <NLPTab
                            token={token}
                            datasets={datasetHook.datasets}
                            models={modelHook.models}
                        />
                    )}
                    </div>
                </main>
            </div>

            {/* ── Upload Modals ── */}
            <UploadModal
                isOpen={isDatasetModalOpen}
                onClose={() => setIsDatasetModalOpen(false)}
                title="Upload Dataset"
                description="Upload tabular data (.csv, .xlsx, .parquet) or image archives (.zip)."
                fileAccept=".csv,.xlsx,.parquet,.zip"
                onUpload={datasetHook.uploadDataset}
                fields={[
                    {
                        name: "name",
                        label: "Dataset Name",
                        type: "text",
                        placeholder: "e.g. Brain MRI 2024",
                        required: true,
                    },
                    {
                        name: "description",
                        label: "Description",
                        type: "textarea",
                        placeholder: "Describe the contents of the dataset",
                    },
                    {
                        name: "is_dvc_tracked",
                        label: "Track using DVC?",
                        type: "checkbox",
                        defaultValue: false,
                    },
                    {
                        name: "dvc_registry_name",
                        label: "DVC Registry Name",
                        type: "text",
                        placeholder: "e.g. medical_images_v1",
                    },
                ]}
            />

            <UploadModal
                isOpen={isModelModalOpen}
                onClose={() => setIsModelModalOpen(false)}
                title="Register ML Model"
                description="Upload a PyTorch (.pth) file to register it in MLFlow."
                fileAccept=".pth"
                onUpload={modelHook.uploadModel}
                fields={[
                    {
                        name: "name",
                        label: "Model Name",
                        type: "text",
                        placeholder: "e.g. UNet Segmentation v1",
                        required: true,
                    },
                    { name: "description", label: "Description", type: "textarea" },
                    { name: "architecture", label: "Architecture", type: "text", defaultValue: "UNet" },
                    { name: "num_classes", label: "Classes", type: "number", defaultValue: 2 },
                    {
                        name: "is_public",
                        label: "Visibility",
                        type: "select",
                        options: [
                            { label: "Private (Only you)", value: "false" },
                            { label: "Public (Everyone)", value: "true" },
                        ],
                        defaultValue: "false",
                    },
                ]}
            />

            <UploadModal
                isOpen={isTenantModalOpen}
                onClose={() => {
                    if (userTenants.length > 0) setIsTenantModalOpen(false);
                    else toast.error("You must create an organization to continue.");
                }}
                title={
                    userTenants.length === 0
                        ? "Welcome! Let's get started."
                        : "Create New Organization"
                }
                description="Create an isolated Workspace/Organization to manage datasets and models."
                onUpload={handleTenantCreate}
                requireFile={false}
                hideCloseButton={userTenants.length === 0}
                fields={[
                    {
                        name: "name",
                        label: "Organization Name",
                        type: "text",
                        placeholder: "e.g. Acme MedCorp",
                        required: true,
                    },
                ]}
            />

            {/* ── Prediction Results Modal ── */}
            {selectedPredictionView && (
                <PredictionResultsModal
                    prediction={selectedPredictionView}
                    datasets={datasetHook.datasets}
                    models={modelHook.models}
                    token={token}
                    onClose={() => setSelectedPredictionView(null)}
                />
            )}

            <CommandPalette
                isOpen={isCommandOpen}
                onClose={() => setIsCommandOpen(false)}
                onNavigate={navigateTo}
                onUploadDataset={() => setIsDatasetModalOpen(true)}
                onOpenDataset={(datasetId) => {
                    navigateTo("datasets");
                    datasetHook.previewDataset(datasetId);
                }}
                onDeployModel={handleDeployModel}
                datasets={datasetHook.datasets}
                models={modelHook.models}
                userRole={userRole}
            />
        </div>
    );
}
