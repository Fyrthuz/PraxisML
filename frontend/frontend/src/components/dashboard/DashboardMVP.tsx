"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Loader2 } from "lucide-react";
import toast from "react-hot-toast";

import { useAuth } from "@/components/AuthContext";
import Sidebar from "./Sidebar";
import UploadModal from "./UploadModal";
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

import { useDatasets } from "@/hooks/useDatasets";
import { useModels } from "@/hooks/useModels";
import { usePredictions } from "@/hooks/usePredictions";
import { useTraining } from "@/hooks/useTraining";

import { Prediction } from "@/lib/api";
import { config } from '@/lib/config';

type View = "datasets" | "preprocessing" | "models" | "predictions" | "training" | "registry" | "data_registry" | "streaming" | "users";

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

    const [activeView, setActiveView] = useState<View>("datasets");
    const [isLoading, setIsLoading] = useState(true);
    const [isTenantModalOpen, setIsTenantModalOpen] = useState(false);
    const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);
    const [isModelModalOpen, setIsModelModalOpen] = useState(false);
    const [isSwitchingTenant, setIsSwitchingTenant] = useState(false);
    const [userTenants, setUserTenants] = useState<any[]>([]);
    const [selectedPredictionView, setSelectedPredictionView] = useState<Prediction | null>(null);

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
            <div className="flex items-center justify-center h-screen bg-neutral-900 text-white">
                <Loader2 className="w-10 h-10 animate-spin text-indigo-500" />
            </div>
        );
    }

    const tenantName = tenant?.name || "My Organization";
    const tenantInitials = tenantName.substring(0, 2).toUpperCase();

    return (
        <div className="flex h-screen bg-neutral-950 text-white font-sans selection:bg-indigo-500/30">
            {/* ── Sidebar ── */}
            <Sidebar
                activeView={activeView}
                onNavigate={setActiveView}
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
            />

            {/* ── Main ── */}
            <main className="flex-1 p-12 overflow-auto bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-indigo-500/5 via-transparent to-transparent">
                <header className="mb-12 flex justify-between items-end">
                    <div>
                        <div className="flex items-center gap-2 text-indigo-400 mb-1">
                            <div className="w-1 h-1 rounded-full bg-indigo-400" />
                            <span className="text-[10px] font-bold uppercase tracking-widest">
                                Active Workspace
                            </span>
                        </div>
                        <h2 className="text-4xl font-bold tracking-tight capitalize">
                            {activeView}
                        </h2>
                    </div>
                </header>

                <div className="max-w-6xl">
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
                        />
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
                        />
                    )}

                    {activeView === "data_registry" && (
                        <DataRegistryTab
                            tenantId={tenant?.id}
                            token={token}
                            onRefresh={() => datasetHook.fetchDatasets()}
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
                </div>
            </main>

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
        </div>
    );
}
