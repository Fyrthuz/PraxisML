const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// Helper for authenticated requests
async function fetchAuth(url: string, options: RequestInit = {}) {
    const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
    const headers: Record<string, string> = { ...((options.headers as any) || {}) };
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }

    // Convert FormData headers: if body is FormData, don't set Content-Type so browser sets boundary
    if (options.body instanceof FormData && headers["Content-Type"]) {
        delete headers["Content-Type"];
    }

    return fetch(url, { ...options, headers });
}

export interface Tenant {
    id: string;
    name: string;
    created_at: string;
    is_active: boolean;
}

export interface Dataset {
    id: string;
    name: string;
    description?: string;
    file_path: string;
    file_size_bytes: number;
    tenant_id: string;
    created_at: string;
    // ── Nuevos campos Fase 1 ──
    file_type?: string;
    num_rows?: number;
    num_columns?: number;
    column_names?: string[];
    version?: number;
    mlflow_artifact_uri?: string;
    pipeline_path?: string;
    // ── DVC Versioning ──
    dvc_remote?: string;
    dvc_hash?: string;
    is_dvc_tracked?: boolean;
    dvc_registry_name?: string;
    dvc_version?: number;
    parent_dataset_id?: string;
}

export interface MLModel {
    id: string;
    name: string;
    description?: string;
    mlflow_run_id: string;
    metrics_metadata: any;
    is_public: boolean;
    tenant_id: string;
    created_at: string;
    // Phase 1: Model Registry
    version?: string;
    stage?: "Staging" | "Production" | "Archived";
    promoted_at?: string;
    promoted_by?: string;
    mlflow_registry_name?: string;
    mlflow_version?: string;
}

export interface Prediction {
    id: string;
    task_id: string;
    status: string;
    method: string;
    dataset_id: string;
    model_id: string;
    tenant_id: string;
    mlflow_inference_run_id?: string;
    created_at: string;
    explain?: boolean;
}

export interface SHAPResult {
    shap_values: number[][];
    expected_value: number;
    feature_names: string[];
}

export interface PredictionExplainResult {
    shap_values: number[];
    expected_value: number;
    feature_names: string[];
    total_importance: number;
    positive_contributors: { feature: string; value: number }[];
    negative_contributors: { feature: string; value: number }[];
}

export interface PredictionRequest {
    tenant_id: string;
    dataset_id: string;
    model_id: string;
    uncertainty_method: string;
}

export interface PredictionStatusResponse {
    task_id: string;
    status: string;
    result?: any;
    error?: string;
    progress?: string;
}

export interface DatasetPreview {
    dataset_id: string;
    file_type: string;
    num_rows: number;
    num_columns: number;
    column_names: string[];
    column_dtypes: Record<string, string>;
    preview_rows: Record<string, any>[];
}

export interface PreprocessingStep {
    type: "impute" | "scale" | "encode" | "feature_eng" | "drop";
    columns: string[];
    strategy?: string;
    method?: string;
    fill_value?: string;
}

export interface PreprocessingConfig {
    dataset_id: string;
    target_column?: string;
    steps: PreprocessingStep[];
}

export interface PreprocessingPreviewResponse {
    original_columns: string[];
    transformed_columns: string[];
    original_shape: number[];
    transformed_shape: number[];
    preview_rows: Record<string, any>[];
    pipeline_path?: string;
}

export interface DatasetProfile {
    dataset_id: string;
    dataset_name: string;
    num_rows: number;
    num_columns: number;
    profile: Record<string, {
        type: string;
        null_count: number;
        null_pct: number;
        distinct_count: number;
        min?: number;
        max?: number | string;
        mean?: number;
        median?: number;
        std?: number;
        zeros?: number;
        histogram?: { counts: number[]; bins: number[] };
        top_values?: { value: string; count: number }[];
    }>;
}

// ── Fase 2: Training ────────────────────────────────────────────────────────

export interface AlgorithmHyperparam {
    name: string;
    label: string;
    type: "int" | "float" | "bool" | "select";
    min?: number;
    max?: number;
    default: any;
    options?: { label: string; value: any }[];
}

export interface AlgorithmInfo {
    id: string;
    display_name: string;
    framework: "sklearn" | "pytorch";
    task_types: string[];
    supports_proba: boolean;
    supports_tree_variance: boolean;
    hyperparams: AlgorithmHyperparam[];
}

export interface ValidationConfig {
    strategy: 'holdout' | 'cross_validation';
    test_size: number;       // for holdout (default 0.2)
    n_folds: number;         // for CV (default 5)
    shuffle: boolean;
    random_state: number;
}

export interface TrainRequest {
    dataset_id: string;
    target_column: string;
    algorithm: string;
    task_type: string;
    hyperparams: Record<string, any>;
    validation: ValidationConfig;
    model_name?: string;
    model_description?: string;
}

export interface TrainResponse {
    message: string;
    task_id: string;
    status_url: string;
}

export interface TrainingStatus {
    task_id: string;
    status: string;
    result?: {
        model_id: string;
        mlflow_run_id: string;
        metrics: Record<string, number>;
        algorithm: string;
        framework: string;
        task_type: string;
        validation_strategy: string;
        n_folds?: number;
        cv_detail?: Record<string, { per_fold: number[]; mean: number; std: number }>;
    };
    error?: string;
}

export const api = {
    // ── Tenants ──────────────────────────────────────────────────────────────
    async getTenant(tenantId: string): Promise<Tenant> {
        const res = await fetchAuth(`${API_BASE_URL}/tenants/${tenantId}`);
        if (!res.ok) throw new Error("Failed to fetch tenant");
        return res.json();
    },

    // ── Datasets ────────────────────────────────────────────────────────────
    async getDatasets(tenantId: string): Promise<Dataset[]> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/?tenant_id=${tenantId}`);
        if (!res.ok) throw new Error("Failed to fetch datasets");
        return res.json();
    },

    async uploadDataset(tenantId: string, file: File, name: string, description?: string, is_dvc_tracked?: boolean, dvc_registry_name?: string): Promise<Dataset> {
        const formData = new FormData();
        formData.append("tenant_id", tenantId);
        formData.append("name", name);
        if (description) formData.append("description", description);
        if (is_dvc_tracked) formData.append("is_dvc_tracked", String(is_dvc_tracked));
        if (dvc_registry_name) formData.append("dvc_registry_name", dvc_registry_name);
        formData.append("file", file);

        const res = await fetchAuth(`${API_BASE_URL}/datasets/`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Failed to upload dataset");
        return res.json();
    },

    async deleteDataset(datasetId: string, tenantId: string): Promise<void> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/${datasetId}?tenant_id=${tenantId}`, {
            method: "DELETE",
        });
        if (!res.ok) throw new Error("Failed to delete dataset");
    },

    async previewDataset(datasetId: string, tenantId: string, maxRows: number = 20): Promise<DatasetPreview> {
        const res = await fetchAuth(
            `${API_BASE_URL}/datasets/${datasetId}/preview?tenant_id=${tenantId}&max_rows=${maxRows}`
        );
        if (!res.ok) throw new Error("Failed to preview dataset");
        return res.json();
    },

    async getDatasetProfile(datasetId: string, tenantId: string): Promise<DatasetProfile> {
        const res = await fetchAuth(`${API_BASE_URL}/profiling/${datasetId}/profile?tenant_id=${tenantId}`);
        if (!res.ok) throw new Error("Failed to profile dataset");
        return res.json();
    },

    // ── DVC Dataset Registry ─────────────────────────────────────────────────
    async getDatasetRegistries(tenantId: string): Promise<{ name: string; versions: number; datasets: any[] }[]> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/registry?tenant_id=${tenantId}`);
        if (!res.ok) throw new Error("Failed to fetch dataset registries");
        return res.json();
    },

    async getDatasetVersions(registryName: string, tenantId: string): Promise<{ registry_name: string; versions: any[] }> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/registry/${encodeURIComponent(registryName)}/versions?tenant_id=${tenantId}`);
        if (!res.ok) throw new Error("Failed to fetch dataset versions");
        return res.json();
    },

    async promoteDataset(datasetId: string, tenantId: string): Promise<{ message: string; dataset_id: string; is_active: boolean }> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/${datasetId}/promote?tenant_id=${tenantId}`, {
            method: "POST",
        });
        if (!res.ok) throw new Error("Failed to promote dataset");
        return res.json();
    },

    async pushDatasetToRemote(datasetId: string, tenantId: string): Promise<{ message: string }> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/${datasetId}/dvc/push?tenant_id=${tenantId}`, {
            method: "POST",
        });
        if (!res.ok) throw new Error("Failed to push dataset to DVC remote");
        return res.json();
    },

    async pullDatasetFromRemote(datasetId: string, tenantId: string): Promise<{ message: string }> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/${datasetId}/dvc/pull?tenant_id=${tenantId}`, {
            method: "POST",
        });
        if (!res.ok) throw new Error("Failed to pull dataset from DVC remote");
        return res.json();
    },

    // ── Models ──────────────────────────────────────────────────────────────
    async getModels(tenantId: string): Promise<MLModel[]> {
        const res = await fetchAuth(`${API_BASE_URL}/models/?tenant_id=${tenantId}`);
        if (!res.ok) throw new Error("Failed to fetch models");
        return res.json();
    },

    async uploadModel(formData: FormData): Promise<MLModel> {
        const res = await fetchAuth(`${API_BASE_URL}/models/upload`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Failed to upload model");
        return res.json();
    },

    async getMLFlowInfo() {
        const res = await fetchAuth(`${API_BASE_URL}/models/mlflow-info`);
        if (!res.ok) throw new Error("Failed to fetch MLFlow info");
        return res.json();
    },

    async deleteModel(modelId: string, tenantId: string): Promise<void> {
        const res = await fetchAuth(`${API_BASE_URL}/models/${modelId}?tenant_id=${tenantId}`, {
            method: "DELETE",
        });
        if (!res.ok) throw new Error("Failed to delete model");
    },

    // ── Predictions ─────────────────────────────────────────────────────────
    async triggerPrediction(req: PredictionRequest): Promise<{
        message: string;
        prediction_id: string;
        task_id: string;
        status_url: string;
        result_url: string;
    }> {
        const res = await fetchAuth(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(req),
        });
        if (!res.ok) throw new Error("Failed to trigger prediction");
        return res.json();
    },

    async getPredictionStatus(taskId: string): Promise<PredictionStatusResponse> {
        const res = await fetchAuth(`${API_BASE_URL}/predictions/status/${taskId}`);
        if (!res.ok) throw new Error("Failed to fetch prediction status");
        return res.json();
    },

    async getPredictionResult(predictionId: string): Promise<Prediction> {
        const res = await fetchAuth(`${API_BASE_URL}/predictions/${predictionId}`);
        if (!res.ok) throw new Error("Failed to fetch prediction result");
        return res.json();
    },

    async listPredictions(tenantId: string): Promise<Prediction[]> {
        const res = await fetchAuth(`${API_BASE_URL}/predictions?tenant_id=${tenantId}`);
        if (!res.ok) throw new Error("Failed to list predictions");
        return res.json();
    },

    async getPredictionExplain(predictionId: string): Promise<PredictionExplainResult> {
        const res = await fetchAuth(`${API_BASE_URL}/predictions/${predictionId}/explain`);
        if (!res.ok) throw new Error("Failed to fetch SHAP values");
        return res.json();
    },

    // ── Preprocessing ───────────────────────────────────────────────────────
    async previewPreprocessing(
        config: PreprocessingConfig,
        tenantId: string,
    ): Promise<PreprocessingPreviewResponse> {
        const res = await fetchAuth(`${API_BASE_URL}/preprocessing/preview?tenant_id=${tenantId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config),
        });
        if (!res.ok) throw new Error("Failed to preview preprocessing");
        return res.json();
    },

    async applyPreprocessing(
        datasetId: string,
        config: PreprocessingConfig,
        tenantId: string,
    ): Promise<{ message: string; new_dataset_id: string; new_dataset_name: string; pipeline_path: string; transformed_shape: number[] }> {
        const res = await fetchAuth(`${API_BASE_URL}/preprocessing/apply/${datasetId}?tenant_id=${tenantId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config),
        });
        if (!res.ok) throw new Error("Failed to apply preprocessing");
        return res.json();
    },

    async getDatasetPipeline(datasetId: string, tenantId: string): Promise<{ steps: PreprocessingStep[]; target_column?: string; message?: string }> {
        const res = await fetchAuth(`${API_BASE_URL}/preprocessing/pipeline/${datasetId}?tenant_id=${tenantId}`);
        if (!res.ok) throw new Error("Failed to fetch dataset pipeline");
        return res.json();
    },

    // ── Training (Fase 2) ───────────────────────────────────────────────────
    async getAlgorithms(): Promise<AlgorithmInfo[]> {
        const res = await fetchAuth(`${API_BASE_URL}/training/algorithms`);
        if (!res.ok) throw new Error("Failed to fetch algorithms");
        return res.json();
    },

    async trainModel(
        req: TrainRequest,
        tenantId: string,
    ): Promise<TrainResponse> {
        const res = await fetchAuth(`${API_BASE_URL}/training/train?tenant_id=${tenantId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(req),
        });
        if (!res.ok) throw new Error("Failed to start training");
        return res.json();
    },

    async getTrainingStatus(taskId: string): Promise<TrainingStatus> {
        const res = await fetchAuth(`${API_BASE_URL}/training/status/${taskId}`);
        if (!res.ok) throw new Error("Failed to fetch training status");
        return res.json();
    },

    // ── Model Registry (Phase 1) ─────────────────────────────────────────────────
    async createRegisteredModel(name: string, description: string = ""): Promise<{ name: string; description: string }> {
        const formData = new FormData();
        formData.append("name", name);
        formData.append("description", description);

        const res = await fetchAuth(`${API_BASE_URL}/models/registry`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Failed to create registered model");
        return res.json();
    },

    async getRegisteredModels(): Promise<{ models: any[] }> {
        const res = await fetchAuth(`${API_BASE_URL}/models/registry`);
        if (!res.ok) throw new Error("Failed to fetch registered models");
        return res.json();
    },

    async deleteRegisteredModel(name: string): Promise<void> {
        const res = await fetchAuth(`${API_BASE_URL}/models/registry/${encodeURIComponent(name)}`, {
            method: "DELETE",
        });
        if (!res.ok) throw new Error("Failed to delete registered model");
    },

    async promoteModel(modelId: string, targetStage: string): Promise<MLModel> {
        const res = await fetchAuth(`${API_BASE_URL}/models/${modelId}/promote`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ target_stage: targetStage }),
        });
        if (!res.ok) throw new Error("Failed to promote model");
        return res.json();
    },

    async archiveModel(modelId: string): Promise<MLModel> {
        const res = await fetchAuth(`${API_BASE_URL}/models/${modelId}/archive`, {
            method: "POST",
        });
        if (!res.ok) throw new Error("Failed to archive model");
        return res.json();
    },

    async getModelVersions(modelId: string): Promise<{
        current_version: string;
        stage: string;
        promoted_at?: string;
        promoted_by?: string;
        mlflow_versions: any[];
    }> {
        const res = await fetchAuth(`${API_BASE_URL}/models/${modelId}/versions`);
        if (!res.ok) throw new Error("Failed to fetch model versions");
        return res.json();
    },

    async getRegistryVersions(modelName: string): Promise<{ versions: any[] }> {
        const res = await fetchAuth(`${API_BASE_URL}/models/registry/${encodeURIComponent(modelName)}/versions`);
        if (!res.ok) throw new Error("Failed to fetch registry versions");
        return res.json();
    },

    async getRunDetails(runId: string): Promise<any> {
        const res = await fetchAuth(`${API_BASE_URL}/models/runs/${runId}/details`);
        if (!res.ok) throw new Error("Failed to fetch run details");
        return res.json();
    },

    async downloadModelByRunId(runId: string, modelName: string): Promise<void> {
        const res = await fetchAuth(`${API_BASE_URL}/models/runs/${runId}/download`);
        if (!res.ok) throw new Error("Failed to download model version");
        
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${modelName.replace(/\s+/g, "_")}_${runId.substring(0, 8)}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    },

    async downloadDataset(datasetId: string, filename: string): Promise<void> {
        const res = await fetchAuth(`${API_BASE_URL}/datasets/${datasetId}/download`);
        if (!res.ok) throw new Error("Failed to download dataset");
        
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    },
};

