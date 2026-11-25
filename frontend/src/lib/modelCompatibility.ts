import { MLModel } from "@/lib/api";

export interface ModelCompatibilityProfile {
    domain: string;
    taskFamily: string;
    datasetLabel: string | null;
    scopeLabel: string;
}

export function getModelCompatibilityProfile(model: MLModel): ModelCompatibilityProfile {
    const meta = (model.metrics_metadata || {}) as Record<string, any>;
    const framework = String(meta.framework || "").toLowerCase();
    const rawDomain = String(meta.domain || "").toLowerCase();
    const domain =
        rawDomain ||
        (framework === "huggingface" || framework === "onnx_nlp" ? "nlp" : "tabular");

    const rawTask = String(meta.task_type || "").toLowerCase();
    let taskFamily = "general";
    if (
        rawTask.includes("token") ||
        rawTask === "ner"
    ) {
        taskFamily = "token_classification";
    } else if (
        rawTask.includes("class") ||
        rawTask.includes("sentiment")
    ) {
        taskFamily = "classification";
    } else if (rawTask.includes("regress")) {
        taskFamily = "regression";
    } else if (rawTask.includes("cluster")) {
        taskFamily = "clustering";
    } else if (rawTask) {
        taskFamily = rawTask;
    }

    const datasetLabel =
        meta.dataset_name ||
        (model as any).dataset_dvc_registry_name ||
        meta.dataset_id ||
        null;

    return {
        domain,
        taskFamily,
        datasetLabel,
        scopeLabel: `${domain} · ${taskFamily}`,
    };
}

export function areModelsCompatible(a: MLModel, b: MLModel): boolean {
    const pa = getModelCompatibilityProfile(a);
    const pb = getModelCompatibilityProfile(b);
    return pa.domain === pb.domain && pa.taskFamily === pb.taskFamily;
}
