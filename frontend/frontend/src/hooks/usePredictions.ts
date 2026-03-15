import { useState, useEffect, useCallback } from "react";
import { Prediction } from "@/lib/api";
import toast from "react-hot-toast";
import { config } from '@/lib/config';

const API = config.getFullApiUrl("");

const PENDING_STATUSES = ["PENDING", "STARTED", "IN_PROGRESS", "RUNNING"];

export function usePredictions(
    token: string | null,
    onPredictionComplete?: (prediction: Prediction) => void
) {
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [isActionLoading, setIsActionLoading] = useState(false);
    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [selectedModel, setSelectedModel] = useState<string>("");
    const [uncertaintyMethod, setUncertaintyMethod] = useState("none");

    const fetchPredictions = useCallback(async () => {
        if (!token) return;
        const res = await fetch(`${API}/predictions`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
            const data = await res.json();
            setPredictions(Array.isArray(data) ? data : []);
        }
    }, [token]);

    // Auto-poll while any prediction is in progress
    useEffect(() => {
        if (!token || !predictions.some((p) => PENDING_STATUSES.includes(p.status)))
            return;

        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${API}/predictions`, {
                    headers: { Authorization: `Bearer ${token}` },
                });
                const updatedPreds: Prediction[] = await res.json();

                if (Array.isArray(updatedPreds)) {
                    setPredictions((prevPreds) => {
                        updatedPreds.forEach((updated) => {
                            const prev = prevPreds.find((p) => p.id === updated.id);
                            if (prev && prev.status !== updated.status) {
                                if (updated.status === "COMPLETED") {
                                    toast.success(
                                        `Inference ${updated.id.substring(0, 8)} completed successfully`
                                    );
                                    onPredictionComplete?.(updated);
                                } else if (updated.status === "FAILED") {
                                    toast.error(
                                        `Inference ${updated.id.substring(0, 8)} failed! Check backend logs.`
                                    );
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
    }, [predictions, token, onPredictionComplete]);

    const runInference = useCallback(
        async (batchFile?: File) => {
            if ((!selectedDataset && !batchFile) || !selectedModel || !token) return;
            setIsActionLoading(true);
            try {
                let res: Response;

                if (batchFile) {
                    const formData = new FormData();
                    formData.append("file", batchFile);
                    formData.append("model_id", selectedModel);
                    formData.append("uncertainty_method", uncertaintyMethod);
                    res = await fetch(`${API}/predictions/predict/batch`, {
                        method: "POST",
                        headers: { Authorization: `Bearer ${token}` },
                        body: formData,
                    });
                } else {
                    res = await fetch(`${API}/predictions/predict`, {
                        method: "POST",
                        headers: {
                            Authorization: `Bearer ${token}`,
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                            dataset_id: selectedDataset,
                            model_id: selectedModel,
                            uncertainty_method: uncertaintyMethod,
                        }),
                    });
                }

                if (!res.ok) {
                    const data = await res.json();
                    throw new Error(data.detail || "Error triggering inference.");
                }

                toast.success("Batch inference task enqueued!");
                await fetchPredictions();
            } catch (error: any) {
                toast.error(error.message || "Error triggering inference.");
                console.error(error);
            } finally {
                setIsActionLoading(false);
            }
        },
        [token, selectedDataset, selectedModel, uncertaintyMethod, fetchPredictions]
    );

    return {
        predictions,
        setPredictions,
        fetchPredictions,
        isActionLoading,
        selectedDataset,
        setSelectedDataset,
        selectedModel,
        setSelectedModel,
        uncertaintyMethod,
        setUncertaintyMethod,
        runInference,
    };
}
