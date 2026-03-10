import { useState, useEffect, useCallback } from "react";
import { AlgorithmInfo, TrainingStatus } from "@/lib/api";
import toast from "react-hot-toast";

const API = "http://localhost:8000/api/v1";

export function useTraining(token: string | null, onComplete?: () => void) {
    // Algorithm catalog
    const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);

    // Form fields
    const [trainDataset, setTrainDataset] = useState<string>("");
    const [trainTarget, setTrainTarget] = useState<string>("");
    const [trainTaskType, setTrainTaskType] = useState<string>("classification");
    const [trainAlgorithm, setTrainAlgorithm] = useState<string>("");
    const [trainHyperparams, setTrainHyperparams] = useState<Record<string, any>>({});
    const [trainModelName, setTrainModelName] = useState<string>("");
    const [trainRegistryName, setTrainRegistryName] = useState<string>("");

    // Validation
    const [validationStrategy, setValidationStrategy] = useState<
        "holdout" | "cross_validation"
    >("holdout");
    const [testSize, setTestSize] = useState<number>(0.2);
    const [nFolds, setNFolds] = useState<number>(5);

    // Run state
    const [trainingTaskId, setTrainingTaskId] = useState<string | null>(null);
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
    const [isTraining, setIsTraining] = useState(false);

    // Load algorithms once token is available
    useEffect(() => {
        if (!token) return;
        fetch(`${API}/training/algorithms`, {
            headers: { Authorization: `Bearer ${token}` },
        })
            .then((res) => res.json())
            .then((data) => {
                if (Array.isArray(data)) setAlgorithms(data);
            })
            .catch((err) => console.error("Failed to load algorithms:", err));
    }, [token]);

    // Poll training status
    useEffect(() => {
        if (!trainingTaskId || !token) return;

        const interval = setInterval(async () => {
            try {
                const res = await fetch(
                    `${API}/training/status/${trainingTaskId}`,
                    { headers: { Authorization: `Bearer ${token}` } }
                );
                const data: TrainingStatus = await res.json();
                setTrainingStatus(data);

                if (data.status === "SUCCESS") {
                    toast.success("Model training completed!");
                    setIsTraining(false);
                    setTrainingTaskId(null);
                    onComplete?.();
                } else if (data.status === "FAILURE") {
                    toast.error(`Training failed: ${data.error || "Unknown error"}`);
                    setIsTraining(false);
                    setTrainingTaskId(null);
                }
            } catch (err) {
                console.error("Training poll error", err);
            }
        }, 3000);

        return () => clearInterval(interval);
    }, [trainingTaskId, token, onComplete]);

    const startTraining = useCallback(async () => {
        if (!trainDataset || !trainTarget || !trainAlgorithm || !token) return;
        setIsTraining(true);
        setTrainingStatus(null);
        try {
            const res = await fetch(`${API}/training/train`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
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
                    registry_name: trainRegistryName || undefined,
                }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Training failed");
            }

            const data = await res.json();
            setTrainingTaskId(data.task_id);
            toast.success("Training task launched!");
        } catch (err: any) {
            toast.error(err.message || "Failed to start training");
            setIsTraining(false);
        }
    }, [
        token,
        trainDataset,
        trainTarget,
        trainAlgorithm,
        trainTaskType,
        trainHyperparams,
        trainModelName,
        trainRegistryName,
        validationStrategy,
        testSize,
        nFolds,
    ]);

    return {
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
        startTraining,
    };
}
