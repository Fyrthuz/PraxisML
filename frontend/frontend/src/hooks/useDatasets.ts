import { useState, useCallback } from "react";
import { Dataset, DatasetPreview } from "@/lib/api";
import toast from "react-hot-toast";
import { config } from '@/lib/config';

const API = config.getFullApiUrl("");

export function useDatasets(token: string | null) {
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [previewData, setPreviewData] = useState<DatasetPreview | null>(null);
    const [deletingDatasetId, setDeletingDatasetId] = useState<string | null>(null);

    const fetchDatasets = useCallback(async () => {
        if (!token) return;
        const res = await fetch(`${API}/datasets/`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
            const data = await res.json();
            // API returns a paginated envelope { items: [...], total, page, ... }
            const items = Array.isArray(data) ? data : (data.items ?? []);
            setDatasets(items);
        }
    }, [token]);

    const uploadDataset = useCallback(
        async (formData: FormData) => {
            if (!token) return;
            await fetch(`${API}/datasets/`, {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
                body: formData,
            });
            await fetchDatasets();
        },
        [token, fetchDatasets]
    );

    const deleteDataset = useCallback(
        async (datasetId: string) => {
            if (!token) return;
            try {
                await fetch(`${API}/datasets/${datasetId}`, {
                    method: "DELETE",
                    headers: { Authorization: `Bearer ${token}` },
                });
                toast.success("Dataset deleted successfully");
                setDeletingDatasetId(null);
                await fetchDatasets();
            } catch {
                toast.error("Failed to delete dataset");
            }
        },
        [token, fetchDatasets]
    );

    const previewDataset = useCallback(
        async (datasetId: string) => {
            if (!token) return;
            try {
                const res = await fetch(`${API}/datasets/${datasetId}/preview`, {
                    headers: { Authorization: `Bearer ${token}` },
                });
                if (!res.ok) throw new Error("Preview failed");
                setPreviewData(await res.json());
            } catch (err: any) {
                toast.error(err.message || "Failed to preview dataset");
            }
        },
        [token]
    );

    return {
        datasets,
        setDatasets,
        previewData,
        setPreviewData,
        deletingDatasetId,
        setDeletingDatasetId,
        fetchDatasets,
        uploadDataset,
        deleteDataset,
        previewDataset,
    };
}
