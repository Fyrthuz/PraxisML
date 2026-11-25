import { useState, useCallback } from "react";
import { MLModel } from "@/lib/api";
import toast from "react-hot-toast";
import { config } from '@/lib/config';

const API = config.getFullApiUrl("");

export function useModels(token: string | null) {
    const [models, setModels] = useState<MLModel[]>([]);

    const fetchModels = useCallback(async () => {
        if (!token) return;
        const res = await fetch(`${API}/models/`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
            const data = await res.json();
            // API returns a paginated envelope { items: [...], total, page, ... }
            const items = Array.isArray(data) ? data : (data.items ?? []);
            setModels(items);
        }
    }, [token]);

    const uploadModel = useCallback(
        async (formData: FormData) => {
            if (!token) return;
            await fetch(`${API}/models/upload`, {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
                body: formData,
            });
            await fetchModels();
        },
        [token, fetchModels]
    );

    const deleteModel = useCallback(
        async (modelId: string, tenantId: string): Promise<boolean> => {
            if (!token) return false;
            try {
                const res = await fetch(
                    `${API}/models/${modelId}?tenant_id=${tenantId}`,
                    {
                        method: "DELETE",
                        headers: { Authorization: `Bearer ${token}` },
                    }
                );
                if (!res.ok) throw new Error("Failed to delete model");
                toast.success("Model deleted successfully");
                await fetchModels();
                return true;
            } catch {
                toast.error("Failed to delete model");
                return false;
            }
        },
        [token, fetchModels]
    );

    return {
        models,
        setModels,
        fetchModels,
        uploadModel,
        deleteModel,
    };
}
