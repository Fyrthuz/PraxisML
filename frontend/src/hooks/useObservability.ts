import { useState, useCallback } from "react";
import toast from "react-hot-toast";
import { config } from "@/lib/config";
import type {
    UsageMetrics,
    CostBreakdown,
    CostForecast,
    Alert,
    AlertCreate,
    DriftHistory,
    HealthScore,
    ResourcePricing,
} from "@/lib/api";

const API = config.getFullApiUrl("/observability");

export function useObservability(token: string | null) {
    const [usageMetrics, setUsageMetrics] = useState<UsageMetrics | null>(null);
    const [costBreakdown, setCostBreakdown] = useState<CostBreakdown | null>(null);
    const [costForecast, setCostForecast] = useState<CostForecast | null>(null);
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [pricing, setPricing] = useState<ResourcePricing[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [activeTab, setActiveTab] = useState<"overview" | "usage" | "costs" | "alerts" | "pricing">("overview");

    const fetchUsage = useCallback(async (domain?: string) => {
        if (!token) return;
        setIsLoading(true);
        try {
            const url = domain
                ? `${API}/usage?domain=${encodeURIComponent(domain)}`
                : `${API}/usage`;
            const res = await fetch(url, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) setUsageMetrics(await res.json());
        } catch {
            toast.error("Failed to fetch usage metrics");
        } finally {
            setIsLoading(false);
        }
    }, [token]);

    const fetchModelUsage = useCallback(async (modelId: string) => {
        if (!token) return;
        setIsLoading(true);
        try {
            const res = await fetch(`${API}/usage/models/${modelId}`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) setUsageMetrics(await res.json());
        } catch {
            toast.error("Failed to fetch model usage");
        } finally {
            setIsLoading(false);
        }
    }, [token]);

    const fetchCosts = useCallback(async () => {
        if (!token) return;
        setIsLoading(true);
        try {
            const res = await fetch(`${API}/costs`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) setCostBreakdown(await res.json());
        } catch {
            toast.error("Failed to fetch costs");
        } finally {
            setIsLoading(false);
        }
    }, [token]);

    const fetchForecast = useCallback(async () => {
        if (!token) return;
        try {
            const res = await fetch(`${API}/costs/forecast`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) setCostForecast(await res.json());
        } catch {
            toast.error("Failed to fetch cost forecast");
        }
    }, [token]);

    const fetchAlerts = useCallback(async () => {
        if (!token) return;
        setIsLoading(true);
        try {
            const res = await fetch(`${API}/alerts`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) setAlerts(await res.json());
        } catch {
            toast.error("Failed to fetch alerts");
        } finally {
            setIsLoading(false);
        }
    }, [token]);

    const createAlert = useCallback(async (data: AlertCreate) => {
        if (!token) return;
        try {
            const res = await fetch(`${API}/alerts`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(data),
            });
            if (res.ok) {
                toast.success("Alert created");
                await fetchAlerts();
            } else {
                toast.error("Failed to create alert");
            }
        } catch {
            toast.error("Failed to create alert");
        }
    }, [token, fetchAlerts]);

    const toggleAlert = useCallback(async (alertId: string, enabled: boolean) => {
        if (!token) return;
        try {
            const res = await fetch(`${API}/alerts/${alertId}`, {
                method: "PATCH",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ enabled }),
            });
            if (res.ok) {
                toast.success(enabled ? "Alert enabled" : "Alert disabled");
                await fetchAlerts();
            }
        } catch {
            toast.error("Failed to update alert");
        }
    }, [token, fetchAlerts]);

    const deleteAlert = useCallback(async (alertId: string) => {
        if (!token) return;
        try {
            const res = await fetch(`${API}/alerts/${alertId}`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                toast.success("Alert deleted");
                await fetchAlerts();
            }
        } catch {
            toast.error("Failed to delete alert");
        }
    }, [token, fetchAlerts]);

    const fetchPricing = useCallback(async () => {
        if (!token) return;
        setIsLoading(true);
        try {
            const res = await fetch(`${API}/pricing`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) setPricing(await res.json());
        } catch {
            toast.error("Failed to fetch pricing");
        } finally {
            setIsLoading(false);
        }
    }, [token]);

    const fetchDriftHistory = useCallback(async (modelId: string): Promise<DriftHistory | null> => {
        if (!token) return null;
        try {
            const res = await fetch(`${API}/drift/history/${modelId}`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) return await res.json();
        } catch {
            toast.error("Failed to fetch drift history");
        }
        return null;
    }, [token]);

    const fetchModelHealth = useCallback(async (modelId: string): Promise<HealthScore | null> => {
        if (!token) return null;
        try {
            const res = await fetch(`${API}/models/${modelId}/health`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) return await res.json();
        } catch {
            toast.error("Failed to fetch model health");
        }
        return null;
    }, [token]);

    return {
        // State
        usageMetrics,
        costBreakdown,
        costForecast,
        alerts,
        pricing,
        isLoading,
        activeTab,
        setActiveTab,

        // Actions
        fetchUsage,
        fetchModelUsage,
        fetchCosts,
        fetchForecast,
        fetchAlerts,
        createAlert,
        toggleAlert,
        deleteAlert,
        fetchPricing,
        fetchDriftHistory,
        fetchModelHealth,
    };
}
