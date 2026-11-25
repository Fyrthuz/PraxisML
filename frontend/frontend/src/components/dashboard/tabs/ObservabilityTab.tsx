"use client";
import React, { useEffect, useState } from "react";
import {
    Activity,
    DollarSign,
    Bell,
    BarChart3,
    TrendingUp,
    AlertTriangle,
    CheckCircle2,
    XCircle,
    ToggleLeft,
    ToggleRight,
    Trash2,
    Plus,
    RefreshCw,
    Cpu,
    Zap,
    Server,
    Database,
    Layers,
    ShieldCheck,
    Radio,
} from "lucide-react";
import type { Alert, AlertCreate, ResourcePricing, HealthScore } from "@/lib/api";
import { useObservability } from "@/hooks/useObservability";
import { config } from "@/lib/config";

type SubTab = "overview" | "usage" | "costs" | "alerts" | "pricing";

interface PlatformSummary {
    tenant_id: string;
    services_health: {
        postgres: { status: string };
        redis: { status: string; queues?: Record<string, number> };
        celery_workers?: { status: string; workers?: number };
        storage: { status: string; backend?: string };
        mlflow: { status: string; tracking_uri?: string };
        prometheus?: { status: string; url?: string };
        grafana?: { status: string; url?: string };
        serving_plane?: { status: string; active_endpoints?: number; runtimes?: string[] };
        vector_search?: { status: string; indexed_vectors?: number; embedding_model?: string };
        llm_gateway?: { status: string; provider?: string; model?: string };
        event_bus?: { status: string; backend?: string };
    };
    serving: {
        total_endpoints: number;
        active_endpoints: number;
        paused_endpoints: number;
        total_requests: number;
        total_errors: number;
        error_rate_pct: number;
        avg_latency_ms: number;
        endpoints: Array<{
            id: string;
            name: string;
            slug: string;
            status: string;
            traffic_split_pct: number;
            request_count: number;
            error_count: number;
            avg_latency_ms: number;
            last_invoked_at: string | null;
        }>;
    };
    tracking_and_data: {
        total_datasets: number;
        dvc_tracked_datasets: number;
        datasets_storage_bytes: number;
        datasets_storage_mb: number;
        total_models: number;
        models_by_stage: Record<string, number>;
    };
    compute_and_governance: {
        runtime_environments: number;
        user_pipelines: number;
        workspaces: number;
        active_api_keys: number;
        audit_events_24h: number;
    };
    usage_by_domain: Record<string, number>;
}

interface ObservabilityTabProps {
    token: string | null;
}

export default function ObservabilityTab({ token }: ObservabilityTabProps) {
    const obs = useObservability(token);
    const [showCreateAlert, setShowCreateAlert] = useState(false);
    const [platformSummary, setPlatformSummary] = useState<PlatformSummary | null>(null);
    const [usageDomain, setUsageDomain] = useState<string>("");

    const fetchSummary = () => {
        if (!token) return;
        fetch(config.getFullApiUrl("/observability/platform-summary"), {
            headers: { Authorization: `Bearer ${token}` },
        })
            .then((r) => (r.ok ? r.json() : null))
            .then((d: PlatformSummary | null) => {
                if (d) setPlatformSummary(d);
            })
            .catch(() => {});
    };

    useEffect(() => {
        obs.fetchUsage();
        obs.fetchCosts();
        obs.fetchForecast();
        obs.fetchAlerts();
        obs.fetchPricing();
        fetchSummary();
    }, [token]);

    const handleTabChange = (tab: SubTab) => {
        obs.setActiveTab(tab);
        switch (tab) {
            case "overview":
                fetchSummary();
                break;
            case "usage":
                obs.fetchUsage(usageDomain || undefined);
                break;
            case "costs":
                obs.fetchCosts();
                obs.fetchForecast();
                break;
            case "alerts":
                obs.fetchAlerts();
                break;
            case "pricing":
                obs.fetchPricing();
                break;
        }
    };

    return (
        <div className="flex-1 flex flex-col space-y-5">
            {/* Sub-tabs & refresh */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-3">
                <div className="flex flex-wrap items-center gap-1">
                    {([
                        { id: "overview", label: "Overview", icon: <BarChart3 className="w-3.5 h-3.5" /> },
                        { id: "usage", label: "Usage", icon: <Activity className="w-3.5 h-3.5" /> },
                        { id: "costs", label: "Costs", icon: <DollarSign className="w-3.5 h-3.5" /> },
                        { id: "alerts", label: "Alerts", icon: <Bell className="w-3.5 h-3.5" /> },
                        { id: "pricing", label: "Pricing", icon: <Cpu className="w-3.5 h-3.5" /> },
                    ] as { id: SubTab; label: string; icon: React.ReactNode }[]).map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => handleTabChange(tab.id)}
                            className={`h-8 flex items-center gap-1.5 px-3 rounded-md text-xs font-medium transition-colors ${
                                obs.activeTab === tab.id
                                    ? "bg-zinc-800 text-zinc-100"
                                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                            }`}
                        >
                            {tab.icon}
                            {tab.label}
                        </button>
                    ))}
                </div>
                <button
                    onClick={() => {
                        obs.fetchUsage(usageDomain || undefined);
                        obs.fetchCosts();
                        obs.fetchForecast();
                        obs.fetchAlerts();
                        obs.fetchPricing();
                        fetchSummary();
                    }}
                    className="h-8 flex items-center gap-1.5 px-3 bg-[#101217] border border-zinc-800 text-zinc-300 text-xs rounded-md hover:bg-zinc-800 transition-colors"
                >
                    <RefreshCw className={`w-3.5 h-3.5 ${obs.isLoading ? "animate-spin" : ""}`} />
                    Refresh
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto">
                {obs.activeTab === "overview" && (
                    <OverviewPanel obs={obs} summary={platformSummary} />
                )}
                {obs.activeTab === "usage" && (
                    <UsagePanel
                        obs={obs}
                        domain={usageDomain}
                        domainCounts={platformSummary?.usage_by_domain}
                        onDomainChange={(dom) => {
                            setUsageDomain(dom);
                            obs.fetchUsage(dom || undefined);
                        }}
                    />
                )}
                {obs.activeTab === "costs" && <CostsPanel obs={obs} />}
                {obs.activeTab === "alerts" && (
                    <AlertsPanel
                        obs={obs}
                        showCreate={showCreateAlert}
                        setShowCreate={setShowCreateAlert}
                    />
                )}
                {obs.activeTab === "pricing" && <PricingPanel obs={obs} />}
            </div>
        </div>
    );
}

// ── Overview Panel ─────────────────────────────────────────────────────────

function OverviewPanel({
    obs,
    summary,
}: {
    obs: ReturnType<typeof useObservability>;
    summary: PlatformSummary | null;
}) {
    const totalCost = obs.costBreakdown?.total_cost_usd ?? 0;
    const totalRecords = obs.costBreakdown?.records.length ?? 0;
    const activeAlerts = obs.alerts.filter((a) => a.enabled).length;
    const budget = obs.costForecast?.budget_usd;
    const overrun = obs.costForecast?.overrun_risk_pct;
    const projected = obs.costForecast?.projected_month_end_usd;

    const serving = summary?.serving;
    const health = summary?.services_health;
    const tracking = summary?.tracking_and_data;
    const compute = summary?.compute_and_governance;
    const domains = summary?.usage_by_domain ?? {};
    const totalDomainEvents = Object.values(domains).reduce((acc, v) => acc + v, 0);
    const totalQueued = health?.redis.queues
        ? Object.values(health.redis.queues).reduce((a, b) => a + b, 0)
        : 0;

    const statusBadge = (status?: string) => {
        if (status === "ok") {
            return (
                <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    healthy
                </span>
            );
        }
        if (status === "rag_local" || status === "idle") {
            return (
                <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-sky-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
                    {status === "rag_local" ? "rag-local" : "idle"}
                </span>
            );
        }
        if (status === "degraded") {
            return (
                <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-amber-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    degraded
                </span>
            );
        }
        return (
            <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-red-400">
                <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                {status ?? "unknown"}
            </span>
        );
    };

    return (
        <div className="space-y-5">
            {/* Row 1: Primary KPIs */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                    icon={<DollarSign className="w-4 h-4 text-emerald-400" />}
                    label="Monthly Cost"
                    value={`$${totalCost.toFixed(2)}`}
                    sub={`${totalRecords} billing records`}
                    color="emerald"
                />
                <StatCard
                    icon={<Radio className="w-4 h-4 text-orange-400" />}
                    label="Serving Endpoints"
                    value={
                        serving
                            ? `${serving.active_endpoints}/${serving.total_endpoints}`
                            : "—"
                    }
                    sub={
                        serving
                            ? `${serving.total_requests} reqs · ${serving.avg_latency_ms.toFixed(1)}ms · ${serving.error_rate_pct}% err`
                            : "Loading serving telemetry..."
                    }
                    color="orange"
                />
                <StatCard
                    icon={<Activity className="w-4 h-4 text-sky-400" />}
                    label="Usage Telemetry"
                    value={totalDomainEvents || (obs.usageMetrics?.metrics.length ?? 0)}
                    sub={`${Object.keys(domains).length} active domains`}
                    color="blue"
                />
                <StatCard
                    icon={<Bell className="w-4 h-4 text-amber-400" />}
                    label="Alerts & Budget"
                    value={budget ? `$${budget.toFixed(0)}` : `${activeAlerts} active`}
                    sub={
                        overrun
                            ? `${overrun.toFixed(0)}% projected · ${activeAlerts} alerts`
                            : `${activeAlerts}/${obs.alerts.length} alerts enabled`
                    }
                    color={overrun && overrun > 100 ? "red" : "amber"}
                />
            </div>

            {/* Row 2: Live Infrastructure Health + Tracking/Data/Compute Footprint */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Platform Services & Queues Health */}
                <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5">
                    <div className="flex items-center justify-between mb-3.5">
                        <h3 className="text-zinc-100 text-sm font-medium flex items-center gap-2">
                            <Server className="w-4 h-4 text-zinc-400" />
                            Platform Services &amp; Queues
                        </h3>
                        <span className="text-[11px] font-mono text-zinc-500">
                            {totalQueued} queued jobs
                        </span>
                    </div>
                    <div className="space-y-1.5 text-xs">
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <span className="text-zinc-300">PostgreSQL Database &amp; RLS</span>
                            {statusBadge(health?.postgres.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div className="flex flex-col gap-1">
                                <span className="text-zinc-300">Redis Broker &amp; Celery Queues</span>
                                {health?.redis.queues && (
                                    <div className="flex flex-wrap gap-1">
                                        {Object.entries(health.redis.queues).map(([q, depth]) => (
                                            <span
                                                key={q}
                                                className="px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-[10px] font-mono text-zinc-400"
                                            >
                                                {q}:{depth}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                            {statusBadge(health?.redis.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div>
                                <span className="text-zinc-300">Celery Workers</span>
                                {health?.celery_workers?.workers !== undefined && (
                                    <span className="ml-2 text-[11px] font-mono text-zinc-500">
                                        ({health.celery_workers.workers} online)
                                    </span>
                                )}
                            </div>
                            {statusBadge(health?.celery_workers?.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div>
                                <span className="text-zinc-300">Model Serving Plane</span>
                                {health?.serving_plane?.runtimes && (
                                    <span className="ml-2 text-[11px] font-mono text-zinc-500">
                                        ({health.serving_plane.active_endpoints ?? 0} active ·{" "}
                                        {health.serving_plane.runtimes.join(", ")})
                                    </span>
                                )}
                            </div>
                            {statusBadge(health?.serving_plane?.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div>
                                <span className="text-zinc-300">Object Storage</span>
                                {health?.storage.backend && (
                                    <span className="ml-2 text-[11px] font-mono text-zinc-500">
                                        ({health.storage.backend})
                                    </span>
                                )}
                            </div>
                            {statusBadge(health?.storage.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div className="truncate pr-3">
                                <span className="text-zinc-300">MLflow Tracking</span>
                                {health?.mlflow.tracking_uri && (
                                    <a
                                        href={
                                            health.mlflow.tracking_uri.startsWith("http")
                                                ? health.mlflow.tracking_uri
                                                : undefined
                                        }
                                        target="_blank"
                                        rel="noreferrer"
                                        className="ml-2 text-[11px] font-mono text-orange-400 hover:underline"
                                    >
                                        {health.mlflow.tracking_uri}
                                    </a>
                                )}
                            </div>
                            {statusBadge(health?.mlflow.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div className="truncate pr-3">
                                <span className="text-zinc-300">Prometheus &amp; Grafana</span>
                                {health?.grafana?.url && (
                                    <a
                                        href={health.grafana.url}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="ml-2 text-[11px] font-mono text-orange-400 hover:underline"
                                    >
                                        Grafana (:3001)
                                    </a>
                                )}
                                {health?.prometheus?.url && (
                                    <a
                                        href={health.prometheus.url}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="ml-2 text-[11px] font-mono text-zinc-500 hover:text-zinc-300 hover:underline"
                                    >
                                        Prom (:9090)
                                    </a>
                                )}
                            </div>
                            <div className="flex items-center gap-2.5">
                                {statusBadge(health?.prometheus?.status)}
                                <span className="text-zinc-700">/</span>
                                {statusBadge(health?.grafana?.status)}
                            </div>
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div>
                                <span className="text-zinc-300">Vector Search (pgvector)</span>
                                {health?.vector_search && (
                                    <span className="ml-2 text-[11px] font-mono text-zinc-500">
                                        ({health.vector_search.indexed_vectors ?? 0} vectors)
                                    </span>
                                )}
                            </div>
                            {statusBadge(health?.vector_search?.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
                            <div>
                                <span className="text-zinc-300">LLM Gateway / Copilot</span>
                                {health?.llm_gateway?.provider && (
                                    <span className="ml-2 text-[11px] font-mono text-zinc-500">
                                        ({health.llm_gateway.provider})
                                    </span>
                                )}
                            </div>
                            {statusBadge(health?.llm_gateway?.status)}
                        </div>
                        <div className="flex items-center justify-between py-1.5">
                            <div>
                                <span className="text-zinc-300">Domain Event Bus</span>
                                {health?.event_bus?.backend && (
                                    <span className="ml-2 text-[11px] font-mono text-zinc-500">
                                        ({health.event_bus.backend})
                                    </span>
                                )}
                            </div>
                            {statusBadge(health?.event_bus?.status)}
                        </div>
                    </div>
                </div>

                {/* Tracking, Data, Compute & Governance Footprint */}
                <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5">
                    <h3 className="text-zinc-100 text-sm font-medium mb-4 flex items-center gap-2">
                        <Layers className="w-4 h-4 text-zinc-400" />
                        Tracking, Data &amp; Compute Footprint
                    </h3>
                    <div className="grid grid-cols-2 gap-3 text-xs mb-4">
                        <div className="bg-[#0b0c0f] border border-zinc-800/80 rounded-md p-3">
                            <div className="flex items-center gap-1.5 text-zinc-400 mb-1">
                                <Database className="w-3.5 h-3.5" />
                                Datasets &amp; DVC
                            </div>
                            <p className="text-lg font-mono font-semibold text-zinc-100 tabular-nums">
                                {tracking?.total_datasets ?? "—"}
                            </p>
                            <p className="text-[11px] text-zinc-500 font-mono">
                                {tracking
                                    ? `${tracking.dvc_tracked_datasets} DVC · ${tracking.datasets_storage_mb} MB`
                                    : "—"}
                            </p>
                        </div>
                        <div className="bg-[#0b0c0f] border border-zinc-800/80 rounded-md p-3">
                            <div className="flex items-center gap-1.5 text-zinc-400 mb-1">
                                <Zap className="w-3.5 h-3.5" />
                                MLflow Models
                            </div>
                            <p className="text-lg font-mono font-semibold text-zinc-100 tabular-nums">
                                {tracking?.total_models ?? "—"}
                            </p>
                            <p className="text-[11px] text-zinc-500 font-mono">
                                {tracking
                                    ? `${tracking.models_by_stage.Production ?? 0} Prod · ${tracking.models_by_stage.Staging ?? 0} Stg`
                                    : "—"}
                            </p>
                        </div>
                        <div className="bg-[#0b0c0f] border border-zinc-800/80 rounded-md p-3">
                            <div className="flex items-center gap-1.5 text-zinc-400 mb-1">
                                <Cpu className="w-3.5 h-3.5" />
                                Envs &amp; Pipelines
                            </div>
                            <p className="text-lg font-mono font-semibold text-zinc-100 tabular-nums">
                                {compute
                                    ? `${compute.runtime_environments} / ${compute.user_pipelines}`
                                    : "—"}
                            </p>
                            <p className="text-[11px] text-zinc-500 font-mono">
                                {compute ? `${compute.workspaces} workspaces` : "—"}
                            </p>
                        </div>
                        <div className="bg-[#0b0c0f] border border-zinc-800/80 rounded-md p-3">
                            <div className="flex items-center gap-1.5 text-zinc-400 mb-1">
                                <ShieldCheck className="w-3.5 h-3.5" />
                                Governance &amp; Audit
                            </div>
                            <p className="text-lg font-mono font-semibold text-zinc-100 tabular-nums">
                                {compute?.active_api_keys ?? "—"} PATs
                            </p>
                            <p className="text-[11px] text-zinc-500 font-mono">
                                {compute ? `${compute.audit_events_24h} audit events (24h)` : "—"}
                            </p>
                        </div>
                    </div>
                    <div className="pt-3 border-t border-zinc-800/80">
                        <p className="text-[11px] uppercase tracking-wider text-zinc-500 mb-2">
                            Usage Telemetry by Domain
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                            {Object.keys(domains).length > 0 ? (
                                Object.entries(domains).map(([dom, cnt]) => (
                                    <span
                                        key={dom}
                                        className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-[11px] font-mono text-zinc-300"
                                    >
                                        {dom}: <span className="text-orange-400">{cnt}</span>
                                    </span>
                                ))
                            ) : (
                                <span className="text-xs text-zinc-500">
                                    No domain usage events recorded yet
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Row 3: Deployed Model Serving Telemetry */}
            <div className="bg-[#101217] border border-zinc-800 rounded-lg overflow-hidden">
                <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between">
                    <h3 className="text-zinc-100 text-sm font-medium flex items-center gap-2">
                        <Radio className="w-4 h-4 text-orange-400" />
                        Deployed Model Serving Telemetry
                    </h3>
                    {serving && (
                        <span className="text-xs font-mono text-zinc-400 tabular-nums">
                            {serving.active_endpoints} active · {serving.paused_endpoints} paused ·{" "}
                            {serving.total_requests} reqs ({serving.error_rate_pct}% errors)
                        </span>
                    )}
                </div>
                {serving && serving.endpoints.length > 0 ? (
                    <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                            <thead className="bg-[#0b0c0f] text-zinc-400 border-b border-zinc-800">
                                <tr>
                                    <th className="px-4 py-2.5 text-left font-medium">Endpoint</th>
                                    <th className="px-4 py-2.5 text-left font-medium">Status</th>
                                    <th className="px-4 py-2.5 text-right font-medium">Canary Split</th>
                                    <th className="px-4 py-2.5 text-right font-medium">Requests</th>
                                    <th className="px-4 py-2.5 text-right font-medium">Errors</th>
                                    <th className="px-4 py-2.5 text-right font-medium">Avg Latency</th>
                                    <th className="px-4 py-2.5 text-left font-medium">Last Invoked</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-zinc-800/70">
                                {serving.endpoints.map((ep) => (
                                    <tr key={ep.id} className="hover:bg-zinc-800/30">
                                        <td className="px-4 py-2.5">
                                            <div className="text-zinc-100 font-medium">{ep.name}</div>
                                            <div className="text-[11px] font-mono text-zinc-500">
                                                /serving/v1/{ep.slug}/invocations
                                            </div>
                                        </td>
                                        <td className="px-4 py-2.5">
                                            <span
                                                className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono ${
                                                    ep.status === "active"
                                                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                                        : "bg-zinc-800 text-zinc-400 border border-zinc-700"
                                                }`}
                                            >
                                                <span
                                                    className={`w-1.5 h-1.5 rounded-full ${
                                                        ep.status === "active"
                                                            ? "bg-emerald-400"
                                                            : "bg-zinc-500"
                                                    }`}
                                                />
                                                {ep.status}
                                            </span>
                                        </td>
                                        <td className="px-4 py-2.5 text-right font-mono text-zinc-300 tabular-nums">
                                            {ep.traffic_split_pct > 0
                                                ? `${ep.traffic_split_pct}% canary`
                                                : "100% primary"}
                                        </td>
                                        <td className="px-4 py-2.5 text-right font-mono text-zinc-100 tabular-nums">
                                            {ep.request_count}
                                        </td>
                                        <td
                                            className={`px-4 py-2.5 text-right font-mono tabular-nums ${
                                                ep.error_count > 0 ? "text-red-400" : "text-zinc-500"
                                            }`}
                                        >
                                            {ep.error_count}
                                        </td>
                                        <td className="px-4 py-2.5 text-right font-mono text-emerald-400 tabular-nums">
                                            {ep.avg_latency_ms.toFixed(1)} ms
                                        </td>
                                        <td className="px-4 py-2.5 font-mono text-zinc-400">
                                            {ep.last_invoked_at
                                                ? new Date(ep.last_invoked_at).toLocaleString()
                                                : "Never"}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <p className="text-zinc-500 text-xs p-5 text-center">
                        No model serving endpoints deployed yet. Deploy a model from Models or Model Registry to stream live serving metrics here.
                    </p>
                )}
            </div>

            {/* Row 4: FinOps Cost Breakdown & Forecast */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5">
                    <h3 className="text-zinc-100 text-sm font-medium mb-4 flex items-center gap-2">
                        <DollarSign className="w-4 h-4 text-emerald-400" />
                        Cost Breakdown by Type
                    </h3>
                    {obs.costBreakdown?.records.length ? (
                        <div className="space-y-2.5 text-xs">
                            {Object.entries(
                                obs.costBreakdown.records.reduce<Record<string, number>>((acc, r) => {
                                    acc[r.cost_type] = (acc[r.cost_type] || 0) + r.amount;
                                    return acc;
                                }, {})
                            ).map(([type, amount]) => (
                                <div key={type} className="flex items-center justify-between">
                                    <span className="text-zinc-300 capitalize">{type}</span>
                                    <span className="text-zinc-100 font-mono tabular-nums">
                                        ${amount.toFixed(4)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-zinc-500 text-xs">No cost records yet</p>
                    )}
                </div>

                <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5">
                    <h3 className="text-zinc-100 text-sm font-medium mb-4 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-zinc-400" />
                        Monthly Forecast
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <p className="text-zinc-400 text-xs">Projected end of month</p>
                            <p className="text-zinc-100 text-xl font-mono font-semibold tabular-nums mt-1">
                                ${projected?.toFixed(2) ?? "—"}
                            </p>
                        </div>
                        <div>
                            <p className="text-zinc-400 text-xs">Daily trend</p>
                            <p className="text-zinc-100 text-xl font-mono font-semibold tabular-nums mt-1">
                                ${obs.costForecast?.daily_trend_usd?.toFixed(4) ?? "—"}/day
                            </p>
                        </div>
                    </div>
                    {overrun && overrun > 100 && (
                        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-md flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-red-400" />
                            <span className="text-red-300 text-xs">
                                Projected to exceed budget by {overrun.toFixed(0)}%
                            </span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// ── Usage Panel ────────────────────────────────────────────────────────────

const USAGE_DOMAINS: Array<{ id: string; label: string }> = [
    { id: "", label: "All domains" },
    { id: "serving", label: "Model Serving" },
    { id: "tabular", label: "Tabular" },
    { id: "nlp", label: "NLP" },
    { id: "pipelines", label: "User Pipelines" },
    { id: "llm", label: "LLM" },
];

function UsagePanel({
    obs,
    domain,
    domainCounts,
    onDomainChange,
}: {
    obs: ReturnType<typeof useObservability>;
    domain: string;
    domainCounts?: Record<string, number>;
    onDomainChange: (domain: string) => void;
}) {
    const metrics = obs.usageMetrics?.metrics ?? [];
    const agg = obs.usageMetrics?.aggregation;

    return (
        <div className="space-y-5">
            {/* Domain filter pills */}
            <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-1.5">
                    {USAGE_DOMAINS.map((d) => {
                        const active = domain === d.id;
                        const count = d.id
                            ? domainCounts?.[d.id]
                            : domainCounts
                            ? Object.values(domainCounts).reduce((a, b) => a + b, 0)
                            : undefined;
                        return (
                            <button
                                key={d.id || "all"}
                                onClick={() => onDomainChange(d.id)}
                                className={`h-7 px-2.5 rounded-md text-xs font-medium border transition-colors flex items-center gap-1.5 ${
                                    active
                                        ? "bg-zinc-100 text-zinc-950 border-zinc-100"
                                        : "bg-[#101217] text-zinc-400 border-zinc-800 hover:text-zinc-200 hover:border-zinc-700"
                                }`}
                            >
                                <span>{d.label}</span>
                                {count !== undefined && (
                                    <span
                                        className={`text-[10px] font-mono tabular-nums ${
                                            active ? "text-zinc-600" : "text-zinc-500"
                                        }`}
                                    >
                                        {count}
                                    </span>
                                )}
                            </button>
                        );
                    })}
                </div>
                <span className="text-xs font-mono text-zinc-500 tabular-nums">
                    Showing {metrics.length} events
                </span>
            </div>

            {agg && (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                    <MiniStat label="Count" value={agg.count} />
                    <MiniStat label="Avg" value={agg.avg.toFixed(2)} />
                    <MiniStat label="Min" value={agg.min.toFixed(2)} />
                    <MiniStat label="P50" value={agg.p50.toFixed(2)} />
                    <MiniStat label="P95" value={agg.p95.toFixed(2)} />
                    <MiniStat label="P99" value={agg.p99.toFixed(2)} />
                    <MiniStat label="Max" value={agg.max.toFixed(2)} />
                </div>
            )}
            <div className="bg-[#101217] border border-zinc-800 rounded-lg overflow-hidden">
                <table className="w-full text-xs">
                    <thead className="bg-[#0b0c0f] border-b border-zinc-800">
                        <tr>
                            <th className="px-4 py-2.5 text-left text-zinc-400 font-medium">Time</th>
                            <th className="px-4 py-2.5 text-left text-zinc-400 font-medium">Metric</th>
                            <th className="px-4 py-2.5 text-right text-zinc-400 font-medium">Value</th>
                            <th className="px-4 py-2.5 text-left text-zinc-400 font-medium">Tags</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/70">
                        {metrics.slice(0, 50).map((m, i) => (
                            <tr key={i} className="hover:bg-zinc-800/30">
                                <td className="px-4 py-2.5 text-zinc-400 font-mono">
                                    {new Date(m.recorded_at).toLocaleString()}
                                </td>
                                <td className="px-4 py-2.5 text-zinc-100 font-mono">
                                    {m.metric_name}
                                </td>
                                <td className="px-4 py-2.5 text-right text-emerald-400 font-mono tabular-nums">
                                    {m.metric_value.toFixed(2)}
                                </td>
                                <td className="px-4 py-2.5 text-zinc-500 font-mono">
                                    {m.tags && Object.keys(m.tags).length > 0
                                        ? Object.entries(m.tags)
                                              .map(([k, v]) => `${k}=${v}`)
                                              .join(", ")
                                        : "—"}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {metrics.length === 0 && (
                    <p className="text-zinc-500 text-xs p-6 text-center">
                        No usage metrics recorded yet{domain ? ` for domain "${domain}"` : ""}
                    </p>
                )}
            </div>
        </div>
    );
}

// ── Costs Panel ────────────────────────────────────────────────────────────

function CostsPanel({ obs }: { obs: ReturnType<typeof useObservability> }) {
    const records = obs.costBreakdown?.records ?? [];

    return (
        <div className="space-y-5">
            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-zinc-100 text-sm font-medium">Total Cost (Period)</h3>
                        <p className="text-zinc-500 text-xs mt-0.5">
                            Includes training, serving, NLP inference, and LLM gateway usage
                        </p>
                    </div>
                    <span className="text-2xl font-mono font-semibold text-emerald-400 tabular-nums">
                        ${obs.costBreakdown?.total_cost_usd.toFixed(4) ?? "0.0000"}
                    </span>
                </div>
            </div>

            <div className="bg-[#101217] border border-zinc-800 rounded-lg overflow-hidden">
                <table className="w-full text-xs">
                    <thead className="bg-[#0b0c0f] border-b border-zinc-800">
                        <tr>
                            <th className="px-4 py-2.5 text-left text-zinc-400 font-medium">Type</th>
                            <th className="px-4 py-2.5 text-left text-zinc-400 font-medium">Model</th>
                            <th className="px-4 py-2.5 text-right text-zinc-400 font-medium">Amount</th>
                            <th className="px-4 py-2.5 text-left text-zinc-400 font-medium">Source</th>
                            <th className="px-4 py-2.5 text-left text-zinc-400 font-medium">Period</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/70">
                        {records.map((r) => (
                            <tr key={r.id} className="hover:bg-zinc-800/30">
                                <td className="px-4 py-2.5">
                                    <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300 font-mono text-[11px] capitalize">
                                        {r.cost_type}
                                    </span>
                                </td>
                                <td className="px-4 py-2.5 text-zinc-300 font-mono">
                                    {r.model_id ?? "—"}
                                </td>
                                <td className="px-4 py-2.5 text-right text-emerald-400 font-mono tabular-nums">
                                    ${r.amount.toFixed(4)}
                                </td>
                                <td className="px-4 py-2.5 text-zinc-400 font-mono">{r.source ?? "—"}</td>
                                <td className="px-4 py-2.5 text-zinc-500 font-mono">
                                    {new Date(r.period_start).toLocaleDateString()}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {records.length === 0 && (
                    <p className="text-zinc-500 text-xs p-6 text-center">No cost records</p>
                )}
            </div>
        </div>
    );
}

// ── Alerts Panel ───────────────────────────────────────────────────────────

function AlertsPanel({
    obs,
    showCreate,
    setShowCreate,
}: {
    obs: ReturnType<typeof useObservability>;
    showCreate: boolean;
    setShowCreate: (v: boolean) => void;
}) {
    return (
        <div className="space-y-5">
            <div className="flex items-center justify-between">
                <h3 className="text-zinc-100 text-sm font-medium">
                    {obs.alerts.length} Alert{obs.alerts.length !== 1 ? "s" : ""} Configured
                </h3>
                <button
                    onClick={() => setShowCreate(!showCreate)}
                    className="h-8 flex items-center gap-1.5 px-3 bg-zinc-100 text-zinc-950 text-xs font-medium rounded-md hover:bg-white transition-colors"
                >
                    <Plus className="w-3.5 h-3.5" />
                    New Alert
                </button>
            </div>

            {showCreate && <CreateAlertForm obs={obs} onClose={() => setShowCreate(false)} />}

            <div className="space-y-2.5">
                {obs.alerts.map((alert) => (
                    <AlertRow key={alert.id} alert={alert} obs={obs} />
                ))}
                {obs.alerts.length === 0 && (
                    <p className="text-zinc-500 text-xs text-center py-8">
                        No alerts configured
                    </p>
                )}
            </div>
        </div>
    );
}

function AlertRow({
    alert,
    obs,
}: {
    alert: Alert;
    obs: ReturnType<typeof useObservability>;
}) {
    return (
        <div
            className={`bg-[#101217] border rounded-lg p-4 flex items-center justify-between ${
                alert.enabled ? "border-zinc-800" : "border-zinc-800/50 opacity-60"
            }`}
        >
            <div className="flex items-center gap-3">
                <div
                    className={`w-2 h-2 rounded-full ${
                        alert.enabled ? "bg-emerald-400" : "bg-zinc-600"
                    }`}
                />
                <div>
                    <p className="text-zinc-100 text-sm font-medium capitalize">
                        {alert.alert_type.replace(/_/g, " ")}
                    </p>
                    <p className="text-zinc-400 font-mono text-xs mt-0.5">
                        {alert.metric_name} {alert.comparison} {alert.threshold}
                    </p>
                </div>
            </div>
            <div className="flex items-center gap-3">
                <span className="text-zinc-500 font-mono text-xs">
                    Cooldown: {alert.cooldown_min}m
                </span>
                <button
                    onClick={() => obs.toggleAlert(alert.id, !alert.enabled)}
                    className="text-zinc-400 hover:text-zinc-100"
                >
                    {alert.enabled ? (
                        <ToggleRight className="w-5 h-5 text-emerald-400" />
                    ) : (
                        <ToggleLeft className="w-5 h-5" />
                    )}
                </button>
                <button
                    onClick={() => obs.deleteAlert(alert.id)}
                    className="text-zinc-400 hover:text-red-400"
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            </div>
        </div>
    );
}

function CreateAlertForm({
    obs,
    onClose,
}: {
    obs: ReturnType<typeof useObservability>;
    onClose: () => void;
}) {
    const [form, setForm] = useState<AlertCreate>({
        alert_type: "cost_spike",
        metric_name: "inference_cost_usd",
        threshold: 2.0,
        comparison: "pct_change",
        cooldown_min: 60,
        channels: [],
    });

    const handleSubmit = async () => {
        await obs.createAlert(form);
        onClose();
    };

    return (
        <div className="bg-[#101217] border border-zinc-700 rounded-lg p-5 space-y-4">
            <h4 className="text-zinc-100 text-sm font-semibold">Create Alert</h4>
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="text-zinc-400 text-[11px] block mb-1">Alert Type</label>
                    <select
                        value={form.alert_type}
                        onChange={(e) => setForm({ ...form, alert_type: e.target.value })}
                        className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 text-zinc-100 rounded-md px-3 text-xs"
                    >
                        <option value="cost_spike">Cost Spike</option>
                        <option value="latency_degradation">Latency Degradation</option>
                        <option value="drift_warning">Drift Warning</option>
                        <option value="drift_critical">Drift Critical</option>
                        <option value="error_rate_spike">Error Rate Spike</option>
                        <option value="budget_near_limit">Budget Near Limit</option>
                    </select>
                </div>
                <div>
                    <label className="text-zinc-400 text-[11px] block mb-1">Metric</label>
                    <input
                        type="text"
                        value={form.metric_name}
                        onChange={(e) => setForm({ ...form, metric_name: e.target.value })}
                        className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 text-zinc-100 font-mono rounded-md px-3 text-xs"
                    />
                </div>
                <div>
                    <label className="text-zinc-400 text-[11px] block mb-1">Threshold</label>
                    <input
                        type="number"
                        step="0.01"
                        value={form.threshold}
                        onChange={(e) => setForm({ ...form, threshold: parseFloat(e.target.value) })}
                        className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 text-zinc-100 font-mono rounded-md px-3 text-xs"
                    />
                </div>
                <div>
                    <label className="text-zinc-400 text-[11px] block mb-1">Comparison</label>
                    <select
                        value={form.comparison}
                        onChange={(e) => setForm({ ...form, comparison: e.target.value })}
                        className="w-full h-9 bg-[#0b0c0f] border border-zinc-800 text-zinc-100 rounded-md px-3 text-xs"
                    >
                        <option value="gt">Greater than</option>
                        <option value="lt">Less than</option>
                        <option value="gte">≥</option>
                        <option value="lte">≤</option>
                        <option value="pct_change">% Change</option>
                    </select>
                </div>
            </div>
            <div className="flex gap-2 justify-end">
                <button
                    onClick={onClose}
                    className="h-8 px-3 text-zinc-400 hover:text-zinc-100 text-xs"
                >
                    Cancel
                </button>
                <button
                    onClick={handleSubmit}
                    className="h-8 px-4 bg-zinc-100 text-zinc-950 font-medium rounded-md text-xs hover:bg-white"
                >
                    Create
                </button>
            </div>
        </div>
    );
}

// ── Pricing Panel ──────────────────────────────────────────────────────────

function PricingPanel({ obs }: { obs: ReturnType<typeof useObservability> }) {
    const grouped = obs.pricing.reduce<Record<string, ResourcePricing[]>>((acc, p) => {
        const cat = p.resource_category;
        if (!acc[cat]) acc[cat] = [];
        acc[cat].push(p);
        return acc;
    }, {});

    const categoryLabels: Record<string, string> = {
        compute_gpu: "GPU Compute",
        compute_cpu: "CPU Compute",
        storage: "Storage",
        llm_tokens: "LLM Tokens",
    };

    return (
        <div className="space-y-5">
            {Object.entries(grouped).map(([category, items]) => (
                <div key={category} className="bg-[#101217] border border-zinc-800 rounded-lg overflow-hidden">
                    <div className="px-4 py-2.5 bg-[#0b0c0f] border-b border-zinc-800">
                        <h3 className="text-zinc-200 text-xs font-semibold uppercase tracking-wider">
                            {categoryLabels[category] ?? category}
                        </h3>
                    </div>
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="border-b border-zinc-800/80">
                                <th className="px-4 py-2 text-left text-zinc-400 font-medium">Resource</th>
                                <th className="px-4 py-2 text-left text-zinc-400 font-medium">Unit</th>
                                <th className="px-4 py-2 text-right text-zinc-400 font-medium">Price</th>
                                <th className="px-4 py-2 text-left text-zinc-400 font-medium">Currency</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/70">
                            {items.map((p) => (
                                <tr key={p.id} className="hover:bg-zinc-800/30">
                                    <td className="px-4 py-2.5 text-zinc-100 font-mono font-medium">{p.resource_id}</td>
                                    <td className="px-4 py-2.5 text-zinc-400 font-mono">{p.pricing_unit}</td>
                                    <td className="px-4 py-2.5 text-right text-emerald-400 font-mono tabular-nums">
                                        ${p.price_per_unit.toFixed(4)}
                                    </td>
                                    <td className="px-4 py-2.5 text-zinc-500 font-mono">{p.currency}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ))}
            {obs.pricing.length === 0 && (
                <p className="text-zinc-500 text-xs text-center py-8">No pricing configured</p>
            )}
        </div>
    );
}

// ── Helper Components ──────────────────────────────────────────────────────

function StatCard({
    icon,
    label,
    value,
    sub,
    color,
}: {
    icon: React.ReactNode;
    label: string;
    value: string | number;
    sub: string;
    color: string;
}) {
    const borderAccent: Record<string, string> = {
        emerald: "border-zinc-800",
        orange: "border-zinc-800",
        blue: "border-zinc-800",
        red: "border-red-500/30",
        amber: "border-zinc-800",
    };

    return (
        <div
            className={`bg-[#101217] border ${
                borderAccent[color] ?? "border-zinc-800"
            } rounded-lg p-4`}
        >
            <div className="flex items-center justify-between mb-2">
                <span className="text-zinc-400 text-xs font-medium">{label}</span>
                {icon}
            </div>
            <p className="text-2xl font-mono font-semibold text-zinc-100 tabular-nums">{value}</p>
            <p className="text-zinc-500 text-[11px] font-mono mt-1 truncate">{sub}</p>
        </div>
    );
}

function MiniStat({ label, value }: { label: string; value: number | string }) {
    return (
        <div className="bg-[#101217] border border-zinc-800 rounded-md p-3 text-center">
            <p className="text-zinc-400 text-[11px] uppercase tracking-wider">{label}</p>
            <p className="text-zinc-100 font-mono font-semibold tabular-nums mt-1">{value}</p>
        </div>
    );
}
