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
} from "lucide-react";
import type { Alert, AlertCreate, ResourcePricing, HealthScore } from "@/lib/api";
import { useObservability } from "@/hooks/useObservability";

type SubTab = "overview" | "usage" | "costs" | "alerts" | "pricing";

interface ObservabilityTabProps {
    token: string | null;
}

export default function ObservabilityTab({ token }: ObservabilityTabProps) {
    const obs = useObservability(token);
    const [showCreateAlert, setShowCreateAlert] = useState(false);

    useEffect(() => {
        obs.fetchUsage();
        obs.fetchCosts();
        obs.fetchForecast();
        obs.fetchAlerts();
        obs.fetchPricing();
    }, [token]);

    const handleTabChange = (tab: SubTab) => {
        obs.setActiveTab(tab);
        switch (tab) {
            case "usage":
                obs.fetchUsage();
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
        <div className="flex-1 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                        <Activity className="w-7 h-7 text-indigo-400" />
                        Observability
                    </h2>
                    <p className="text-neutral-400 text-sm mt-1">
                        Monitor usage, costs, alerts and resource pricing
                    </p>
                </div>
                <button
                    onClick={() => {
                        obs.fetchUsage();
                        obs.fetchCosts();
                        obs.fetchForecast();
                        obs.fetchAlerts();
                        obs.fetchPricing();
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-neutral-800 text-neutral-300 rounded-xl hover:bg-neutral-700 transition-colors"
                >
                    <RefreshCw className={`w-4 h-4 ${obs.isLoading ? "animate-spin" : ""}`} />
                    Refresh
                </button>
            </div>

            {/* Sub-tabs */}
            <div className="flex gap-1 bg-neutral-900/50 p-1 rounded-xl mb-6 w-fit">
                {([
                    { id: "overview", label: "Overview", icon: <BarChart3 className="w-4 h-4" /> },
                    { id: "usage", label: "Usage", icon: <Activity className="w-4 h-4" /> },
                    { id: "costs", label: "Costs", icon: <DollarSign className="w-4 h-4" /> },
                    { id: "alerts", label: "Alerts", icon: <Bell className="w-4 h-4" /> },
                    { id: "pricing", label: "Pricing", icon: <Cpu className="w-4 h-4" /> },
                ] as { id: SubTab; label: string; icon: React.ReactNode }[]).map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => handleTabChange(tab.id)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            obs.activeTab === tab.id
                                ? "bg-indigo-600 text-white"
                                : "text-neutral-400 hover:text-white hover:bg-neutral-800"
                        }`}
                    >
                        {tab.icon}
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto">
                {obs.activeTab === "overview" && <OverviewPanel obs={obs} />}
                {obs.activeTab === "usage" && <UsagePanel obs={obs} />}
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

function OverviewPanel({ obs }: { obs: ReturnType<typeof useObservability> }) {
    const totalCost = obs.costBreakdown?.total_cost_usd ?? 0;
    const totalRecords = obs.costBreakdown?.records.length ?? 0;
    const activeAlerts = obs.alerts.filter((a) => a.enabled).length;
    const budget = obs.costForecast?.budget_usd;
    const overrun = obs.costForecast?.overrun_risk_pct;
    const projected = obs.costForecast?.projected_month_end_usd;

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
                icon={<DollarSign className="w-5 h-5" />}
                label="Monthly Cost"
                value={`$${totalCost.toFixed(2)}`}
                sub={`${totalRecords} records`}
                color="emerald"
            />
            <StatCard
                icon={<Activity className="w-5 h-5" />}
                label="Usage Metrics"
                value={obs.usageMetrics?.metrics.length ?? 0}
                sub="data points"
                color="blue"
            />
            <StatCard
                icon={<Bell className="w-5 h-5" />}
                label="Active Alerts"
                value={activeAlerts}
                sub={`${obs.alerts.length} total`}
                color={activeAlerts > 5 ? "red" : "amber"}
            />
            <StatCard
                icon={<TrendingUp className="w-5 h-5" />}
                label="Budget Forecast"
                value={budget ? `$${budget.toFixed(0)}` : "N/A"}
                sub={overrun ? `${overrun.toFixed(0)}% projected use` : "No budget set"}
                color={overrun && overrun > 100 ? "red" : "indigo"}
            />

            {/* Cost breakdown by type */}
            <div className="col-span-1 md:col-span-2 bg-neutral-900 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                    <DollarSign className="w-5 h-5 text-emerald-400" />
                    Cost Breakdown by Type
                </h3>
                {obs.costBreakdown?.records.length ? (
                    <div className="space-y-3">
                        {Object.entries(
                            obs.costBreakdown.records.reduce<Record<string, number>>((acc, r) => {
                                acc[r.cost_type] = (acc[r.cost_type] || 0) + r.amount;
                                return acc;
                            }, {})
                        ).map(([type, amount]) => (
                            <div key={type} className="flex items-center justify-between">
                                <span className="text-neutral-300 capitalize">{type}</span>
                                <span className="text-white font-mono">${amount.toFixed(4)}</span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-neutral-500 text-sm">No cost records yet</p>
                )}
            </div>

            {/* Forecast */}
            <div className="col-span-1 md:col-span-2 bg-neutral-900 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-indigo-400" />
                    Monthly Forecast
                </h3>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <p className="text-neutral-400 text-sm">Projected end of month</p>
                        <p className="text-white text-2xl font-bold">
                            ${projected?.toFixed(2) ?? "—"}
                        </p>
                    </div>
                    <div>
                        <p className="text-neutral-400 text-sm">Daily trend</p>
                        <p className="text-white text-2xl font-bold">
                            ${obs.costForecast?.daily_trend_usd?.toFixed(4) ?? "—"}/day
                        </p>
                    </div>
                </div>
                {overrun && overrun > 100 && (
                    <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-red-300 text-sm">
                            Projected to exceed budget by {overrun.toFixed(0)}%
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}

// ── Usage Panel ────────────────────────────────────────────────────────────

function UsagePanel({ obs }: { obs: ReturnType<typeof useObservability> }) {
    const metrics = obs.usageMetrics?.metrics ?? [];
    const agg = obs.usageMetrics?.aggregation;

    return (
        <div className="space-y-6">
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
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-neutral-800/50">
                        <tr>
                            <th className="px-4 py-3 text-left text-neutral-400 font-medium">Time</th>
                            <th className="px-4 py-3 text-left text-neutral-400 font-medium">Metric</th>
                            <th className="px-4 py-3 text-right text-neutral-400 font-medium">Value</th>
                            <th className="px-4 py-3 text-left text-neutral-400 font-medium">Tags</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-800">
                        {metrics.slice(0, 50).map((m, i) => (
                            <tr key={i} className="hover:bg-neutral-800/30">
                                <td className="px-4 py-2.5 text-neutral-300 font-mono text-xs">
                                    {new Date(m.recorded_at).toLocaleString()}
                                </td>
                                <td className="px-4 py-2.5 text-white">{m.metric_name}</td>
                                <td className="px-4 py-2.5 text-right text-emerald-400 font-mono">
                                    {m.metric_value.toFixed(2)}
                                </td>
                                <td className="px-4 py-2.5 text-neutral-500 text-xs">
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
                    <p className="text-neutral-500 text-sm p-6 text-center">
                        No usage metrics recorded yet
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
        <div className="space-y-6">
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-white font-semibold">Total Cost (Period)</h3>
                    <span className="text-2xl font-bold text-emerald-400">
                        ${obs.costBreakdown?.total_cost_usd.toFixed(2) ?? "0.00"}
                    </span>
                </div>
            </div>

            <div className="bg-neutral-900 border border-neutral-800 rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-neutral-800/50">
                        <tr>
                            <th className="px-4 py-3 text-left text-neutral-400 font-medium">Type</th>
                            <th className="px-4 py-3 text-left text-neutral-400 font-medium">Model</th>
                            <th className="px-4 py-3 text-right text-neutral-400 font-medium">Amount</th>
                            <th className="px-4 py-3 text-left text-neutral-400 font-medium">Source</th>
                            <th className="px-4 py-3 text-left text-neutral-400 font-medium">Period</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-800">
                        {records.map((r) => (
                            <tr key={r.id} className="hover:bg-neutral-800/30">
                                <td className="px-4 py-2.5">
                                    <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 text-xs capitalize">
                                        {r.cost_type}
                                    </span>
                                </td>
                                <td className="px-4 py-2.5 text-neutral-300 font-mono text-xs">
                                    {r.model_id ?? "—"}
                                </td>
                                <td className="px-4 py-2.5 text-right text-emerald-400 font-mono">
                                    ${r.amount.toFixed(4)}
                                </td>
                                <td className="px-4 py-2.5 text-neutral-500 text-xs">{r.source ?? "—"}</td>
                                <td className="px-4 py-2.5 text-neutral-500 text-xs">
                                    {new Date(r.period_start).toLocaleDateString()}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {records.length === 0 && (
                    <p className="text-neutral-500 text-sm p-6 text-center">No cost records</p>
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
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-white font-semibold">
                    {obs.alerts.length} Alert{obs.alerts.length !== 1 ? "s" : ""} Configured
                </h3>
                <button
                    onClick={() => setShowCreate(!showCreate)}
                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-xl hover:bg-indigo-500 transition-colors"
                >
                    <Plus className="w-4 h-4" />
                    New Alert
                </button>
            </div>

            {showCreate && <CreateAlertForm obs={obs} onClose={() => setShowCreate(false)} />}

            <div className="space-y-3">
                {obs.alerts.map((alert) => (
                    <AlertRow key={alert.id} alert={alert} obs={obs} />
                ))}
                {obs.alerts.length === 0 && (
                    <p className="text-neutral-500 text-sm text-center py-8">
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
            className={`bg-neutral-900 border rounded-xl p-4 flex items-center justify-between ${
                alert.enabled ? "border-neutral-800" : "border-neutral-800/50 opacity-60"
            }`}
        >
            <div className="flex items-center gap-4">
                <div
                    className={`w-2 h-2 rounded-full ${
                        alert.enabled ? "bg-emerald-400" : "bg-neutral-600"
                    }`}
                />
                <div>
                    <p className="text-white font-medium capitalize">{alert.alert_type.replace(/_/g, " ")}</p>
                    <p className="text-neutral-400 text-xs mt-0.5">
                        {alert.metric_name} {alert.comparison} {alert.threshold}
                    </p>
                </div>
            </div>
            <div className="flex items-center gap-3">
                <span className="text-neutral-500 text-xs">
                    Cooldown: {alert.cooldown_min}m
                </span>
                <button
                    onClick={() => obs.toggleAlert(alert.id, !alert.enabled)}
                    className="text-neutral-400 hover:text-white"
                >
                    {alert.enabled ? (
                        <ToggleRight className="w-5 h-5 text-emerald-400" />
                    ) : (
                        <ToggleLeft className="w-5 h-5" />
                    )}
                </button>
                <button
                    onClick={() => obs.deleteAlert(alert.id)}
                    className="text-neutral-400 hover:text-red-400"
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
        <div className="bg-neutral-900 border border-indigo-500/30 rounded-xl p-6 space-y-4">
            <h4 className="text-white font-semibold">Create Alert</h4>
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="text-neutral-400 text-xs block mb-1">Alert Type</label>
                    <select
                        value={form.alert_type}
                        onChange={(e) => setForm({ ...form, alert_type: e.target.value })}
                        className="w-full bg-neutral-800 text-white rounded-lg px-3 py-2 text-sm"
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
                    <label className="text-neutral-400 text-xs block mb-1">Metric</label>
                    <input
                        type="text"
                        value={form.metric_name}
                        onChange={(e) => setForm({ ...form, metric_name: e.target.value })}
                        className="w-full bg-neutral-800 text-white rounded-lg px-3 py-2 text-sm"
                    />
                </div>
                <div>
                    <label className="text-neutral-400 text-xs block mb-1">Threshold</label>
                    <input
                        type="number"
                        step="0.01"
                        value={form.threshold}
                        onChange={(e) => setForm({ ...form, threshold: parseFloat(e.target.value) })}
                        className="w-full bg-neutral-800 text-white rounded-lg px-3 py-2 text-sm"
                    />
                </div>
                <div>
                    <label className="text-neutral-400 text-xs block mb-1">Comparison</label>
                    <select
                        value={form.comparison}
                        onChange={(e) => setForm({ ...form, comparison: e.target.value })}
                        className="w-full bg-neutral-800 text-white rounded-lg px-3 py-2 text-sm"
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
                    className="px-4 py-2 text-neutral-400 hover:text-white text-sm"
                >
                    Cancel
                </button>
                <button
                    onClick={handleSubmit}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-500"
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
        <div className="space-y-6">
            {Object.entries(grouped).map(([category, items]) => (
                <div key={category} className="bg-neutral-900 border border-neutral-800 rounded-xl overflow-hidden">
                    <div className="px-6 py-3 bg-neutral-800/50 border-b border-neutral-800">
                        <h3 className="text-white font-semibold capitalize">
                            {categoryLabels[category] ?? category}
                        </h3>
                    </div>
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-neutral-800">
                                <th className="px-4 py-2 text-left text-neutral-400 font-medium">Resource</th>
                                <th className="px-4 py-2 text-left text-neutral-400 font-medium">Unit</th>
                                <th className="px-4 py-2 text-right text-neutral-400 font-medium">Price</th>
                                <th className="px-4 py-2 text-left text-neutral-400 font-medium">Currency</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-800">
                            {items.map((p) => (
                                <tr key={p.id} className="hover:bg-neutral-800/30">
                                    <td className="px-4 py-2.5 text-white font-medium">{p.resource_id}</td>
                                    <td className="px-4 py-2.5 text-neutral-400 text-xs">{p.pricing_unit}</td>
                                    <td className="px-4 py-2.5 text-right text-emerald-400 font-mono">
                                        ${p.price_per_unit.toFixed(4)}
                                    </td>
                                    <td className="px-4 py-2.5 text-neutral-500">{p.currency}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ))}
            {obs.pricing.length === 0 && (
                <p className="text-neutral-500 text-sm text-center py-8">No pricing configured</p>
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
    const colorClasses: Record<string, string> = {
        emerald: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
        blue: "text-blue-400 bg-blue-400/10 border-blue-400/20",
        red: "text-red-400 bg-red-400/10 border-red-400/20",
        amber: "text-amber-400 bg-amber-400/10 border-amber-400/20",
        indigo: "text-indigo-400 bg-indigo-400/10 border-indigo-400/20",
    };

    return (
        <div className={`bg-neutral-900 border border-neutral-800 rounded-xl p-5 ${colorClasses[color] ?? ""}`}>
            <div className="flex items-center gap-2 mb-3">
                {icon}
                <span className="text-neutral-400 text-xs font-medium">{label}</span>
            </div>
            <p className="text-2xl font-bold text-white">{value}</p>
            <p className="text-neutral-500 text-xs mt-1">{sub}</p>
        </div>
    );
}

function MiniStat({ label, value }: { label: string; value: number | string }) {
    return (
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3 text-center">
            <p className="text-neutral-400 text-xs">{label}</p>
            <p className="text-white font-mono font-semibold mt-1">{value}</p>
        </div>
    );
}
