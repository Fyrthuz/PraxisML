"use client";
import {
    Database,
    Filter,
    Cpu,
    Clock,
    FlaskConical,
    Settings,
    Plus,
    BookMarked,
    GitBranch,
    Wifi,
    Users,
    Activity,
    MessageSquareCode,
    LayoutGrid,
    Terminal,
    LogOut,
    Server,
    ShieldCheck,
    Box,
} from "lucide-react";
import toast from "react-hot-toast";

export type View =
    | "overview"
    | "datasets"
    | "preprocessing"
    | "environments"
    | "models"
    | "predictions"
    | "endpoints"
    | "training"
    | "registry"
    | "data_registry"
    | "streaming"
    | "users"
    | "governance"
    | "observability"
    | "nlp"
    | "copilot";

interface NavItem {
    id: View;
    label: string;
    icon: React.ReactNode;
    adminOnly?: boolean;
    countKey?: "datasets" | "models" | "predictions";
}

interface NavSection {
    title: string;
    items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
    {
        title: "Workspace",
        items: [
            { id: "overview", label: "Overview", icon: <LayoutGrid className="w-4 h-4" /> },
        ],
    },
    {
        title: "Data",
        items: [
            { id: "datasets", label: "Datasets", icon: <Database className="w-4 h-4" />, countKey: "datasets" },
            { id: "data_registry", label: "Data Registry", icon: <GitBranch className="w-4 h-4" /> },
            { id: "preprocessing", label: "Preprocessing", icon: <Filter className="w-4 h-4" /> },
            { id: "environments", label: "Environments & Code", icon: <Box className="w-4 h-4" /> },
        ],
    },
    {
        title: "Training & Models",
        items: [
            { id: "training", label: "Tabular Training", icon: <FlaskConical className="w-4 h-4" /> },
            { id: "nlp", label: "NLP Engine", icon: <MessageSquareCode className="w-4 h-4" /> },
            { id: "models", label: "Models", icon: <Cpu className="w-4 h-4" />, countKey: "models" },
            { id: "registry", label: "Model Registry", icon: <BookMarked className="w-4 h-4" /> },
        ],
    },
    {
        title: "Serving",
        items: [
            { id: "endpoints", label: "Model Endpoints", icon: <Server className="w-4 h-4" /> },
            { id: "predictions", label: "Predictions", icon: <Clock className="w-4 h-4" />, countKey: "predictions" },
            { id: "streaming", label: "Streaming WS", icon: <Wifi className="w-4 h-4" /> },
        ],
    },
    {
        title: "Operations",
        items: [
            { id: "observability", label: "Observability", icon: <Activity className="w-4 h-4" /> },
            { id: "copilot", label: "Plugins & Tools", icon: <Terminal className="w-4 h-4" /> },
            { id: "governance", label: "API Keys & Audit", icon: <ShieldCheck className="w-4 h-4" /> },
            { id: "users", label: "Users & RBAC", icon: <Users className="w-4 h-4" />, adminOnly: true },
        ],
    },
];

interface SidebarProps {
    activeView: View;
    onNavigate: (view: View) => void;
    tenantName: string;
    tenantInitials: string;
    userTenants: any[];
    isSwitchingTenant: boolean;
    onToggleSwitchTenant: () => void;
    onCreateTenant: () => void;
    currentTenantId?: string;
    onLogout: () => void;
    userRole?: string;
    counts?: {
        datasets: number;
        models: number;
        predictions: number;
    };
}

export default function Sidebar({
    activeView,
    onNavigate,
    tenantName,
    tenantInitials,
    userTenants,
    isSwitchingTenant,
    onToggleSwitchTenant,
    onCreateTenant,
    currentTenantId,
    onLogout,
    userRole = "viewer",
    counts,
}: SidebarProps) {
    return (
        <aside className="w-60 border-r border-zinc-800/90 bg-[#101217] flex flex-col shrink-0 h-screen select-none">
            {/* Brand bar */}
            <div className="h-14 px-4 border-b border-zinc-800/80 flex items-center justify-between">
                <button
                    onClick={() => onNavigate("overview")}
                    className="flex items-center gap-2.5 text-left"
                >
                    <div className="w-6 h-6 rounded bg-orange-500 flex items-center justify-center text-[11px] font-bold text-zinc-950 font-mono">
                        P
                    </div>
                    <span className="text-sm font-semibold tracking-tight text-zinc-100">
                        PraxisML
                    </span>
                </button>
                <span className="text-[10px] font-mono text-zinc-500 border border-zinc-800 px-1.5 py-0.5 rounded">
                    local
                </span>
            </div>

            {/* Navigation */}
            <nav className="flex-1 overflow-y-auto px-2.5 py-3 space-y-4">
                {NAV_SECTIONS.map((section) => {
                    const visibleItems = section.items.filter(
                        (item) => !item.adminOnly || userRole === "admin"
                    );
                    if (visibleItems.length === 0) return null;

                    return (
                        <div key={section.title}>
                            <p className="px-2.5 mb-1 text-[11px] font-medium text-zinc-500">
                                {section.title}
                            </p>
                            <div className="space-y-0.5">
                                {visibleItems.map((item) => {
                                    const badgeCount =
                                        item.countKey && counts ? counts[item.countKey] : undefined;
                                    const isActive = activeView === item.id;
                                    return (
                                        <button
                                            key={item.id}
                                            onClick={() => onNavigate(item.id)}
                                            className={`relative w-full flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-md text-[13px] transition-colors ${
                                                isActive
                                                    ? "bg-zinc-800/80 text-zinc-100 font-medium"
                                                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                                            }`}
                                        >
                                            {isActive && (
                                                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-orange-500" />
                                            )}
                                            <span className="flex items-center gap-2.5 truncate">
                                                <span className={isActive ? "text-zinc-100" : "text-zinc-500"}>
                                                    {item.icon}
                                                </span>
                                                <span className="truncate">{item.label}</span>
                                            </span>
                                            {badgeCount !== undefined && badgeCount > 0 && (
                                                <span className="text-[11px] font-mono tabular-nums text-zinc-500">
                                                    {badgeCount}
                                                </span>
                                            )}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </nav>

            {/* Organization footer */}
            <div className="p-3 border-t border-zinc-800/80 space-y-2">
                <div className="flex items-center gap-2.5 px-1.5 py-1">
                    <div className="w-7 h-7 rounded bg-zinc-800 border border-zinc-700 flex items-center justify-center text-[11px] font-mono font-semibold text-zinc-200 shrink-0">
                        {tenantInitials}
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-zinc-200 truncate">{tenantName}</p>
                        <p className="text-[10px] font-mono uppercase text-zinc-500">{userRole}</p>
                    </div>
                    <button
                        onClick={onToggleSwitchTenant}
                        className="p-1.5 rounded hover:bg-zinc-800 text-zinc-500 hover:text-zinc-200 transition-colors"
                        title="Organizations"
                    >
                        <Settings className="w-3.5 h-3.5" />
                    </button>
                    <button
                        onClick={onLogout}
                        className="p-1.5 rounded hover:bg-zinc-800 text-zinc-500 hover:text-red-400 transition-colors"
                        title="Log out"
                    >
                        <LogOut className="w-3.5 h-3.5" />
                    </button>
                </div>

                {isSwitchingTenant && (
                    <div className="pt-2 border-t border-zinc-800/80 space-y-1">
                        <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500 px-1.5">
                            Organizations
                        </span>
                        {userTenants.map((t) => (
                            <button
                                key={t.id}
                                className={`w-full text-left px-2 py-1 text-xs rounded transition-colors ${
                                    t.id === currentTenantId
                                        ? "bg-zinc-800 text-zinc-100 font-medium"
                                        : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
                                }`}
                                onClick={() => {
                                    if (t.id !== currentTenantId) {
                                        toast("Sign in with the target organization credentials to switch workspace.");
                                    }
                                }}
                            >
                                {t.name}
                            </button>
                        ))}
                        <button
                            onClick={onCreateTenant}
                            className="w-full flex items-center justify-center gap-1 mt-1 px-2 py-1.5 text-xs text-zinc-300 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded transition-colors"
                        >
                            <Plus className="w-3 h-3" /> New organization
                        </button>
                    </div>
                )}
            </div>
        </aside>
    );
}
