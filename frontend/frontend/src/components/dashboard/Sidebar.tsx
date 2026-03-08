"use client";
import { BarChart3, Database, Filter, Cpu, Clock, FlaskConical, Settings, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

type View = "datasets" | "preprocessing" | "models" | "predictions" | "training";

interface NavItem {
    id: View;
    label: string;
    icon: React.ReactNode;
}

const NAV_ITEMS: NavItem[] = [
    { id: "datasets", label: "Datasets", icon: <Database className="w-5 h-5" /> },
    { id: "preprocessing", label: "Preprocessing", icon: <Filter className="w-5 h-5" /> },
    { id: "models", label: "Models", icon: <Cpu className="w-5 h-5" /> },
    { id: "predictions", label: "Predictions", icon: <Clock className="w-5 h-5" /> },
    { id: "training", label: "Training", icon: <FlaskConical className="w-5 h-5" /> },
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
}: SidebarProps) {
    return (
        <aside className="w-72 border-r border-neutral-900 bg-neutral-950 p-8 flex flex-col shrink-0">
            {/* Logo */}
            <div className="flex items-center gap-3 mb-12">
                <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
                    <BarChart3 className="w-5 h-5" />
                </div>
                <h1 className="text-xl font-bold tracking-tight">PraxisML</h1>
            </div>

            {/* Navigation */}
            <nav className="space-y-1.5 flex-1">
                {NAV_ITEMS.map((item) => (
                    <button
                        key={item.id}
                        onClick={() => onNavigate(item.id)}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 ${activeView === item.id
                            ? "bg-indigo-600 text-white"
                            : "text-neutral-400 hover:text-white hover:bg-neutral-900"
                            }`}
                    >
                        {item.icon}
                        {item.label}
                    </button>
                ))}
            </nav>

            {/* Tenant / User info */}
            <div className="mt-auto pt-6 border-t border-neutral-900">
                <div className="flex flex-col gap-2 p-3 bg-neutral-900/50 rounded-2xl relative">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-sm">
                            {tenantInitials}
                        </div>
                        <div className="flex-1 overflow-hidden">
                            <p className="text-sm font-medium truncate">{tenantName}</p>
                            <p className="text-[10px] text-neutral-500">Free Tier</p>
                        </div>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0 hover:bg-neutral-800 text-neutral-400"
                            onClick={onToggleSwitchTenant}
                        >
                            <Settings className="w-4 h-4" />
                        </Button>
                    </div>

                    {isSwitchingTenant && (
                        <div className="mt-2 pt-2 border-t border-neutral-800 space-y-1">
                            <span className="text-[10px] font-bold text-neutral-500 uppercase ml-1">
                                Switch Organization
                            </span>
                            {userTenants.map((t) => (
                                <button
                                    key={t.id}
                                    className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors ${t.id === currentTenantId
                                        ? "bg-indigo-500/10 text-indigo-400"
                                        : "text-neutral-400 hover:bg-neutral-800 hover:text-white"
                                        }`}
                                    onClick={() =>
                                        alert("To switch tenant, you must re-login and select it!")
                                    }
                                >
                                    {t.name}
                                </button>
                            ))}
                            <Button
                                className="w-full mt-2 bg-neutral-800 hover:bg-neutral-700 text-xs"
                                size="sm"
                                onClick={onCreateTenant}
                            >
                                <Plus className="w-3 h-3 mr-1" /> New Organization
                            </Button>
                        </div>
                    )}

                    <Button
                        variant="destructive"
                        size="sm"
                        className="w-full mt-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20"
                        onClick={onLogout}
                    >
                        Log Out
                    </Button>
                </div>
            </div>
        </aside>
    );
}
