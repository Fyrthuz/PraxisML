"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
    Activity,
    BookMarked,
    Box,
    Clock,
    Cpu,
    Database,
    Filter,
    FlaskConical,
    GitBranch,
    LayoutGrid,
    MessageSquareCode,
    Plus,
    Search,
    Server,
    ShieldCheck,
    Terminal,
    Users,
    Wifi,
} from "lucide-react";
import { View } from "@/components/layout/Sidebar";
import { Dataset, MLModel } from "@/lib/api";

interface CommandItem {
    id: string;
    label: string;
    group: "Navigation" | "Actions" | "Datasets" | "Models";
    hint?: string;
    icon: React.ReactNode;
    onSelect: () => void;
}

interface CommandPaletteProps {
    isOpen: boolean;
    onClose: () => void;
    onNavigate: (view: View) => void;
    onUploadDataset: () => void;
    onOpenDataset: (datasetId: string) => void;
    onDeployModel: (modelId: string) => void;
    datasets: Dataset[];
    models: MLModel[];
    userRole: string;
}

export default function CommandPalette({
    isOpen,
    onClose,
    onNavigate,
    onUploadDataset,
    onOpenDataset,
    onDeployModel,
    datasets,
    models,
    userRole,
}: CommandPaletteProps) {
    const [query, setQuery] = useState("");
    const [selectedIdx, setSelectedIdx] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (isOpen) {
            setQuery("");
            setSelectedIdx(0);
            setTimeout(() => inputRef.current?.focus(), 20);
        }
    }, [isOpen]);

    const items = useMemo<CommandItem[]>(() => {
        const nav: CommandItem[] = [
            { id: "nav-overview", label: "Workspace Overview", group: "Navigation", icon: <LayoutGrid className="w-3.5 h-3.5" />, onSelect: () => onNavigate("overview") },
            { id: "nav-datasets", label: "Datasets", group: "Navigation", icon: <Database className="w-3.5 h-3.5" />, onSelect: () => onNavigate("datasets") },
            { id: "nav-data-reg", label: "Data Registry (DVC)", group: "Navigation", icon: <GitBranch className="w-3.5 h-3.5" />, onSelect: () => onNavigate("data_registry") },
            { id: "nav-prep", label: "Preprocessing Pipelines", group: "Navigation", icon: <Filter className="w-3.5 h-3.5" />, onSelect: () => onNavigate("preprocessing") },
            { id: "nav-envs", label: "Runtime Environments & Code", group: "Navigation", icon: <Box className="w-3.5 h-3.5" />, onSelect: () => onNavigate("environments") },
            { id: "nav-train", label: "Tabular Training", group: "Navigation", icon: <FlaskConical className="w-3.5 h-3.5" />, onSelect: () => onNavigate("training") },
            { id: "nav-nlp", label: "NLP Engine", group: "Navigation", icon: <MessageSquareCode className="w-3.5 h-3.5" />, onSelect: () => onNavigate("nlp") },
            { id: "nav-models", label: "Models", group: "Navigation", icon: <Cpu className="w-3.5 h-3.5" />, onSelect: () => onNavigate("models") },
            { id: "nav-reg", label: "Model Registry (MLflow)", group: "Navigation", icon: <BookMarked className="w-3.5 h-3.5" />, onSelect: () => onNavigate("registry") },
            { id: "nav-endpoints", label: "Model Endpoints", group: "Navigation", icon: <Server className="w-3.5 h-3.5" />, onSelect: () => onNavigate("endpoints") },
            { id: "nav-preds", label: "Predictions", group: "Navigation", icon: <Clock className="w-3.5 h-3.5" />, onSelect: () => onNavigate("predictions") },
            { id: "nav-ws", label: "Streaming WebSockets", group: "Navigation", icon: <Wifi className="w-3.5 h-3.5" />, onSelect: () => onNavigate("streaming") },
            { id: "nav-obs", label: "Observability & FinOps", group: "Navigation", icon: <Activity className="w-3.5 h-3.5" />, onSelect: () => onNavigate("observability") },
            { id: "nav-copilot", label: "Plugins & Copilot Tools", group: "Navigation", icon: <Terminal className="w-3.5 h-3.5" />, onSelect: () => onNavigate("copilot") },
            { id: "nav-gov", label: "API Keys, Workspaces & Audit", group: "Navigation", icon: <ShieldCheck className="w-3.5 h-3.5" />, onSelect: () => onNavigate("governance") },
        ];
        if (userRole === "admin") {
            nav.push({ id: "nav-users", label: "Users & RBAC", group: "Navigation", icon: <Users className="w-3.5 h-3.5" />, onSelect: () => onNavigate("users") });
        }

        const actions: CommandItem[] =
            userRole !== "viewer"
                ? [
                      {
                          id: "act-upload",
                          label: "Upload new dataset",
                          group: "Actions",
                          hint: "CSV / Parquet / Excel / ZIP",
                          icon: <Plus className="w-3.5 h-3.5 text-orange-400" />,
                          onSelect: onUploadDataset,
                      },
                  ]
                : [];

        const dsItems: CommandItem[] = datasets.slice(0, 12).map((ds) => ({
            id: `ds-${ds.id}`,
            label: ds.name,
            group: "Datasets",
            hint: `${ds.num_rows ?? "?"}×${ds.num_columns ?? "?"} · v${ds.version || 1}`,
            icon: <Database className="w-3.5 h-3.5 text-zinc-400" />,
            onSelect: () => onOpenDataset(ds.id),
        }));

        const modelItems: CommandItem[] = models.slice(0, 12).map((m) => ({
            id: `mdl-${m.id}`,
            label: `Deploy "${m.name}"`,
            group: "Models",
            hint: `${m.stage || "Staging"} · v${m.version || "1.0.0"}`,
            icon: <Server className="w-3.5 h-3.5 text-orange-400" />,
            onSelect: () => onDeployModel(m.id),
        }));

        return [...actions, ...nav, ...dsItems, ...modelItems];
    }, [datasets, models, userRole, onNavigate, onUploadDataset, onOpenDataset, onDeployModel]);

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!q) return items.slice(0, 18);
        return items.filter(
            (item) =>
                item.label.toLowerCase().includes(q) ||
                item.group.toLowerCase().includes(q) ||
                (item.hint || "").toLowerCase().includes(q)
        );
    }, [items, query]);

    useEffect(() => {
        setSelectedIdx(0);
    }, [query]);

    useEffect(() => {
        if (!isOpen) return;
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                onClose();
            } else if (e.key === "ArrowDown") {
                e.preventDefault();
                setSelectedIdx((i) => Math.min(i + 1, Math.max(filtered.length - 1, 0)));
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setSelectedIdx((i) => Math.max(i - 1, 0));
            } else if (e.key === "Enter" && filtered[selectedIdx]) {
                e.preventDefault();
                filtered[selectedIdx].onSelect();
                onClose();
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [isOpen, filtered, selectedIdx, onClose]);

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-50 bg-black/70 flex items-start justify-center pt-[14vh] px-4"
            onClick={onClose}
        >
            <div
                className="w-full max-w-xl rounded-lg border border-zinc-800 bg-[#101217] shadow-2xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="flex items-center gap-2.5 px-4 h-12 border-b border-zinc-800">
                    <Search className="w-4 h-4 text-zinc-500 shrink-0" />
                    <input
                        ref={inputRef}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Jump to view, dataset, model or action..."
                        className="flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none"
                    />
                    <span className="text-[10px] font-mono text-zinc-500 border border-zinc-800 px-1.5 py-0.5 rounded">
                        ESC
                    </span>
                </div>

                <div className="max-h-80 overflow-y-auto py-1.5">
                    {filtered.length === 0 ? (
                        <p className="py-8 text-center text-xs text-zinc-500">No matching commands.</p>
                    ) : (
                        filtered.map((item, idx) => (
                            <button
                                key={item.id}
                                onClick={() => {
                                    item.onSelect();
                                    onClose();
                                }}
                                onMouseEnter={() => setSelectedIdx(idx)}
                                className={`w-full px-4 py-2 flex items-center justify-between gap-3 text-left transition-colors ${
                                    idx === selectedIdx ? "bg-zinc-800/80 text-zinc-100" : "text-zinc-300"
                                }`}
                            >
                                <span className="flex items-center gap-2.5 min-w-0">
                                    <span className="text-zinc-400 shrink-0">{item.icon}</span>
                                    <span className="text-xs font-medium truncate">{item.label}</span>
                                </span>
                                <span className="flex items-center gap-2 shrink-0">
                                    {item.hint && (
                                        <span className="text-[11px] font-mono text-zinc-500">
                                            {item.hint}
                                        </span>
                                    )}
                                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-[#0b0c0f] border border-zinc-800 text-zinc-500">
                                        {item.group}
                                    </span>
                                </span>
                            </button>
                        ))
                    )}
                </div>

                <div className="px-4 py-2 border-t border-zinc-800 bg-[#0b0c0f] flex items-center justify-between text-[10px] font-mono text-zinc-500">
                    <span>↑↓ navigate · ↵ select</span>
                    <span>⌘K / Ctrl+K</span>
                </div>
            </div>
        </div>
    );
}
