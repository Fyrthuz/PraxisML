"use client";
import React from "react";
import { Search } from "lucide-react";
import { SemanticSearchResult } from "@/lib/api";

interface Props {
    onSearch: (query: string, index: "models" | "datasets") => void;
    results: SemanticSearchResult[];
}

export default function SemanticSearchTab({ onSearch, results }: Props) {
    const [query, setQuery] = React.useState("");
    const [index, setIndex] = React.useState<"models" | "datasets">("models");

    return (
        <div className="space-y-6">
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-violet-500/20 rounded-xl text-violet-400">
                        <Search className="w-5 h-5" />
                    </div>
                    <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider">Búsqueda Semántica</h4>
                </div>
                <div className="flex gap-3">
                    <input
                        className="flex-1 bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="¿Qué estás buscando? p.ej. modelo de sentimiento"
                    />
                    <button
                        onClick={() => onSearch(query, index)}
                        disabled={!query}
                        className="px-5 py-3 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded-xl text-sm font-semibold"
                    >
                        Buscar
                    </button>
                </div>
                <div className="flex gap-2">
                    {(["models", "datasets"] as const).map((i) => (
                        <button
                            key={i}
                            onClick={() => setIndex(i)}
                            className={`px-4 py-2 rounded-lg text-xs font-semibold ${index === i ? "bg-violet-600 text-white" : "bg-neutral-800 text-neutral-400"}`}
                        >
                            {i === "models" ? "Modelos" : "Datasets"}
                        </button>
                    ))}
                </div>
            </div>

            <div className="space-y-3">
                {results.map((r, i) => (
                    <div key={i} className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-5 flex items-center justify-between">
                        <div>
                            <p className="font-semibold text-white">{r.name || r.model_id || r.dataset_id}</p>
                            <p className="text-xs text-neutral-500 mt-1">
                                {r.metadata?.algorithm ? `${r.metadata.algorithm} · ` : ""}
                                {r.metadata?.task_type ? `${r.metadata.task_type} · ` : ""}
                                {r.metadata?.stage ? `stage ${r.metadata.stage}` : ""}
                            </p>
                            <p className="text-xs text-neutral-600 mt-1">{r.metadata?.text_preview || ""}</p>
                        </div>
                        <span className="text-sm font-bold text-violet-400">
                            {(r.score * 100).toFixed(1)}%
                        </span>
                    </div>
                ))}
                {results.length === 0 && (
                    <p className="text-sm text-neutral-500">Sin resultados. Ejecuta una búsqueda.</p>
                )}
            </div>
        </div>
    );
}
