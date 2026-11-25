"use client";
import React from "react";
import { Search } from "lucide-react";
import { SemanticSearchResult } from "@/lib/api";

interface Props {
    onSearch: (query: string, index: "models" | "datasets") => void;
    results: SemanticSearchResult[];
}

const EXAMPLE_QUERIES = [
    "sentiment classification NLP",
    "titanic survival prediction",
    "preprocessed customer dataset",
];

export default function SemanticSearchTab({ onSearch, results }: Props) {
    const [query, setQuery] = React.useState("");
    const [index, setIndex] = React.useState<"models" | "datasets">("models");

    const run = (q?: string) => {
        const target = (q ?? query).trim();
        if (!target) return;
        if (q !== undefined) setQuery(q);
        onSearch(target, index);
    };

    return (
        <div className="space-y-5">
            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 pb-3">
                    <div className="flex items-center gap-2">
                        <Search className="w-4 h-4 text-orange-400" />
                        <h4 className="text-xs font-semibold text-zinc-100 uppercase tracking-wider">
                            Semantic Vector Search (pgvector + SentenceTransformers)
                        </h4>
                    </div>
                    <div className="flex items-center gap-1">
                        {(["models", "datasets"] as const).map((i) => (
                            <button
                                key={i}
                                onClick={() => setIndex(i)}
                                className={`h-7 px-2.5 rounded-md text-xs font-medium border transition-colors ${
                                    index === i
                                        ? "bg-zinc-100 text-zinc-950 border-zinc-100"
                                        : "bg-[#0b0c0f] text-zinc-400 border-zinc-800 hover:text-zinc-200"
                                }`}
                            >
                                {i === "models" ? "Models Index" : "Datasets Index"}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="flex gap-2">
                    <input
                        className="flex-1 h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && run()}
                        placeholder="Natural language query, e.g. sentiment classifier or titanic survival model..."
                    />
                    <button
                        onClick={() => run()}
                        disabled={!query.trim()}
                        className="h-9 px-4 bg-zinc-100 hover:bg-white disabled:opacity-40 text-zinc-950 rounded-md text-xs font-medium"
                    >
                        Search
                    </button>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[11px] text-zinc-500">Quick queries:</span>
                    {EXAMPLE_QUERIES.map((q) => (
                        <button
                            key={q}
                            type="button"
                            onClick={() => run(q)}
                            className="px-2 py-0.5 rounded bg-[#0b0c0f] border border-zinc-800 text-[11px] font-mono text-zinc-400 hover:text-zinc-200 hover:border-zinc-700"
                        >
                            {q}
                        </button>
                    ))}
                </div>
            </div>

            <div className="space-y-2.5">
                {results.map((r, i) => (
                    <div
                        key={i}
                        className="bg-[#101217] border border-zinc-800 rounded-lg p-4 flex items-center justify-between gap-4"
                    >
                        <div className="min-w-0">
                            <p className="text-sm font-medium text-zinc-100 truncate">
                                {r.name || r.model_id || r.dataset_id}
                            </p>
                            <p className="text-[11px] font-mono text-zinc-400 mt-1">
                                {r.metadata?.domain ? `${r.metadata.domain} · ` : ""}
                                {r.metadata?.algorithm ? `${r.metadata.algorithm} · ` : ""}
                                {r.metadata?.task_type ? `${r.metadata.task_type} · ` : ""}
                                {r.metadata?.stage ? `stage: ${r.metadata.stage}` : ""}
                            </p>
                            {r.metadata?.text_preview && (
                                <p className="text-xs text-zinc-500 mt-1 truncate">
                                    {r.metadata.text_preview}
                                </p>
                            )}
                        </div>
                        <div className="shrink-0 text-right">
                            <span className="text-sm font-mono font-semibold text-orange-400 tabular-nums">
                                {(r.score * 100).toFixed(1)}%
                            </span>
                            <p className="text-[10px] font-mono text-zinc-500">cosine sim</p>
                        </div>
                    </div>
                ))}
                {results.length === 0 && (
                    <div className="bg-[#101217] border border-dashed border-zinc-800 rounded-lg p-8 text-center text-xs text-zinc-500">
                        Run a semantic search to retrieve models or datasets by vector similarity.
                    </div>
                )}
            </div>
        </div>
    );
}
