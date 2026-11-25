"use client";
import React from "react";
import { Send, Loader2, Sparkles, Trash2, RefreshCw } from "lucide-react";

interface Props {
    isChatLoading: boolean;
    chatAnswer: string;
    chatSources: { model_id?: string; name?: string; type?: string }[];
    onSend: (message: string, conversationId?: string) => void;
    onFetchHistory: () => void;
    onClearHistory: () => void;
}

const QUICK_PROMPTS = [
    "¿Cómo están mis model endpoints y su latencia?",
    "Resume mis modelos de producción, NLP y métricas",
    "¿Cuál es el estado de drift y costes FinOps?",
];

export default function AssistantTab({
    isChatLoading,
    chatAnswer,
    chatSources,
    onSend,
    onFetchHistory,
    onClearHistory,
}: Props) {
    const [message, setMessage] = React.useState("");

    const submit = (override?: string) => {
        const text = (override ?? message).trim();
        if (!text) return;
        onSend(text);
        if (override === undefined) setMessage("");
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-2 bg-[#101217] border border-zinc-800 rounded-lg p-5 flex flex-col space-y-4 min-h-[420px]">
                <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                    <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-orange-400" />
                        <h4 className="text-xs font-semibold text-zinc-100 uppercase tracking-wider">
                            PraxisML Platform Copilot (RAG + LLM Gateway)
                        </h4>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-500">metadata-only RAG</span>
                </div>

                <div className="flex-1 space-y-3">
                    {chatAnswer ? (
                        <div className="bg-[#0b0c0f] border border-zinc-800 rounded-md p-4 text-xs text-zinc-200 leading-relaxed whitespace-pre-wrap font-mono">
                            {chatAnswer}
                        </div>
                    ) : (
                        <div className="space-y-3 py-4">
                            <p className="text-xs text-zinc-400">
                                Ask about your live models, deployed Model Serving Endpoints, drift anomalies, datasets, or FinOps costs. Raw dataset rows are never exposed to the LLM.
                            </p>
                            <div className="flex flex-wrap gap-1.5">
                                {QUICK_PROMPTS.map((q) => (
                                    <button
                                        key={q}
                                        type="button"
                                        onClick={() => submit(q)}
                                        className="px-2.5 py-1.5 rounded-md bg-[#0b0c0f] border border-zinc-800 text-xs text-zinc-300 hover:border-zinc-700 hover:text-white text-left"
                                    >
                                        {q}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {chatSources.length > 0 && (
                        <div className="flex flex-wrap items-center gap-1.5 pt-1">
                            <span className="text-[10px] font-mono uppercase text-zinc-500">Sources:</span>
                            {chatSources.map((s, i) => (
                                <span
                                    key={i}
                                    className="text-[10px] font-mono px-2 py-0.5 bg-zinc-900 border border-zinc-800 text-zinc-300 rounded"
                                >
                                    {s.name || s.model_id}
                                </span>
                            ))}
                        </div>
                    )}

                    {isChatLoading && (
                        <div className="flex items-center gap-2 text-xs text-zinc-400">
                            <Loader2 className="w-3.5 h-3.5 animate-spin text-orange-400" />
                            Querying RAG context &amp; generating response...
                        </div>
                    )}
                </div>

                <div className="flex gap-2 pt-2 border-t border-zinc-800">
                    <input
                        className="flex-1 h-9 bg-[#0b0c0f] border border-zinc-800 rounded-md px-3 text-xs text-zinc-100 focus:outline-none focus:border-zinc-600"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && submit()}
                        placeholder="Ask about endpoints, models, drift, or FinOps costs..."
                    />
                    <button
                        onClick={() => submit()}
                        disabled={isChatLoading || !message.trim()}
                        className="h-9 px-4 bg-zinc-100 hover:bg-white text-zinc-950 font-medium disabled:opacity-40 rounded-md flex items-center gap-1.5 text-xs"
                    >
                        <Send className="w-3.5 h-3.5" />
                        Send
                    </button>
                </div>
            </div>

            <div className="bg-[#101217] border border-zinc-800 rounded-lg p-5 space-y-3">
                <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider border-b border-zinc-800 pb-3">
                    Conversation Controls
                </h4>
                <button
                    onClick={onFetchHistory}
                    className="w-full h-8 flex items-center justify-center gap-1.5 px-3 bg-[#0b0c0f] hover:bg-zinc-800 border border-zinc-800 text-zinc-300 rounded-md text-xs"
                >
                    <RefreshCw className="w-3.5 h-3.5" /> Reload History
                </button>
                <button
                    onClick={onClearHistory}
                    className="w-full h-8 flex items-center justify-center gap-1.5 px-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 rounded-md text-xs"
                >
                    <Trash2 className="w-3.5 h-3.5" /> Clear History
                </button>
                <div className="pt-3 border-t border-zinc-800 space-y-1.5 text-[11px] text-zinc-500">
                    <p className="font-mono text-zinc-400">RAG Context Sources:</p>
                    <ul className="space-y-1 list-disc list-inside">
                        <li>Active MLModels &amp; MLflow metrics</li>
                        <li>Deployed Model Serving Endpoints</li>
                        <li>Datasets &amp; DVC lineage status</li>
                        <li>Recent Drift checks &amp; FinOps records</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}
