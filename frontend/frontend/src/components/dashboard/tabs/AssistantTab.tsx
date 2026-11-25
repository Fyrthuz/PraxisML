"use client";
import React from "react";
import { Send, Loader2, Sparkles, Trash2 } from "lucide-react";

interface Props {
    isChatLoading: boolean;
    chatAnswer: string;
    chatSources: { model_id?: string; name?: string; type?: string }[];
    onSend: (message: string, conversationId?: string) => void;
    onFetchHistory: () => void;
    onClearHistory: () => void;
}

export default function AssistantTab({
    isChatLoading,
    chatAnswer,
    chatSources,
    onSend,
    onFetchHistory,
    onClearHistory,
}: Props) {
    const [message, setMessage] = React.useState("");

    const submit = () => {
        if (!message.trim()) return;
        onSend(message.trim());
        setMessage("");
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 flex flex-col space-y-4 min-h-[420px]">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-zinc-800 rounded-xl text-zinc-300">
                        <Sparkles className="w-5 h-5" />
                    </div>
                    <h4 className="text-sm font-bold text-neutral-300 uppercase tracking-wider">Asistente PraxisML</h4>
                </div>

                <div className="flex-1 space-y-3">
                    {chatAnswer ? (
                        <div className="bg-zinc-900 border border-zinc-700 rounded-xl p-4 text-sm text-neutral-200 whitespace-pre-wrap">
                            {chatAnswer}
                        </div>
                    ) : (
                        <p className="text-sm text-neutral-500">
                            Pregunta sobre tus modelos: "¿Cómo está mi modelo de producción?", "¿Hay drift?",
                            "¿Cuánto cuesta la inferencia este mes?"
                        </p>
                    )}
                    {chatSources.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                            {chatSources.map((s, i) => (
                                <span key={i} className="text-[10px] px-2 py-1 bg-neutral-800 text-neutral-400 rounded-full">
                                    {s.name || s.model_id}
                                </span>
                            ))}
                        </div>
                    )}
                    {isChatLoading && (
                        <div className="flex items-center gap-2 text-xs text-neutral-400">
                            <Loader2 className="w-4 h-4 animate-spin" /> Generando respuesta...
                        </div>
                    )}
                </div>

                <div className="flex gap-3">
                    <input
                        className="flex-1 bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-3 text-sm"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && submit()}
                        placeholder="Escribe tu pregunta..."
                    />
                    <button
                        onClick={submit}
                        disabled={isChatLoading || !message.trim()}
                        className="px-5 py-3 bg-zinc-100 hover:bg-white text-zinc-950 font-medium disabled:opacity-40 text-white rounded-xl"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 space-y-3">
                <h4 className="text-sm font-bold text-neutral-400 uppercase tracking-wider">Historial</h4>
                <button onClick={onFetchHistory} className="w-full px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-xl text-sm">
                    Cargar historial
                </button>
                <button onClick={onClearHistory} className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 rounded-xl text-sm">
                    <Trash2 className="w-4 h-4" /> Borrar historial
                </button>
            </div>
        </div>
    );
}
