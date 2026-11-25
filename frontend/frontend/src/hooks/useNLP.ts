import { useState, useCallback, useEffect } from "react";
import toast from "react-hot-toast";
import { config } from "@/lib/config";
import type {
    NLPAlgorithm,
    NLPTrainingStatus,
    NLPPrediction,
    EmbeddingResponse,
    SemanticSearchResult,
    ChatResponse,
    ChatHistoryItem,
} from "@/lib/api";

const API = config.getFullApiUrl("/nlp");

export function useNLP(token: string | null) {
    const [algorithms, setAlgorithms] = useState<NLPAlgorithm[]>([]);
    const [trainingTaskId, setTrainingTaskId] = useState<string | null>(null);
    const [trainingStatus, setTrainingStatus] = useState<NLPTrainingStatus | null>(null);
    const [isTraining, setIsTraining] = useState(false);
    const [prediction, setPrediction] = useState<NLPPrediction | null>(null);
    const [isPredicting, setIsPredicting] = useState(false);
    const [embeddings, setEmbeddings] = useState<EmbeddingResponse | null>(null);
    const [searchResults, setSearchResults] = useState<SemanticSearchResult[]>([]);
    const [chatAnswer, setChatAnswer] = useState<string>("");
    const [chatSources, setChatSources] = useState<ChatResponse["sources"]>([]);
    const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
    const [isChatLoading, setIsChatLoading] = useState(false);

    useEffect(() => {
        if (!token) return;
        fetch(`${API}/algorithms`, { headers: { Authorization: `Bearer ${token}` } })
            .then((res) => res.json())
            .then((data) => {
                if (Array.isArray(data)) setAlgorithms(data);
            })
            .catch((err) => console.error("Failed to load NLP algorithms:", err));
    }, [token]);

    // ── Training ──────────────────────────────────────────────────────────
    const startNLPTraining = useCallback(
        async (body: Record<string, any>) => {
            if (!token) return;
            setIsTraining(true);
            setTrainingStatus(null);
            try {
                const res = await fetch(`${API}/train`, {
                    method: "POST",
                    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Error al lanzar el entrenamiento NLP");
                setTrainingTaskId(data.task_id);
                toast.success("Entrenamiento NLP lanzado");
            } catch (err: any) {
                toast.error(err.message);
                setIsTraining(false);
            }
        },
        [token]
    );

    useEffect(() => {
        if (!trainingTaskId || !token) return;
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${API}/training/status/${trainingTaskId}`, {
                    headers: { Authorization: `Bearer ${token}` },
                });
                const data: NLPTrainingStatus = await res.json();
                setTrainingStatus(data);
                if (data.status === "SUCCESS") {
                    toast.success("Entrenamiento NLP completado");
                    setIsTraining(false);
                    setTrainingTaskId(null);
                } else if (data.status === "FAILURE") {
                    toast.error(`Entrenamiento falló: ${data.error || "Error desconocido"}`);
                    setIsTraining(false);
                    setTrainingTaskId(null);
                }
            } catch (err) {
                console.error("NLP training poll error", err);
            }
        }, 3000);
        return () => clearInterval(interval);
    }, [trainingTaskId, token]);

    // ── Inference ─────────────────────────────────────────────────────────
    const predictNLP = useCallback(
        async (modelId: string, text: string, explain = false) => {
            if (!token) return;
            setIsPredicting(true);
            try {
                const res = await fetch(`${API}/predict`, {
                    method: "POST",
                    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
                    body: JSON.stringify({ model_id: modelId, text, explain }),
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Error en la predicción");
                setPrediction(data);
            } catch (err: any) {
                toast.error(err.message);
            } finally {
                setIsPredicting(false);
            }
        },
        [token]
    );

    const batchPredictNLP = useCallback(
        async (modelId: string, textColumn: string, file: File) => {
            if (!token) return;
            const form = new FormData();
            form.append("model_id", modelId);
            form.append("text_column", textColumn);
            form.append("file", file);
            try {
                const res = await fetch(`${API}/predict/batch`, {
                    method: "POST",
                    headers: { Authorization: `Bearer ${token}` },
                    body: form,
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Error en batch");
                toast.success(`Batch encolado: ${data.prediction_id}`);
                return data;
            } catch (err: any) {
                toast.error(err.message);
            }
        },
        [token]
    );

    // ── Embeddings / Semantic Search ─────────────────────────────────────
    const embedText = useCallback(
        async (texts: string[]) => {
            if (!token) return;
            try {
                const res = await fetch(`${API}/embed`, {
                    method: "POST",
                    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
                    body: JSON.stringify({ texts }),
                });
                if (res.ok) setEmbeddings(await res.json());
            } catch {
                toast.error("Error generando embeddings");
            }
        },
        [token]
    );

    const semanticSearch = useCallback(
        async (query: string, index: "models" | "datasets" = "models") => {
            if (!token || !query) return;
            try {
                const res = await fetch(`${API}/search?q=${encodeURIComponent(query)}&index=${index}`, {
                    headers: { Authorization: `Bearer ${token}` },
                });
                const data = await res.json();
                if (res.ok) setSearchResults(data.results || []);
            } catch {
                toast.error("Error en la búsqueda semántica");
            }
        },
        [token]
    );

    // ── Assistant ─────────────────────────────────────────────────────────
    const sendChat = useCallback(
        async (message: string, conversationId?: string) => {
            if (!token) return;
            setIsChatLoading(true);
            setChatAnswer("");
            try {
                const res = await fetch(`${API}/assistant/chat`, {
                    method: "POST",
                    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
                    body: JSON.stringify({ message, conversation_id: conversationId, stream: true }),
                });
                if (!res.ok || !res.body) {
                    toast.error("Error en la llamada al asistente");
                    setIsChatLoading(false);
                    return;
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, { stream: true });
                    const events = chunk.split("\n\n").filter((e) => e.startsWith("data: "));
                    for (const ev of events) {
                        const payload = ev.slice(6);
                        if (payload === "[DONE]") continue;
                        try {
                            const json = JSON.parse(payload);
                            if (json.delta) setChatAnswer((prev) => prev + json.delta);
                            if (json.sources) setChatSources(json.sources);
                        } catch {
                            /* chunk parcial */
                        }
                    }
                }
            } catch {
                toast.error("Error de conexión con el asistente");
            } finally {
                setIsChatLoading(false);
            }
        },
        [token]
    );

    const fetchChatHistory = useCallback(async () => {
        if (!token) return;
        try {
            const res = await fetch(`${API}/assistant/history`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) setChatHistory(await res.json());
        } catch {
            toast.error("Error cargando historial");
        }
    }, [token]);

    const clearChatHistory = useCallback(async () => {
        if (!token) return;
        try {
            await fetch(`${API}/assistant/history`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` },
            });
            setChatHistory([]);
            toast.success("Historial borrado");
        } catch {
            toast.error("Error borrando historial");
        }
    }, [token]);

    return {
        algorithms,
        startNLPTraining,
        isTraining,
        trainingStatus,
        predictNLP,
        isPredicting,
        prediction,
        batchPredictNLP,
        embedText,
        embeddings,
        semanticSearch,
        searchResults,
        sendChat,
        chatAnswer,
        chatSources,
        isChatLoading,
        fetchChatHistory,
        chatHistory,
        clearChatHistory,
    };
}
