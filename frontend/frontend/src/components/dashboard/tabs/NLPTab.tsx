"use client";
import React, { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";
import { Dataset, MLModel } from "@/lib/api";
import { useNLP } from "@/hooks/useNLP";
import NLPTrainingTab from "./NLPTrainingTab";
import NLPInferenceTab from "./NLPInferenceTab";
import SemanticSearchTab from "./SemanticSearchTab";
import AssistantTab from "./AssistantTab";

type SubTab = "training" | "inference" | "search" | "assistant";

interface NLPTabProps {
    token: string | null;
    datasets: Dataset[];
    models: MLModel[];
}

const SUB_TABS: { id: SubTab; label: string }[] = [
    { id: "training", label: "Training" },
    { id: "inference", label: "Inference" },
    { id: "search", label: "Semantic Search" },
    { id: "assistant", label: "Assistant" },
];

export default function NLPTab({ token, datasets, models }: NLPTabProps) {
    const nlp = useNLP(token);
    const [activeTab, setActiveTab] = useState<SubTab>("training");

    useEffect(() => {
        nlp.fetchChatHistory();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [token]);

    const nlpModels = models.filter(
        (m) => (m.metrics_metadata as any)?.domain === "nlp"
    );

    return (
        <div className="flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                        <Sparkles className="w-7 h-7 text-indigo-400" />
                        NLP Engine
                    </h2>
                    <p className="text-neutral-400 text-sm mt-1">
                        Entrena modelos transformer, infiere, busca semánticamente y pregunta al asistente.
                    </p>
                </div>
            </div>

            <div className="flex gap-1 bg-neutral-900/50 p-1 rounded-xl mb-6 w-fit">
                {SUB_TABS.map((t) => (
                    <button
                        key={t.id}
                        onClick={() => setActiveTab(t.id)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            activeTab === t.id
                                ? "bg-indigo-600 text-white"
                                : "text-neutral-400 hover:text-white"
                        }`}
                    >
                        {t.label}
                    </button>
                ))}
            </div>

            {activeTab === "training" && (
                <NLPTrainingTab
                    datasets={datasets}
                    algorithms={nlp.algorithms}
                    isTraining={nlp.isTraining}
                    trainingStatus={nlp.trainingStatus}
                    onTrain={nlp.startNLPTraining}
                />
            )}
            {activeTab === "inference" && (
                <NLPInferenceTab
                    nlpModels={nlpModels}
                    isPredicting={nlp.isPredicting}
                    prediction={nlp.prediction}
                    onPredict={nlp.predictNLP}
                    onBatch={nlp.batchPredictNLP}
                />
            )}
            {activeTab === "search" && (
                <SemanticSearchTab onSearch={nlp.semanticSearch} results={nlp.searchResults} />
            )}
            {activeTab === "assistant" && (
                <AssistantTab
                    isChatLoading={nlp.isChatLoading}
                    chatAnswer={nlp.chatAnswer}
                    chatSources={nlp.chatSources}
                    onSend={nlp.sendChat}
                    onFetchHistory={nlp.fetchChatHistory}
                    onClearHistory={nlp.clearChatHistory}
                />
            )}
        </div>
    );
}
