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
        <div className="flex-1 flex flex-col space-y-5">
            <div className="flex items-center gap-1 border-b border-zinc-800 pb-3">
                {SUB_TABS.map((t) => (
                    <button
                        key={t.id}
                        onClick={() => setActiveTab(t.id)}
                        className={`h-8 px-3 rounded-md text-xs font-medium transition-colors ${
                            activeTab === t.id
                                ? "bg-zinc-800 text-zinc-100"
                                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
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
