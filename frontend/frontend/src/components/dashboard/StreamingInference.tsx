"use client";
import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  StopCircle,
  Wifi,
  WifiOff,
  Loader2,
  Send,
  Trash2,
  BarChart3,
  AlertCircle,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { MLModel } from '@/lib/api';
import { useStreamingInference } from '@/hooks/useStreamingInference';

interface StreamingInferenceProps {
  models: MLModel[];
  token: string | null;
}

export default function StreamingInference({ models, token }: StreamingInferenceProps) {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [explain, setExplain] = useState(true);
  const [rowInput, setRowInput] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const {
    isConnected,
    isConnecting,
    results,
    error,
    connect,
    disconnect,
    sendRow,
    clearResults,
  } = useStreamingInference(token, selectedModel, explain);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [rowInput]);

  const handleSend = () => {
    if (!rowInput.trim()) return;
    
    try {
      const rowData = JSON.parse(rowInput);
      sendRow(rowData);
      setRowInput('');
      setJsonError(null);
    } catch (err) {
      setJsonError('JSON inválido: ' + (err as Error).message);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Streaming Inference</h2>
          <p className="text-neutral-400 text-sm mt-1">
            Predicciones en tiempo real mediante WebSocket
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            isConnected 
              ? 'bg-emerald-500/10 text-emerald-400' 
              : 'bg-red-500/10 text-red-400'
          }`}>
            {isConnected ? (
              <>
                <Wifi className="w-4 h-4" />
                Conectado
              </>
            ) : isConnecting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Conectando...
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4" />
                Desconectado
              </>
            )}
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4 space-y-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-neutral-500 mb-1 uppercase tracking-wider">
              Modelo
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-neutral-800 border border-neutral-700 text-neutral-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
              disabled={isConnected}
            >
              <option value="">Seleccionar modelo...</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.stage})
                </option>
              ))}
            </select>
          </div>
          
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={explain}
                onChange={(e) => setExplain(e.target.checked)}
                className="w-4 h-4 rounded border-neutral-600 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm text-neutral-300">Incluir explicabilidad (SHAP)</span>
            </label>
          </div>
          
          <div className="flex gap-2">
            <Button
              onClick={isConnected ? disconnect : connect}
              disabled={!selectedModel || isConnecting}
              variant={isConnected ? "destructive" : "default"}
              className="gap-2"
            >
              {isConnected ? (
                <>
                  <StopCircle className="w-4 h-4" />
                  Desconectar
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Conectar
                </>
              )}
            </Button>
          </div>
        </div>
        
        {error && (
          <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 px-3 py-2 rounded-lg">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex gap-6 min-h-0">
        {/* Input Panel */}
        <div className="flex-1 flex flex-col bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-neutral-800 flex items-center justify-between">
            <h3 className="text-sm font-medium text-neutral-300">Entrada de Datos (JSON)</h3>
            <span className="text-xs text-neutral-500">Ctrl+Enter para enviar</span>
          </div>
          <div className="flex-1 p-4 flex flex-col">
            <textarea
              ref={textareaRef}
              value={rowInput}
              onChange={(e) => {
                setRowInput(e.target.value);
                setJsonError(null);
              }}
              onKeyDown={handleKeyPress}
              placeholder='{"feature1": 1.5, "feature2": 2.3, ...}'
              className="flex-1 bg-neutral-800 border border-neutral-700 rounded-lg p-3 text-neutral-200 font-mono text-sm resize-none focus:outline-none focus:border-indigo-500"
            />
            {jsonError && (
              <p className="text-red-400 text-xs mt-2">{jsonError}</p>
            )}
            <div className="mt-4 flex justify-end gap-2">
              <Button
                onClick={() => setRowInput('')}
                variant="outline"
                className="bg-transparent border-neutral-700 text-neutral-400 hover:bg-neutral-800"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Limpiar
              </Button>
              <Button
                onClick={handleSend}
                disabled={!isConnected || !rowInput.trim()}
                className="gap-2"
              >
                <Send className="w-4 h-4" />
                Enviar Fila
              </Button>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="flex-1 flex flex-col bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-neutral-800 flex items-center justify-between">
            <h3 className="text-sm font-medium text-neutral-300">Resultados en Tiempo Real</h3>
            <span className="text-xs text-neutral-500">{results.length} resultados</span>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {results.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-neutral-500">
                <BarChart3 className="w-12 h-12 mb-3 opacity-50" />
                <p className="text-sm">Esperando datos...</p>
                <p className="text-xs mt-1">Envía filas JSON para comenzar</p>
              </div>
            ) : (
              results.slice().reverse().map((result) => (
                <div
                  key={result.id}
                  className="bg-neutral-800/50 border border-neutral-700 rounded-lg p-3 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {result.error ? (
                        <XCircle className="w-4 h-4 text-red-400" />
                      ) : (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      )}
                      <span className="text-xs text-neutral-500">
                        Predicción: 
                        <span className={`font-mono ml-1 ${
                          result.error ? 'text-red-400' : 'text-emerald-400'
                        }`}>
                          {result.error || (result.prediction?.toFixed(4) || 'N/A')}
                        </span>
                      </span>
                    </div>
                    {result.uncertainty !== undefined && (
                      <span className="text-xs text-neutral-500">
                        Incertidumbre: 
                        <span className="font-mono text-amber-400 ml-1">
                          {result.uncertainty.toFixed(4)}
                        </span>
                      </span>
                    )}
                  </div>
                  
                  {result.shapValues && result.featureNames && (
                    <div className="pt-2 border-t border-neutral-700">
                      <div className="text-xs text-neutral-500 mb-1">SHAP (Top 3):</div>
                      <div className="flex flex-wrap gap-2">
                        {result.shapValues.slice(0, 3).map((value, idx) => (
                          <span
                            key={idx}
                            className={`px-2 py-0.5 rounded text-xs font-mono ${
                              value >= 0 
                                ? 'bg-emerald-500/20 text-emerald-400' 
                                : 'bg-red-500/20 text-red-400'
                            }`}
                          >
                            {result.featureNames?.[idx] || `F${idx}`}: {value.toFixed(4)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
          {results.length > 0 && (
            <div className="p-3 border-t border-neutral-800 flex justify-end">
              <Button
                onClick={clearResults}
                variant="outline"
                className="bg-transparent border-neutral-700 text-neutral-400 hover:bg-neutral-800 text-xs py-1"
              >
                Limpiar resultados
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Footer Info */}
      <div className="text-xs text-neutral-500 text-center">
        Conexiones WebSocket mantienen 30 minutos de inactividad antes de cerrarse automáticamente
      </div>
    </div>
  );
}