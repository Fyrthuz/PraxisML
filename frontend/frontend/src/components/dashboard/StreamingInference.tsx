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
  Expand,
  X,
} from 'lucide-react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);
import { Button } from '@/components/ui/button';
import { MLModel } from '@/lib/api';
import { useStreamingInference, StreamingResult } from '@/hooks/useStreamingInference';

interface StreamingInferenceProps {
  models: MLModel[];
  token: string | null;
}

export default function StreamingInference({ models, token }: StreamingInferenceProps) {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [explain, setExplain] = useState(true);
  const [rowInput, setRowInput] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [selectedShapResult, setSelectedShapResult] = useState<StreamingResult | null>(null);
  const [showAllShap, setShowAllShap] = useState(false);
  
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

  // Helper function to render SHAP chart
  const renderShapChart = (shapValues: number[], featureNames: string[], title: string) => {
    // Sort by absolute impact (showing all variables)
    const sortedIndices = shapValues
      .map((value, index) => ({ value: Math.abs(value), index }))
      .sort((a, b) => b.value - a.value)
      .map(item => item.index);

    const sortedFeatureNames = sortedIndices.map(i => featureNames[i]);
    const sortedShapValues = sortedIndices.map(i => shapValues[i]);

    const data = {
      labels: sortedFeatureNames,
      datasets: [
        {
          label: 'Impacto en predicción (SHAP)',
          data: sortedShapValues,
          backgroundColor: sortedShapValues.map(v =>
            v >= 0 ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)'
          ),
          borderColor: sortedShapValues.map(v =>
            v >= 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)'
          ),
          borderWidth: 1,
        },
      ],
    };

    const options = {
      indexAxis: 'y' as const,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: title,
          color: '#e5e5e5',
          font: {
            size: 14,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#a3a3a3',
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
          },
        },
        y: {
          ticks: {
            color: '#e5e5e5',
          },
          grid: {
            display: false,
          },
        },
      },
    };

    return { data, options };
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
        <div className="flex items-center gap-3">
          {results.some(r => r.shapValues && r.featureNames) && (
            <Button
              onClick={() => setShowAllShap(true)}
              variant="outline"
              size="sm"
              className="bg-neutral-800 border-neutral-700 text-neutral-300 hover:bg-neutral-700"
            >
              <BarChart3 className="w-4 h-4 mr-1" />
              Ver todo el gráfico
            </Button>
          )}
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
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-xs text-neutral-500">SHAP (Top 3):</div>
                        <Button
                          onClick={() => setSelectedShapResult(result)}
                          variant="ghost"
                          size="sm"
                          className="h-5 px-2 text-xs text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700"
                        >
                          <Expand className="w-3 h-3 mr-1" />
                          Ver gráfico completo
                        </Button>
                      </div>
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

      {/* SHAP Chart Modal */}
      {selectedShapResult && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-neutral-900 border border-neutral-700 rounded-xl w-full max-w-3xl max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-700">
              <h3 className="text-lg font-semibold text-neutral-200">Gráfico SHAP - Predicción #{selectedShapResult.id}</h3>
              <Button
                onClick={() => setSelectedShapResult(null)}
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[60vh]">
              {selectedShapResult.shapValues && selectedShapResult.featureNames && (
                <div className="bg-neutral-800/50 rounded-lg p-4">
                  <div className="grid grid-cols-3 gap-4 text-sm mb-4">
                    <div>
                      <span className="text-neutral-500">Predicción:</span>{' '}
                      <span className="text-emerald-400 font-mono">
                        {selectedShapResult.prediction?.toFixed(4) || 'N/A'}
                      </span>
                    </div>
                    {selectedShapResult.uncertainty !== undefined && (
                      <div>
                        <span className="text-neutral-500">Incertidumbre:</span>{' '}
                        <span className="text-amber-400 font-mono">
                          {selectedShapResult.uncertainty.toFixed(4)}
                        </span>
                      </div>
                    )}
                    <div>
                      <span className="text-neutral-500">Total Features:</span>{' '}
                      <span className="text-indigo-400 font-mono">
                        {selectedShapResult.shapValues.length}
                      </span>
                    </div>
                  </div>
                  <div className="min-h-[400px]">
                    <Bar data={renderShapChart(selectedShapResult.shapValues, selectedShapResult.featureNames, 'Contribución de características (SHAP)').data} 
                         options={renderShapChart(selectedShapResult.shapValues, selectedShapResult.featureNames, 'Contribución de características (SHAP)').options} />
                  </div>
                  <div className="mt-4 text-xs text-neutral-500">
                    <p>
                      <span className="text-emerald-400">Verde:</span> Características que aumentan la predicción
                    </p>
                    <p>
                      <span className="text-red-400">Rojo:</span> Características que disminuyen la predicción
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* All SHAP Charts Modal */}
      {showAllShap && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-neutral-900 border border-neutral-700 rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-700">
              <h3 className="text-lg font-semibold text-neutral-200">Todos los gráficos SHAP</h3>
              <Button
                onClick={() => setShowAllShap(false)}
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[75vh] space-y-4">
              {results.filter(r => r.shapValues && r.featureNames).slice(-10).reverse().map((result) => (
                <div key={result.id} className="bg-neutral-800/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span className="text-sm text-neutral-400">
                        Predicción #{result.id}: <span className="text-emerald-400 font-mono ml-1">{result.prediction?.toFixed(4) || 'N/A'}</span>
                      </span>
                    </div>
                    {result.uncertainty !== undefined && (
                      <span className="text-xs text-neutral-500">
                        Incertidumbre: <span className="text-amber-400 font-mono">{result.uncertainty.toFixed(4)}</span>
                      </span>
                    )}
                  </div>
                  <div className="h-80">
                    <Bar data={renderShapChart(result.shapValues!, result.featureNames!, '').data} 
                         options={renderShapChart(result.shapValues!, result.featureNames!, '').options} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}