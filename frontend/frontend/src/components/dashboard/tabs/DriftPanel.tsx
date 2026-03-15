"use client";
import React, { useState } from 'react';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Settings,
  Loader2,
  BarChart3,
  LineChart,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dataset } from '@/lib/api';
import { useDrift, DriftReport } from '@/hooks/useDrift';

interface DriftPanelProps {
  dataset: Dataset;
  token: string | null;
}

export default function DriftPanel({ dataset, token }: DriftPanelProps) {
  const [showConfig, setShowConfig] = useState(false);
  const [psiThreshold, setPsiThreshold] = useState(0.2);
  const [ksThreshold, setKsThreshold] = useState(0.05);
  
  const {
    driftReports,
    loading,
    error,
    fetchDriftReport,
    updateDriftThresholds,
  } = useDrift(token);

  const handleFetchReport = async () => {
    if (dataset.id) {
      await fetchDriftReport(undefined, dataset.id);
    }
  };

  const handleUpdateThresholds = async () => {
    await updateDriftThresholds(dataset.id, 'dataset', psiThreshold, ksThreshold);
    setShowConfig(false);
  };

  const latestReport = driftReports.length > 0 
    ? driftReports[driftReports.length - 1] 
    : null;

  const hasDrift = latestReport?.metrics?.dataset_drift;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-indigo-400" />
          <h4 className="text-sm font-medium">Data Drift Monitor</h4>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowConfig(!showConfig)}
            className="bg-transparent border-neutral-700 text-neutral-400 hover:bg-neutral-800"
          >
            <Settings className="w-4 h-4 mr-1" />
            Configurar
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleFetchReport}
            disabled={loading}
            className="bg-neutral-800 border-neutral-700 text-neutral-300 hover:bg-neutral-700"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <BarChart3 className="w-4 h-4 mr-1" />
            )}
            Analizar
          </Button>
        </div>
      </div>

      {/* Configuration Panel */}
      {showConfig && (
        <div className="bg-neutral-800/50 border border-neutral-700 rounded-lg p-4 space-y-3">
          <h5 className="text-xs text-neutral-500 uppercase tracking-wider">
            Umbrales de Alerta
          </h5>
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[150px]">
              <label className="block text-xs text-neutral-400 mb-1">
                PSI (Population Stability Index)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="1.0"
                value={psiThreshold}
                onChange={(e) => setPsiThreshold(parseFloat(e.target.value))}
                className="w-full bg-neutral-900 border border-neutral-700 rounded px-3 py-1.5 text-sm text-neutral-200"
              />
            </div>
            <div className="flex-1 min-w-[150px]">
              <label className="block text-xs text-neutral-400 mb-1">
                KS Test (p-valor)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.001"
                max="0.5"
                value={ksThreshold}
                onChange={(e) => setKsThreshold(parseFloat(e.target.value))}
                className="w-full bg-neutral-900 border border-neutral-700 rounded px-3 py-1.5 text-sm text-neutral-200"
              />
            </div>
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowConfig(false)}
              className="bg-transparent border-neutral-700 text-neutral-400 hover:bg-neutral-800"
            >
              Cancelar
            </Button>
            <Button
              size="sm"
              onClick={handleUpdateThresholds}
              className="bg-indigo-600 hover:bg-indigo-500"
            >
              Guardar
            </Button>
          </div>
        </div>
      )}

      {/* Drift Status */}
      {latestReport && (
        <div className={`border rounded-lg p-4 ${
          hasDrift 
            ? 'border-amber-500/50 bg-amber-500/10' 
            : 'border-emerald-500/50 bg-emerald-500/10'
        }`}>
          <div className="flex items-center gap-3">
            {hasDrift ? (
              <>
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                <div>
                  <p className="text-amber-400 font-medium">Drift Detectado</p>
                  <p className="text-xs text-neutral-400">
                    Algunas distribuciones han cambiado significativamente
                  </p>
                </div>
              </>
            ) : (
              <>
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                <div>
                  <p className="text-emerald-400 font-medium">Sin Drift</p>
                  <p className="text-xs text-neutral-400">
                    Las distribuciones se mantienen estables
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Resumen por columna */}
          <div className="mt-4 space-y-2">
            <h6 className="text-xs text-neutral-500 uppercase tracking-wider">
              Detalles por columna
            </h6>
            <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
              {Object.entries(latestReport.metrics.drift_by_columns || {}).slice(0, 6).map(([col, data]) => (
                <div key={col} className="flex items-center justify-between text-xs bg-neutral-900/50 px-2 py-1 rounded">
                  <span className="text-neutral-300 truncate max-w-[100px]">{col}</span>
                  <span className={data.drift_detected ? 'text-amber-400' : 'text-emerald-400'}>
                    {data.drift_detected ? 'Drift' : 'OK'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* No report yet */}
      {!latestReport && !loading && (
        <div className="text-center py-6 border border-dashed border-neutral-700 rounded-lg">
          <BarChart3 className="w-8 h-8 text-neutral-600 mx-auto mb-2" />
          <p className="text-neutral-500 text-sm">
            Sin análisis de drift
          </p>
          <p className="text-neutral-600 text-xs mt-1">
            Haz clic en &quot;Analizar&quot; para verificar la estabilidad del dataset
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="text-red-400 text-sm bg-red-500/10 px-3 py-2 rounded-lg">
          {error}
        </div>
      )}
    </div>
  );
}