import { useState, useCallback } from 'react';
import toast from 'react-hot-toast';

export interface DriftMetrics {
  dataset_drift: boolean;
  drift_by_columns: Record<string, any>;
  psi_threshold: number;
  ks_threshold: number;
}

export interface DriftReport {
  model_id?: string;
  dataset_id?: string;
  metrics: DriftMetrics;
  timestamp: string;
}

export function useDrift(token: string | null) {
  const [driftReports, setDriftReports] = useState<DriftReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDriftReport = useCallback(async (modelId: string, datasetId?: string) => {
    if (!token) return;

    setLoading(true);
    setError(null);

    try {
      const endpoint = datasetId
        ? `http://localhost:8000/api/v1/drift/report/${datasetId}`
        : `http://localhost:8000/api/v1/drift/report/${modelId}`;

      const res = await fetch(endpoint, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      }

      const report = await res.json();
      setDriftReports(prev => [...prev, report]);
      return report;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error fetching drift report');
      toast.error('Error al cargar reporte de drift');
    } finally {
      setLoading(false);
    }
  }, [token]);

  const updateDriftThresholds = useCallback(async (
    entityId: string,
    entityType: 'dataset' | 'model',
    psiThreshold?: number,
    ksThreshold?: number
  ) => {
    if (!token) return;

    try {
      const endpoint = entityType === 'dataset'
        ? `http://localhost:8000/api/v1/datasets/${entityId}/drift-thresholds`
        : `http://localhost:8000/api/v1/models/${entityId}/drift-thresholds`;

      const res = await fetch(endpoint, {
        method: 'PATCH',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          psi_threshold: psiThreshold,
          ks_threshold: ksThreshold,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      }

      toast.success('Umbrales de drift actualizados');
      return await res.json();
    } catch (err) {
      toast.error('Error al actualizar umbrales de drift');
      throw err;
    }
  }, [token]);

  const clearReports = useCallback(() => {
    setDriftReports([]);
  }, []);

  return {
    driftReports,
    loading,
    error,
    fetchDriftReport,
    updateDriftThresholds,
    clearReports,
  };
}