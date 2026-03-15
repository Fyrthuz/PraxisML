import { useState, useCallback } from 'react';
import toast from 'react-hot-toast';
import { config } from '@/lib/config';

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

  const fetchDriftReport = useCallback(async (modelId?: string, datasetId?: string) => {
    if (!token) return;

    setLoading(true);
    setError(null);

    try {
      let endpoint = '';
      if (datasetId) {
        endpoint = config.getFullApiUrl(`/drift/report/dataset/${datasetId}`);
      } else if (modelId) {
        endpoint = config.getFullApiUrl(`/drift/report/model/${modelId}`);
      } else {
        throw new Error('Must provide either modelId or datasetId');
      }

      const res = await fetch(endpoint, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}: ${res.statusText}`);
      }

      const report = await res.json();
      setDriftReports(prev => [...prev, report]);
      return report;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Error fetching drift report';
      setError(msg);
      toast.error(msg);
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
      // Build query string for thresholds
      const params = new URLSearchParams();
      if (psiThreshold !== undefined) params.append('psi_threshold', psiThreshold.toString());
      if (ksThreshold !== undefined) params.append('ks_threshold', ksThreshold.toString());
      
      const endpoint = entityType === 'dataset'
        ? config.getFullApiUrl(`/datasets/${entityId}/drift-thresholds?${params.toString()}`)
        : config.getFullApiUrl(`/models/${entityId}/drift-thresholds?${params.toString()}`);

      const res = await fetch(endpoint, {
        method: 'PATCH',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        }
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}: ${res.statusText}`);
      }

      toast.success('Umbrales de drift actualizados');
      return await res.json();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Error al actualizar umbrales de drift';
      toast.error(msg);
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