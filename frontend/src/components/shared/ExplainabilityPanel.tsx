"use client";
import React from 'react';
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

interface ExplainabilityPanelProps {
  shapValues: number[];
  featureNames: string[];
  expectedValue: number;
  prediction: number;
}

export default function ExplainabilityPanel({
  shapValues,
  featureNames,
  expectedValue,
  prediction,
}: ExplainabilityPanelProps) {
  // Ordenar características por impacto absoluto
  const sortedIndices = shapValues
    .map((value, index) => ({ value: Math.abs(value), index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10) // Mostrar solo top 10
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
        text: 'Contribución de características (SHAP)',
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

  const totalImpact = sortedShapValues.reduce((sum, val) => sum + Math.abs(val), 0);

  return (
    <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
      <div className="mb-4">
        <h4 className="text-lg font-semibold text-neutral-200 mb-2">
          Explicabilidad (SHAP)
        </h4>
        <div className="flex gap-4 text-sm text-neutral-400">
          <div>
            <span className="text-neutral-500">Valor esperado:</span>{' '}
            <span className="text-indigo-400 font-mono">
              {expectedValue.toFixed(4)}
            </span>
          </div>
          <div>
            <span className="text-neutral-500">Predicción:</span>{' '}
            <span className="text-emerald-400 font-mono">
              {prediction.toFixed(4)}
            </span>
          </div>
          <div>
            <span className="text-neutral-500">Impacto total:</span>{' '}
            <span className="text-amber-400 font-mono">
              {totalImpact.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
      
      <div className="h-64">
        <Bar data={data} options={options} />
      </div>
      
      <div className="mt-4 text-xs text-neutral-500">
        <p>
          <span className="text-emerald-400">Verde:</span> Características que aumentan la predicción
        </p>
        <p>
          <span className="text-red-400">Rojo:</span> Características que disminuyen la predicción
        </p>
        <p className="mt-2">
          El valor SHAP muestra cuánto contribuye cada característica a la predicción final.
        </p>
      </div>
    </div>
  );
}