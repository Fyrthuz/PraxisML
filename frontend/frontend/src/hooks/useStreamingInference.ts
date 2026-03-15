import { useState, useEffect, useCallback, useRef } from 'react';
import toast from 'react-hot-toast';
import { config } from '@/lib/config';

export interface StreamingResult {
    id: number;
    row: any;
    prediction?: number;
    uncertainty?: number;
    shapValues?: number[];
    featureNames?: string[];
    error?: string;
}

export function useStreamingInference(
    token: string | null,
    modelId: string,
    explain: boolean = false,
    onResult?: (result: StreamingResult) => void
) {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [results, setResults] = useState<StreamingResult[]>([]);
    const [error, setError] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const onResultRef = useRef(onResult);

    // Update the ref whenever the callback changes to prevent reconnect loops
    useEffect(() => {
        onResultRef.current = onResult;
    }, [onResult]);

    const connect = useCallback(() => {
        if (!token) {
            setError('Usuario no autenticado. Por favor, inicie sesión.');
            return;
        }
        if (!modelId) {
            setError('Por favor, seleccione un modelo.');
            return;
        }

        setIsConnecting(true);
        setError(null);

        const wsUrl = `${config.WS_BASE_URL}/api/v1/streaming/predict/${modelId}?token=${encodeURIComponent(token)}&explain=${explain}`;
        
        try {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setIsConnecting(false);
                toast.success('Conexión de streaming establecida');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('WebSocket message received:', data);
                    
                    if (data.error) {
                        setError(data.error);
                        return;
                    }

                    const result: StreamingResult = {
                        id: Date.now(),
                        row: data,
                        prediction: data.prediction,
                        uncertainty: data.uncertainty,
                        shapValues: data.shap_values,
                        featureNames: data.feature_names,
                    };

                    setResults(prev => [...prev.slice(-99), result]); // Mantener últimos 100 resultados
                    onResultRef.current?.(result);
                } catch (err) {
                    console.error('Error parsing WebSocket message:', err);
                }
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
                setError('Error en la conexión de streaming');
            };

            ws.onclose = (event) => {
                console.log('WebSocket disconnected', event.code, event.reason);
                setIsConnected(false);
                setIsConnecting(false);
                
                if (event.code !== 1000 && event.code !== 1001) {
                    const reason = event.reason ? `: ${event.reason}` : '';
                    setError(`Conexión terminada (Código ${event.code})${reason}`);
                }
                
                // Intentar reconexión después de 5 segundos si no fue una desconexión intencionada
                if (event.code !== 1000) {
                    reconnectTimeoutRef.current = setTimeout(() => {
                        // Usamos una función anónima para obtener el valor más reciente de connect
                        connect();
                    }, 5000);
                }
            };
        } catch (err) {
            console.error('Error establishing WebSocket:', err);
            setError('No se pudo establecer la conexión WebSocket');
            setIsConnecting(false);
        }
    }, [token, modelId, explain]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
        setIsConnecting(false);
    }, []);

    const sendRow = useCallback((rowData: any) => {
        if (wsRef.current && isConnected) {
            wsRef.current.send(JSON.stringify(rowData));
        } else {
            toast.error('No hay conexión de streaming activa');
        }
    }, [isConnected]);

    const clearResults = useCallback(() => {
        setResults([]);
    }, []);

    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        isConnected,
        isConnecting,
        results,
        error,
        connect,
        disconnect,
        sendRow,
        clearResults,
    };
}