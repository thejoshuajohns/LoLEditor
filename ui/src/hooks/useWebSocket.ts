import { useEffect, useRef, useState, useCallback } from 'react';

export interface ProgressUpdate {
  type: 'progress';
  project_id: string;
  stage: string;
  progress: number;
  message: string;
  eta_seconds: number | null;
  timestamp: number;
}

export interface LogUpdate {
  type: 'log';
  project_id: string;
  level: string;
  message: string;
  timestamp: number;
}

type WSMessage = ProgressUpdate | LogUpdate;

export function useWebSocket(projectId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [progress, setProgress] = useState<Record<string, ProgressUpdate>>({});
  const [logs, setLogs] = useState<LogUpdate[]>([]);

  const connect = useCallback(() => {
    if (!projectId) return;

    const ws = new WebSocket(`ws://127.0.0.1:8420/ws/${projectId}`);

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const msg: WSMessage = JSON.parse(event.data);

      if (msg.type === 'progress') {
        setProgress((prev) => ({
          ...prev,
          [msg.stage]: msg as ProgressUpdate,
        }));
      } else if (msg.type === 'log') {
        setLogs((prev) => [...prev.slice(-100), msg as LogUpdate]);
      }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = () => {
      setConnected(false);
    };

    wsRef.current = ws;
  }, [projectId]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const cancel = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'cancel' }));
  }, []);

  useEffect(() => {
    setProgress({});
    setLogs([]);
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { connected, progress, logs, cancel, reconnect: connect };
}
