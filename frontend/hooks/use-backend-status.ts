import { useState, useEffect } from 'react';

interface BackendStatus {
  cameraFeed: boolean;
  dataApi: boolean;
  isLoading: boolean;
  error: string | null;
}

export function useBackendStatus() {
  const [status, setStatus] = useState<BackendStatus>({
    cameraFeed: false,
    dataApi: false,
    isLoading: true,
    error: null,
  });

  const checkBackendConnectivity = async () => {
    setStatus(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Check camera feed port (999)
      const cameraPromise = fetch('http://localhost:999/health', {
        method: 'GET',
        signal: AbortSignal.timeout(3000), // 3 second timeout
      }).then(response => response.ok).catch(() => false);

      // Check data API port (666)
      const dataPromise = fetch('http://localhost:666/health', {
        method: 'GET',
        signal: AbortSignal.timeout(3000), // 3 second timeout
      }).then(response => response.ok).catch(() => false);

      const [cameraFeed, dataApi] = await Promise.all([cameraPromise, dataPromise]);

      setStatus({
        cameraFeed,
        dataApi,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setStatus({
        cameraFeed: false,
        dataApi: false,
        isLoading: false,
        error: 'Failed to check backend connectivity',
      });
    }
  };

  useEffect(() => {
    checkBackendConnectivity();
    
    // Check connectivity every 30 seconds
    const interval = setInterval(checkBackendConnectivity, 30000);
    
    return () => clearInterval(interval);
  }, []);

  return {
    ...status,
    refresh: checkBackendConnectivity,
    isBackendRunning: status.cameraFeed && status.dataApi,
  };
}
