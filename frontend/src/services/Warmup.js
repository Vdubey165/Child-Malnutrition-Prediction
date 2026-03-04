// src/services/warmup.js
// Pings the backend on app load so Render wakes up before the user needs it

const API_URL = process.env.REACT_APP_API_URL 
  || 'https://child-malnutrition-prediction-api.onrender.com';

export const warmupBackend = async (onReady, onWaiting) => {
  const TIMEOUT = 60000; // 60s max wait
  const INTERVAL = 3000; // ping every 3s
  const start = Date.now();

  const ping = async () => {
    try {
      const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        onReady();
        return;
      }
    } catch (_) {
      // still sleeping
    }

    if (Date.now() - start < TIMEOUT) {
      onWaiting(Math.round((Date.now() - start) / 1000));
      setTimeout(ping, INTERVAL);
    } else {
      // give up after 60s — let user try anyway
      onReady();
    }
  };

  ping();
};