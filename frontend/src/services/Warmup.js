// src/services/warmup.js
// Pings the backend on app load so Render wakes up before the user needs it.
// Also installs a keep-alive interval so the server never sleeps while a tab is open.

const API_URL = process.env.REACT_APP_API_URL
  || 'https://child-malnutrition-prediction-api.onrender.com';

let _keepAliveTimer = null;

/** Start pinging /health every 10 min to prevent Render from sleeping. */
const startKeepAlive = () => {
  if (_keepAliveTimer) return; // already running
  _keepAliveTimer = setInterval(() => {
    fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) }).catch(() => {});
  }, 10 * 60 * 1000); // 10 minutes

  // Stop when the tab closes / navigates away
  window.addEventListener('beforeunload', stopKeepAlive, { once: true });
};

export const stopKeepAlive = () => {
  if (_keepAliveTimer) {
    clearInterval(_keepAliveTimer);
    _keepAliveTimer = null;
  }
};

/**
 * warmupBackend — call once on app mount.
 * @param {() => void}      onReady   — called when /health returns 200
 * @param {(secs: number) => void} onWaiting — called each retry with elapsed seconds
 */
export const warmupBackend = (onReady, onWaiting) => {
  const TIMEOUT  = 90000; // 90 s max wait (Render can be slow to wake)
  const INTERVAL = 3000;  // retry every 3 s
  const start    = Date.now();

  const ping = async () => {
    try {
      const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(8000) });
      if (res.ok) {
        startKeepAlive(); // 🔑 keep it awake from now on
        onReady();
        return;
      }
    } catch (_) {
      // still waking up — swallow the error
    }

    const elapsed = Date.now() - start;
    if (elapsed < TIMEOUT) {
      onWaiting(Math.round(elapsed / 1000));
      setTimeout(ping, INTERVAL);
    } else {
      // Give up waiting — let the user try; keep-alive not started
      onReady();
    }
  };

  ping();
};