import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SideBar from './components/SideBar';
import Landing from './pages/Landing';
import DashBoard from './pages/DashBoard';
import Prediction from './pages/Prediction';
import Simulate from './pages/Simulate';
import DistrictExplorer from './pages/DistrictExplorer';
import About from './pages/About';
import WakeupBanner from './components/WakeupBanner';
import { warmupBackend } from './services/Warmup';
import './App.css';

// Set at build time: `REACT_APP_KIOSK_MODE=true npm run build` produces the
// Pi field-deployment build (Prediction only, no sidebar, no other pages).
// The regular `npm run build` (flag unset) still produces the full
// Vercel-hosted app, unchanged.
const KIOSK = process.env.REACT_APP_KIOSK_MODE === 'true';

function App() {
  const [backendReady, setBackendReady] = useState(KIOSK);
  const [waitSeconds, setWaitSeconds]   = useState(0);

  useEffect(() => {
    // The warmup/cold-start banner exists for the Cloud Run backend, which
    // can sleep after inactivity. The Pi's backend is local and always
    // running, so there's no cold start to wait out — skip the ping
    // entirely in kiosk mode rather than showing a pointless "waking up"
    // banner to a field worker.
    if (KIOSK) return;
    warmupBackend(
      () => setBackendReady(true),
      (s) => setWaitSeconds(s)
    );
  }, []);

  if (KIOSK) {
    return (
      <Router>
        <Routes>
          <Route path="*" element={<Prediction />} />
        </Routes>
      </Router>
    );
  }

  return (
    <>
      {!backendReady && <WakeupBanner seconds={waitSeconds} />}
      <Router>
        <Routes>
          {/* Landing page - no sidebar */}
          <Route path="/" element={<Landing />} />

          {/* Dashboard pages - with sidebar */}
          <Route path="/*" element={
            <div className="app">
              <SideBar />
              <main className="main-content">
                <Routes>
                  <Route path="/dashboard"  element={<DashBoard />} />
                  <Route path="/predict"    element={<Prediction />} />
                  <Route path="/simulate"   element={<Simulate />} />
                  <Route path="/districts"  element={<DistrictExplorer />} />
                  <Route path="/about"      element={<About />} />
                </Routes>
              </main>
            </div>
          } />
        </Routes>
      </Router>
    </>
  );
}

export default App;