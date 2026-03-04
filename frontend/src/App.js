import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SideBar from './components/SideBar';
import Landing from './pages/Landing';
import DashBoard from './pages/DashBoard';
import Prediction from './pages/Prediction';
import DistrictExplorer from './pages/DistrictExplorer';
import About from './pages/About';
import WakeUpBanner from './components/WakeUpBanner';
import { warmupBackend } from './services/warmup';
import './App.css';

function App() {
  const [backendReady, setBackendReady] = useState(false);
  const [waitSeconds, setWaitSeconds]   = useState(0);

  useEffect(() => {
    warmupBackend(
      () => setBackendReady(true),
      (s) => setWaitSeconds(s)
    );
  }, []);

  return (
    <>
      {!backendReady && <WakeUpBanner seconds={waitSeconds} />}
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