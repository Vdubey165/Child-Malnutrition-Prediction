import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SideBar from './components/SideBar';
import Landing from './pages/Landing';  // ADD THIS
import DashBoard from './pages/DashBoard';
import Prediction from './pages/Prediction';
import DistrictExplorer from './pages/DistrictExplorer';
import About from './pages/About';
import './App.css';

function App() {
  return (
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
                <Route path="/dashboard" element={<DashBoard />} />
                <Route path="/predict" element={<Prediction />} />
                <Route path="/districts" element={<DistrictExplorer />} />
                <Route path="/about" element={<About />} />
              </Routes>
            </main>
          </div>
        } />
      </Routes>
    </Router>
  );
}

export default App;