import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Home, Activity, Map, Info, Menu, X } from 'lucide-react';
import './SideBar.css';

const Sidebar = () => {
  const [mobileOpen, setMobileOpen] = useState(false);

  const navItems = [
    { path: '/dashboard', icon: Home,     label: 'Dashboard'  },
    { path: '/predict',   icon: Activity, label: 'Prediction' },
    { path: '/districts', icon: Map,      label: 'Districts'  },
    { path: '/about',     icon: Info,     label: 'About'      },
  ];

  const closeMobile = () => setMobileOpen(false);

  return (
    <>
      {/* ── Mobile top bar ── */}
      <div className="mobile-topbar">
        <div className="logo">
          <span className="logo-icon">🍎</span>
          <span className="logo-text">Malnutrition Predictor</span>
        </div>
        <button
          className="hamburger"
          onClick={() => setMobileOpen(v => !v)}
          aria-label="Toggle menu"
          type="button"
        >
          {mobileOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* ── Overlay ── */}
      {mobileOpen && (
        <div className="sidebar-overlay" onClick={closeMobile} />
      )}

      {/* ── Sidebar ── */}
      <div className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">🍎</span>
            <span className="logo-text">Malnutrition Predictor</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              onClick={closeMobile}
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p className="footer-text">NFHS-5 Data • 2019-21</p>
          <p className="footer-subtext">707 Districts • 232K Children</p>
        </div>
      </div>
    </>
  );
};

export default Sidebar;