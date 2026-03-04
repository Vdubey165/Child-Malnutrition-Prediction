import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, Activity, Map, Info } from 'lucide-react';
import './SideBar.css';

const Sidebar = () => {
  const navItems = [
    { path: '/dashboard', icon: Home, label: 'Dashboard' },  // Changed from '/'
    { path: '/predict', icon: Activity, label: 'Prediction' },
    { path: '/districts', icon: Map, label: 'Districts' },
    { path: '/about', icon: Info, label: 'About' },
  ];

  return (
    <div className="sidebar">
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
  );
};

export default Sidebar;