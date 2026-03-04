import React from 'react';
import { Bell, Settings, User } from 'lucide-react';
import './Header.css';

const Header = ({ title, subtitle }) => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-text">
          <h1 className="header-title">{title}</h1>
          {subtitle && <p className="header-subtitle">{subtitle}</p>}
        </div>
        
        
      </div>
    </header>
  );
};

export default Header;