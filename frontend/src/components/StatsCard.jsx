import React from 'react';
import './StatsCard.css';

const StatsCard = ({ title, value, subtitle, color, icon: Icon, trend }) => {
  return (
    <div className="stats-card card">
      <div className="stats-header">
        <div>
          <p className="stats-title">{title}</p>
          <h2 className="stats-value" style={{ color: `var(--${color})` }}>
            {value}
          </h2>
          {subtitle && <p className="stats-subtitle">{subtitle}</p>}
        </div>
        {Icon && (
          <div className="stats-icon" style={{ background: `var(--${color}-light)` }}>
            <Icon size={24} style={{ color: `var(--${color})` }} />
          </div>
        )}
      </div>
      {trend && (
        <div className="stats-trend">
          <span className={trend > 0 ? 'trend-up' : 'trend-down'}>
            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
          </span>
          <span className="trend-text">vs national avg</span>
        </div>
      )}
    </div>
  );
};

export default StatsCard;