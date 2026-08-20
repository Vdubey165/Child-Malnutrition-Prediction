import React from 'react';
import './StatsCard.css';

const StatsCard = ({ title, value, subtitle, color, icon: Icon, trend, higherIsBetter = false }) => {
  // Direction is always shown literally (↑/↓ = the value actually went up/down).
  // Color reflects whether that direction is good or bad for THIS metric —
  // for malnutrition rates (the default), a rise is bad, so don't reuse the
  // arrow direction as the color without checking higherIsBetter first.
  const isIncrease = trend > 0;
  const isGood = higherIsBetter ? isIncrease : !isIncrease;

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
      {!!trend && (
        <div className="stats-trend">
          <span className={isGood ? 'trend-good' : 'trend-bad'}>
            {isIncrease ? '↑' : '↓'} {Math.abs(trend)}%
          </span>
          <span className="trend-text">vs national avg</span>
        </div>
      )}
    </div>
  );
};

export default StatsCard;