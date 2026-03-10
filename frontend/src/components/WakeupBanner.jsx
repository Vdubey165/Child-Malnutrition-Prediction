// src/components/WakeupBanner.jsx
import React from 'react';
import './WakeupBanner.css';

const WakeUpBanner = ({ seconds }) => (
  <div className="wakeup-banner">
    <div className="wakeup-spinner" />
    <div className="wakeup-text">
      <strong>Waking up the server…</strong>
      <span>Free hosting spins down after inactivity. Ready in ~30s. ({seconds}s elapsed)</span>
    </div>
  </div>
);

export default WakeUpBanner;