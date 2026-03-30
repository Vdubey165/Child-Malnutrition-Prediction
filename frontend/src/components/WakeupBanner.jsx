// src/components/WakeupBanner.jsx
import React from 'react';
import './WakeupBanner.css';

const getMessage = (seconds) => {
  if (seconds < 15) return 'Waking up the server — usually takes 20–40s on free hosting…';
  if (seconds < 35) return 'Almost there — server is loading ML models…';
  if (seconds < 60) return 'Taking a bit longer than usual — nearly ready…';
  return 'Server is taking a while. It will load — please hang tight…';
};

const WakeUpBanner = ({ seconds }) => (
  <div className="wakeup-banner">
    <div className="wakeup-spinner" />
    <div className="wakeup-text">
      <strong>{getMessage(seconds)}</strong>
      <span>({seconds}s elapsed) · Once loaded, the server stays warm while you browse</span>
    </div>
  </div>
);

export default WakeUpBanner;