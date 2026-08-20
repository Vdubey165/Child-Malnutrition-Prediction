import React, { useState, useEffect } from 'react';
import { Sliders, AlertTriangle } from 'lucide-react';
import Header from '../components/Header';
import { getAllDistricts, simulateScenario } from '../services/api';
import './Simulate.css';

// Only expose policy-relevant, interpretable levers — not every raw feature.
// Keep to 1 active slider at a time in the UI to avoid compounding
// aggregation error (see backend disclaimer).
const SLIDERS = [
  { key: 'wealth_index',     label: 'Wealth Index',              step: 0.1, range: [-1, 1] },
  { key: 'mother_edu_years', label: "Mother's Education (yrs)",  step: 0.5, range: [-3, 3] },
  { key: 'bcg_vaccination',  label: 'BCG Vaccination Rate',      step: 0.1, range: [-0.3, 0.3] },
];

export default function Simulate() {
  const [districts, setDistricts] = useState([]);
  const [districtId, setDistrictId] = useState(null);
  const [activeFeature, setActiveFeature] = useState(SLIDERS[0].key);
  const [delta, setDelta] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAllDistricts().then(data => {
      setDistricts(data.districts);
      if (data.districts.length) setDistrictId(data.districts[0].district);
    });
  }, []);

  const runSimulation = async () => {
    if (districtId == null) return;
    setLoading(true);
    setError(null);
    try {
      const res = await simulateScenario(districtId, { [activeFeature]: delta });
      setResult(res);
    } catch (e) {
      setError('Simulation failed — is the backend v1 model loaded?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="simulate-container">
      <Header
        title="Scenario Simulator"
        subtitle="Estimate how a district-level shift in one factor associates with predicted malnutrition rates."
      />

      <div className="simulate-page">
        <div className="card simulate-controls">
          <label>
            District
            <select value={districtId ?? ''} onChange={e => setDistrictId(Number(e.target.value))}>
              {districts.map(d => (
                <option key={d.district} value={d.district}>
                  {d.district_name} ({d.state_name})
                </option>
              ))}
            </select>
          </label>

          <label>
            Factor to adjust
            <select value={activeFeature} onChange={e => { setActiveFeature(e.target.value); setDelta(0); }}>
              {SLIDERS.map(s => <option key={s.key} value={s.key}>{s.label}</option>)}
            </select>
          </label>

          {SLIDERS.filter(s => s.key === activeFeature).map(s => (
            <label key={s.key}>
              Change: {delta > 0 ? '+' : ''}{delta}
              <input
                type="range"
                min={s.range[0]}
                max={s.range[1]}
                step={s.step}
                value={delta}
                onChange={e => setDelta(Number(e.target.value))}
              />
            </label>
          ))}

          <button className="simulate-run-btn" onClick={runSimulation} disabled={loading}>
            <Sliders size={16} />
            {loading ? 'Simulating…' : 'Run Scenario'}
          </button>
        </div>

        {error && (
          <div className="card simulate-error">
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        {result && (
          <div className="card simulate-results">
            <h2>{result.district_name}, {result.state_name}</h2>
            {['stunting', 'wasting', 'underweight'].map(k => (
              <div key={k} className="simulate-metric-row">
                <span className="metric-label">{k}</span>
                <span className="metric-values">{result.baseline[k]}% &rarr; {result.scenario[k]}%</span>
                <span className={result.delta[k] < 0 ? 'delta-good' : 'delta-bad'}>
                  {result.delta[k] > 0 ? '+' : ''}{result.delta[k]} pts
                </span>
                <span className="risk-badge">
                  {result.risk_level_baseline[k]} &rarr; {result.risk_level_scenario[k]}
                </span>
              </div>
            ))}
            <p className="simulate-disclaimer">{result.disclaimer}</p>
          </div>
        )}
      </div>
    </div>
  );
}
