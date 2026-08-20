import React, { useState, useEffect } from 'react';
import {
  Sliders, AlertTriangle, Info, ChevronDown, ChevronUp,
  TrendingDown, TrendingUp, Minus, BookOpen,
} from 'lucide-react';
import Header from '../components/Header';
import { getAllDistricts, simulateScenario } from '../services/api';
import './Simulate.css';

// Only expose policy-relevant, interpretable levers — not every raw feature.
// Keep to 1 active slider at a time in the UI to avoid compounding
// aggregation error (see backend disclaimer + "How this tool works" below).
const SLIDERS = [
  {
    key: 'wealth_index', label: 'Wealth Index', step: 0.1, range: [-1, 1],
    hint: 'Household asset/wealth score for the district average.',
  },
  {
    key: 'mother_edu_years', label: "Mother's Education (yrs)", step: 0.5, range: [-3, 3],
    hint: "Average years of schooling completed by mothers in the district.",
  },
  {
    key: 'bcg_vaccination', label: 'BCG Vaccination Rate', step: 0.1, range: [-0.3, 0.3],
    hint: 'Average BCG immunization coverage across the district.',
  },
];

// Same identity used on the Prediction page — label, plain-language meaning,
// and brand color per metric — kept consistent across the app.
const METRICS = [
  { key: 'stunting',    label: 'Stunting',    desc: 'Height-for-age deficit — chronic malnutrition',    color: 'var(--stunting)',    bg: 'var(--stunting-light)' },
  { key: 'wasting',     label: 'Wasting',     desc: 'Weight-for-height deficit — acute malnutrition',   color: 'var(--wasting)',     bg: 'var(--wasting-light)' },
  { key: 'underweight', label: 'Underweight', desc: 'Weight-for-age deficit — combined indicator',      color: 'var(--underweight)', bg: 'var(--underweight-light)' },
];

// Same Low/Medium/High palette used on Dashboard, District Explorer, and
// Prediction — kept identical so a risk badge means the same thing everywhere.
const RISK_COLORS = { Low: '#22c55e', Medium: '#f97316', High: '#ef4444' };
const RISK_BG     = { Low: '#f0fdf4', Medium: '#fff7ed', High: '#fef2f2' };

export default function Simulate() {
  const [districts, setDistricts] = useState([]);
  const [districtId, setDistrictId] = useState(null);
  const [activeFeature, setActiveFeature] = useState(SLIDERS[0].key);
  const [delta, setDelta] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showContext, setShowContext] = useState(false);

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

  const activeSlider = SLIDERS.find(s => s.key === activeFeature);

  return (
    <div className="simulate-container">
      <Header
        title="Scenario Simulator"
        subtitle="Estimate how a district-level shift in one factor associates with predicted malnutrition rates."
      />

      <div className="page-container fade-in simulate-page">

        {/* ── Conceptual framing banner — mirrors Prediction's "How this tool works" ── */}
        <div className="card context-banner" onClick={() => setShowContext(v => !v)}>
          <div className="context-banner-header">
            <div className="context-banner-left">
              <Info size={18} style={{ color: '#6366f1' }} />
              <strong>How this tool works</strong>
            </div>
            {showContext ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>
          {showContext && (
            <div className="context-banner-body">
              <p>
                This simulator runs on a <strong>different, older model</strong> than the Prediction
                page — trained on 707 district-wide averages instead of individual children. That's
                deliberate: a district-average scenario matches this model's input format exactly, while
                feeding an averaged profile into the individual-child model would be a distorted input
                it was never trained to handle.
              </p>
              <p>
                Only <strong>three factors</strong> are adjustable — wealth, mother's education, and BCG
                coverage — because these are things a program can realistically shift over time. Traits
                like a child's age, sex, or birth weight aren't shown here since no policy changes them.
              </p>
              <p>
                Only <strong>one factor moves per run</strong>. Shifting several at once would build a
                combination of values that likely never occurred in any real district, which this model
                has no reliable way to judge.
              </p>
              <p>
                This model explains <strong>43–69% of district-to-district variation</strong> (R²) —
                meaningfully weaker than the Prediction page's model. Treat results as a talking point
                for discussion, not a forecast.
              </p>
            </div>
          )}
        </div>

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
          {activeSlider && <p className="simulate-slider-hint">{activeSlider.hint}</p>}

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
          <p className="simulate-slider-hint simulate-model-note">
            This model responds in steps, not a smooth curve — small changes sometimes shift the
            estimate and sometimes don't. That's expected behavior, not a bug.
          </p>

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

            {/* ── Legend — explains what the colors/icons/units mean before showing them ── */}
            <div className="simulate-legend">
              <span className="legend-item">
                <TrendingDown size={13} style={{ color: 'var(--success)' }} /> falling risk (improvement)
              </span>
              <span className="legend-item">
                <TrendingUp size={13} style={{ color: 'var(--stunting)' }} /> rising risk
              </span>
              <span className="legend-item">
                <Minus size={13} style={{ color: 'var(--text-light)' }} /> no meaningful change
              </span>
            </div>
            <p className="simulate-legend-note">
              "pts" = percentage points — the raw change in predicted rate (e.g. 24% → 22% is "&minus;2 pts"),
              not a percent of the original number.
            </p>

            {METRICS.map(({ key, label, desc, color, bg }) => {
              const d = result.delta[key];
              const improved = d < 0;
              const flat = d === 0;
              const DirIcon = flat ? Minus : improved ? TrendingDown : TrendingUp;
              const dirColor = flat ? 'var(--text-light)' : improved ? 'var(--success)' : 'var(--stunting)';
              const baseRisk = result.risk_level_baseline[key];
              const scenRisk = result.risk_level_scenario[key];
              const riskChanged = baseRisk !== scenRisk;

              return (
                <div key={key} className="simulate-metric-row">
                  <div className="simulate-metric-head">
                    <span className="metric-icon-dot" style={{ background: bg, color }}>●</span>
                    <div>
                      <div className="metric-label">{label}</div>
                      <div className="metric-desc">{desc}</div>
                    </div>
                  </div>

                  <div className="simulate-metric-body">
                    <span className="metric-values">
                      {result.baseline[key]}% <span className="metric-arrow">&rarr;</span> {result.scenario[key]}%
                    </span>
                    <span className="metric-delta" style={{ color: dirColor }}>
                      <DirIcon size={14} />
                      {d > 0 ? '+' : ''}{d} pts
                    </span>
                  </div>

                  <div className="simulate-risk-row">
                    <span className="risk-badge" style={{ background: RISK_BG[baseRisk], color: RISK_COLORS[baseRisk] }}>
                      {baseRisk}
                    </span>
                    <span className="metric-arrow">&rarr;</span>
                    <span className="risk-badge" style={{ background: RISK_BG[scenRisk], color: RISK_COLORS[scenRisk] }}>
                      {scenRisk}
                    </span>
                    {riskChanged && (
                      <span className="risk-changed-flag" style={{ color: RISK_COLORS[scenRisk] }}>
                        Crosses into {scenRisk} risk band
                      </span>
                    )}
                  </div>
                </div>
              );
            })}

            {Object.keys(result.clamped_features).length > 0 && (
              <p className="simulate-clamp-note">
                <AlertTriangle size={13} />
                {' '}Some values were capped to this district's observed range:{' '}
                {Object.entries(result.clamped_features).map(([f, reason]) => `${f} (${reason})`).join('; ')}
              </p>
            )}

            {Object.keys(result.large_shift_features).length > 0 && (
              <p className="simulate-clamp-note simulate-large-shift-note">
                <AlertTriangle size={13} />
                {' '}{Object.entries(result.large_shift_features).map(([f, reason]) => `${f}: ${reason}`).join('; ')}
              </p>
            )}
          </div>
        )}

        {/* ── Model transparency — promoted from a small caption to its own card, ── */}
        {/* mirroring Prediction's "Model transparency" note.                       */}
        {result && (
          <div className="card transparency-note">
            <BookOpen size={16} style={{ color: 'var(--text-secondary)' }} />
            <p><strong>Model transparency:</strong> {result.disclaimer}</p>
          </div>
        )}
      </div>
    </div>
  );
}