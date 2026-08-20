import React, { useState } from 'react';
import { Activity, TrendingDown, AlertCircle, AlertTriangle, Info, ChevronDown, ChevronUp, Target, Zap, BookOpen, Users, Landmark, HeartHandshake, Microscope, MapPin, UserRound, Baby, Syringe, Search } from 'lucide-react';
import Header from '../components/Header';
import { predictMalnutrition } from '../services/api';
import './Prediction.css';

// ─── Feature importance from actual model outputs (child-level XGBoost, v2) ───
const FEATURE_IMPORTANCE = {
  stunting: [
    { label: 'Wealth Index',               value: 31.5, color: '#ef4444' },
    { label: "Mother's Education (years)", value: 10.5, color: '#f97316' },
    { label: "Mother's Education Level",   value: 9.0,  color: '#eab308' },
    { label: 'Child Age (years)',          value: 8.4,  color: '#84cc16' },
    { label: 'Child Age (months)',         value: 6.7,  color: '#22c55e' },
    { label: 'Birth Weight',               value: 6.4,  color: '#14b8a6' },
    { label: 'State',                      value: 5.0,  color: '#6366f1' },
    { label: "Mother's BMI",               value: 3.9,  color: '#8b5cf6' },
    { label: 'Child Sex',                  value: 3.5,  color: '#ec4899' },
    { label: 'Birth Interval',             value: 3.2,  color: '#06b6d4' },
  ],
  wasting: [
    { label: 'State',                      value: 12.4, color: '#6366f1' },
    { label: 'Child Age (years)',          value: 11.2, color: '#84cc16' },
    { label: 'Child Age (months)',         value: 9.4,  color: '#22c55e' },
    { label: 'Birth Weight',               value: 9.0,  color: '#14b8a6' },
    { label: "Mother's BMI",               value: 8.2,  color: '#ef4444' },
    { label: 'Wealth Index',               value: 6.1,  color: '#f97316' },
    { label: 'Child Sex',                  value: 5.3,  color: '#ec4899' },
    { label: 'Measles Vaccination',        value: 4.6,  color: '#8b5cf6' },
    { label: "Mother's Education (years)", value: 4.4,  color: '#eab308' },
    { label: 'Urban/Rural',                value: 4.1,  color: '#06b6d4' },
  ],
  underweight: [
    { label: 'Wealth Index',               value: 19.7, color: '#ef4444' },
    { label: 'Birth Weight',               value: 14.3, color: '#84cc16' },
    { label: 'State',                      value: 10.0, color: '#6366f1' },
    { label: "Mother's BMI",               value: 9.8,  color: '#f97316' },
    { label: "Mother's Education (years)", value: 8.9,  color: '#eab308' },
    { label: "Mother's Education Level",   value: 7.5,  color: '#22c55e' },
    { label: 'Child Age (months)',         value: 4.7,  color: '#14b8a6' },
    { label: 'Child Age (years)',          value: 4.5,  color: '#8b5cf6' },
    { label: 'Child Sex',                  value: 4.4,  color: '#ec4899' },
    { label: 'Birth Interval',             value: 2.8,  color: '#06b6d4' },
  ],
};

// NFHS-5 state codes the model was trained on
const STATE_OPTIONS = [
  [1,'Jammu & Kashmir'],[2,'Himachal Pradesh'],[3,'Punjab'],[4,'Chandigarh'],
  [5,'Uttarakhand'],[6,'Haryana'],[7,'NCT of Delhi'],[8,'Rajasthan'],
  [9,'Uttar Pradesh'],[10,'Bihar'],[11,'Sikkim'],[12,'Arunachal Pradesh'],
  [13,'Nagaland'],[14,'Manipur'],[15,'Mizoram'],[16,'Tripura'],
  [17,'Meghalaya'],[18,'Assam'],[19,'West Bengal'],[20,'Jharkhand'],
  [21,'Odisha'],[22,'Chhattisgarh'],[23,'Madhya Pradesh'],[24,'Gujarat'],
  [25,'Dadra & Nagar Haveli and Daman & Diu'],[27,'Maharashtra'],
  [28,'Andhra Pradesh'],[29,'Karnataka'],[30,'Goa'],[31,'Lakshadweep'],
  [32,'Kerala'],[33,'Tamil Nadu'],[34,'Puducherry'],
  [35,'Andaman & Nicobar Islands'],[36,'Telangana'],[37,'Ladakh'],
];

// ─── Persona-aware action plans ───
const ACTION_PLANS = {
  policymaker: {
    label: 'Policymaker',
    icon: Landmark,
    description: 'District-level resource allocation & policy response',
    high: [
      { priority: 'P1', action: 'Declare nutritional emergency — activate ICDS supplementary feeding for all U5 children', dept: 'Health & WCD Ministry' },
      { priority: 'P1', action: 'Deploy mobile health units to cover bottom-quartile districts with no facility access', dept: 'NHM / State Health' },
      { priority: 'P2', action: 'Increase PM-POSHAN allocations by 40% for high-stunting districts in next budget', dept: 'Finance / Education' },
      { priority: 'P2', action: 'Mandate universal BCG + DPT coverage verification — tie to district performance metrics', dept: 'Health Ministry' },
      { priority: 'P3', action: 'Launch conditional cash transfer linked to maternal BMI check-ups (Antenatal care)', dept: 'WCD / DBT Cell' },
    ],
    medium: [
      { priority: 'P1', action: 'Strengthen Anganwadi centres — ensure VHSND days are conducted monthly', dept: 'WCD Department' },
      { priority: 'P2', action: 'Run district-level awareness campaigns on exclusive breastfeeding (0-6 months)', dept: 'Health & NHM' },
      { priority: 'P3', action: 'Pilot kitchen garden programs in rural schools to improve dietary diversity', dept: 'Agriculture / Education' },
    ],
    low: [
      { priority: 'P1', action: 'Sustain current intervention coverage — quarterly NFHS sub-sampling for monitoring', dept: 'Statistics Ministry' },
      { priority: 'P2', action: 'Document best practices for replication in adjacent high-burden districts', dept: 'NITI Aayog' },
    ],
  },
  ngo: {
    label: 'NGO / Field Worker',
    icon: HeartHandshake,
    description: 'Community-level intervention & outreach targeting',
    high: [
      { priority: 'P1', action: 'Identify and enroll SAM/MAM children in nearest NRC — conduct door-to-door screening', dept: 'Community Outreach' },
      { priority: 'P1', action: 'Distribute Ready-to-Use Therapeutic Food (RUTF) through community health workers', dept: 'Nutrition Team' },
      { priority: 'P2', action: 'Train 10 ASHAs per block on malnutrition early-warning signs and referral pathways', dept: 'Capacity Building' },
      { priority: 'P2', action: 'Establish community kitchen with SHG participation for hot cooked meals', dept: 'Livelihoods Team' },
      { priority: 'P3', action: 'Sensitize fathers and grandmothers on feeding practices — behaviour change comms', dept: 'Social & Behavioural' },
    ],
    medium: [
      { priority: 'P1', action: 'Conduct growth monitoring camps bi-monthly — flag children falling off growth curve', dept: 'Health Program' },
      { priority: 'P2', action: 'Facilitate linkages with PDS for Above Poverty Line households not receiving benefits', dept: 'Advocacy' },
    ],
    low: [
      { priority: 'P1', action: 'Continue monthly home visits — track immunisation and feeding progress per child', dept: 'Field Team' },
    ],
  },
  researcher: {
    label: 'Researcher / Analyst',
    icon: Microscope,
    description: 'Data-driven insights & evidence generation',
    high: [
      { priority: 'P1', action: 'Conduct geospatial clustering analysis to identify malnutrition hotspot corridors', dept: 'Spatial Analytics' },
      { priority: 'P1', action: 'Disaggregate stunting by caste/tribal status — check for social determinant confounding', dept: 'Equity Analysis' },
      { priority: 'P2', action: 'Run longitudinal cohort sub-study in top-5 high-burden districts from this prediction set', dept: 'Field Research' },
      { priority: 'P2', action: 'Quantify healthcare access barriers using distance-to-facility as covariate in model v2', dept: 'Model Improvement' },
    ],
    medium: [
      { priority: 'P1', action: 'Test model generalisability on NFHS-4 (2015-16) — check temporal stability of predictors', dept: 'Validation' },
      { priority: 'P2', action: 'Analyse prediction residuals — identify districts where model underperforms (OOD regions)', dept: 'Model Diagnostics' },
    ],
    low: [
      { priority: 'P1', action: 'Publish district-level risk scores with confidence intervals as open data for replication', dept: 'Dissemination' },
    ],
  },
};

const RISK_THRESHOLDS = {
  stunting:    { low: 20, high: 35 },
  wasting:     { low: 10, high: 20 },
  underweight: { low: 20, high: 35 },
};

const getRisk = (metric, value) => {
  const { low, high } = RISK_THRESHOLDS[metric];
  if (value < low) return 'Low';
  if (value < high) return 'Medium';
  return 'High';
};

const RISK_COLORS = { Low: '#22c55e', Medium: '#f97316', High: '#ef4444' };
const PRIORITY_COLORS = { P1: '#ef4444', P2: '#f97316', P3: '#6366f1' };

const Prediction = () => {
  const [formData, setFormData] = useState({
    wealth_index: 3,
    mother_edu_level: 1,
    mother_age: 27,
    mother_edu_years: 8,
    mother_bmi: 2200,
    mother_works: 0,
    female_headed_hh: 1,
    child_age_months: 30,
    child_sex: 1,
    child_age_years: 2,
    birth_interval: 32,
    birth_weight: 2800,
    breastfeed_duration: 12,
    bcg_vaccination: 1,
    dpt_vaccination: 1,
    measles_vaccination: 1,
    knows_ors: 1,
    urban_rural: 2,
    state: 9,
  });
  const [firstBorn, setFirstBorn] = useState(false);

  const [prediction, setPrediction]   = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [persona, setPersona]         = useState('policymaker');
  const [showExplain, setShowExplain] = useState(false);
  const [activeExplain, setActiveExplain] = useState('stunting');
  const [showContext, setShowContext] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = { ...formData, birth_interval: firstBorn ? null : formData.birth_interval };
      const result = await predictMalnutrition(payload);
      setPrediction(result);
    } catch (err) {
      setError('Prediction failed — please check your inputs and try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Derive overall risk level for action plan selection
  const overallRisk = prediction
    ? (['High', 'Medium', 'Low'].find(r =>
        [prediction.risk_level.stunting, prediction.risk_level.wasting, prediction.risk_level.underweight].includes(r)
      ))
    : null;

  const currentPlan = prediction
    ? ACTION_PLANS[persona][overallRisk?.toLowerCase() || 'medium']
    : [];

  // Highlight which features are most impactful given current inputs
  const getInputRiskFlags = () => {
    const flags = [];
    if (formData.wealth_index <= 2)  flags.push('Low wealth index — strongest predictor across all three outcomes');
    if (formData.mother_bmi < 1850)  flags.push('Mother BMI is low — elevates wasting and underweight risk');
    if (formData.mother_edu_years < 5) flags.push('Low maternal education — linked to poorer feeding practices');
    if (formData.birth_weight < 2500) flags.push('Low birth weight — strong risk factor for underweight');
    if (formData.bcg_vaccination < 1) flags.push('Incomplete BCG vaccination — reduces immunity, worsens nutritional outcomes');
    return flags;
  };

  const inputFlags = getInputRiskFlags();

  return (
    <div className="prediction-page">
      <Header
        title="Malnutrition Risk Estimator"
        subtitle="Estimate an individual child's malnutrition risk from their household and health profile"
      />

      <div className="page-container fade-in">

        {/* ── Conceptual framing banner ── */}
        <div className="context-banner card" onClick={() => setShowContext(v => !v)}>
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
                This tool operates at the <strong>individual child level</strong>. The inputs below describe
                one specific child and their mother/household — not a district average.
              </p>
              <p>
                The model was trained on 200k+ individual child records from NFHS-5 (India's national health
                survey), so a single child's profile drives the prediction directly, rather than an averaged
                district profile.
              </p>
              <p>
                To estimate a district or region's overall burden, run this for a representative sample of
                children from that area and average the results — that's how the underlying model itself
                was validated at the district level.
              </p>
              <p>
                This enables <strong>individual risk screening as well as policy planning and resource
                allocation</strong> when aggregated — but predictions are probabilistic estimates, not
                clinical diagnoses.
              </p>
            </div>
          )}
        </div>

        {/* ── Persona selector ── */}
        <div className="persona-bar card">
          <span className="persona-label"><Users size={16} /> I am a:</span>
          {Object.entries(ACTION_PLANS).map(([key, val]) => (
            <button
              key={key}
              className={`persona-btn ${persona === key ? 'active' : ''}`}
              onClick={() => setPersona(key)}
              type="button"
            >
              <val.icon size={14} />
              {val.label}
            </button>
          ))}
          <span className="persona-desc">{ACTION_PLANS[persona].description}</span>
        </div>

        {/* ── Input risk flags ── */}
        {inputFlags.length > 0 && (
          <div className="input-flags card">
            <h4 className="risk-flags-title"><AlertTriangle size={16} /> Profile Risk Signals</h4>
            <div className="flags-list">
              {inputFlags.map((f, i) => (
                <div key={i} className="flag-item">{f}</div>
              ))}
            </div>
          </div>
        )}

        <div className="prediction-grid">
          {/* ── Form ── */}
          <div className="form-section card">
            <h3 className="form-title">Child Profile Inputs</h3>

            <div className="form-group-section">
              <h4 className="group-title"><MapPin size={16} /> Location</h4>
              <div className="form-group">
                <label>State</label>
                <select name="state" value={formData.state} onChange={handleChange}>
                  {STATE_OPTIONS.map(([code, name]) => (
                    <option key={code} value={code}>{name}</option>
                  ))}
                </select>
                <span className="help-text"><strong>Top-3 predictor for wasting and underweight</strong></span>
              </div>
              <div className="form-group">
                <label>Area Type</label>
                <select name="urban_rural" value={formData.urban_rural} onChange={handleChange}>
                  <option value="1">Urban</option>
                  <option value="2">Rural</option>
                </select>
              </div>
            </div>

            <div className="form-group-section">
              <h4 className="group-title"><UserRound size={16} /> Maternal Characteristics</h4>
              <div className="form-group">
                <label>Household Wealth Index (1–5)</label>
                <input type="number" name="wealth_index" min="1" max="5" step="1" value={formData.wealth_index} onChange={handleChange} />
                <span className="help-text">1 = Poorest quintile, 5 = Richest quintile | <strong>Strongest overall predictor</strong></span>
              </div>
              <div className="form-group">
                <label>Mother's Age (years)</label>
                <input type="number" name="mother_age" min="15" max="49" value={formData.mother_age} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Mother's Education (years)</label>
                <input type="number" name="mother_edu_years" min="0" max="20" value={formData.mother_edu_years} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Mother's Education Level</label>
                <select name="mother_edu_level" value={formData.mother_edu_level} onChange={handleChange}>
                  <option value="0">No education</option>
                  <option value="1">Primary</option>
                  <option value="2">Secondary</option>
                  <option value="3">Higher</option>
                </select>
              </div>
              <div className="form-group">
                <label>Mother's BMI (×100)</label>
                <input type="number" name="mother_bmi" min="1200" max="6000" step="50" value={formData.mother_bmi} onChange={handleChange} />
                <span className="help-text">Normal range: 1850–2500</span>
              </div>
              <div className="form-group">
                <label>Mother Currently Working?</label>
                <select name="mother_works" value={formData.mother_works} onChange={handleChange}>
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>
              <div className="form-group">
                <label>Head of Household</label>
                <select name="female_headed_hh" value={formData.female_headed_hh} onChange={handleChange}>
                  <option value="1">Male</option>
                  <option value="2">Female</option>
                </select>
              </div>
            </div>

            <div className="form-group-section">
              <h4 className="group-title"><Baby size={16} /> Child Profile</h4>
              <div className="form-group">
                <label>Child Age (months)</label>
                <input type="number" name="child_age_months" min="0" max="59" value={formData.child_age_months} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Child Age (years, rounded)</label>
                <input type="number" name="child_age_years" min="0" max="4" value={formData.child_age_years} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Child's Sex</label>
                <select name="child_sex" value={formData.child_sex} onChange={handleChange}>
                  <option value="1">Male</option>
                  <option value="2">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label>Birth Weight (grams)</label>
                <input type="number" name="birth_weight" min="500" max="5850" step="50" value={formData.birth_weight} onChange={handleChange} />
                <span className="help-text">Below 2500g = low birth weight risk zone</span>
              </div>
              <div className="form-group">
                <label>
                  <input type="checkbox" checked={firstBorn} onChange={(e) => setFirstBorn(e.target.checked)} style={{ marginRight: 6 }} />
                  First-born child (no preceding birth)
                </label>
              </div>
              {!firstBorn && (
                <div className="form-group">
                  <label>Birth Interval (months since previous sibling)</label>
                  <input type="number" name="birth_interval" min="5" max="280" value={formData.birth_interval} onChange={handleChange} />
                </div>
              )}
            </div>

            <div className="form-group-section">
              <h4 className="group-title"><Syringe size={16} /> Healthcare & Nutrition</h4>
              <div className="form-group">
                <label>Breastfeeding Duration (months)</label>
                <input type="number" name="breastfeed_duration" min="0" max="90" value={formData.breastfeed_duration} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Mother Knows About ORS?</label>
                <select name="knows_ors" value={formData.knows_ors} onChange={handleChange}>
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>
              <div className="form-group">
                <label>BCG Vaccination</label>
                <input type="number" name="bcg_vaccination" min="0" max="3" step="1" value={formData.bcg_vaccination} onChange={handleChange} />
                <span className="help-text">0 = None, 1 = Vaccination card, 2 = Mother's report</span>
              </div>
              <div className="form-group">
                <label>DPT Vaccination</label>
                <input type="number" name="dpt_vaccination" min="0" max="3" step="1" value={formData.dpt_vaccination} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Measles Vaccination</label>
                <input type="number" name="measles_vaccination" min="0" max="3" step="1" value={formData.measles_vaccination} onChange={handleChange} />
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button className="btn btn-primary predict-submit-btn" disabled={loading} onClick={handleSubmit} type="button">
              {loading ? 'Estimating...' : (<><Search size={16} /> Run Prediction</>)}
            </button>
          </div>

          {/* ── Results ── */}
          <div className="results-section">
            {prediction ? (
              <div className="results-container fade-in">

                {/* Risk cards */}
                {[
                  { key: 'stunting',    label: 'Stunting',    icon: TrendingDown, bgVar: 'var(--stunting-light)',    iconVar: 'var(--stunting)',    desc: 'Height-for-age deficit' },
                  { key: 'wasting',     label: 'Wasting',     icon: AlertCircle,  bgVar: 'var(--wasting-light)',     iconVar: 'var(--wasting)',     desc: 'Acute malnutrition' },
                  { key: 'underweight', label: 'Underweight', icon: Activity,     bgVar: 'var(--underweight-light)', iconVar: 'var(--underweight)', desc: 'Weight-for-age deficit' },
                ].map(({ key, label, icon: Icon, bgVar, iconVar, desc }) => {
                  const risk = prediction.risk_level[key];
                  return (
                    <div key={key} className="result-card card">
                      <div className="result-header" style={{ background: bgVar }}>
                        <Icon size={24} style={{ color: iconVar }} />
                        <div>
                          <h3>{label}</h3>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{desc}</span>
                        </div>
                      </div>
                      <div className="result-body">
                        <div className="result-value">{prediction[key].toFixed(1)}%</div>
                        <div>
                          <div className="result-badge" style={{ background: RISK_COLORS[risk], color: 'white' }}>
                            {risk} Risk
                          </div>
                          <div className="result-vs-national">
                            vs national avg {key === 'stunting' ? '35.5' : key === 'wasting' ? '19.3' : '32.1'}%
                          </div>
                        </div>
                      </div>
                      {/* Mini gauge bar */}
                      <div className="gauge-track">
                        <div className="gauge-fill" style={{
                          width: `${Math.min(prediction[key], 100)}%`,
                          background: RISK_COLORS[risk]
                        }} />
                      </div>
                    </div>
                  );
                })}

                {/* ── Feature Importance explainability ── */}
                <div className="card explain-card">
                  <button
                    className="explain-toggle"
                    type="button"
                    onClick={() => {
                      if (!showExplain) {
                        // auto-select the highest-risk target when opening
                        const targets = ['stunting', 'wasting', 'underweight'];
                        const highest = targets.reduce((a, b) =>
                          prediction[a] > prediction[b] ? a : b
                        );
                        setActiveExplain(highest);
                      }
                      setShowExplain(v => !v);
                    }}
                  >
                    <div className="explain-toggle-left">
                      <Zap size={18} style={{ color: '#6366f1' }} />
                      <span>What drives this prediction?</span>
                    </div>
                    {showExplain ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                  {showExplain && (
                    <div className="explain-body">
                      <div className="explain-tabs">
                        {[
                          { key: 'stunting',    label: 'Stunting',    model: 'XGB' },
                          { key: 'wasting',     label: 'Wasting',     model: 'XGB' },
                          { key: 'underweight', label: 'Underweight', model: 'XGB' },
                        ].map(({ key, label, model }) => (
                          <button
                            key={key}
                            type="button"
                            className={`explain-tab ${activeExplain === key ? 'active' : ''}`}
                            onClick={() => setActiveExplain(key)}
                          >
                            {label}
                            <span className="explain-tab-model">{model}</span>
                          </button>
                        ))}
                      </div>
                      <p className="explain-subtitle">
                        Feature importance — XGBoost {activeExplain} model (child-level)
                      </p>
                      {FEATURE_IMPORTANCE[activeExplain].map((f) => (
                        <div key={f.label} className="fi-row">
                          <span className="fi-label">{f.label}</span>
                          <div className="fi-track">
                            <div className="fi-fill" style={{ width: `${Math.min(f.value * 1.8, 100)}%`, background: f.color }} />
                          </div>
                          <span className="fi-pct">{f.value}%</span>
                        </div>
                      ))}
                      <p className="explain-note">
                        * From actual <code>feature_importances_</code> outputs. Higher value = more influence on prediction.
                      </p>
                    </div>
                  )}
                </div>

                {/* ── Action Plan ── */}
                <div className="card action-plan">
                  <div className="action-plan-header">
                    <Target size={20} style={{ color: '#6366f1' }} />
                    <div>
                      <h4>Action Plan — {ACTION_PLANS[persona].label}</h4>
                      <span className="action-plan-risk" style={{ color: RISK_COLORS[overallRisk] }}>
                        Overall Risk: {overallRisk}
                      </span>
                    </div>
                  </div>
                  <div className="action-items">
                    {currentPlan.map((item, i) => (
                      <div key={i} className="action-item">
                        <div className="action-priority" style={{ background: PRIORITY_COLORS[item.priority] }}>
                          {item.priority}
                        </div>
                        <div className="action-content">
                          <p className="action-text">{item.action}</p>
                          <span className="action-dept">{item.dept}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* ── Model transparency note ── */}
                <div className="card transparency-note">
                  <BookOpen size={16} style={{ color: 'var(--text-secondary)' }} />
                  <p>
                    <strong>Model transparency:</strong> Child-level XGBoost classifiers trained on
                    206k–210k individual NFHS-5 child records. Aggregated to district level, the models
                    explain R² 61% (stunting), 50% (wasting), and 76% (underweight) of district-to-district
                    variation — a large improvement over the previous district-aggregate model. Individual
                    predictions remain probabilistic estimates, not clinical diagnoses.
                  </p>
                </div>

              </div>
            ) : (
              <div className="empty-state card">
                <Activity size={48} style={{ color: 'var(--text-light)' }} />
                <h3>No Estimate Yet</h3>
                <p>Fill in this child's profile on the left and click <strong>Run Prediction</strong> to see risk estimates and action plan.</p>
                <div className="empty-persona-hint">
                  Selected persona: <strong>{ACTION_PLANS[persona].label}</strong><br/>
                  <span>{ACTION_PLANS[persona].description}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Prediction;