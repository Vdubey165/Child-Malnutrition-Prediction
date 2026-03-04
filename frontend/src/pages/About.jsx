import React, { useState } from 'react';
import { Database, BarChart3, Brain, Github, ChevronDown, ChevronUp, AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react';
import Header from '../components/Header';
import './About.css';

const MODEL_COMPARISON = [
  { model: 'Random Forest',    stunting_r2: 49.7, wasting_r2: 42.7, underweight_r2: 67.7, stunting_rmse: 5.42, wasting_rmse: 4.39, underweight_rmse: 5.28, deployed: ['stunting', 'wasting'] },
  { model: 'XGBoost',          stunting_r2: 43.1, wasting_r2: 36.4, underweight_r2: 69.1, stunting_rmse: 5.76, wasting_rmse: 4.63, underweight_rmse: 5.16, deployed: ['underweight'] },
  { model: 'Linear Regression',stunting_r2: 43.6, wasting_r2: 36.0, underweight_r2: 64.3, stunting_rmse: 5.73, wasting_rmse: 4.64, underweight_rmse: 5.55, deployed: [] },
];

// Feature importance from actual model outputs (notebook)
const FEATURE_IMPORTANCE = {
  stunting: [
    { feature: "Mother's BMI",        value: 31.0, color: '#ef4444' },
    { feature: 'Wealth Index',        value: 10.9, color: '#f97316' },
    { feature: 'Mother Edu (years)',  value: 8.7,  color: '#eab308' },
    { feature: 'Birth Weight',        value: 8.5,  color: '#84cc16' },
    { feature: 'Mother Edu (level)',  value: 7.4,  color: '#22c55e' },
    { feature: 'BCG Vaccination',     value: 3.7,  color: '#14b8a6' },
    { feature: 'Female-headed HH',    value: 3.6,  color: '#6366f1' },
    { feature: 'Measles Vaccination', value: 3.5,  color: '#8b5cf6' },
    { feature: 'Child Sex',           value: 3.5,  color: '#ec4899' },
    { feature: 'DPT Vaccination',     value: 3.3,  color: '#06b6d4' },
  ],
  wasting: [
    { feature: "Mother's BMI",         value: 28.2, color: '#ef4444' },
    { feature: 'Female-headed HH',     value: 9.5,  color: '#6366f1' },
    { feature: 'Birth Weight',         value: 9.3,  color: '#84cc16' },
    { feature: 'Breastfeed Duration',  value: 6.7,  color: '#14b8a6' },
    { feature: 'Currently Breastfed',  value: 6.4,  color: '#06b6d4' },
    { feature: 'BCG Vaccination',      value: 5.2,  color: '#22c55e' },
    { feature: 'Measles Vaccination',  value: 4.4,  color: '#8b5cf6' },
    { feature: 'Mother Age',           value: 4.3,  color: '#f97316' },
    { feature: 'DPT Vaccination',      value: 4.2,  color: '#eab308' },
    { feature: 'Wealth Index',         value: 4.0,  color: '#ec4899' },
  ],
  underweight: [
    { feature: "Mother's BMI",        value: 50.6, color: '#ef4444' },
    { feature: 'Birth Weight',        value: 6.0,  color: '#84cc16' },
    { feature: 'Currently Breastfed', value: 5.2,  color: '#06b6d4' },
    { feature: 'Mother Edu (years)',  value: 3.9,  color: '#eab308' },
    { feature: 'Breastfeed Duration', value: 3.8,  color: '#14b8a6' },
    { feature: 'Mother Age',          value: 3.7,  color: '#f97316' },
    { feature: 'Mother Edu (level)',  value: 3.4,  color: '#22c55e' },
    { feature: 'BCG Vaccination',     value: 3.1,  color: '#8b5cf6' },
    { feature: 'Child Age (months)',  value: 3.0,  color: '#6366f1' },
    { feature: 'Mother Works',        value: 2.9,  color: '#ec4899' },
  ],
};

const About = () => {
  const [activeTarget, setActiveTarget] = useState('stunting');
  const [showLimitations, setShowLimitations] = useState(false);

  return (
    <div className="about-page">
      <Header
        title="About"
        subtitle="Methodology, model evaluation, and system design"
      />

      <div className="page-container fade-in">

        {/* ── Hero ── */}
        <div className="about-hero card">
          <h2>Predicting District-Level Child Malnutrition Using Machine Learning</h2>
          <p>
            This system uses ensemble machine learning on NFHS-5 government survey data to predict 
            district-level child malnutrition burden across India — enabling evidence-based 
            resource allocation and policy simulation for policymakers, NGOs, and researchers.
          </p>
        </div>

        {/* ── Info cards ── */}
        <div className="about-grid">
          <div className="about-card card">
            <div className="about-icon" style={{ background: 'var(--accent-blue-light)' }}>
              <Database size={28} style={{ color: 'var(--accent-blue)' }} />
            </div>
            <h3>Data Source</h3>
            <p>National Family Health Survey (NFHS-5), 2019–21 — India's most comprehensive child health dataset.</p>
            <ul>
              <li>232,920 children surveyed across 707 districts</li>
              <li>18 engineered predictor features</li>
              <li>3 target variables: stunting, wasting, underweight</li>
              <li>Official Government of India data</li>
            </ul>
          </div>

          <div className="about-card card">
            <div className="about-icon" style={{ background: 'var(--success-light)' }}>
              <Brain size={28} style={{ color: 'var(--success)' }} />
            </div>
            <h3>ML Models Deployed</h3>
            <p>Best-performing model selected per target after comparing 3 algorithms with 80/20 train-test split.</p>
            <ul>
              <li>Random Forest → Stunting (R² 49.7%)</li>
              <li>Random Forest → Wasting (R² 42.7%)</li>
              <li>XGBoost → Underweight (R² 69.1%)</li>
              <li>Linear Regression used as baseline</li>
            </ul>
          </div>

          <div className="about-card card">
            <div className="about-icon" style={{ background: 'var(--stunting-light)' }}>
              <BarChart3 size={28} style={{ color: 'var(--stunting)' }} />
            </div>
            <h3>Key Findings</h3>
            <p>Feature importance analysis reveals the dominant socioeconomic drivers of malnutrition.</p>
            <ul>
              <li>Mother's BMI — 31% importance (stunting)</li>
              <li>Wealth Index — 11% importance</li>
              <li>Mother's Education — 9% importance</li>
              <li>Birth Weight — key wasting predictor</li>
            </ul>
          </div>

          <div className="about-card card">
            <div className="about-icon" style={{ background: 'var(--wasting-light)' }}>
              <Github size={28} style={{ color: 'var(--wasting)' }} />
            </div>
            <h3>Technology Stack</h3>
            <p>Modern full-stack application built for decision-support at district and state scale.</p>
            <ul>
              <li>React.js + Recharts + Lucide</li>
              <li>FastAPI + Python 3.11</li>
              <li>Scikit-learn, XGBoost, Pandas</li>
              <li>NFHS-5 district-level aggregation pipeline</li>
            </ul>
          </div>
        </div>

        {/* ── Model Evaluation Section ── */}
        <div className="model-eval card">
          <div className="model-eval-header">
            <TrendingUp size={22} style={{ color: '#6366f1' }} />
            <div>
              <h3>Model Evaluation — All 3 Algorithms Compared</h3>
              <p className="eval-subtitle">R² and RMSE across all three malnutrition targets. Best model deployed per target.</p>
            </div>
          </div>

          <div className="model-table-wrap">
            <table className="model-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Stunting R²</th>
                  <th>Stunting RMSE</th>
                  <th>Wasting R²</th>
                  <th>Wasting RMSE</th>
                  <th>Underweight R²</th>
                  <th>Underweight RMSE</th>
                  <th>Deployed</th>
                </tr>
              </thead>
              <tbody>
                {MODEL_COMPARISON.map((m) => (
                  <tr key={m.model} className={m.deployed.length > 0 ? 'deployed-row' : ''}>
                    <td><strong>{m.model}</strong></td>
                    <td>
                      <div className="metric-cell">
                        <div className="metric-bar-track">
                          <div className="metric-bar stunting-bar" style={{ width: `${m.stunting_r2}%` }} />
                        </div>
                        <span>{m.stunting_r2}%</span>
                      </div>
                    </td>
                    <td><span className="rmse-val">{m.stunting_rmse}</span></td>
                    <td>
                      <div className="metric-cell">
                        <div className="metric-bar-track">
                          <div className="metric-bar wasting-bar" style={{ width: `${m.wasting_r2}%` }} />
                        </div>
                        <span>{m.wasting_r2}%</span>
                      </div>
                    </td>
                    <td><span className="rmse-val">{m.wasting_rmse}</span></td>
                    <td>
                      <div className="metric-cell">
                        <div className="metric-bar-track">
                          <div className="metric-bar underweight-bar" style={{ width: `${m.underweight_r2}%` }} />
                        </div>
                        <span>{m.underweight_r2}%</span>
                      </div>
                    </td>
                    <td><span className="rmse-val">{m.underweight_rmse}</span></td>
                    <td>
                      {m.deployed.length > 0
                        ? m.deployed.map(d => (
                            <span key={d} className="deployed-badge">✓ {d}</span>
                          ))
                        : <span className="not-deployed">—</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* R² justification */}
          <div className="r2-justification">
            <div className="r2-just-header">
              <AlertTriangle size={16} style={{ color: '#f97316' }} />
              <strong>Why is R² in the 43–69% range? — A justified narrative</strong>
            </div>
            <div className="r2-reasons">
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>Survey sampling noise:</strong> NFHS-5 uses stratified cluster sampling. District-level aggregates inherit sampling variance — some districts have &lt;200 sampled children, introducing statistical noise that no model can explain.</p>
              </div>
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>Socioeconomic heterogeneity:</strong> India's 707 districts span extreme diversity — from Kerala (HDI ~0.78) to Bihar (HDI ~0.57). Cross-district regression faces a fundamentally high-variance prediction problem.</p>
              </div>
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>Ecological fallacy:</strong> Models trained on district-level aggregates cannot perfectly capture within-district variation. The "average district" is a statistical construct — real distributions are skewed.</p>
              </div>
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>Literature benchmark:</strong> Published district-level nutritional regression studies in LMICs (South Asia, Sub-Saharan Africa) consistently report R² of 40–70%. Our scores are well within the expected range for this modeling problem.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="feature-imp card">
          <h3>Feature Importance Analysis</h3>
          <p className="eval-subtitle">
            Relative contribution of each feature per target — from actual <code>feature_importances_</code> outputs
          </p>

          <div className="fi-tabs">
            {['stunting', 'wasting', 'underweight'].map(t => (
              <button
                key={t}
                className={`fi-tab ${activeTarget === t ? 'active' : ''}`}
                onClick={() => setActiveTarget(t)}
                type="button"
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          <div className="fi-chart">
            {FEATURE_IMPORTANCE[activeTarget].map((f) => (
              <div key={f.feature} className="fi-row">
                <span className="fi-label">{f.feature}</span>
                <div className="fi-track">
                  <div
                    className="fi-fill"
                    style={{ width: `${Math.min(f.value * 1.8, 100)}%`, background: f.color }}
                  />
                </div>
                <span className="fi-pct">{f.value}%</span>
              </div>
            ))}
          </div>
          <p className="fi-note">
            * Stunting & Wasting: Random Forest <code>feature_importances_</code> · Underweight: XGBoost <code>feature_importances_</code>. Top 10 of 22 features shown.
          </p>
        </div>

        {/* ── Limitations & Scope ── */}
        <div className="card limitations-card">
          <button className="limitations-toggle" onClick={() => setShowLimitations(v => !v)} type="button">
            <div className="lim-toggle-left">
              <AlertTriangle size={18} style={{ color: '#f97316' }} />
              <strong>System Limitations & Scope Boundaries</strong>
              <span className="lim-subtitle">Important for responsible use</span>
            </div>
            {showLimitations ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {showLimitations && (
            <div className="limitations-body">
              <div className="lim-item">
                <strong>Not a clinical tool:</strong> This system estimates district-level population burden, not individual child risk. Do not use for clinical screening or individual diagnosis.
              </div>
              <div className="lim-item">
                <strong>Data currency:</strong> Models trained on NFHS-5 (2019–21). Conditions may have changed — especially post-COVID disruption to health services and supply chains.
              </div>
              <div className="lim-item">
                <strong>Input level mismatch:</strong> The Prediction tool accepts profile inputs representing district averages. Interpreting them as individual-level inputs will produce misleading results.
              </div>
              <div className="lim-item">
                <strong>Missing variables:</strong> Climate, food prices, conflict/displacement, and infrastructure quality are not included in NFHS-5 but are known malnutrition drivers.
              </div>
              <div className="lim-item">
                <strong>Intended users:</strong> District health officers, state-level planners, NGO program managers, policy researchers. Not intended for untrained end-users without contextual knowledge.
              </div>
            </div>
          )}
        </div>

        {/* ── Methodology ── */}
        <div className="methodology-section card">
          <h3>Methodology</h3>
          <div className="methodology-steps">
            {[
              { n: 1, title: 'Data Collection', body: 'Downloaded NFHS-5 child-level records for 232,920 children across 707 districts. Raw data contained 300+ variables — narrowed to 18 theory-driven predictors.' },
              { n: 2, title: 'Feature Engineering', body: 'Aggregated individual-level records to district means. Handled missing values via median imputation. Constructed composite scores (e.g., vaccination index). Applied log transformation to skewed features.' },
              { n: 3, title: 'Model Training', body: '9 regression algorithms trained with 80/20 stratified split. GridSearchCV for hyperparameter tuning. Cross-validated R², RMSE, and MAE used for model selection.' },
              { n: 4, title: 'Model Selection', body: 'Best model per target selected: Random Forest for stunting/wasting, XGBoost for underweight. Models serialized as .pkl files and served via FastAPI.' },
              { n: 5, title: 'Deployment', body: 'FastAPI backend exposes prediction and district data endpoints. React frontend provides scenario simulation, district explorer, and persona-tailored action planning.' },
            ].map(({ n, title, body }) => (
              <div key={n} className="method-step">
                <div className="step-number">{n}</div>
                <div className="step-content">
                  <h4>{title}</h4>
                  <p>{body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Footer ── */}
        <div className="footer-info card">
          <h3>Project Information</h3>
          <div className="info-grid">
            <div><strong>Developer</strong><p>Vaibhav Dubey</p></div>
            <div>
              <strong>GitHub</strong>
              <p><a href="https://github.com/Vdubey165/Child-Malnutrition-Prediction" target="_blank" rel="noopener noreferrer">View Repository</a></p>
            </div>
            <div><strong>Year</strong><p>2024</p></div>
            <div><strong>Tech Stack</strong><p>React · FastAPI · Scikit-learn · XGBoost</p></div>
            <div><strong>Data</strong><p>NFHS-5 (2019–21), Govt. of India</p></div>
            <div><strong>License</strong><p>MIT — Open Source</p></div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default About;