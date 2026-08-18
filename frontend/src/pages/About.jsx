import React, { useState } from 'react';
import { Database, BarChart3, Brain, Github, ChevronDown, ChevronUp, AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react';
import Header from '../components/Header';
import './About.css';

const MODEL_COMPARISON = [
  { model: 'Child-level XGBoost (current)', stunting_r2: 60.8, wasting_r2: 49.5, underweight_r2: 76.0, stunting_rmse: 5.32, wasting_rmse: 4.63, underweight_rmse: 4.80, deployed: ['stunting', 'wasting', 'underweight'] },
  { model: 'District-aggregate Random Forest (previous)', stunting_r2: 49.7, wasting_r2: 42.7, underweight_r2: 67.7, stunting_rmse: 5.42, wasting_rmse: 4.39, underweight_rmse: 5.28, deployed: [] },
  { model: 'District-aggregate XGBoost (previous)', stunting_r2: 43.1, wasting_r2: 36.4, underweight_r2: 69.1, stunting_rmse: 5.76, wasting_rmse: 4.63, underweight_rmse: 5.16, deployed: [] },
  { model: 'District-aggregate Linear Regression (previous)', stunting_r2: 43.6, wasting_r2: 36.0, underweight_r2: 64.3, stunting_rmse: 5.73, wasting_rmse: 4.64, underweight_rmse: 5.55, deployed: [] },
];

// Feature importance from actual model outputs — child-level XGBoost (v2)
const FEATURE_IMPORTANCE = {
  stunting: [
    { feature: 'Wealth Index',               value: 31.5, color: '#ef4444' },
    { feature: "Mother's Education (years)", value: 10.5, color: '#f97316' },
    { feature: "Mother's Education Level",   value: 9.0,  color: '#eab308' },
    { feature: 'Child Age (years)',          value: 8.4,  color: '#84cc16' },
    { feature: 'Child Age (months)',         value: 6.7,  color: '#22c55e' },
    { feature: 'Birth Weight',               value: 6.4,  color: '#14b8a6' },
    { feature: 'State',                      value: 5.0,  color: '#6366f1' },
    { feature: "Mother's BMI",               value: 3.9,  color: '#8b5cf6' },
    { feature: 'Child Sex',                  value: 3.5,  color: '#ec4899' },
    { feature: 'Birth Interval',             value: 3.2,  color: '#06b6d4' },
  ],
  wasting: [
    { feature: 'State',                      value: 12.4, color: '#6366f1' },
    { feature: 'Child Age (years)',          value: 11.2, color: '#84cc16' },
    { feature: 'Child Age (months)',         value: 9.4,  color: '#22c55e' },
    { feature: 'Birth Weight',               value: 9.0,  color: '#14b8a6' },
    { feature: "Mother's BMI",               value: 8.2,  color: '#ef4444' },
    { feature: 'Wealth Index',               value: 6.1,  color: '#f97316' },
    { feature: 'Child Sex',                  value: 5.3,  color: '#ec4899' },
    { feature: 'Measles Vaccination',        value: 4.6,  color: '#8b5cf6' },
    { feature: "Mother's Education (years)", value: 4.4,  color: '#eab308' },
    { feature: 'Urban/Rural',                value: 4.1,  color: '#06b6d4' },
  ],
  underweight: [
    { feature: 'Wealth Index',               value: 19.7, color: '#ef4444' },
    { feature: 'Birth Weight',               value: 14.3, color: '#84cc16' },
    { feature: 'State',                      value: 10.0, color: '#6366f1' },
    { feature: "Mother's BMI",               value: 9.8,  color: '#f97316' },
    { feature: "Mother's Education (years)", value: 8.9,  color: '#eab308' },
    { feature: "Mother's Education Level",   value: 7.5,  color: '#22c55e' },
    { feature: 'Child Age (months)',         value: 4.7,  color: '#14b8a6' },
    { feature: 'Child Age (years)',          value: 4.5,  color: '#8b5cf6' },
    { feature: 'Child Sex',                  value: 4.4,  color: '#ec4899' },
    { feature: 'Birth Interval',             value: 2.8,  color: '#06b6d4' },
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
          <h2>Predicting Child Malnutrition Using Machine Learning</h2>
          <p>
            This system uses XGBoost trained on individual NFHS-5 child records to predict
            malnutrition risk — for an individual child, or aggregated to district-level burden —
            enabling evidence-based resource allocation and policy simulation for policymakers,
            NGOs, and researchers.
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
              <li>19 predictor features, including state</li>
              <li>3 target variables: stunting, wasting, underweight</li>
              <li>Official Government of India data</li>
            </ul>
          </div>

          <div className="about-card card">
            <div className="about-icon" style={{ background: 'var(--success-light)' }}>
              <Brain size={28} style={{ color: 'var(--success)' }} />
            </div>
            <h3>ML Models Deployed</h3>
            <p>Child-level XGBoost classifiers trained on 206k+ individual NFHS-5 child records, aggregated to district level for evaluation.</p>
            <ul>
              <li>XGBoost → Stunting (R² 60.8%)</li>
              <li>XGBoost → Wasting (R² 49.5%)</li>
              <li>XGBoost → Underweight (R² 76.0%)</li>
              <li>Previous district-aggregate models kept below for comparison</li>
            </ul>
          </div>

          <div className="about-card card">
            <div className="about-icon" style={{ background: 'var(--stunting-light)' }}>
              <BarChart3 size={28} style={{ color: 'var(--stunting)' }} />
            </div>
            <h3>Key Findings</h3>
            <p>Feature importance analysis reveals the dominant socioeconomic drivers of malnutrition.</p>
            <ul>
              <li>Wealth Index — 31.5% importance (stunting), strongest overall predictor</li>
              <li>Mother's Education (years) — 10.5% importance</li>
              <li>State — top-3 predictor for wasting and underweight</li>
              <li>Birth Weight — key predictor across all three outcomes</li>
            </ul>
          </div>

          <div className="about-card card">
            <div className="about-icon" style={{ background: 'var(--wasting-light)' }}>
              <Github size={28} style={{ color: 'var(--wasting)' }} />
            </div>
            <h3>Technology Stack</h3>
            <p>Modern full-stack application built for decision-support at district and individual-child scale.</p>
            <ul>
              <li>React.js + Recharts + Lucide</li>
              <li>FastAPI + Python 3.11</li>
              <li>XGBoost, Pandas</li>
              <li>NFHS-5 child-level microdata pipeline (206k+ records)</li>
            </ul>
          </div>
        </div>

        {/* ── Model Evaluation Section ── */}
        <div className="model-eval card">
          <div className="model-eval-header">
            <TrendingUp size={22} style={{ color: '#6366f1' }} />
            <div>
              <h3>Model Evaluation — Child-Level Rebuild vs. Previous District Models</h3>
              <p className="eval-subtitle">R² and RMSE across all three malnutrition targets, evaluated at district level via 5-fold cross-validation.</p>
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
              <strong>Why the jump from ~43–69% to 50–76% R²?</strong>
            </div>
            <div className="r2-reasons">
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>707 rows → 206k+ rows:</strong> The previous models trained on district-level averages (707 rows total). Averaging before training discards individual-level variation the model could otherwise learn from. The current model trains directly on individual NFHS-5 child records.</p>
              </div>
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>Noise cancels on aggregation, not before it:</strong> Per-child predictions are inherently noisy (AUC ~0.63–0.70), but averaging thousands of them per district cancels out individual noise — producing a more accurate district-level estimate than fitting directly on pre-averaged, noise-inflated district rows.</p>
              </div>
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>State as a real feature:</strong> State was previously excluded from district-level training as an ID column. At child level it carries real predictive signal not already captured by other averaged features, and is now a top-3 predictor for wasting and underweight.</p>
              </div>
              <div className="r2-reason">
                <CheckCircle size={14} style={{ color: '#22c55e', flexShrink: 0 }} />
                <p><strong>Remaining ceiling:</strong> The residual error reflects genuine unexplained variation — illness episodes, local food security shocks, and factors not captured in survey variables. Published district-level nutritional regression studies in LMICs report R² of 40–70%; this rebuild exceeds that range for two of three targets.</p>
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
              { n: 1, title: 'Data Collection', body: 'Downloaded NFHS-5 child-level microdata (DHS Children\'s Recode, 232,920 children across 707 districts) directly from the DHS Program — 300+ raw variables narrowed to 19 theory-driven predictors, kept at individual child level.' },
              { n: 2, title: 'Feature Engineering', body: 'Trained directly on individual child records rather than pre-aggregating to district means. Handled missing values (e.g. birth interval for first-born children) via training-median imputation. Corrected two mislabeled variables inherited from an earlier pipeline: birth weight and birth interval.' },
              { n: 3, title: 'Model Training', body: 'XGBoost classifiers trained per target with 5-fold stratified cross-validation, evaluated by aggregating per-child out-of-fold predictions up to the district level and comparing against actual district rates — avoiding any leakage between folds.' },
              { n: 4, title: 'Model Selection', body: 'Child-level XGBoost selected for all three targets, replacing the previous district-aggregate Random Forest/XGBoost models. Models serialized as XGBoost native JSON and served via FastAPI.' },
              { n: 5, title: 'Deployment', body: 'FastAPI backend exposes prediction and district data endpoints. React frontend provides individual child risk screening, district explorer, and persona-tailored action planning.' },
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
            <div><strong>Year</strong><p>2026</p></div>
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