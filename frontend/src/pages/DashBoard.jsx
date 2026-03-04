import React, { useState, useEffect, useMemo } from 'react';
import { Activity, TrendingDown, Users, AlertCircle, AlertTriangle, BarChart2, ArrowRight,TrendingUp } from 'lucide-react';
import Header from '../components/Header';
import StatsCard from '../components/StatsCard';
import MalnutritionChart from '../components/MalnutritionChart';
import { getStatistics, getAllDistricts } from '../services/api';
import './DashBoard.css';

// Composite risk score — same formula as DistrictExplorer (keep consistent)
const compositeScore = (d) =>
  d.actual_stunting * 0.4 + d.actual_wasting * 0.35 + d.actual_underweight * 0.25;

const getRiskLevel = (score) => {
  if (score >= 30) return 'High';
  if (score >= 18) return 'Medium';
  return 'Low';
};

const RISK_COLORS = { High: '#ef4444', Medium: '#f97316', Low: '#22c55e' };
const RISK_BG     = { High: '#fef2f2', Medium: '#fff7ed', Low: '#f0fdf4' };

// Model performance data (from notebooks — all 9 models)
const MODEL_METRICS = [
  { model: 'Random Forest',    stunting: 49.7, wasting: 42.7, underweight: 67.7, best: true  },
  { model: 'XGBoost',          stunting: 43.1, wasting: 36.4, underweight: 69.1, best: true  },
  { model: 'Linear Regression',stunting: 43.6, wasting: 36.0, underweight: 64.3, best: false },
];

const Dashboard = () => {
  const [stats, setStats]       = useState(null);
  const [districts, setDistricts] = useState([]);
  const [loading, setLoading]   = useState(true);
  const [showAllModels, setShowAllModels] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, districtsData] = await Promise.all([
          getStatistics(),
          getAllDistricts(707),
        ]);
        setStats(statsData);
        setDistricts(districtsData.districts);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Top 5 highest composite risk districts
  const topRiskDistricts = useMemo(() => {
    return [...districts]
      .sort((a, b) => compositeScore(b) - compositeScore(a))
      .slice(0, 5);
  }, [districts]);

  // Risk distribution counts
  const riskCounts = useMemo(() => {
    const counts = { High: 0, Medium: 0, Low: 0 };
    districts.forEach(d => counts[getRiskLevel(compositeScore(d))]++);
    return counts;
  }, [districts]);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  const chartData = [
    {
      name: 'National Avg',
      stunting:    stats?.national_average.stunting    || 0,
      wasting:     stats?.national_average.wasting     || 0,
      underweight: stats?.national_average.underweight || 0,
    },
  ];

  // All 3 models shown
  const visibleModels = MODEL_METRICS;

  return (
    <div className="dashboard">
      <Header
        title="Dashboard"
        subtitle="National overview of child malnutrition — India NFHS-5 (2019–21)"
      />

      <div className="page-container fade-in">

        {/* ── KPI cards ── */}
        <div className="stats-grid">
          <StatsCard
            title="Stunting Rate"
            value={`${stats?.national_average.stunting || 0}%`}
            subtitle="Height-for-age deficit"
            color="stunting"
            icon={TrendingDown}
          />
          <StatsCard
            title="Wasting Rate"
            value={`${stats?.national_average.wasting || 0}%`}
            subtitle="Acute malnutrition"
            color="wasting"
            icon={AlertCircle}
          />
          <StatsCard
            title="Underweight Rate"
            value={`${stats?.national_average.underweight || 0}%`}
            subtitle="Weight-for-age deficit"
            color="underweight"
            icon={Activity}
          />
          <StatsCard
            title="Total Children"
            value="232,920"
            subtitle={`Across ${stats?.total_districts || 707} districts`}
            color="accent-blue"
            icon={Users}
          />
        </div>

        {/* ── Risk distribution banner ── */}
        <div className="risk-distribution card">
          <div className="risk-dist-title">
            <AlertTriangle size={16} style={{ color: '#f97316' }} />
            <strong>District Risk Distribution</strong>
            <span className="risk-dist-note">Composite score: 40% stunting + 35% wasting + 25% underweight</span>
          </div>
          <div className="risk-dist-bars">
            {Object.entries(riskCounts).map(([level, count]) => (
              <div key={level} className="risk-dist-item">
                <span className="risk-dist-label" style={{ color: RISK_COLORS[level] }}>{level}</span>
                <div className="risk-dist-track">
                  <div
                    className="risk-dist-fill"
                    style={{
                      width: `${(count / districts.length) * 100}%`,
                      background: RISK_COLORS[level],
                    }}
                  />
                </div>
                <span className="risk-dist-count">{count} districts ({((count / districts.length) * 100).toFixed(0)}%)</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Chart + Insights ── */}
        <div className="dashboard-grid">
          <div className="chart-section">
            <MalnutritionChart data={chartData} />
          </div>

          <div className="insights-section card">
            <h3 className="section-title">Key Insights</h3>
            <div className="insight-list">
              <div className="insight-item">
                <div className="insight-icon" style={{ background: 'var(--stunting-light)' }}>
                  <TrendingDown size={20} style={{ color: 'var(--stunting)' }} />
                </div>
                <div>
                  <h4>Stunting — Chronic</h4>
                  <p>35.5% of children under 5 stunted. Reflects long-term deprivation — driven by maternal BMI & wealth.</p>
                </div>
              </div>
              <div className="insight-item">
                <div className="insight-icon" style={{ background: 'var(--wasting-light)' }}>
                  <AlertCircle size={20} style={{ color: 'var(--wasting)' }} />
                </div>
                <div>
                  <h4>Wasting — Acute</h4>
                  <p>19.3% acutely malnourished. Immediate intervention required — linked to birth weight & vaccination gaps.</p>
                </div>
              </div>
              <div className="insight-item">
                <div className="insight-icon" style={{ background: 'var(--underweight-light)' }}>
                  <Activity size={20} style={{ color: 'var(--underweight)' }} />
                </div>
                <div>
                  <h4>Underweight — Combined</h4>
                  <p>32.1% underweight. XGBoost achieves 69% R² predicting this — strongest model signal.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Top Priority Districts ── */}
        <div className="districts-preview card">
          <div className="section-header">
            <div>
              <h3 className="section-title">🚨 Highest Risk Districts — Priority Intervention Targets</h3>
              <p className="section-subtitle">Sorted by composite malnutrition burden score</p>
            </div>
            <a href="/districts" className="view-all-link">
              View All <ArrowRight size={14} />
            </a>
          </div>
          <div className="districts-table">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>District</th>
                  <th>State</th>
                  <th>Stunting</th>
                  <th>Wasting</th>
                  <th>Underweight</th>
                  <th>Risk Score</th>
                  <th>Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {topRiskDistricts.map((district, idx) => {
                  const score = compositeScore(district);
                  const risk  = getRiskLevel(score);
                  return (
                    <tr key={district.district} className="priority-row">
                      <td>
                        <span className="rank-badge" style={{ background: idx === 0 ? '#ef4444' : idx === 1 ? '#f97316' : '#eab308' }}>
                          #{idx + 1}
                        </span>
                      </td>
                      <td><strong>{district.district_name || `District ${district.district}`}</strong></td>
                      <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        {district.state_name || `State ${district.state}`}
                      </td>
                      <td><span className="badge badge-stunting">{district.actual_stunting.toFixed(1)}%</span></td>
                      <td><span className="badge badge-wasting">{district.actual_wasting.toFixed(1)}%</span></td>
                      <td><span className="badge badge-underweight">{district.actual_underweight.toFixed(1)}%</span></td>
                      <td><strong style={{ color: RISK_COLORS[risk] }}>{score.toFixed(1)}</strong></td>
                      <td>
                        <span
                          className="risk-level-badge"
                          style={{ background: RISK_BG[risk], color: RISK_COLORS[risk], border: `1px solid ${RISK_COLORS[risk]}` }}
                        >
                          {risk}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Model Performance Section ── */}
        <div className="model-perf card">
            <div>
              <h3 className="model-eval-header">
                <TrendingUp size={22} style={{ color: '#6366f1' }} />
                Model Evaluation — 3 Algorithms Compared
              </h3>
              <p className="eval-subtitle">R² and RMSE across all three malnutrition targets. Best model deployed per target.</p>
            </div>

          <div className="model-table-wrap">
            <table className="model-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Stunting R²</th>
                  <th>Wasting R²</th>
                  <th>Underweight R²</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {visibleModels.map((m) => (
                  <tr key={m.model} className={m.best ? 'best-model-row' : ''}>
                    <td><strong>{m.model}</strong></td>
                    <td>
                      <div className="perf-cell">
                        <div className="perf-bar-track">
                          <div className="perf-bar-fill stunting-bar" style={{ width: `${m.stunting}%` }} />
                        </div>
                        <span>{m.stunting}%</span>
                      </div>
                    </td>
                    <td>
                      <div className="perf-cell">
                        <div className="perf-bar-track">
                          <div className="perf-bar-fill wasting-bar" style={{ width: `${m.wasting}%` }} />
                        </div>
                        <span>{m.wasting}%</span>
                      </div>
                    </td>
                    <td>
                      <div className="perf-cell">
                        <div className="perf-bar-track">
                          <div className="perf-bar-fill underweight-bar" style={{ width: `${m.underweight}%` }} />
                        </div>
                        <span>{m.underweight}%</span>
                      </div>
                    </td>
                    <td>
                      {m.best && <span className="best-badge">★ Deployed</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="model-perf-note">
            <strong>Why 43–69% R²?</strong> District-level regression on survey data has inherent ceiling — 
            NFHS-5 sampling noise, socioeconomic variability across 707 districts, and ecological fallacy 
            constraints (aggregated data masks individual-level variation). These scores are consistent with 
            published literature on district-level nutritional modeling in LMICs.
          </div>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;