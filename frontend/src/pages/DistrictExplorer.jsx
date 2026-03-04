import React, { useState, useEffect, useMemo } from 'react';
import { Search, Filter, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import Header from '../components/Header';
import { getAllDistricts } from '../services/api';
import './DistrictExplorer.css';

// Compute composite risk score (0–100) weighted by severity
const compositeScore = (d) =>
  d.actual_stunting * 0.4 + d.actual_wasting * 0.35 + d.actual_underweight * 0.25;

const getRiskLevel = (score) => {
  if (score >= 30) return 'High';
  if (score >= 18) return 'Medium';
  return 'Low';
};

const RISK_COLORS  = { High: '#ef4444', Medium: '#f97316', Low: '#22c55e' };
const RISK_BG      = { High: '#fef2f2', Medium: '#fff7ed', Low: '#f0fdf4' };

const DistrictExplorer = () => {
  const [districts, setDistricts]               = useState([]);
  const [filteredDistricts, setFilteredDistricts] = useState([]);
  const [searchTerm, setSearchTerm]             = useState('');
  const [sortBy, setSortBy]                     = useState('composite');
  const [riskFilter, setRiskFilter]             = useState('All');
  const [stateFilter, setStateFilter]           = useState('All');
  const [loading, setLoading]                   = useState(true);

  useEffect(() => {
    const fetchDistricts = async () => {
      try {
        const data = await getAllDistricts(707);
        setDistricts(data.districts);
        setFilteredDistricts(data.districts);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching districts:', error);
        setLoading(false);
      }
    };
    fetchDistricts();
  }, []);

  // Derive unique state names for filter
  const stateNames = useMemo(() => {
    const names = [...new Set(districts.map(d => d.state_name || `State ${d.state}`))].sort();
    return ['All', ...names];
  }, [districts]);

  // National averages for comparison
  const nationalAvg = useMemo(() => {
    if (!districts.length) return null;
    return {
      stunting:    districts.reduce((s, d) => s + d.actual_stunting, 0) / districts.length,
      wasting:     districts.reduce((s, d) => s + d.actual_wasting, 0) / districts.length,
      underweight: districts.reduce((s, d) => s + d.actual_underweight, 0) / districts.length,
    };
  }, [districts]);

  useEffect(() => {
    let filtered = [...districts];

    // Search
    if (searchTerm) {
      filtered = filtered.filter(d =>
        d.district.toString().includes(searchTerm) ||
        (d.district_name && d.district_name.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    // State filter
    if (stateFilter !== 'All') {
      filtered = filtered.filter(d => (d.state_name || `State ${d.state}`) === stateFilter);
    }

    // Risk filter
    if (riskFilter !== 'All') {
      filtered = filtered.filter(d => getRiskLevel(compositeScore(d)) === riskFilter);
    }

    // Sort
    filtered.sort((a, b) => {
      if (sortBy === 'composite')   return compositeScore(b) - compositeScore(a);
      if (sortBy === 'stunting')    return b.actual_stunting - a.actual_stunting;
      if (sortBy === 'wasting')     return b.actual_wasting - a.actual_wasting;
      if (sortBy === 'underweight') return b.actual_underweight - a.actual_underweight;
      if (sortBy === 'district')    return a.district - b.district;
      return 0;
    });

    setFilteredDistricts(filtered);
  }, [searchTerm, sortBy, riskFilter, stateFilter, districts]);

  const riskCounts = useMemo(() => {
    const counts = { High: 0, Medium: 0, Low: 0 };
    districts.forEach(d => counts[getRiskLevel(compositeScore(d))]++);
    return counts;
  }, [districts]);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading districts...</p>
      </div>
    );
  }

  return (
    <div className="district-explorer">
      <Header
        title="District Explorer"
        subtitle="Explore and prioritise districts by malnutrition burden across India"
      />

      <div className="page-container fade-in">

        {/* ── Summary risk bar ── */}
        <div className="risk-summary card">
          <div className="risk-summary-title">
            <Info size={16} /> District Risk Distribution (Composite Score)
          </div>
          <div className="risk-pills">
            {(['High', 'Medium', 'Low']).map(r => (
              <button
                key={r}
                className={`risk-pill ${riskFilter === r ? 'active' : ''}`}
                style={{
                  borderColor: RISK_COLORS[r],
                  background: riskFilter === r ? RISK_COLORS[r] : RISK_BG[r],
                  color: riskFilter === r ? 'white' : RISK_COLORS[r],
                }}
                onClick={() => setRiskFilter(prev => prev === r ? 'All' : r)}
                type="button"
              >
                <AlertTriangle size={12} /> {r}: {riskCounts[r]}
              </button>
            ))}
            <button
              className={`risk-pill ${riskFilter === 'All' ? 'active' : ''}`}
              style={{ borderColor: '#94a3b8', background: riskFilter === 'All' ? '#64748b' : '#f8fafc', color: riskFilter === 'All' ? 'white' : '#64748b' }}
              onClick={() => setRiskFilter('All')}
              type="button"
            >
              All: {districts.length}
            </button>
          </div>
          {nationalAvg && (
            <div className="national-avg-row">
              <span>National Avg →</span>
              <span className="avg-chip stunting-chip">Stunting {nationalAvg.stunting.toFixed(1)}%</span>
              <span className="avg-chip wasting-chip">Wasting {nationalAvg.wasting.toFixed(1)}%</span>
              <span className="avg-chip underweight-chip">Underweight {nationalAvg.underweight.toFixed(1)}%</span>
            </div>
          )}
        </div>

        {/* ── Controls ── */}
        <div className="explorer-controls card">
          <div className="search-box">
            <Search size={20} style={{ color: 'var(--text-secondary)' }} />
            <input
              type="text"
              placeholder="Search by district name or number..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div className="sort-controls">
            <Filter size={20} style={{ color: 'var(--text-secondary)' }} />
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="composite">Composite Risk (High → Low)</option>
              <option value="stunting">Stunting (High → Low)</option>
              <option value="wasting">Wasting (High → Low)</option>
              <option value="underweight">Underweight (High → Low)</option>
              <option value="district">District Number</option>
            </select>
          </div>

          <div className="sort-controls">
            <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
              {stateNames.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div className="results-count">
            Showing {filteredDistricts.length} districts
          </div>
        </div>

        <div className="districts-grid">
          {filteredDistricts.map((district) => {
            const score     = compositeScore(district);
            const riskLevel = getRiskLevel(score);
            const isAboveAvg = nationalAvg && score > compositeScore({ actual_stunting: nationalAvg.stunting, actual_wasting: nationalAvg.wasting, actual_underweight: nationalAvg.underweight });

            return (
              <div
                key={district.district}
                className="district-card card"
                style={{ borderTop: `3px solid ${RISK_COLORS[riskLevel]}` }}
              >
                <div className="district-header">
                  <div>
                    <h3>{district.district_name || `District ${district.district}`}</h3>
                    <span className="state-badge">{district.state_name || `State ${district.state}`}</span>
                  </div>
                  <div className="district-risk-badge" style={{
                    background: RISK_BG[riskLevel],
                    color: RISK_COLORS[riskLevel],
                    border: `1px solid ${RISK_COLORS[riskLevel]}`,
                  }}>
                    {riskLevel === 'High' ? <AlertTriangle size={11} /> : <CheckCircle size={11} />}
                    {riskLevel}
                  </div>
                </div>

                <div className="district-stats">
                  {[
                    { label: 'Stunting',    value: district.actual_stunting,    cls: 'stunting',    avg: nationalAvg?.stunting },
                    { label: 'Wasting',     value: district.actual_wasting,     cls: 'wasting',     avg: nationalAvg?.wasting },
                    { label: 'Underweight', value: district.actual_underweight, cls: 'underweight', avg: nationalAvg?.underweight },
                  ].map(({ label, value, cls, avg }) => (
                    <div key={label} className="stat-item">
                      <span className="stat-label">{label}</span>
                      <div className="stat-bar">
                        <div className={`stat-fill ${cls}-fill`} style={{ width: `${value}%` }} />
                        {avg && (
                          <div className="stat-avg-marker" style={{ left: `${avg}%` }} title={`National avg: ${avg.toFixed(1)}%`} />
                        )}
                      </div>
                      <span className={`stat-value ${value > (avg || 0) ? 'above-avg' : 'below-avg'}`}>
                        {value.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>

                <div className="district-footer">
                  <span className="sample-size">Sample: {district.sample_size} children</span>
                  <span className="composite-score" style={{ color: RISK_COLORS[riskLevel] }}>
                    Risk Score: {score.toFixed(1)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default DistrictExplorer;