import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, BarChart3, Map, Brain, Database, Github } from 'lucide-react';
import { TrendingDown, AlertCircle, Scale } from 'lucide-react';

import './Landing.css';

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="landing">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-overlay"></div>
        <div className="hero-content">
          <div className="hero-badge">
            <span className="badge-icon"></span>
            <span>Powered by Machine Learning</span>
          </div>
          
          <h1 className="hero-title">
            Child Malnutrition
            <span className="gradient-text"> Prediction System</span>
          </h1>
          
          <p className="hero-subtitle">
            Leveraging NFHS-5 data and advanced machine learning to predict 
            district-level child malnutrition across India
          </p>

          <div className="hero-stats">
            <div className="hero-stat">
              <div className="stat-number">232,920</div>
              <div className="stat-label">Children Analyzed</div>
            </div>
            <div className="hero-stat">
              <div className="stat-number">707</div>
              <div className="stat-label">Districts Covered</div>
            </div>
            <div className="hero-stat">
              <div className="stat-number">69%</div>
              <div className="stat-label">Prediction Accuracy</div>
            </div>
          </div>

          <button className="cta-button" onClick={() => navigate('/dashboard')}>
            Enter Dashboard
            <ArrowRight size={20} />
          </button>

          <p className="hero-footnote">
            Based on National Family Health Survey (NFHS-5) • 2019-21
          </p>
        </div>
      </section>

      {/* Problem Section */}
      <section className="problem-section">
        <div className="container">
          <div className="section-header">
            <span className="section-badge">The Challenge</span>
            <h2 className="section-title">Understanding India's Malnutrition Crisis</h2>
          </div>

          <div className="problem-grid">
            <div className="problem-content">
              <p className="problem-text">
                Child malnutrition remains one of India's most pressing public health challenges, 
                affecting millions of children and their future potential. Understanding and 
                predicting malnutrition patterns is crucial for effective policy intervention.
              </p>

              <div className="problem-stats">
            <div className="problem-stat">
                <div className="stat-icon stunting-bg">
                <TrendingDown size={24} style={{ color: 'var(--stunting)' }} />
                </div>
                <div>
                <div className="stat-value">35.5%</div>
                <div className="stat-desc">Children are stunted (height-for-age deficit)</div>
                </div>
            </div>

            <div className="problem-stat">
                <div className="stat-icon wasting-bg">
                <AlertCircle size={24} style={{ color: 'var(--wasting)' }} />
                </div>
                <div>
                <div className="stat-value">19.3%</div>
                <div className="stat-desc">Show signs of wasting (acute malnutrition)</div>
                </div>
            </div>

            <div className="problem-stat">
                <div className="stat-icon underweight-bg">
                <Scale size={24} style={{ color: 'var(--underweight)' }} />
                </div>
                <div>
                <div className="stat-value">32.1%</div>
                <div className="stat-desc">Are underweight for their age</div>
                </div>
            </div>
            </div>
            </div>

            <div className="problem-visual">
            <img 
                src="https://images.unsplash.com/photo-1488521787991-ed7bbaae773c?w=800&q=80"
                alt="Indian children - malnutrition challenge"
                style={{ width: '100%', borderRadius: '16px', objectFit: 'cover', height: '100%' }}
            />
            </div>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="solution-section">
        <div className="container">
          <div className="solution-grid">
            <div className="solution-visual">
              <img 
                src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&q=80" 
                alt="Data analytics and technology"
                style={{ width: '100%', borderRadius: '16px', objectFit: 'cover', height: '100%' }}
              />
            </div>

            <div className="solution-content">
              <span className="section-badge">Our Solution</span>
              <h2 className="section-title">Machine Learning for Public Health</h2>
              
              <p className="solution-text">
                This system uses advanced machine learning models trained on NFHS-5 data to 
                predict malnutrition rates at the district level, enabling targeted interventions 
                and resource allocation.
              </p>

              <div className="solution-features">
                <div className="feature-item">
                  <div className="feature-icon">✓</div>
                  <div>
                    <h4>Predictive Analytics</h4>
                    <p>Random Forest and XGBoost models achieving up to 69% accuracy</p>
                  </div>
                </div>

                <div className="feature-item">
                  <div className="feature-icon">✓</div>
                  <div>
                    <h4>District-Level Insights</h4>
                    <p>Granular predictions across all 707 districts in India</p>
                  </div>
                </div>

                <div className="feature-item">
                  <div className="feature-icon">✓</div>
                  <div>
                    <h4>Key Factor Analysis</h4>
                    <p>Identifies maternal health, wealth, and education as top predictors</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="container">
          <div className="section-header centered">
            <span className="section-badge">Features</span>
            <h2 className="section-title">What You Can Do</h2>
            <p className="section-subtitle">
              Explore comprehensive malnutrition predictions and analytics
            </p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-card-icon" style={{ background: 'var(--stunting-light)' }}>
                <BarChart3 size={32} style={{ color: 'var(--stunting)' }} />
              </div>
              <h3>Interactive Dashboard</h3>
              <p>
                Visualize national malnutrition statistics with real-time data 
                from 232,920 children across India
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-card-icon" style={{ background: 'var(--wasting-light)' }}>
                <Brain size={32} style={{ color: 'var(--wasting)' }} />
              </div>
              <h3>ML Predictions</h3>
              <p>
                Get instant malnutrition predictions based on maternal health, 
                socioeconomic, and healthcare factors
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-card-icon" style={{ background: 'var(--underweight-light)' }}>
                <Map size={32} style={{ color: 'var(--underweight)' }} />
              </div>
              <h3>District Explorer</h3>
              <p>
                Explore detailed malnutrition data for all 707 districts with 
                search and filtering capabilities
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-card-icon" style={{ background: 'var(--accent-blue-light)' }}>
                <Database size={32} style={{ color: 'var(--accent-blue)' }} />
              </div>
              <h3>NFHS-5 Data</h3>
              <p>
                Built on official government survey data ensuring accuracy 
                and reliability in predictions
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Tech Stack Section */}
      <section className="tech-section">
        <div className="container">
          <div className="section-header centered">
            <span className="section-badge">Technology</span>
            <h2 className="section-title">Built With Modern Tools</h2>
          </div>

          <div className="tech-grid">
            <div className="tech-category">
              <h4>Frontend</h4>
              <div className="tech-tags">
                <span className="tech-tag">React.js</span>
                <span className="tech-tag">Recharts</span>
                <span className="tech-tag">React Router</span>
              </div>
            </div>

            <div className="tech-category">
              <h4>Backend</h4>
              <div className="tech-tags">
                <span className="tech-tag">FastAPI</span>
                <span className="tech-tag">Python</span>
                <span className="tech-tag">REST API</span>
              </div>
            </div>

            <div className="tech-category">
              <h4>Machine Learning</h4>
              <div className="tech-tags">
                <span className="tech-tag">Scikit-learn</span>
                <span className="tech-tag">XGBoost</span>
                <span className="tech-tag">Random Forest</span>
              </div>
            </div>

            <div className="tech-category">
              <h4>Data</h4>
              <div className="tech-tags">
                <span className="tech-tag">Pandas</span>
                <span className="tech-tag">NumPy</span>
                <span className="tech-tag">NFHS-5</span>
              </div>
            </div>
          </div>

          <div className="github-section">
            <a 
              href="https://github.com/Vdubey165/Child-Malnutrition-Prediction" 
              target="_blank" 
              rel="noopener noreferrer"
              className="github-button"
            >
              <Github size={20} />
              View on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-content">
            <h2 className="cta-title">Ready to Explore?</h2>
            <p className="cta-text">
              Start analyzing district-level malnutrition data and make data-driven decisions
            </p>
            <button className="cta-button-large" onClick={() => navigate('/dashboard')}>
              Launch Dashboard
              <ArrowRight size={24} />
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="container">
          <p>© 2024 Child Malnutrition Predictor • Built by Vaibhav Dubey</p>
          <p className="footer-note">Data Source: NFHS-5 (2019-21) • For educational purposes</p>
        </div>
      </footer>
    </div>
  );
};

export default Landing;