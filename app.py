"""
Child Malnutrition Prediction Dashboard
Predicting District-Level Malnutrition Using NFHS-5 Data

Author: [Your Name]
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Child Malnutrition Prediction",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498DB;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_csv('Data/Processed/district_malnutrition_enhanced.csv')
        predictions = pd.read_csv('Data/Processed/district_predictions_all_types.csv')
        state_summary = pd.read_csv('Data/Processed/state_level_summary.csv')
        return df, predictions, state_summary
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        with open('Models/random_forest_stunting.pkl', 'rb') as f:
            rf_stunt = pickle.load(f)
        with open('Models/random_forest_wasting.pkl', 'rb') as f:
            rf_wast = pickle.load(f)
        with open('Models/xgboost_underweight.pkl', 'rb') as f:
            xgb_under = pickle.load(f)
        return rf_stunt, rf_wast, xgb_under
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load data
df, predictions, state_summary = load_data()
rf_stunt, rf_wast, xgb_under = load_models()

# Main header
st.markdown('<p class="main-header">🍎 Child Malnutrition Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown("### Predicting District-Level Malnutrition Using Machine Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/3498DB/FFFFFF?text=NFHS-5+Data", use_container_width=True)
    st.markdown("## 📊 Navigation")
    page = st.radio(
        "Select Page:",
        ["🏠 Overview", "📈 Model Performance", "🗺️ District Predictions", 
         "🎯 State Analysis", "🔮 Make Prediction"]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Project Info")
    st.info("""
    **Data Source:** NFHS-5 (2019-21)  
    **Sample:** 232,920 children  
    **Districts:** 707  
    **Models:** RF + XGBoost
    """)

# Page 1: Overview
if page == "🏠 Overview":
    st.markdown("## 🏠 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Children",
            value="232,920",
            delta="Across India"
        )
    
    with col2:
        st.metric(
            label="🗺️ Districts Covered",
            value="707",
            delta="All states"
        )
    
    with col3:
        st.metric(
            label="🎯 ML Models",
            value="9",
            delta="3 algorithms × 3 types"
        )
    
    with col4:
        st.metric(
            label="📈 Best R² Score",
            value="69.1%",
            delta="Underweight prediction"
        )
    
    st.markdown("---")
    
    # National statistics
    st.markdown("## 📊 National Malnutrition Statistics")
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        stunting_avg = df['stunting_rate'].mean()
        wasting_avg = df['wasting_rate'].mean()
        underweight_avg = df['underweight_rate'].mean()
        
        with col1:
            st.markdown("### 🔴 Stunting")
            st.markdown(f"<h1 style='text-align: center; color: #E74C3C;'>{stunting_avg:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Height-for-age deficit</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🟠 Wasting")
            st.markdown(f"<h1 style='text-align: center; color: #F39C12;'>{wasting_avg:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Weight-for-height deficit</p>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("### 🟡 Underweight")
            st.markdown(f"<h1 style='text-align: center; color: #F1C40F;'>{underweight_avg:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Weight-for-age deficit</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key findings
    st.markdown("## 💡 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Model Performance:**
        - Underweight most predictable (69% R²)
        - Stunting moderately predictable (50% R²)
        - Wasting most challenging (43% R²)
        """)
        
        st.info("""
        **📊 Top Predictors:**
        1. Mother's BMI (31% importance)
        2. Wealth Index (11%)
        3. Mother's Education (9%)
        4. Birth Weight (9%)
        """)
    
    with col2:
        st.warning("""
        **⚠️ High-Risk States:**
        - State 17: Highest stunting (40%)
        - State 24: High wasting (24%)
        - State 24: High underweight (38%)
        """)
        
        st.success("""
        **🎯 Policy Implications:**
        - Focus on maternal health & nutrition
        - Target low-wealth districts
        - Improve maternal education access
        """)

# Page 2: Model Performance
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Analysis")
    
    # Model comparison
    model_results = pd.DataFrame({
        'Model': ['Linear Reg', 'Random Forest', 'XGBoost'] * 3,
        'Malnutrition Type': ['Stunting']*3 + ['Wasting']*3 + ['Underweight']*3,
        'R² Score': [0.436, 0.497, 0.431, 0.360, 0.427, 0.364, 0.643, 0.677, 0.691],
        'RMSE': [5.73, 5.42, 5.76, 4.64, 4.39, 4.63, 5.55, 5.28, 5.16]
    })
    
    # R² Score comparison
    fig1 = px.bar(
        model_results,
        x='Model',
        y='R² Score',
        color='Malnutrition Type',
        barmode='group',
        title='Model Performance Comparison (R² Score)',
        color_discrete_map={'Stunting': '#E74C3C', 'Wasting': '#F39C12', 'Underweight': '#F1C40F'}
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # RMSE comparison
    fig2 = px.bar(
        model_results,
        x='Model',
        y='RMSE',
        color='Malnutrition Type',
        barmode='group',
        title='Model Error Comparison (RMSE - Lower is Better)',
        color_discrete_map={'Stunting': '#E74C3C', 'Wasting': '#F39C12', 'Underweight': '#F1C40F'}
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Best model summary
    st.markdown("---")
    st.markdown("## 🏆 Best Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Stunting")
        st.success("**Random Forest**")
        st.metric("R² Score", "49.7%")
        st.metric("RMSE", "5.42%")
    
    with col2:
        st.markdown("### Wasting")
        st.success("**Random Forest**")
        st.metric("R² Score", "42.7%")
        st.metric("RMSE", "4.39%")
    
    with col3:
        st.markdown("### Underweight")
        st.success("**XGBoost**")
        st.metric("R² Score", "69.1%")
        st.metric("RMSE", "5.16%")

# Page 3: District Predictions
elif page == "🗺️ District Predictions":
    st.markdown("## 🗺️ District-Level Predictions")
    
    if predictions is not None:
        # District selector
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_district = st.selectbox(
                "Select District:",
                options=sorted(predictions['district'].unique()),
                format_func=lambda x: f"District {int(x)}"
            )
        
        # Get district data
        district_data = predictions[predictions['district'] == selected_district].iloc[0]
        
        st.markdown(f"### District {int(selected_district)} - Detailed Analysis")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("State", int(district_data['state']))
        with col2:
            st.metric("Sample Size", int(district_data['sample_size']))
        with col3:
            st.metric("Children Surveyed", int(district_data['sample_size']))
        with col4:
            rank_stunt = (predictions['actual_stunting'] > district_data['actual_stunting']).sum() + 1
            st.metric("Stunting Rank", f"{rank_stunt}/707")
        
        st.markdown("---")
        
        # Predictions vs Actual
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔴 Stunting")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=district_data['actual_stunting'],
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': district_data['predicted_stunting']},
                gauge={'axis': {'range': [None, 60]},
                       'bar': {'color': "#E74C3C"},
                       'steps': [
                           {'range': [0, 20], 'color': "lightgreen"},
                           {'range': [20, 35], 'color': "yellow"},
                           {'range': [35, 60], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75,
                                   'value': district_data['predicted_stunting']}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Predicted: {district_data['predicted_stunting']:.1f}%")
        
        with col2:
            st.markdown("#### 🟠 Wasting")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=district_data['actual_wasting'],
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': district_data['predicted_wasting']},
                gauge={'axis': {'range': [None, 40]},
                       'bar': {'color': "#F39C12"},
                       'steps': [
                           {'range': [0, 10], 'color': "lightgreen"},
                           {'range': [10, 20], 'color': "yellow"},
                           {'range': [20, 40], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "orange", 'width': 4},
                                   'thickness': 0.75,
                                   'value': district_data['predicted_wasting']}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Predicted: {district_data['predicted_wasting']:.1f}%")
        
        with col3:
            st.markdown("#### 🟡 Underweight")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=district_data['actual_underweight'],
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': district_data['predicted_underweight']},
                gauge={'axis': {'range': [None, 60]},
                       'bar': {'color': "#F1C40F"},
                       'steps': [
                           {'range': [0, 20], 'color': "lightgreen"},
                           {'range': [20, 35], 'color': "yellow"},
                           {'range': [35, 60], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "gold", 'width': 4},
                                   'thickness': 0.75,
                                   'value': district_data['predicted_underweight']}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Predicted: {district_data['predicted_underweight']:.1f}%")
        
        st.markdown("---")
        
        # Top/Bottom districts comparison
        st.markdown("### 📊 How Does This District Compare?")
        
        tab1, tab2, tab3 = st.tabs(["Stunting", "Wasting", "Underweight"])
        
        with tab1:
            top10_stunt = predictions.nlargest(10, 'actual_stunting')[['district', 'actual_stunting', 'predicted_stunting']]
            fig = px.bar(top10_stunt, x='district', y=['actual_stunting', 'predicted_stunting'],
                        title='Top 10 Districts by Stunting Rate',
                        labels={'value': 'Rate (%)', 'district': 'District'},
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            top10_wast = predictions.nlargest(10, 'actual_wasting')[['district', 'actual_wasting', 'predicted_wasting']]
            fig = px.bar(top10_wast, x='district', y=['actual_wasting', 'predicted_wasting'],
                        title='Top 10 Districts by Wasting Rate',
                        labels={'value': 'Rate (%)', 'district': 'District'},
                        barmode='group', color_discrete_sequence=['#F39C12', '#F8C471'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            top10_under = predictions.nlargest(10, 'actual_underweight')[['district', 'actual_underweight', 'predicted_underweight']]
            fig = px.bar(top10_under, x='district', y=['actual_underweight', 'predicted_underweight'],
                        title='Top 10 Districts by Underweight Rate',
                        labels={'value': 'Rate (%)', 'district': 'District'},
                        barmode='group', color_discrete_sequence=['#F1C40F', '#F9E79F'])
            st.plotly_chart(fig, use_container_width=True)

# Page 4: State Analysis
elif page == "🎯 State Analysis":
    st.markdown("## 🎯 State-Level Analysis")
    
    if state_summary is not None:
        # State comparison
        malnutrition_type = st.radio(
            "Select Malnutrition Type:",
            ["Stunting", "Wasting", "Underweight"],
            horizontal=True
        )
        
        if malnutrition_type == "Stunting":
            actual_col, pred_col = 'actual_stunting', 'predicted_stunting'
            color = '#E74C3C'
        elif malnutrition_type == "Wasting":
            actual_col, pred_col = 'actual_wasting', 'predicted_wasting'
            color = '#F39C12'
        else:
            actual_col, pred_col = 'actual_underweight', 'predicted_underweight'
            color = '#F1C40F'
        
        # Top 15 states
        top_states = state_summary.nlargest(15, actual_col).copy()
        top_states['state'] = top_states.index.astype(str)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_states['state'],
            x=top_states[actual_col],
            name='Actual',
            orientation='h',
            marker_color=color
        ))
        fig.add_trace(go.Bar(
            y=top_states['state'],
            x=top_states[pred_col],
            name='Predicted',
            orientation='h',
            marker_color=color,
            opacity=0.6
        ))
        
        fig.update_layout(
            title=f'Top 15 States by {malnutrition_type} Rate',
            xaxis_title=f'{malnutrition_type} Rate (%)',
            yaxis_title='State',
            barmode='overlay',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # State summary table
        st.markdown("### 📋 Complete State Summary")
        
        display_df = state_summary.copy()
        display_df.index.name = 'State'
        display_df = display_df.round(2)
        display_df = display_df.sort_values(actual_col, ascending=False)
        
        st.dataframe(
            display_df[['num_districts', actual_col, pred_col, 'sample_size']],
            use_container_width=True,
            height=400
        )

# Page 5: Make Prediction
elif page == "🔮 Make Prediction":
    st.markdown("## 🔮 Make a Prediction")
    st.markdown("Enter district characteristics to predict malnutrition rates")
    
    if rf_stunt is not None and rf_wast is not None and xgb_under is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👩 Mother Characteristics")
            mother_bmi = st.slider("Mother's BMI", 1500, 3000, 2200, step=50,
                                   help="Mother's Body Mass Index (×100)")
            mother_edu_years = st.slider("Mother's Education (years)", 0, 15, 8)
            mother_edu_level = st.selectbox("Mother's Education Level", 
                                           [0, 1, 2, 3],
                                           format_func=lambda x: ["No education", "Primary", "Secondary", "Higher"][x])
            mother_age = st.slider("Mother's Age", 15, 49, 27)
            mother_works = st.selectbox("Mother Works", [0, 1], 
                                       format_func=lambda x: ["No", "Yes"][x])
        
        with col2:
            st.markdown("### 🏠 Household & Child Characteristics")
            wealth_index = st.slider("Wealth Index", 1, 5, 3,
                                    help="1=Poorest, 5=Richest")
            birth_weight = st.slider("Birth Weight (grams)", 400, 5000, 2800, step=100)
            child_age_months = st.slider("Child Age (months)", 0, 59, 30)
            child_sex = st.selectbox("Child Sex", [1, 2],
                                    format_func=lambda x: ["Male", "Female"][x-1])
            birth_interval = st.slider("Birth Interval (years)", 1.0, 3.0, 2.0, step=0.1)
            
            female_headed_hh = st.selectbox("Female-Headed Household", [1, 2],
                                           format_func=lambda x: ["No", "Yes"][x-1])
            breastfeed_duration = st.slider("Breastfeeding Duration (months)", 0, 90, 70)
            currently_breastfeed = st.slider("Currently Breastfeeding Score", 2500, 8000, 3500, step=100)
            
            bcg_vaccination = st.slider("BCG Vaccination Rate", 0.0, 2.0, 1.0, step=0.1)
            dpt_vaccination = st.slider("DPT Vaccination Rate", 0.0, 2.0, 1.0, step=0.1)
            measles_vaccination = st.slider("Measles Vaccination Rate", 0.0, 3.0, 1.5, step=0.1)
        
        if st.button("🎯 Predict Malnutrition Rates", type="primary"):
            # Prepare input features (in correct order)
            features = np.array([[
                wealth_index,
                mother_edu_level,
                mother_age,
                mother_edu_years,
                mother_bmi,
                mother_works,
                female_headed_hh,
                child_age_months,
                child_sex,
                birth_interval,
                birth_weight,
                breastfeed_duration,
                currently_breastfeed,
                bcg_vaccination,
                dpt_vaccination,
                measles_vaccination
            ]])
            
            # Make predictions
            pred_stunting = rf_stunt.predict(features)[0]
            pred_wasting = rf_wast.predict(features)[0]
            pred_underweight = xgb_under.predict(features)[0]
            
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🔴 Stunting")
                st.markdown(f"<h1 style='text-align: center; color: #E74C3C;'>{pred_stunting:.1f}%</h1>", 
                           unsafe_allow_html=True)
                if pred_stunting < 20:
                    st.success("✅ Low Risk")
                elif pred_stunting < 35:
                    st.warning("⚠️ Medium Risk")
                else:
                    st.error("🚨 High Risk")
            
            with col2:
                st.markdown("### 🟠 Wasting")
                st.markdown(f"<h1 style='text-align: center; color: #F39C12;'>{pred_wasting:.1f}%</h1>", 
                           unsafe_allow_html=True)
                if pred_wasting < 10:
                    st.success("✅ Low Risk")
                elif pred_wasting < 20:
                    st.warning("⚠️ Medium Risk")
                else:
                    st.error("🚨 High Risk")
            
            with col3:
                st.markdown("### 🟡 Underweight")
                st.markdown(f"<h1 style='text-align: center; color: #F1C40F;'>{pred_underweight:.1f}%</h1>", 
                           unsafe_allow_html=True)
                if pred_underweight < 20:
                    st.success("✅ Low Risk")
                elif pred_underweight < 35:
                    st.warning("⚠️ Medium Risk")
                else:
                    st.error("🚨 High Risk")
            
            st.markdown("---")
            st.info("💡 **Recommendations:** Focus on improving maternal health, wealth status, and access to healthcare services.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7F8C8D; padding: 2rem;'>
        <p>📊 Built with Streamlit | 🎓 ML Project 2024</p>
        <p>Data Source: NFHS-5 (2019-21) | Models: Random Forest & XGBoost</p>
    </div>
""", unsafe_allow_html=True)