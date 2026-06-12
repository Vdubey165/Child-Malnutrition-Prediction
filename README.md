# 🩺 Child Malnutrition Prediction

A full-stack ML web application for predicting child malnutrition rates across Indian districts using NFHS-5 data. Supports both individual-level prediction (stunting, wasting, underweight) and district-level analytics across all 707 districts.

**Live Demo:** [child-malnutrition-prediction.vercel.app](https://child-malnutrition-prediction.vercel.app)  
**API:** [child-malnutrition-prediction-api.onrender.com](https://child-malnutrition-prediction-api.onrender.com)

---

## 📸 Screenshots

**Landing Page**
![Landing Page](screenshots/landing.png)
> 232,920 children analyzed · 707 districts covered · 69% prediction accuracy (R² underweight)

**Dashboard — National Overview**
![Dashboard](screenshots/dashboard.png)
> National averages: Stunting 29.7% · Wasting 15.9% · Underweight 26.6% · District risk distribution across 707 districts

**District Explorer**
![Districts](screenshots/districts.png)
> Browse and filter all 707 districts by composite risk score. High-risk districts (119) flagged with stunting, wasting, and underweight rates.

**Malnutrition Risk Estimator**
![Prediction](screenshots/prediction.png)
> Input a district socioeconomic profile to get predicted stunting, wasting, and underweight rates with risk classification vs national average.

**Feature Importance Analysis**
![Feature Importance](screenshots/feature-importance.png)
> Mother's BMI (31%) is the dominant predictor for stunting, followed by Wealth Index (10.9%) and Mother's Education (8.7%).

**About — Model Evaluation**
![About](screenshots/about.png)
> R² and RMSE comparison across all 3 algorithms per malnutrition target.

---

## 🧠 ML Models

Three models were trained per malnutrition target — Linear Regression (baseline), Random Forest, and XGBoost — with the best selected per outcome by test R²:

| Target | Best Model | Notes |
|---|---|---|
| Stunting | Random Forest | `random_forest_stunting.pkl` |
| Wasting | Random Forest | `random_forest_wasting.pkl` |
| Underweight | XGBoost | `xgboost_underweight.pkl` |

**16 input features** (district-level averages from NFHS-5): wealth index, mother's education level & years, mother's age & BMI, mother employment status, female-headed household, child age/sex, birth interval, birth weight, breastfeeding duration, BCG/DPT/Measles vaccination status.

**WHO z-score thresholds used:** height-for-age, weight-for-height, and weight-for-age z-scores < −2 SD define stunting, wasting, and underweight respectively.

**Risk thresholds:**
- Stunting / Underweight: Low < 20% · Medium < 35% · High ≥ 35%
- Wasting: Low < 10% · Medium < 20% · High ≥ 20%

---

## 📓 Notebooks

The `Notebook/` directory contains the full ML pipeline:

| Notebook | Purpose |
|---|---|
| `Data_exploration.ipynb` | Loads NFHS-5 Stata files, cleans DHS missing codes, computes malnutrition flags using WHO z-score thresholds (z < -2), aggregates to district level, exports processed CSVs and district name mappings |
| `02_feature_engineering_and_modeling.ipynb` | Trains Linear Regression, Random Forest, and XGBoost for all 3 targets; runs hyperparameter tuning; saves `.pkl` models and district/state prediction CSVs |
| `Feature-Engineering.ipynb` | Placeholder — feature engineering is handled inside the two notebooks above |

> **Note:** `Data/Raw/*.DTA` (NFHS-5 Stata files) are not included in the repo. To re-run `Data_exploration.ipynb`, place `Children.DTA`, `Household.DTA`, and `Individuals.DTA` in `Data/Raw/`. The processed outputs are already included so the app runs without them.

---

## 🗂 Project Structure

```
child-malnutrition/
├── backend/
│   ├── main.py                  # FastAPI app & routes
│   ├── district_mapping.py      # District name enrichment (707 districts)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.js
│   └── package.json
├── Notebook/
│   ├── Data_exploration.ipynb
│   ├── 02_feature_engineering_and_modeling.ipynb
│   └── Feature-Engineering.ipynb   # placeholder
├── Models/
│   ├── random_forest_stunting.pkl
│   ├── random_forest_wasting.pkl
│   └── xgboost_underweight.pkl
├── Data/
│   ├── Raw/                     # NFHS-5 Stata files (not in repo)
│   └── Processed/
│       ├── district_malnutrition_enhanced.csv
│       ├── district_predictions_all_types.csv
│       ├── district_name_mapping.csv
│       └── state_level_summary.csv
└── .github/
    └── workflows/
        └── keep-alive.yml       # Prevents Render cold starts
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+

### 1. Clone the repo

```bash
git clone https://github.com/Vdubey165/Child-Malnutrition-Prediction.git
cd Child-Malnutrition-Prediction
```

### 2. Backend setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

API will be running at `http://localhost:8000`

### 3. Frontend setup

```bash
cd frontend
npm install
```

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000
```

```bash
npm run dev
```

Frontend will be running at `http://localhost:5173`

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check — models & data status |
| `POST` | `/api/predict` | Predict stunting, wasting, underweight |
| `GET` | `/api/districts` | All district data (paginated via `?limit=`) |
| `GET` | `/api/districts/{id}` | Single district by ID |
| `GET` | `/api/statistics` | National averages across all districts |

### Sample prediction request

```json
POST /api/predict
{
  "wealth_index": 3,
  "mother_edu_level": 2,
  "mother_age": 25,
  "mother_edu_years": 10,
  "mother_bmi": 2200,
  "mother_works": 0,
  "female_headed_hh": 1,
  "child_age_months": 24,
  "child_sex": 1,
  "birth_interval": 2,
  "birth_weight": 2800,
  "breastfeed_duration": 12,
  "currently_breastfeed": 5000,
  "bcg_vaccination": 1,
  "dpt_vaccination": 1,
  "measles_vaccination": 1
}
```

### Sample response

```json
{
  "stunting": 28.45,
  "wasting": 11.20,
  "underweight": 22.80,
  "risk_level": {
    "stunting": "Medium",
    "wasting": "Medium",
    "underweight": "Medium"
  }
}
```

---

## 🌐 Deployment

| Layer | Platform |
|---|---|
| Frontend | Vercel |
| Backend | Render (Free tier) |

A GitHub Actions workflow (`.github/workflows/keep-alive.yml`) pings the Render backend periodically to prevent cold starts on the free tier.

---

## 📊 Data

Built on **NFHS-5 (National Family Health Survey 5)** district-level data covering **707 Indian districts**. Processed CSV is included in `Data/Processed/`.

---

## 🛠 Tech Stack

- **Frontend:** React, Vite
- **Backend:** FastAPI, Python
- **ML:** scikit-learn (Random Forest), XGBoost
- **Data:** Pandas, NumPy
- **Deployment:** Vercel + Render

---

## 👤 Author

**Vaibhav Dubey** — [github.com/Vdubey165](https://github.com/Vdubey165)