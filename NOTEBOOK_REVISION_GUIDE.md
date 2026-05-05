# Notebook Revision Guide (Cell-by-Cell)

This document explains every notebook in [Notebook/](file:///e:/Child-Malnutrition/Notebook) cell-by-cell so you can quickly revise what the project does, what each cell outputs, and how the notebooks connect to the rest of the repository.

## Notebooks Covered

1. [Data_exploration.ipynb](file:///e:/Child-Malnutrition/Notebook/Data_exploration.ipynb) (21 cells): builds district-level datasets from NFHS Stata files and exports processed CSVs + mapping tables.
2. [02_feature_engineering_and_modeling.ipynb](file:///e:/Child-Malnutrition/Notebook/02_feature_engineering_and_modeling.ipynb) (52 cells): trains ML models to predict district malnutrition and saves trained models + CSV outputs + figures.
3. [Feature-Engineering.ipynb](file:///e:/Child-Malnutrition/Notebook/Feature-Engineering.ipynb) (1 cell): currently empty/placeholder.

## Big Picture Flow (How the Notebooks Connect)

- `Data_exploration.ipynb` reads NFHS Stata files from `Data/Raw/` (not present in this repo), computes child-level malnutrition flags, aggregates to district level, and saves:
  - [Data/Processed/district_malnutrition.csv](file:///e:/Child-Malnutrition/Data/Processed/district_malnutrition.csv)
  - [Data/Processed/district_malnutrition_enhanced.csv](file:///e:/Child-Malnutrition/Data/Processed/district_malnutrition_enhanced.csv)
  - [Data/Processed/district_name_mapping.csv](file:///e:/Child-Malnutrition/Data/Processed/district_name_mapping.csv)
  - [Data/Processed/complete_district_mapping.csv](file:///e:/Child-Malnutrition/Data/Processed/complete_district_mapping.csv)
- `02_feature_engineering_and_modeling.ipynb` consumes `district_malnutrition_enhanced.csv`, builds models (Linear Regression, Random Forest, XGBoost) for:
  - stunting_rate
  - wasting_rate
  - underweight_rate
  and saves trained pickles to [Models/](file:///e:/Child-Malnutrition/Models) plus output CSVs like:
  - [Data/Processed/district_predictions_all_types.csv](file:///e:/Child-Malnutrition/Data/Processed/district_predictions_all_types.csv)
  - [Data/Processed/state_level_summary.csv](file:///e:/Child-Malnutrition/Data/Processed/state_level_summary.csv)

## Important Runtime Notes (So You Don’t Get Stuck)

- The repository does not include `Data/Raw/*.DTA`. If you want to re-run `Data_exploration.ipynb`, you need to place the NFHS Stata files at:
  - `Data/Raw/Children.DTA`
  - `Data/Raw/Household.DTA`
  - `Data/Raw/Individuals.DTA`
- `02_feature_engineering_and_modeling.ipynb` expects `Data/Processed/district_malnutrition_enhanced.csv` to exist (it is included in this repo under [Data/Processed](file:///e:/Child-Malnutrition/Data/Processed)).
- A few cells contain minor inconsistencies/assumptions (noted in-place below) that you should remember during revision.

---

# Notebook 1 — Data_exploration.ipynb

Location: [Data_exploration.ipynb](file:///e:/Child-Malnutrition/Notebook/Data_exploration.ipynb)

## Purpose

- Load child-level NFHS data (Stata).
- Identify anthropometric variables for malnutrition.
- Clean DHS-style missing codes.
- Compute malnutrition prevalence (stunting/wasting/underweight).
- Aggregate to district-level rates for machine learning.
- Engineer a richer district-level feature set.
- Export processed CSV datasets and district name/state mappings for the rest of the project.

## Cell-by-Cell Explanation

### Cell 1 (code) — Imports + plotting/display settings

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

print("✅ Libraries imported successfully!")
```

What it does:
- Imports the core stack used throughout the notebook: pandas/numpy for processing and matplotlib/seaborn for plots.
- Sets notebook-friendly display defaults so you can inspect many columns/rows.
- Sets a consistent plotting style.

Why it matters:
- Everything later depends on these imports and style settings.

---

### Cell 2 (code) — Define input file paths

```python
# Define file paths
children_path = '../Data/Raw/Children.DTA'
household_path = '../Data/Raw/Household.DTA'
individual_path = '../Data/Raw/Individuals.DTA'

print("Loading datasets... This may take a few minutes.\n")
```

What it does:
- Defines where the raw NFHS Stata files are expected to be found.

Key dependency:
- `Data/Raw/` is required to run this notebook end-to-end.

---

### Cell 3 (code) — Load the children dataset

```python
# Load Children's Recode (main dataset)
print("📂 Loading Children's Recode...")
children_df = pd.read_stata(children_path, convert_categoricals=False)
print(f"   Shape: {children_df.shape}")
print(f"   Columns: {children_df.shape[1]}")
print(f"   Rows: {children_df.shape[0]:,}\n")
```

What it does:
- Reads the Stata file into a pandas DataFrame.
- Uses `convert_categoricals=False` so categorical labels don’t become pandas categoricals automatically (keeps raw numeric codes).

What to look for when revising:
- If this fails, your `Children.DTA` path is wrong or the file isn’t present.
- The printed shape gives you a quick sanity-check that you loaded the expected dataset.

---

### Cell 4 (code) — Search the dataset for relevant variables

```python
# Search for malnutrition indicators
print("🔍 Searching for key malnutrition variables...\n")

# Anthropometric indicators
anthro_vars = [col for col in children_df.columns if any(x in col.lower()       
               for x in ['hw70', 'hw71', 'hw72', 'height', 'weight', 'age'])]   

print("📏 Anthropometric variables (height, weight, age):")
for var in anthro_vars[:15]:
    print(f"  - {var}")

# Location variables
print("\n📍 Location variables:")
location_vars = [col for col in children_df.columns if any(x in col.lower()     
                 for x in ['dist', 'state', 'v024', 'shdist'])]
for var in location_vars:
    print(f"  - {var}")

# Socioeconomic variables
print("\n💰 Socioeconomic variables:")
socio_vars = [col for col in children_df.columns if any(x in col.lower()        
              for x in ['wealth', 'educ', 'v106', 'v190'])]
for var in socio_vars[:10]:
    print(f"  - {var}")
```

What it does:
- Scans column names looking for anthropometric variables and likely identifiers:
  - `hw70` (height-for-age z-score × 100)
  - `hw71` (weight-for-age z-score × 100)
  - `hw72` (weight-for-height z-score × 100)
  - `sdist` / district code (used later for aggregation)
  - `v024` / state code
  - `v106` / mother’s education level
  - `v190` / wealth index

Why it matters:
- This is the first “schema discovery” step: you confirm the dataset contains the variables you plan to use.

---

### Cell 5 (code) — Inspect a core set of variables

```python
# Let's look at these key variables
key_vars = ['hw70', 'hw71', 'hw72', 'sdist', 'v024', 'v106', 'v190']

print("📊 Sample data with key variables:\n")
print(children_df[key_vars].head(10))

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)

# Check data types
print("\nData types:")
print(children_df[key_vars].dtypes)

print("\nBasic statistics:")
print(children_df[key_vars].describe())
```

What it does:
- Prints sample rows and basic summary stats for the variables that drive the whole project.

Revision tips:
- If `hw70/hw71/hw72` show huge values (e.g., 9999), that’s your hint you must clean DHS missing codes (next cell).

---

### Cell 6 (code) — Clean DHS missing value codes (9996–9999)

```python
# Replace DHS missing value codes with NaN
print("🧹 Cleaning missing value codes...\n")

# DHS uses 9996-9999 for missing values
missing_codes = [9996, 9997, 9998, 9999]

# Count missing before cleaning
print("Missing values BEFORE cleaning:")
print(f"hw70: {children_df['hw70'].isnull().sum()}")
print(f"hw71: {children_df['hw71'].isnull().sum()}")
print(f"hw72: {children_df['hw72'].isnull().sum()}")

# Replace missing codes
for col in ['hw70', 'hw71', 'hw72']:
    children_df[col] = children_df[col].replace(missing_codes, np.nan)

print("\nMissing values AFTER cleaning:")
print(f"hw70: {children_df['hw70'].isnull().sum()}")
print(f"hw71: {children_df['hw71'].isnull().sum()}")
print(f"hw72: {children_df['hw72'].isnull().sum()}")

print("\n✅ Missing value codes cleaned!")
```

What it does:
- Converts DHS-specific missing codes into real `NaN` values so later calculations don’t treat “9999” as a real measurement.

Why it matters:
- Without this, prevalence rates will be wrong and plots will be distorted.

---

### Cell 7 (code) — Compute overall prevalence (WHO threshold: z < -2)

```python
# Calculate malnutrition indicators
# WHO definition: Z-score < -200 (which is -2 SD)

print("📊 MALNUTRITION PREVALENCE (Overall)\n")
print("="*50)

# Stunting (hw70 < -200)
valid_hw70 = children_df['hw70'].notna()
stunted = (children_df['hw70'] < -200) & valid_hw70
stunting_prev = (stunted.sum() / valid_hw70.sum()) * 100

print(f"🔴 STUNTING (Height-for-age < -2 SD)")
print(f"   Prevalence: {stunting_prev:.2f}%")
print(f"   Children stunted: {stunted.sum():,}")
print(f"   Total measured: {valid_hw70.sum():,}\n")

# Wasting (hw72 < -200)
valid_hw72 = children_df['hw72'].notna()
wasted = (children_df['hw72'] < -200) & valid_hw72
wasting_prev = (wasted.sum() / valid_hw72.sum()) * 100

print(f"🟠 WASTING (Weight-for-height < -2 SD)")
print(f"   Prevalence: {wasting_prev:.2f}%")
print(f"   Children wasted: {wasted.sum():,}")
print(f"   Total measured: {valid_hw72.sum():,}\n")

# Underweight (hw71 < -200)
valid_hw71 = children_df['hw71'].notna()
underweight = (children_df['hw71'] < -200) & valid_hw71
underweight_prev = (underweight.sum() / valid_hw71.sum()) * 100

print(f"🟡 UNDERWEIGHT (Weight-for-age < -2 SD)")
print(f"   Prevalence: {underweight_prev:.2f}%")
print(f"   Children underweight: {underweight.sum():,}")
print(f"   Total measured: {valid_hw71.sum():,}\n")

print("="*50)
```

What it does:
- Uses WHO standard: malnutrition indicator is `True` if z-score < -2 (stored as `< -200` because DHS stores z-scores × 100).
- Carefully restricts denominators to children with valid measurements (`valid_hw70.sum()` etc.).

Why it matters:
- This gives you baseline prevalence numbers for the entire sample and validates that the data is plausible.

---

### Cell 8 (code) — Plot distributions of z-scores

```python
# Visualize the distribution of malnutrition indicators
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Stunting
axes[0].hist(children_df['hw70'].dropna()/100, bins=50, edgecolor='black', alpha=0.7, color='red')
axes[0].axvline(-2, color='darkred', linestyle='--', linewidth=2, label='Stunting threshold (-2 SD)')
axes[0].set_xlabel('Height-for-age Z-score', fontsize=11)
axes[0].set_ylabel('Number of children', fontsize=11)
axes[0].set_title('STUNTING Distribution\n35.47% below -2 SD', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Wasting
axes[1].hist(children_df['hw72'].dropna()/100, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].axvline(-2, color='darkorange', linestyle='--', linewidth=2, label='Wasting threshold (-2 SD)')
axes[1].set_xlabel('Weight-for-height Z-score', fontsize=11)
axes[1].set_ylabel('Number of children', fontsize=11)
axes[1].set_title('WASTING Distribution\n18.62% below -2 SD', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Underweight
axes[2].hist(children_df['hw71'].dropna()/100, bins=50, edgecolor='black', alpha=0.7, color='gold')
axes[2].axvline(-2, color='darkgoldenrod', linestyle='--', linewidth=2, label='Underweight threshold (-2 SD)')
axes[2].set_xlabel('Weight-for-age Z-score', fontsize=11)
axes[2].set_ylabel('Number of children', fontsize=11)
axes[2].set_title('UNDERWEIGHT Distribution\n30.90% below -2 SD', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Visualization complete!")
```

What it does:
- Plots histograms of the three z-score distributions (converted back to standard z units by dividing by 100).
- Draws the -2 threshold line for each.

Revision note:
- The titles hardcode prevalence values (e.g., “35.47%”) and may not match your run if the underlying data differs.

---

### Cell 9 (code) — District-level sample size sanity-check

```python
# Analyze district-level distribution
print("📍 DISTRICT-LEVEL ANALYSIS\n")
print("="*60)

# Count unique districts
n_districts = children_df['sdist'].nunique()
print(f"Total unique districts: {n_districts}")

# Check sample size per district
district_counts = children_df['sdist'].value_counts().sort_index()

print(f"\nSample size per district:")
print(f"  Minimum: {district_counts.min()} children")
print(f"  Maximum: {district_counts.max()} children")
print(f"  Mean: {district_counts.mean():.0f} children")
print(f"  Median: {district_counts.median():.0f} children")

print(f"\n📊 Top 10 districts by sample size:")
top_districts = district_counts.nlargest(10)
for dist, count in top_districts.items():
    print(f"  District {dist}: {count} children")

print(f"\n📊 Districts with smallest samples:")
small_districts = district_counts.nsmallest(10)
for dist, count in small_districts.items():
    print(f"  District {dist}: {count} children")

print("\n" + "="*60)
```

What it does:
- Counts how many districts exist (`sdist` unique).
- Checks distribution of sample sizes by district (important because district-level prevalence estimates become noisy with tiny sample sizes).

Why it matters for modeling:
- `sample_size` later becomes both a metadata field and a signal of confidence.

---

### Cell 10 (code) — Create the basic district-level ML dataset

```python
# Aggregate malnutrition data at district level
print("🎯 Creating District-Level Dataset for ML...\n")

# Create malnutrition flags
children_df['is_stunted'] = (children_df['hw70'] < -200).astype(float)
children_df['is_wasted'] = (children_df['hw72'] < -200).astype(float)
children_df['is_underweight'] = (children_df['hw71'] < -200).astype(float)      

# Aggregate by district
district_malnutrition = children_df.groupby('sdist').agg({
    # Target variables (malnutrition prevalence %)
    'is_stunted': 'mean',
    'is_wasted': 'mean',
    'is_underweight': 'mean',

    # Sample size
    'hw70': 'count',

    # Predictors - socioeconomic
    'v190': 'mean',  # Wealth index
    'v106': 'mean',  # Mother's education
    'v024': 'first'  # State
}).reset_index()

# Rename columns
district_malnutrition.columns = [
    'district',
    'stunting_rate',
    'wasting_rate',
    'underweight_rate',
    'sample_size',
    'avg_wealth_index',
    'avg_mother_education',
    'state'
]

# Convert rates to percentages
district_malnutrition['stunting_rate'] *= 100
district_malnutrition['wasting_rate'] *= 100
district_malnutrition['underweight_rate'] *= 100

print(f"✅ District dataset created!")
print(f"   Shape: {district_malnutrition.shape}")
print(f"\n📊 First 10 districts:\n")
print(district_malnutrition.head(10))

print(f"\n📈 Summary statistics:\n")
print(district_malnutrition[['stunting_rate', 'wasting_rate', 'underweight_rate']].describe())
```

What it does:
- Creates binary flags for each malnutrition type.
- Aggregates by district:
  - target rates = mean(flag) × 100
  - sample size = count of non-null `hw70`
  - basic predictors = wealth, education
  - state code

Important revision note (data correctness):
- For missing anthropometric measurements, `(NaN < -200)` becomes `False` → `0.0` in the flags. That means missing children are implicitly counted as “not malnourished” in district means.
- If you ever re-run or improve this notebook, the statistically cleaner approach is: set the flag to `NaN` when the measurement is missing, so district means ignore missing records.

---

### Cell 11 (code) — Plot district-level rate distributions

```python
# Visualize district-level malnutrition rates
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Stunting by district
axes[0].hist(district_malnutrition['stunting_rate'], bins=30, edgecolor='black', alpha=0.7, color='red')
axes[0].axvline(district_malnutrition['stunting_rate'].mean(), color='darkred', linestyle='--', linewidth=2, label=f"Mean: {district_malnutrition['stunting_rate'].mean():.1f}%")
axes[0].set_xlabel('Stunting Rate (%)', fontsize=11)
axes[0].set_ylabel('Number of Districts', fontsize=11)
axes[0].set_title('District-Level STUNTING Rates', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Wasting by district
axes[1].hist(district_malnutrition['wasting_rate'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[1].axvline(district_malnutrition['wasting_rate'].mean(), color='darkorange', linestyle='--', linewidth=2, label=f"Mean: {district_malnutrition['wasting_rate'].mean():.1f}%")
axes[1].set_xlabel('Wasting Rate (%)', fontsize=11)
axes[1].set_ylabel('Number of Districts', fontsize=11)
axes[1].set_title('District-Level WASTING Rates', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Underweight by district
axes[2].hist(district_malnutrition['underweight_rate'], bins=30, edgecolor='black', alpha=0.7, color='gold')
axes[2].axvline(district_malnutrition['underweight_rate'].mean(), color='darkgoldenrod', linestyle='--', linewidth=2, label=f"Mean: {district_malnutrition['underweight_rate'].mean():.1f}%")
axes[2].set_xlabel('Underweight Rate (%)', fontsize=11)
axes[2].set_ylabel('Number of Districts', fontsize=11)
axes[2].set_title('District-Level UNDERWEIGHT Rates', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ District-level variation visualized!")
print(f"\n📊 Range of malnutrition across districts:")
print(f"   Stunting: {district_malnutrition['stunting_rate'].min():.1f}% to {district_malnutrition['stunting_rate'].max():.1f}%")
print(f"   Wasting: {district_malnutrition['wasting_rate'].min():.1f}% to {district_malnutrition['wasting_rate'].max():.1f}%")
print(f"   Underweight: {district_malnutrition['underweight_rate'].min():.1f}% to {district_malnutrition['underweight_rate'].max():.1f}%")
```

What it does:
- Confirms that districts vary meaningfully (good: there’s something to predict).
- Shows ranges and central tendency.

---

### Cell 12 (code) — Save `district_malnutrition.csv`

```python
# Save the district-level dataset
import os

# Create processed data folder if it doesn't exist
processed_folder = '../Data/Processed'
os.makedirs(processed_folder, exist_ok=True)

# Save to CSV
output_path = os.path.join(processed_folder, 'district_malnutrition.csv')       
district_malnutrition.to_csv(output_path, index=False)

print(f"✅ District dataset saved to: {output_path}")
print(f"\n📊 Dataset summary:")
print(f"   Rows (districts): {len(district_malnutrition)}")
print(f"   Columns: {len(district_malnutrition.columns)}")
print(f"\nColumns saved:")
for col in district_malnutrition.columns:
    print(f"   - {col}")
```

What it does:
- Writes the baseline district dataset used for early experimentation and sanity checks.

Output file:
- [district_malnutrition.csv](file:///e:/Child-Malnutrition/Data/Processed/district_malnutrition.csv)

---

### Cell 13 (code) — Broad feature discovery for richer ML predictors

```python
# Comprehensive feature extraction
print("🔍 EXTRACTING ADDITIONAL FEATURES FOR ML MODEL\n")
print("="*70)

# Let's explore what variables we have
print("Total columns in children's dataset:", len(children_df.columns))

# Search for different categories of variables
categories = {
    'Water & Sanitation': ['water', 'toilet', 'hv201', 'hv205'],
    'Household Assets': ['tv', 'radio', 'bike', 'car', 'phone', 'fridge', 'hv206', 'hv207', 'hv208', 'hv209', 'hv243'],
    'Cooking & Energy': ['cooking', 'fuel', 'electric', 'hv226'],
    'Mother Health': ['bmi', 'anemia', 'v445', 'v457'],
    'Antenatal Care': ['anc', 'v459', 'v461', 'prenatal'],
    'Child Health': ['vaccin', 'immun', 'h2', 'h3', 'breastfeed', 'm4'],        
    'Demographics': ['age', 'birth', 'v008', 'b8', 'v013'],
    'Housing': ['floor', 'wall', 'roof', 'rooms', 'hv213', 'hv214', 'hv215']    
}

found_features = {}

for category, patterns in categories.items():
    found = [col for col in children_df.columns if any(p in col.lower() for p in patterns)]
    if found:
        found_features[category] = found
        print(f"\n📌 {category}: {len(found)} variables found")
        for var in found[:10]:  # Show first 10
            print(f"   - {var}")
        if len(found) > 10:
            print(f"   ... and {len(found) - 10} more")

print("\n" + "="*70)
print(f"\n✅ Total feature categories found: {len(found_features)}")
print(f"✅ Ready to engineer features for district-level aggregation!")
```

What it does:
- Scans the children dataset for columns that likely represent household infrastructure, assets, maternal health, child health, etc.
- This step is exploratory and prepares you to pick predictors that plausibly influence malnutrition outcomes.

---

### Cell 14 (code) — Targeted search for specific “high value” predictors

```python
# More targeted search for specific important variables
print("🔍 TARGETED FEATURE SEARCH\n")
print("="*70)

# Key variables we want to find
target_vars = {
    # Household characteristics
    'hv206': 'Has electricity',
    'hv201': 'Source of drinking water',
    'hv205': 'Type of toilet',
    'hv207': 'Has radio',
    'hv208': 'Has television',
    'hv209': 'Has refrigerator',
    'hv210': 'Has bicycle',
    'hv211': 'Has motorcycle',
    'hv212': 'Has car',
    'hv221': 'Has telephone',
    'hv243a': 'Has mobile phone',
    'hv226': 'Type of cooking fuel',
    'hv213': 'Has floor material',
    'hv025': 'Urban/Rural',

    # Mother characteristics
    'v012': 'Mother age',
    'v133': 'Mother education years',
    'v151': 'Sex of household head',
    'v190': 'Wealth index',
    'v445': 'Mother BMI',
    'v714': 'Mother works',

    # Child characteristics
    'b4': 'Child sex',
    'hw1': 'Child age in months',
    'b8': 'Birth interval',
    'm4': 'Duration of breastfeeding',
    'm19': 'Breastfeeding status',

    # Healthcare access
    'v394': 'Knowledge of ORS',
    'v437': 'Weight at birth',
    'h2': 'BCG vaccination',
    'h3': 'DPT vaccination',
    'h9': 'Measles vaccination'
}

# Check which variables exist
print("Checking for key variables...\n")

available = {}
missing = {}

for var, description in target_vars.items():
    if var in children_df.columns:
        available[var] = description
        print(f"✅ {var:10} - {description}")
    else:
        missing[var] = description

if missing:
    print(f"\n❌ Missing variables ({len(missing)}):")
    for var, desc in list(missing.items())[:10]:
        print(f"   {var:10} - {desc}")

print("\n" + "="*70)
print(f"\n📊 Summary:")
print(f"   Available: {len(available)} variables")
print(f"   Missing: {len(missing)} variables")
print(f"\n✅ We have {len(available)} high-quality features to use!")
```

What it does:
- Checks whether specific NFHS variables exist in this Stata file.

Why it matters:
- Helps you pick predictors that are actually available, avoiding KeyErrors later.

---

### Cell 15 (code) — Explore alternative naming conventions (hv/sh/s variables)

```python
# Search for household variables with different naming patterns
print("🔍 Searching for household assets with alternate names...\n")

# Check all columns that start with 'hv' or 'sh' (state-specific household vars)
hv_vars = [col for col in children_df.columns if col.lower().startswith('hv') or col.lower().startswith('sh')]
print(f"Found {len(hv_vars)} variables starting with 'hv' or 'sh':")
print(hv_vars[:30])  # Show first 30
if len(hv_vars) > 30:
    print(f"... and {len(hv_vars) - 30} more")

# Also check for 's' variables (often state-level household vars in NFHS)       
s_vars = [col for col in children_df.columns if col.lower().startswith('s') and len(col) <= 10]
print(f"\nOther 's' variables found: {len(s_vars)}")
print(s_vars[:20])

print("\n" + "="*70)
```

What it does:
- Prints lists of candidate variables with common NFHS prefixes:
  - `hv...` household variables
  - `sh...` state-specific household variables
  - `s...` often state-specific or survey-specific extra variables

---

### Cell 16 (code) — Inspect distributions of selected `s...` variables

```python
# Explore the 's' variables to understand what they contain
print("🔍 EXPLORING 'S' VARIABLES (likely household/socioeconomic data)\n")     
print("="*70)

# Get all 's' variables
s_vars = [col for col in children_df.columns if col.lower().startswith('s') and len(col) <= 10]

# Look at their data to understand what they are
print(f"Total 's' variables: {len(s_vars)}\n")

# Sample a few to see their values
sample_vars = ['s113', 's116', 's234', 's235', 's236', 's238', 's239', 's240', 's241', 's242', 's243', 's244']

print("Sample of 's' variables with their value distributions:\n")

for var in sample_vars:
    if var in children_df.columns:
        print(f"\n{var}:")
        print(children_df[var].value_counts().head(5))
        print(f"  Non-null count: {children_df[var].notna().sum()}")

# Also check if there are any 'v1' or 'v2' or 'v3' pattern variables (common NFHS naming)
print("\n" + "="*70)
print("\nLet's also check 'v' variables (mother/household characteristics):\n") 

v_vars = [col for col in children_df.columns if col.lower().startswith('v') and len(col) <= 5]
print(f"Found {len(v_vars)} 'v' variables")
print("Sample:", v_vars[:30])
```

What it does:
- Looks at value counts of a handful of `s...` variables to infer what they might represent.
- Prints a sample of `v...` variables (typical DHS/NFHS mother/household variables).

Revision tip:
- This is exploratory; it doesn’t create outputs directly but helps you understand the dataset schema.

---

### Cell 17 (code) — Build the enhanced district-level dataset

```python
# Create enhanced district-level dataset with many more features
print("🎯 CREATING COMPREHENSIVE DISTRICT-LEVEL DATASET\n")
print("="*70)

# Define all features we want to aggregate at district level
feature_list = {
    # Already have these
    'v190': 'mean',      # Wealth index
    'v106': 'mean',      # Mother's education level

    # Mother characteristics
    'v012': 'mean',      # Mother's age
    'v133': 'mean',      # Mother's education in years
    'v445': 'mean',      # Mother's BMI
    'v714': 'mean',      # Mother works (0/1)
    'v151': 'mean',      # Female-headed household

    # Child characteristics
    'hw1': 'mean',       # Child age in months
    'b4': 'mean',        # Child sex (1=male, 2=female)
    'b8': 'mean',        # Birth interval
    'v437': 'mean',      # Birth weight

    # Breastfeeding & nutrition
    'm4': 'mean',        # Duration of breastfeeding
    'm19': 'mean',       # Currently breastfeeding

    # Healthcare & immunization
    'h2': 'mean',        # BCG vaccination
    'h3': 'mean',        # DPT vaccination
    'h9': 'mean',        # Measles vaccination
    'v394': 'mean',      # Knowledge of ORS

    # Target variables
    'is_stunted': 'mean',
    'is_wasted': 'mean',
    'is_underweight': 'mean'
}

# Add state and sample size
print("Aggregating features by district...\n")

# Create aggregation dictionary
agg_dict = feature_list.copy()
agg_dict['v024'] = 'first'  # State (just take first value)
agg_dict['hw70'] = 'count'  # Sample size

# Aggregate
district_enhanced = children_df.groupby('sdist').agg(agg_dict).reset_index()    

# Rename columns for clarity
new_names = {
    'sdist': 'district',
    'v190': 'wealth_index',
    'v106': 'mother_edu_level',
    'v012': 'mother_age',
    'v133': 'mother_edu_years',
    'v445': 'mother_bmi',
    'v714': 'mother_works',
    'v151': 'female_headed_hh',
    'hw1': 'child_age_months',
    'b4': 'child_sex',
    'b8': 'birth_interval',
    'v437': 'birth_weight',
    'm4': 'breastfeed_duration',
    'm19': 'currently_breastfeed',
    'h2': 'bcg_vaccination',
    'h3': 'dpt_vaccination',
    'h9': 'measles_vaccination',
    'v394': 'knows_ors',
    'is_stunted': 'stunting_rate',
    'is_wasted': 'wasting_rate',
    'is_underweight': 'underweight_rate',
    'v024': 'state',
    'hw70': 'sample_size'
}

district_enhanced.rename(columns=new_names, inplace=True)

# Convert rates to percentages
district_enhanced['stunting_rate'] *= 100
district_enhanced['wasting_rate'] *= 100
district_enhanced['underweight_rate'] *= 100

print(f"✅ Enhanced dataset created!")
print(f"\n📊 Dataset shape: {district_enhanced.shape}")
print(f"   Districts: {district_enhanced.shape[0]}")
print(f"   Features: {district_enhanced.shape[1]}")

print(f"\n📋 Feature list ({district_enhanced.shape[1]} total):")
for i, col in enumerate(district_enhanced.columns, 1):
    print(f"   {i:2}. {col}")

print("\n" + "="*70)
```

What it does:
- Aggregates many more predictors by district to increase ML model signal.
- Renames columns into ML-friendly names (`mother_bmi`, `wealth_index`, etc.).

Key output object:
- `district_enhanced`: the dataset later used by the modeling notebook.

---

### Cell 18 (code) — Preview + save `district_malnutrition_enhanced.csv`

```python
# Preview the enhanced dataset
print("📊 ENHANCED DISTRICT DATASET PREVIEW\n")
print("="*70)

print("\nFirst 10 districts:")
print(district_enhanced.head(10))

print("\n\n📈 Summary Statistics of Key Features:\n")
print(district_enhanced.describe().round(2))

print("\n\n🔍 Check for missing values:")
missing = district_enhanced.isnull().sum()
print(missing[missing > 0])

print("\n" + "="*70)

# Save the enhanced dataset
output_path = '../Data/Processed/district_malnutrition_enhanced.csv'
district_enhanced.to_csv(output_path, index=False)

print(f"\n💾 SAVED: {output_path}")
print(f"\n✅ Dataset ready for machine learning!")
print(f"   • 707 districts")
print(f"   • 23 features")
print(f"   • 3 target variables (stunting, wasting, underweight)")
print(f"   • 19 predictor variables")
```

What it does:
- Prints a preview, summary stats, and missingness report.
- Saves the enhanced dataset.

Output file:
- [district_malnutrition_enhanced.csv](file:///e:/Child-Malnutrition/Data/Processed/district_malnutrition_enhanced.csv)

Revision note:
- The printed “23 features / 19 predictor variables” is a narrative summary; always trust the DataFrame shape and column list.

---

### Cell 19 (code) — Extract district code → district name labels from Stata metadata

```python
import pandas as pd
from pandas.io.stata import StataReader

# Read the Stata file with value labels
with StataReader('../Data/Raw/Children.DTA') as reader:
    # Get the value label mappings
    value_labels = reader.value_labels()

    # Get the sdist label mapping
    if 'SDIST' in value_labels:
        sdist_labels = value_labels['SDIST']
        print("District code → name mapping found!")
        print(f"Total districts in label: {len(sdist_labels)}\n")

        # Show first 30
        print("first district mappings:")
        for code, name in list(sdist_labels.items())[:30]:
            print(f"  {code}: {name}")

        # Create proper mapping dataframe
        district_mapping = pd.DataFrame([
            {'district_code': code, 'district_name': name}
            for code, name in sdist_labels.items()
        ])

        # Save it
        district_mapping.to_csv('../Data/Processed/district_name_mapping.csv', index=False)
        print(f"\n✅ Saved {len(district_mapping)} districts to: Data/Processed/district_name_mapping.csv")
    else:
        print("No SDIST label found. Available labels:", list(value_labels.keys()))
```

What it does:
- Uses Stata value labels embedded in the file to map numeric district codes to human-readable district names.

Output file:
- [district_name_mapping.csv](file:///e:/Child-Malnutrition/Data/Processed/district_name_mapping.csv)

Why it matters:
- ML datasets often keep districts as numeric IDs; the mapping lets the frontend/backend show readable names.

---

### Cell 20 (code) — Build a district → state mapping table and a dict for backend use

```python
import pandas as pd

# Load the mapping we just created
mapping_df = pd.read_csv('../Data/Processed/district_name_mapping.csv')

# Also need state mapping - let's get that too
df = pd.read_stata('../Data/Raw/Children.DTA', columns=['sdist', 'v024'], convert_categoricals=False)
district_state = df[['sdist', 'v024']].drop_duplicates()

# Merge to get district_code → state mapping
full_mapping = mapping_df.merge(
    district_state,
    left_on='district_code',
    right_on='sdist',
    how='left'
).drop('sdist', axis=1)

print("Complete mapping with states:")
print(full_mapping.head(30))

# Save complete mapping
full_mapping.to_csv('../Data/Processed/complete_district_mapping.csv', index=False)
print(f"\n✅ Saved complete mapping: Data/Processed/complete_district_mapping.csv")

# Now generate the Python dictionary for backend
print("\n" + "="*70)
print("Generating district_mapping.py...")
print("="*70)

# Create the dictionary format
district_dict = {}
for _, row in full_mapping.iterrows():
    district_dict[int(row['district_code'])] = {
        "name": row['district_name'].title(),  # Capitalize properly
        "state": int(row['v024'])
    }

print(f"\n✅ Generated mapping for {len(district_dict)} districts")
print("\nSample entries:")
for i in range(1, 11):
    print(f"  {i}: {district_dict[i]}")
```

What it does:
- Adds state codes (`v024`) to the district name mapping.
- Saves a “complete mapping” table that includes district_code, district_name, state.
- Builds a Python dictionary structure that could be written into a backend module (this notebook prints it, but does not actually write a `.py` file here).

Output file:
- [complete_district_mapping.csv](file:///e:/Child-Malnutrition/Data/Processed/complete_district_mapping.csv)

How this ties to the codebase:
- The backend already contains [district_mapping.py](file:///e:/Child-Malnutrition/backend/district_mapping.py). That file is likely produced from (or inspired by) this mapping step.

---

### Cell 21 (code) — Empty cell

```python

```

What it means:
- Placeholder / end of notebook.

---

# Notebook 2 — 02_feature_engineering_and_modeling.ipynb

Location: [02_feature_engineering_and_modeling.ipynb](file:///e:/Child-Malnutrition/Notebook/02_feature_engineering_and_modeling.ipynb)

## Purpose

- Load the enhanced district dataset.
- Run correlation analysis.
- Train ML models to predict district-level malnutrition rates.
- Compare models and analyze feature importance.
- Create district-level predictions and state-level summaries.
- Save models, metrics, CSV outputs, and selected figures.

## Cell-by-Cell Explanation

### Cell 1 (markdown) — Notebook goals + planned models

This is the notebook header. It explains:
- The overall workflow (load data → explore → train → compare → save).
- Intended models (Linear Regression, Random Forest, XGBoost, Neural Network).

Revision note:
- A neural network is mentioned here, but this notebook (as written) does not implement one.

---

### Cell 2 (markdown) — Section header

Introduces “Import Libraries & Load Data”.

---

### Cell 3 (code) — Imports + plotting config + random seed

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error   

# Settings
pd.set_option('display.max_columns', 50)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

print("✅ Libraries imported successfully!")
```

What it does:
- Imports scikit-learn model classes and evaluation metrics.
- Sets display and plotting defaults.
- Sets the numpy random seed to make random operations (like model training, splits) reproducible.

---

### Cell 4 (code) — Load `district_malnutrition_enhanced.csv`

```python
# Load the enhanced district dataset
data_path = '../Data/Processed/district_malnutrition_enhanced.csv'
df = pd.read_csv(data_path)

print(f"✅ Data loaded successfully!")
print(f"\n📊 Dataset shape: {df.shape}")
print(f"   Districts: {df.shape[0]}")
print(f"   Features: {df.shape[1]}")

print("\n📋 First 5 rows:")
df.head()
```

What it does:
- Loads the district-level dataset created in the exploration notebook.
- Prints shape + a preview.

Expected input:
- [district_malnutrition_enhanced.csv](file:///e:/Child-Malnutrition/Data/Processed/district_malnutrition_enhanced.csv)

---

### Cell 5 (markdown) — Section header

Introduces “Data Preparation”.

---

### Cell 6 (code) — Missing values report

```python
# Check for missing values
print("🔍 Checking for missing values...\n")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})

print(missing_df[missing_df['Missing_Count'] > 0])

if missing_df['Missing_Count'].sum() == 0:
    print("✅ No missing values found!")
else:
    print("\n⚠️ Found missing values - will handle them")
```

What it does:
- Computes missing counts and missing % per column.
- Prints only columns that have missing values.

Why it matters:
- Missingness directly affects model training and some models (Linear Regression) won’t handle NaNs without preprocessing.

---

### Cell 7 (code) — Drop fully-missing columns

```python
# Drop columns with all missing values (like knows_ors)
df = df.dropna(axis=1, how='all')

print(f"✅ Dropped columns with all missing values")
print(f"   Remaining features: {df.shape[1]}")
print(f"\n📋 Columns in dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")
```

What it does:
- Drops any column that is 100% missing across districts.
- Prints the remaining columns.

Why it matters:
- Some variables may exist in the schema but were not recorded or could not be computed.

---

### Cell 8 (markdown) — Section header

Introduces correlation analysis.

---

### Cell 9 (code) — Define targets and feature columns

```python
# Define target variables and features
target_vars = ['stunting_rate', 'wasting_rate', 'underweight_rate']
id_vars = ['district', 'state', 'sample_size']

# Get feature columns (exclude targets and IDs)
feature_cols = [col for col in df.columns if col not in target_vars + id_vars]  

print(f"📊 Analysis Setup:")
print(f"   Target variables: {len(target_vars)} - {target_vars}")
print(f"   Feature variables: {len(feature_cols)}")
print(f"\n📋 Features to use for prediction:")
for i, feat in enumerate(feature_cols, 1):
    print(f"   {i:2}. {feat}")
```

What it does:
- Declares:
  - targets = malnutrition rates
  - ids = columns not used for prediction
- Builds `feature_cols` for modeling.

Why it matters:
- This defines `X` for all downstream training cells.

---

### Cell 10 (code) — Plot “top correlations with stunting” (but has a dependency gap)

```python
# Visualize top correlations with stunting
fig, ax = plt.subplots(figsize=(10, 8))

# Get top 10 absolute correlations
top_corr = correlations.abs().nlargest(10)
top_features = correlations[top_corr.index]

# Create bar plot
colors = ['red' if x > 0 else 'green' for x in top_features]
top_features.plot(kind='barh', ax=ax, color=colors, edgecolor='black')

ax.set_xlabel('Correlation with Stunting Rate', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Top 10 Features Correlated with Stunting Rate', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Red bars = Positive correlation (bad for nutrition)")
print("✅ Green bars = Negative correlation (good for nutrition)")
```

What it intends to do:
- Plot the 10 features most correlated with `stunting_rate`, with color indicating direction.

Important revision note:
- `correlations` is not defined in any prior cell in this notebook (as written). The intended missing step is likely something like:
  - compute numeric correlations between each feature and `stunting_rate`
  - store that vector in `correlations`

Practical implication:
- If you run the notebook from top to bottom, this cell will raise a `NameError` unless you define `correlations` earlier.

---

### Cell 11 (code) — Feature-to-feature correlation heatmap

```python
# Correlation heatmap of all features
plt.figure(figsize=(12, 10))

# Select numeric features only
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns  
corr_matrix = df[numeric_features].corr()

# Create heatmap
sns.heatmap(corr_matrix,
            cmap='coolwarm',
            center=0,
            annot=False,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})

plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\n✅ Heatmap shows relationships between all features")
print("   (Red = positive correlation, Blue = negative correlation)")
```

What it does:
- Computes the correlation matrix across predictor features.
- Visualizes multicollinearity patterns (highly correlated predictors).

Why it matters:
- Helps explain why some models may overfit or why feature importance may be shared among correlated predictors.

---

### Cell 12 (markdown) — Section header

Introduces ML data preparation.

---

### Cell 13 (code) — Build X/y and split train/test (stunting split)

```python
# Prepare features and target for STUNTING prediction
print("🎯 Preparing data for ML models (Target: Stunting Rate)\n")

# Features (X) and Target (y)
X = df[feature_cols].copy()
y_stunting = df['stunting_rate'].copy()
y_wasting = df['wasting_rate'].copy()
y_underweight = df['underweight_rate'].copy()

print(f"Features (X) shape: {X.shape}")
print(f"Target stunting (y) shape: {y_stunting.shape}")
print(f"Target wasting (y) shape: {y_wasting.shape}")
print(f"Target underweight (y) shape: {y_underweight.shape}")

# Split data: 80% train, 20% test
X_train, X_test, y_train_stunt, y_test_stunt = train_test_split(
    X, y_stunting, test_size=0.2, random_state=42
)

print(f"\n✅ Train-Test Split Complete:")
print(f"   Training set: {X_train.shape[0]} districts ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Test set: {X_test.shape[0]} districts ({X_test.shape[0]/len(X)*100:.0f}%)")
```

What it does:
- Creates the design matrix `X` (predictor features).
- Creates 3 target series: stunting, wasting, underweight.
- Splits ONLY the stunting target into train/test alongside `X`.

Revision note:
- The wasting/underweight splits are performed later using `train_test_split` again, but only to split the target (see Cells 36–37). That assumes the same random split is reproduced.

---

### Cell 14 (code) — Standardize features for scale-sensitive models

```python
# Feature scaling (important for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling complete (StandardScaler)")
print(f"   Mean after scaling: {X_train_scaled.mean():.6f}")
print(f"   Std after scaling: {X_train_scaled.std():.6f}")
```

What it does:
- Fits a `StandardScaler` on training features and applies it to train/test.

Why it matters:
- Linear regression and neural networks typically benefit from standardized inputs.
- Tree models (Random Forest, XGBoost) do not require scaling and the notebook correctly trains them on unscaled `X_train`/`X_test`.

---

### Cell 15 (markdown) — Model plan + evaluation metrics

This cell defines:
- Models trained: Linear Regression, Random Forest, XGBoost.
- Metrics:
  - R² (higher better)
  - RMSE (lower better)
  - MAE (lower better)

---

### Cell 16 (markdown) — Linear regression section header

Introduces the baseline model.

---

### Cell 17 (code) — Train/evaluate Linear Regression (stunting)

```python
print("🤖 MODEL 1: Linear Regression\n")
print("="*60)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_stunt)

# Make predictions
y_pred_train_lr = lr_model.predict(X_train_scaled)
y_pred_test_lr = lr_model.predict(X_test_scaled)

# Evaluate
train_r2_lr = r2_score(y_train_stunt, y_pred_train_lr)
test_r2_lr = r2_score(y_test_stunt, y_pred_test_lr)
test_rmse_lr = np.sqrt(mean_squared_error(y_test_stunt, y_pred_test_lr))        
test_mae_lr = mean_absolute_error(y_test_stunt, y_pred_test_lr)

print(f"Training R² Score: {train_r2_lr:.4f}")
print(f"Test R² Score: {test_r2_lr:.4f}")
print(f"Test RMSE: {test_rmse_lr:.4f}%")
print(f"Test MAE: {test_mae_lr:.4f}%")

print("\n✅ Linear Regression trained successfully!")
print("="*60)
```

What it does:
- Fits a linear model on standardized features.
- Reports train and test R² to spot overfitting.
- Reports error metrics in “percent points” because targets are percentages.

What to look for:
- If train R² is high but test R² is low, the model may overfit or the relationship is not linear enough.

---

### Cell 18 (markdown) — Random Forest section header

Introduces the tree-ensemble model.

---

### Cell 19 (code) — Train/evaluate Random Forest (stunting)

```python
print("🌲 MODEL 2: Random Forest\n")
print("="*60)

# Train model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest... (this may take a minute)")
rf_model.fit(X_train, y_train_stunt)

# Make predictions
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

# Evaluate
train_r2_rf = r2_score(y_train_stunt, y_pred_train_rf)
test_r2_rf = r2_score(y_test_stunt, y_pred_test_rf)
test_rmse_rf = np.sqrt(mean_squared_error(y_test_stunt, y_pred_test_rf))        
test_mae_rf = mean_absolute_error(y_test_stunt, y_pred_test_rf)

print(f"\nTraining R² Score: {train_r2_rf:.4f}")
print(f"Test R² Score: {test_r2_rf:.4f}")
print(f"Test RMSE: {test_rmse_rf:.4f}%")
print(f"Test MAE: {test_mae_rf:.4f}%")

print("\n✅ Random Forest trained successfully!")
print("="*60)
```

What it does:
- Trains a RandomForest regressor to model non-linear relationships.
- Uses `max_depth=10` to reduce overfitting.
- Uses all CPU cores via `n_jobs=-1`.

Interpretation tip:
- A very large gap between train and test R² indicates overfitting.

---

### Cell 20 (markdown) — XGBoost section header

Introduces gradient boosting.

---

### Cell 21 (code) — Ensure xgboost is installed/importable

```python
# Install XGBoost if not already installed
try:
    import xgboost as xgb
    print("✅ XGBoost already installed")
except ImportError:
    print("Installing XGBoost...")
    import sys
    !{sys.executable} -m pip install xgboost
    import xgboost as xgb
    print("✅ XGBoost installed successfully")
```

What it does:
- Attempts to import `xgboost`.
- If missing, installs it via pip from inside the notebook.

Revision note:
- This is notebook-specific magic (`!pip ...`) and won’t work if you run the cell outside Jupyter.

---

### Cell 22 (code) — Train/evaluate XGBoost (stunting)

```python
print("🚀 MODEL 3: XGBoost\n")
print("="*60)

# Train model
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

print("Training XGBoost... (this may take a minute)")
xgb_model.fit(X_train, y_train_stunt)

# Make predictions
y_pred_train_xgb = xgb_model.predict(X_train)
y_pred_test_xgb = xgb_model.predict(X_test)

# Evaluate
train_r2_xgb = r2_score(y_train_stunt, y_pred_train_xgb)
test_r2_xgb = r2_score(y_test_stunt, y_pred_test_xgb)
test_rmse_xgb = np.sqrt(mean_squared_error(y_test_stunt, y_pred_test_xgb))      
test_mae_xgb = mean_absolute_error(y_test_stunt, y_pred_test_xgb)

print(f"\nTraining R² Score: {train_r2_xgb:.4f}")
print(f"Test R² Score: {test_r2_xgb:.4f}")
print(f"Test RMSE: {test_rmse_xgb:.4f}%")
print(f"Test MAE: {test_mae_xgb:.4f}%")

print("\n✅ XGBoost trained successfully!")
print("="*60)
```

What it does:
- Trains a boosted-tree regressor; often strong on tabular data.

Why it matters:
- XGBoost can outperform Random Forest by learning additive ensembles with regularization.

---

### Cell 23 (markdown) — Comparison section header

Introduces model comparison for stunting.

---

### Cell 24 (code) — Build a comparison table and pick the best model (stunting)

```python
# Create comparison dataframe
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'Train R²': [train_r2_lr, train_r2_rf, train_r2_xgb],
    'Test R²': [test_r2_lr, test_r2_rf, test_r2_xgb],
    'Test RMSE': [test_rmse_lr, test_rmse_rf, test_rmse_xgb],
    'Test MAE': [test_mae_lr, test_mae_rf, test_mae_xgb]
})

print("\n📊 MODEL COMPARISON RESULTS\n")
print("="*70)
print(results.to_string(index=False))
print("="*70)

# Find best model
best_model_idx = results['Test R²'].idxmax()
best_model_name = results.loc[best_model_idx, 'Model']
best_r2 = results.loc[best_model_idx, 'Test R²']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {best_r2:.4f}")
print(f"\n💡 Interpretation: The model explains {best_r2*100:.1f}% of variance in stunting rates")
```

What it does:
- Creates a structured comparison table for stunting prediction.
- Picks the best model using maximum test R².

Why it matters:
- This is your main “decision point” for which model to deploy.

---

### Cell 25 (code) — Visualize comparison (R² + error)

This cell produces:
- A bar chart comparing train vs test R² (helps detect overfitting).
- A bar chart comparing RMSE vs MAE (error magnitude).

---

### Cell 26 (markdown) — Feature importance header

Introduces feature importance.

---

### Cell 27 (code) — Random Forest feature importance (stunting)

```python
# Get feature importance from Random Forest
print("📊 FEATURE IMPORTANCE ANALYSIS (Random Forest)\n")
print("="*60)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*60)
```

What it does:
- Extracts `feature_importances_` from the stunting Random Forest model.
- Lists the top predictors.

Interpretation tip:
- For correlated predictors, feature importance may be split across multiple related columns.

---

### Cell 28 (code) — Feature importance for wasting + underweight (depends on later models)

```python
# Feature importance for Wasting (Random Forest)
fi_wasting = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_wast.feature_importances_
}).sort_values('Importance', ascending=False)

print("📊 FEATURE IMPORTANCE — Wasting (Random Forest)")
print(fi_wasting.head(10).to_string(index=False))

print()

# Feature importance for Underweight (XGBoost)
fi_underweight = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_under.feature_importances_
}).sort_values('Importance', ascending=False)

print("📊 FEATURE IMPORTANCE — Underweight (XGBoost)")
print(fi_underweight.head(10).to_string(index=False))
```

What it does:
- Prints feature importance for:
  - wasting via `rf_wast`
  - underweight via `xgb_under`

Revision note (dependency ordering):
- `rf_wast` and `xgb_under` are created later in Cells 36–37.
- If you run this notebook strictly top-to-bottom, this cell will fail unless you move it after the model training cells for wasting/underweight.

---

### Cell 29 (code) — Plot top stunting feature importances

Produces a horizontal bar chart of the top 12 predictors for stunting.

---

### Cell 30 (markdown) — Prediction visualization header

Introduces actual vs predicted plots.

---

### Cell 31 (code) — Scatter: actual vs predicted (stunting) for each model

```python
# Visualize predictions vs actual values (using best model - Random Forest or XGBoost)
# Let's use Random Forest for visualization

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

models = [
    ('Linear Regression', y_pred_test_lr),
    ('Random Forest', y_pred_test_rf),
    ('XGBoost', y_pred_test_xgb)
]

for idx, (model_name, predictions) in enumerate(models):
    axes[idx].scatter(y_test_stunt, predictions, alpha=0.6, edgecolor='black')  
    axes[idx].plot([y_test_stunt.min(), y_test_stunt.max()],
                   [y_test_stunt.min(), y_test_stunt.max()],
                   'r--', linewidth=2, label='Perfect Prediction')

    r2 = r2_score(y_test_stunt, predictions)

    axes[idx].set_xlabel('Actual Stunting Rate (%)', fontsize=11)
    axes[idx].set_ylabel('Predicted Stunting Rate (%)', fontsize=11)
    axes[idx].set_title(f'{model_name}\nR² = {r2:.3f}', fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Prediction scatter plots created!")
print("   Points closer to red line = better predictions")
```

What it does:
- Visual sanity-check of prediction accuracy.
- The red dashed diagonal represents perfect predictions.

---

### Cell 32 (markdown) — Saving header

Introduces model persistence.

---

### Cell 33 (code) — Save stunting models + scaler + comparison CSV

```python
import pickle
import os

# Create models directory
models_dir = '../Models'
os.makedirs(models_dir, exist_ok=True)

# Save models
models_to_save = {
    'linear_regression_stunting.pkl': lr_model,
    'random_forest_stunting.pkl': rf_model,
    'xgboost_stunting.pkl': xgb_model,
    'scaler.pkl': scaler
}

for filename, model in models_to_save.items():
    filepath = os.path.join(models_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved: {filename}")

# Save results
results.to_csv(os.path.join(models_dir, 'model_comparison_results.csv'), index=False)
print(f"✅ Saved: model_comparison_results.csv")

print(f"\n💾 All models saved to: {models_dir}")
```

What it does:
- Serializes trained scikit-learn models to `.pkl` so the backend can load them for inference.
- Saves a CSV table of model metrics for stunting.

Outputs:
- [Models/linear_regression_stunting.pkl](file:///e:/Child-Malnutrition/Models/linear_regression_stunting.pkl)
- [Models/random_forest_stunting.pkl](file:///e:/Child-Malnutrition/Models/random_forest_stunting.pkl)
- [Models/xgboost_stunting.pkl](file:///e:/Child-Malnutrition/Models/xgboost_stunting.pkl)
- [Models/scaler.pkl](file:///e:/Child-Malnutrition/Models/scaler.pkl)
- [Models/model_comparison_results.csv](file:///e:/Child-Malnutrition/Models/model_comparison_results.csv)

---

### Cell 34 (markdown) — Summary header

Introduces the “completion summary”.

---

### Cell 35 (code) — Prints an “ML complete” summary for stunting run

This cell:
- Prints a checklist of steps completed.
- Prints the `best_model_name` and `best_r2` computed earlier for stunting.

Revision note:
- The printed “707 districts, 22 features” is a narrative; your actual `df.shape` is the source of truth.

---

### Cell 36 (code) — Train wasting models (LR/RF/XGB)

Key idea:
- Repeats the modeling process for wasting.
- Uses the same `X_train/X_test` created earlier.
- Produces `lr_wast`, `rf_wast`, `xgb_wast` and their test metrics.

Important detail:
- It splits `y_wasting` using `train_test_split` to produce `y_train_wast` and `y_test_wast`. Because `random_state=42` and `test_size=0.2` match the earlier split, this should align with `X_train/X_test` ordering, but it’s conceptually safer to split `X` and `y` together each time.

---

### Cell 37 (code) — Train underweight models (LR/RF/XGB)

What it does:
- Same as wasting but for underweight.

Revision note (logic inconsistency):
- The cell prints: “Best model for underweight: Random Forest …”
- Later cells (48+) treat XGBoost as best for underweight.
- The correct “best” should be determined by comparing `test_r2_lr_under`, `test_r2_rf_under`, and `test_r2_xgb_under`, not by hardcoding a model name.

---

### Cell 38 (code) — Build the “all outcomes” comparison table

What it does:
- Creates `all_results`, a 9-row table (3 outcomes × 3 models) with R² and RMSE.
- Prints best model per malnutrition type by selecting max R².
- This is the consolidated evaluation across the project.

Output file later:
- [Models/all_models_comparison.csv](file:///e:/Child-Malnutrition/Models/all_models_comparison.csv)

---

### Cell 39 (code) — 4-panel comprehensive visualization

What it does:
- Creates a multi-panel figure summarizing:
  - R² across models and outcomes
  - best R² per outcome
  - RMSE per outcome for best models
  - “top 5 features” per outcome

Important revision note:
- The “underweight top 5 features” here use `rf_under.feature_importances_` (Random Forest), not XGBoost importances. That means the feature-importance comparison panel is not model-consistent if XGBoost is best for underweight.

---

### Cell 40 (code) — Project completion summary + save all trained models

What it does:
- Prints a narrative summary of the whole project.
- Saves additional models for wasting and underweight into [Models/](file:///e:/Child-Malnutrition/Models).
- Saves the full comparison table as `all_models_comparison.csv`.

Saved models include:
- `random_forest_wasting.pkl`, `xgboost_wasting.pkl`, `linear_regression_wasting.pkl`
- `random_forest_underweight.pkl`, `xgboost_underweight.pkl`, `linear_regression_underweight.pkl`

---

### Cell 41 (code) — District-level predictions for all districts + save CSV

What it does:
- Builds `district_results`:
  - district/state/sample_size
  - actual rates
  - predicted rates for each outcome using “best” models
  - error columns (actual - predicted)
- Writes output to:
  - [district_predictions_all_types.csv](file:///e:/Child-Malnutrition/Data/Processed/district_predictions_all_types.csv)

Why it matters:
- This file is what a dashboard or reporting layer can use to show predictions by district.

---

### Cell 42 (code) — Error analysis visualization + identify hardest districts

What it does:
- Plots:
  - histograms of errors for each outcome
  - top-10 largest absolute-error districts per outcome
- Prints the single worst district per outcome by absolute error.

Interpretation:
- Helps you diagnose where the model struggles (outliers, data quality issues, or missing predictors).

---

### Cell 43 (code) — State-level aggregation + plots + save CSV

What it does:
- Aggregates district results up to `state`:
  - mean actual vs predicted rates
  - total sample size
  - number of districts
- Visualizes top states by outcome + a scatter plot actual vs predicted.
- Saves:
  - [state_level_summary.csv](file:///e:/Child-Malnutrition/Data/Processed/state_level_summary.csv)

Why it matters:
- Useful for policy-level insights (state comparisons) beyond district-level views.

---

### Cell 44 (code) — Create `Outputs/Figures` folder and explain how to save plots

What it does:
- Creates `../Outputs/Figures`.
- Prints instructions for adding `plt.savefig(...)` to visualization cells.

Revision note:
- This notebook later actually saves figures in Cells 45–47.

---

### Cell 45 (code) — Re-run and save the comprehensive 4-panel figure

What it does:
- Rebuilds the 4-panel “comprehensive ML analysis” plot.
- Saves:
  - `../Outputs/Figures/1_comprehensive_ml_analysis.png`

---

### Cell 46 (code) — Re-run and save the error analysis figure

What it does:
- Rebuilds the 2×3 error analysis plot.
- Saves:
  - `../Outputs/Figures/2_error_analysis.png`

---

### Cell 47 (code) — Re-run and save the state-level figure

What it does:
- Rebuilds the 2×2 state-level analysis plot.
- Saves:
  - `../Outputs/Figures/3_state_level_analysis.png`

---

### Cell 48 (code) — Hyperparameter tuning setup and baseline printout

What it does:
- Imports `RandomizedSearchCV` etc.
- Prints current baseline best scores (as assumed by the author):
  - stunting best = Random Forest
  - wasting best = Random Forest
  - underweight best = XGBoost

Revision note:
- This baseline assumption should match the actual `all_results` comparison, not a hardcoded narrative.

---

### Cell 49 (code) — Randomized hyperparameter search for stunting Random Forest

What it does:
- Defines a search space (n_estimators, depth, leaf sizes, etc.).
- Runs `RandomizedSearchCV` with:
  - 50 sampled parameter combinations
  - 5-fold CV
  - scoring = R²
- Evaluates the tuned estimator on the held-out test set.
- Stores best estimator in `rf_model_tuned`.

Why it matters:
- This is the “model improvement” stage after a baseline is established.

---

### Cell 50 (code) — Randomized hyperparameter search for underweight XGBoost

What it does:
- Defines a search space for XGBoost (depth, learning rate, subsample, etc.).
- Runs `RandomizedSearchCV`.
- Compares tuned performance to original.
- If improved, stores `xgb_under_tuned`.

Revision note:
- This tuning is only for underweight; wasting is not tuned in this notebook.

---

### Cell 51 (code) — Tuning summary

What it does:
- Prints before/after test R² for tuned models.
- Concludes that originals were already good (narrative conclusion).

Revision tip:
- Always trust the printed numeric deltas here; the conclusion line is generic.

---

### Cell 52 (code) — Empty cell

```python

```

Meaning:
- Placeholder / end of notebook.

---

# Notebook 3 — Feature-Engineering.ipynb

Location: [Feature-Engineering.ipynb](file:///e:/Child-Malnutrition/Notebook/Feature-Engineering.ipynb)

## Cell-by-Cell Explanation

### Cell 1 (code) — Empty notebook cell

```python

```

What it means:
- This notebook currently contains no code or markdown content.
- Treat it as a placeholder; the actual feature engineering work happens inside:
  - [Data_exploration.ipynb](file:///e:/Child-Malnutrition/Notebook/Data_exploration.ipynb) (district-level feature aggregation)
  - [02_feature_engineering_and_modeling.ipynb](file:///e:/Child-Malnutrition/Notebook/02_feature_engineering_and_modeling.ipynb) (model-focused feature selection via `feature_cols`)

