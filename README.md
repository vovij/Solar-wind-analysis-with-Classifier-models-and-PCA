# Solar Wind Analysis with Classifier Models and PCA

A comprehensive machine learning project for analyzing solar wind data and predicting geomagnetic storms using multiple classification algorithms and dimensionality reduction techniques.

## Overview

This project analyzes solar wind measurements from 1988-2020 to predict geomagnetic storms based on the Disturbance Storm Time (Dst) index. The analysis includes exploratory data analysis, linear regression, Principal Component Analysis (PCA), and multiple classification models.

## Dataset

**Source:** OMNI solar wind dataset (1988-2020)
- **Time Range:** January 1988 - December 2020
- **Total Records:** 289,296 hourly measurements
- **Features:**
  - `Bx`, `By`, `Bz`: Interplanetary Magnetic Field components (nT)
  - `density`: Proton density (N/cm³)
  - `V`: Solar wind speed (km/s)
  - `Dst`: Disturbance Storm Time index (nT)

**Data Quality:**
- Missing values (~12-15% per column) represented as 999.9 and 9999
- Missing data primarily concentrated in 1988-1994
- After cleaning: 246,486 complete records

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels missingno dmba
```

## Project Structure

### 1. Data Preprocessing
- Missing value detection and removal
- Temporal analysis of data quality
- Feature engineering (log transformation, binary classification labels)

### 2. Exploratory Data Analysis
- Distribution analysis of all features
- Correlation analysis between variables
- Autocorrelation analysis (1000-hour time lag)
- 2D histogram visualizations

### 3. Linear Regression Analysis
- Ordinary Least Squares (OLS) regression
- Feature selection via backward elimination
- Residual analysis
- Influence plot generation
- **Best Model:** Uses `Bz`, `density`, and `V` (R² = 0.312)

### 4. Principal Component Analysis (PCA)
- Dimensionality reduction on 5 solar wind parameters
- Explained variance analysis
- Component loadings interpretation

### 5. Classification Models

**Target Variable:** Geomagnetic storm detection (Dst < -20 nT)
- **Class Distribution:** ~25% storms, ~75% no storms
- **Dataset Balancing:** Reduced to ~50/50 split for training

#### Models Implemented:

**Gaussian Naive Bayes**
- Training Accuracy: 73.63%
- Testing Accuracy: 88.49%
- ROC AUC: 0.783

**Linear Discriminant Analysis (LDA)**
- Training Accuracy: 72.48%
- Testing Accuracy: 84.73%
- ROC AUC: 0.758

**Logistic Regression**
- Training Accuracy: 75.52%
- L2 regularization with liblinear solver
- ROC AUC: 0.838

## Key Findings

1. **Feature Correlations:**
   - Strongest positive correlation: Dst and Bz (0.305)
   - Strongest negative correlation: Dst and V (-0.472)

2. **Linear Regression Results:**
   - Final model includes: Bz, density, and V
   - R-squared: 0.312
   - All predictors statistically significant (p < 0.001)
   - F-statistic: 3.596e+04

3. **PCA Analysis:**
   - First 3 components explain 75.9% of variance
   - All 5 components needed for 100% variance
   - Component 1 (29%): Bx and By dominant
   - Component 2 (27%): Density and V dominant
   - Component 3 (20%): Bz dominant

4. **Classification Performance:**
   - Best testing accuracy: Gaussian Naive Bayes (88.49%)
   - Best ROC AUC: Logistic Regression (0.838)
   - Model successfully identifies geomagnetic storms

## Usage
```python
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('Omni_98_20.csv')

# Preprocess
df['time'] = pd.to_datetime(df['time'])
df.replace({999.9: pd.NA, 9999: pd.NA}, inplace=True)
df = df.dropna()

# Create storm labels
df['Dst_label'] = df['Dst'] < -20

# Train model
predictors = ["Bx", "By", "Bz", "density", "V"]
X_train = df[predictors]
y_train = df['Dst_label']

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
predictions = gnb.predict(X_train)
```

## Visualizations

The notebook includes:
- Distribution histograms for all features
- Correlation heatmap
- 2D histograms for variable relationships
- Autocorrelation plots (1000-hour lag)
- Residual plots for regression analysis
- ROC curves for all classifiers
- Pie charts for class distribution

## Results Summary

| Model | Training Accuracy | Testing Accuracy | ROC AUC |
|-------|------------------|------------------|---------|
| Gaussian Naive Bayes | 73.63% | 88.49% | 0.783 |
| Linear Discriminant Analysis | 72.48% | 84.73% | 0.758 |
| Logistic Regression | 75.52% | - | 0.838 |

## Files

- `Solar wind analysis code.ipynb` - Main analysis notebook
- `Solar wind analysis online.ipynb` - Online version
- `Solar wind analysis.pdf` - PDF report
- `Omni_98_20.csv` - Dataset (not included in repo)

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- missingno
- dmba

## Author

Created for solar wind and geomagnetic storm analysis research.

## License

This project is available for educational and research purposes.

## Acknowledgments

Data sourced from the OMNI database maintained by NASA's Space Physics Data Facility.
