# Solar Wind Analysis with Classifier Models and PCA

Machine learning project for predicting geomagnetic storms from solar wind data (1988-2020) using classification algorithms and PCA.

## Dataset

- **Source:** OMNI solar wind dataset
- **Records:** 246,486 hourly measurements (after cleaning)
- **Features:** Bx, By, Bz (magnetic field), density, V (velocity), Dst (storm index)

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Key Features

- **Data Preprocessing:** Missing value handling (~12-15% per feature)
- **EDA:** Correlation analysis, autocorrelation, distribution plots
- **Linear Regression:** Feature selection, R² = 0.312
- **PCA:** 5 components, first 3 explain 75.9% variance
- **Classification:** Binary prediction of geomagnetic storms (Dst < -20 nT)

## Models & Results

| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| Gaussian Naive Bayes | 88.49% | 0.783 |
| Linear Discriminant Analysis | 84.73% | 0.758 |
| Logistic Regression | 75.52% | 0.838 |

## Quick Start
```python
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load and preprocess
df = pd.read_csv('Omni_98_20.csv')
df.replace({999.9: pd.NA, 9999: pd.NA}, inplace=True)
df = df.dropna()
df['Dst_label'] = df['Dst'] < -20

# Train
predictors = ["Bx", "By", "Bz", "density", "V"]
model = GaussianNB()
model.fit(df[predictors], df['Dst_label'])
```

## Files

- `Solar wind analysis code.ipynb` - Main notebook
- `Solar wind analysis.pdf` - Report

## Key Findings

- Dst strongly correlated with Bz (0.305) and V (-0.472)
- Gaussian Naive Bayes achieves best testing accuracy (88.49%)
- Successfully identifies geomagnetic storms from solar wind parameters
