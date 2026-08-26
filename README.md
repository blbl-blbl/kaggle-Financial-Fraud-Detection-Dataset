![Tests](https://github.com/blbl-blbl/kaggle-Financial-Fraud-Detection-Dataset/actions/workflows/tests.yml/badge.svg)

# Financial Fraud Detection

A machine learning project for detecting fraudulent financial transactions in a highly imbalanced dataset.

The project focuses on both predictive performance and ML engineering practices, including:

- leakage-aware feature selection
- temporal validation
- class imbalance handling
- hyperparameter optimization
- probability calibration
- operating-threshold selection
- model interpretability
- reproducible training and inference
- automated testing and CI

## Dataset

The project uses the [Financial Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset/data), containing synthetic mobile-money transactions.

Dataset characteristics:

- 6,362,620 transactions
- 8,213 fraudulent transactions
- fraud rate ≈ 0.129%
- target: `isFraud`

Because fraud is extremely rare, accuracy is not suitable as the primary model-selection metric.

**Average Precision (AP)** is used as the main ranking metric.

CSV files are intentionally excluded from Git.

## Methodology

### Leakage-aware feature selection

Several original variables are excluded from the final model.

Balance-related features:

```text
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

may contain target leakage because fraudulent transactions in the synthetic dataset are annulled, allowing post-transaction balances to indirectly reveal information about the target.

`isFlaggedFraud` is excluded because it represents an existing rule-based fraud detector.

High-cardinality account identifiers:

```text
nameOrig
nameDest
```

are not directly encoded in the current model.

### Final features

The selected feature set is:

```text
step
type
amount
log_amount
day
hour_sin
hour_cos
```

`log_amount` reduces the strong right skew of transaction amounts.

Transaction hour is represented using sine and cosine transformations to preserve its cyclical structure.

## Temporal validation

Random splitting is deliberately avoided because fraud prevalence changes significantly over time.

The data is divided into consecutive time periods:

| Dataset | Step range | Purpose |
| --- | ---: | --- |
| Train | 1–400 | Model development and tuning |
| Validation | 401–450 | Model selection / early stopping |
| Calibration | 451–500 | Probability calibration |
| Threshold | 501–550 | Operating threshold selection |
| Test | 551–743 | Final out-of-time evaluation |

After model selection:

```text
development = train + validation
            = steps 1–450
```

The selected model is retrained on the complete development period.

## Temporal cross-validation

Model stability is additionally evaluated using expanding-window folds:

```text
Fold 1
Train:      step <= 250
Validation: step 251–300

Fold 2
Train:      step <= 300
Validation: step 301–350

Fold 3
Train:      step <= 350
Validation: step 351–400

Fold 4
Train:      step <= 400
Validation: step 401–450
```

This provides a more realistic robustness check under temporal distribution shift than random cross-validation.

## Models evaluated

The research stage compares:

```text
Logistic Regression
Decision Tree
Random Forest
XGBoost
CatBoost
LightGBM
```

Class-imbalance strategies include:

```text
class weighting
random undersampling
random oversampling
SMOTENC
```

Sampling is applied only to training data.

Validation, calibration, threshold and test datasets retain their original class distributions.

## Selected model

CatBoost and LightGBM were the strongest candidates in the initial experiments.

Latest saved temporal CV results:

| Model | Mean CV AP | CV AP std |
| --- | ---: | ---: |
| CatBoost | 0.407 | 0.044 |

CatBoost was selected as the primary model.

The current CatBoost pipeline uses RandomOverSampler with an effective minority-class sampling ratio of `0.02`.

Hyperparameters are optimized using Optuna and stored separately in:

```text
configs/catboost.yaml
```

## Probability calibration

The classifier is trained on resampled data, so raw model scores should not automatically be interpreted as calibrated fraud probabilities.

A dedicated out-of-time calibration period is used.

The research stage compared:

```text
isotonic
sigmoid
```

Isotonic calibration produced the best saved Brier score and is currently used by the training pipeline.

## Operating threshold

The final classification threshold is selected on a dedicated threshold dataset rather than the test set.

Threshold optimization uses **F2 score**, placing more emphasis on recall than precision.

This reflects a common fraud-detection assumption: missing a fraudulent transaction can be more costly than investigating an additional legitimate transaction.

## Final out-of-time performance

The final reproducible training pipeline produced the following
results on the untouched test period (steps 551–743):

| Metric | Score |
| --- | ---: |
| Average Precision | 0.345 |
| Precision | 0.384 |
| Recall | 0.379 |
| F1 | 0.382 |
| F2 | 0.380 |

Confusion matrix:

| | Predicted legitimate | Predicted fraud |
| --- | ---: | ---: |
| Actual legitimate | 193144 | 1309 |
| Actual fraud | 1333 | 815 |


The report is generated automatically by the training pipeline and saved to:

```text
reports/metrics.json
```

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── configs/
│   └── catboost.yaml
│
├── data/
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── reports/
│   └── metrics.json
│
├── src/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── calibration.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── predict.py
│   ├── split.py
│   ├── threshold.py
│   ├── train.py
│   └── tuning.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_predict.py
│   ├── test_split.py
│   └── test_threshold.py
│
├── .gitignore
├── .python-version
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Setup

Python 3.12 is used for the project.

Create and activate a virtual environment.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install project and development dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Download the dataset from Kaggle and place:

```text
Synthetic_Financial_datasets_log.csv
```

inside:

```text
data/
```

## Running tests

Run all unit and integration tests:

```bash
python -m pytest -v
```

The tests use small synthetic datasets and do not require the full Kaggle dataset.

Tests cover:

```text
input validation
feature engineering
temporal splitting
threshold selection
prediction logic
raw inference pipeline
```

## Training

Run the complete training pipeline from the repository root:

```bash
python -m src.train
```

The pipeline performs:

```text
Load raw data
      ↓
Feature engineering
      ↓
Temporal splitting
      ↓
Model training
      ↓
Probability calibration
      ↓
Threshold selection
      ↓
Artifact serialization
      ↓
Final test evaluation
```

The trained artifact is saved to:

```text
models/catboost_fraud_model.joblib
```

The final evaluation report is saved to:

```text
reports/metrics.json
```

## Inference

Load the trained artifact:

```python
from src.artifacts import load_artifact

artifact = load_artifact()
```

Predictions can be generated from raw transaction data:

```python
import pandas as pd

from src.predict import predict_raw

transactions = pd.DataFrame({
    "step": [100],
    "type": ["TRANSFER"],
    "amount": [15000.0],
})

result = predict_raw(
    transactions,
    artifact
)

print(result)
```

The result contains:

```text
probabilities
predictions
```

where `predictions` uses the frozen threshold selected during training.

## Hyperparameter tuning

Hyperparameter optimization is separated from normal model training.

`tuning.py` uses Optuna to maximize Average Precision on the temporal validation period.

The selected parameters can be stored in:

```text
configs/catboost.yaml
```

Normal training then uses the frozen configuration instead of rerunning expensive optimization.


## Notebooks

`01_eda.ipynb` contains exploratory data analysis and leakage investigation.

`02_baseline.ipynb` establishes simple leakage-safe baselines.

`03_feature_engineering.ipynb` evaluates additional engineered features.

`04_modeling.ipynb` contains model comparison, imbalance experiments, Optuna tuning, temporal cross-validation, calibration, threshold analysis and SHAP interpretation.

The notebooks document the research process, while reusable production-style logic lives in `src/`.
