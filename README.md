# Financial Fraud Detection

A machine learning project for detecting fraudulent financial transactions in a highly imbalanced dataset.

The project focuses not only on predictive performance, but also on **data leakage prevention, temporal validation, class imbalance, probability calibration, operating-threshold selection, and model interpretability**.

The repository currently contains the research and experimentation stage implemented in Jupyter notebooks. The next stage is to convert the selected approach into a reproducible Python training and inference pipeline.

## Dataset

The project uses the [Financial Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset/data), which contains synthetic mobile-money transactions.

* **6,362,620 transactions**
* **8,213 fraudulent transactions**
* Fraud rate: approximately **0.129%**
* Target variable: `isFraud`

Because the target is extremely imbalanced, metrics such as accuracy are not suitable for model selection. **Average Precision (AP)** is therefore used as the primary ranking metric.

## Methodology

### Leakage-aware feature selection

Several original dataset variables are intentionally excluded from the primary model.

The balance-related features:

* `oldbalanceOrg`
* `newbalanceOrig`
* `oldbalanceDest`
* `newbalanceDest`

may introduce target leakage because fraudulent transactions in the dataset are annulled, meaning post-transaction balances can indirectly contain information about the target.

`isFlaggedFraud` is also excluded because it represents an existing rule-based fraud detection system rather than an independent transaction characteristic.

Raw account identifiers:

* `nameOrig`
* `nameDest`

are not directly encoded because of their extremely high cardinality.

### Engineered features

The current feature set includes:

* `step`
* `type`
* `amount`
* `log_amount`
* `day`
* `hour_sin`
* `hour_cos`

`log_amount` reduces the strong right skew of transaction amounts.

Transaction hour is represented using sine and cosine transformations:

```text
hour_sin
hour_cos
```

which preserve the cyclical nature of time.

## Temporal validation

A random train/test split is deliberately avoided.

Fraud prevalence changes substantially over time in this dataset, which creates a strong temporal distribution shift. A random split would mix earlier and later observations and produce an unrealistically optimistic estimate of model quality.

The current data split is:

| Dataset     | Step range | Purpose                                             |
| ----------- | ---------: | --------------------------------------------------- |
| Train       |    `1–400` | Initial model development and hyperparameter search |
| Validation  |  `401–450` | Model selection and early stopping                  |
| Calibration |  `451–500` | Probability calibration                             |
| Threshold   |  `501–550` | Operating-threshold selection                       |
| Test        |  `551–743` | Final untouched out-of-time evaluation              |

After model selection, `train` and `validation` are combined into a single development dataset:

```text
development = step 1–450
```

The selected model is then retrained on the complete development period.

## Expanding-window cross-validation

The strongest candidate models are additionally evaluated using expanding temporal folds:

```text
Fold 1:
Train      step <= 250
Validation step 251–300

Fold 2:
Train      step <= 300
Validation step 301–350

Fold 3:
Train      step <= 350
Validation step 351–400

Fold 4:
Train      step <= 400
Validation step 401–450
```

This approach provides a more realistic estimate of model stability under temporal drift than evaluating performance on a single validation period.

## Models evaluated

The following algorithms are compared:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* CatBoost
* LightGBM

Different class-imbalance strategies are also evaluated:

* class weighting
* random undersampling
* random oversampling
* SMOTENC

Hyperparameters for the strongest models are optimized with **Optuna**.

## Model selection

CatBoost and LightGBM emerged as the strongest candidates after the initial model comparison and hyperparameter tuning.

The latest saved cross-validation outputs showed:

| Model    | Mean CV AP | CV AP std |
| -------- | ---------: | --------: |
| CatBoost |  **0.407** |     0.044 |
| LightGBM |      0.384 | **0.030** |

CatBoost was selected as the primary model because of its higher mean Average Precision.

LightGBM remained a strong alternative: it showed slightly lower variation across temporal folds and substantially faster training.

> **Important:** these results were produced before the latest refactoring of the modeling notebook. The entire experiment will be rerun before the metrics are treated as final.

## Class imbalance

The positive class represents only approximately 0.13% of all transactions.

Several strategies are evaluated to determine whether changing the effective class distribution improves ranking performance:

```text
Class weights
Random undersampling
Random oversampling
SMOTENC
```

Sampling is applied only to training data. Validation, calibration, threshold, and test datasets retain their original class distributions.

## Probability calibration

The raw output of a classifier trained on highly imbalanced and resampled data should not automatically be interpreted as a well-calibrated fraud probability.

For this reason, the selected model is calibrated using a dedicated out-of-time calibration dataset.

Two approaches are compared:

* Isotonic calibration
* Sigmoid calibration

Calibration quality is evaluated using:

* Brier score
* Calibration curves

In the latest saved experiment, isotonic calibration produced the lowest Brier score and was selected for the final model.

## Operating threshold

The classification threshold is selected on a dedicated threshold dataset rather than on the test set.

The project uses **F2** instead of F1 for threshold optimization.

F2 gives recall more weight than precision:

```text
F2 → recall is more important than precision
```

This is a more natural assumption for fraud detection, where failing to detect fraudulent transactions may be more costly than investigating additional legitimate transactions.

Once selected, the threshold is frozen and applied unchanged to the final test period.

## Evaluation metrics

The following metrics are used:

### Ranking quality

* **Average Precision** — primary model-selection metric

### Classification quality at the selected threshold

* Precision
* Recall
* F1
* F2
* Confusion Matrix

### Probability quality

* Brier score
* Calibration curve

Accuracy is intentionally not used as a primary metric because a classifier that predicts every transaction as legitimate would already achieve extremely high accuracy.

## Interpretability

The final CatBoost model is interpreted using **SHAP**.

During model development, SHAP values are computed on a sample from the development dataset rather than on the final test set.

This avoids indirectly using information from the test period to make further modeling decisions.

The SHAP analysis is intended to answer questions such as:

* Which features drive fraud scores globally?
* What features increase fraud risk for individual transactions?
* Why does the model produce false positives?
* Why are some fraudulent transactions missed?

## Repository structure

```text
.
├── data/
│   └── .gitkeep
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```

## Notebooks

### `01_eda.ipynb`

Exploratory data analysis:

* dataset structure
* missing values and duplicates
* class imbalance
* transaction-type distributions
* amount distributions
* temporal fraud behavior
* feature associations
* target-leakage analysis

### `02_baseline.ipynb`

Establishes simple leakage-safe baseline models.

Main topics:

* temporal splitting
* DummyClassifier baseline
* Logistic Regression
* class weighting
* Average Precision as the primary metric

### `03_feature_engineering.ipynb`

Evaluates additional leakage-safe features.

Examples:

* logarithmic transaction amount
* transaction day
* transaction hour
* cyclical hour encoding

Candidate features are compared using the same out-of-time validation methodology.

### `04_modeling.ipynb`

Main modeling notebook.

Includes:

* multiple ML algorithms
* imbalance strategies
* sampling experiments
* CatBoost
* LightGBM
* Optuna hyperparameter tuning
* temporal cross-validation
* final model retraining
* probability calibration
* threshold selection
* final test evaluation
* confusion matrices
* SHAP analysis
* model artifact serialization

## Kaggle dataset page:

https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset/data

Dataset CSV files are intentionally excluded from Git.

## Running the project

Run the notebooks in order:

```text
01_eda.ipynb
      ↓
02_baseline.ipynb
      ↓
03_feature_engineering.ipynb
      ↓
04_modeling.ipynb
```

Some experiments in `04_modeling.ipynb`, especially model comparison, temporal cross-validation, and Optuna hyperparameter tuning, can take a significant amount of time on the full 6.3 million-row dataset.

For this reason, some expensive cells may remain disabled or commented after their results have been saved.

## Model artifact

The final artifact contains the calibrated fraud-detection model together with the selected operating threshold.

Conceptually:

```text
Transaction
    ↓
Feature engineering
    ↓
CatBoost
    ↓
Probability calibration
    ↓
Fraud probability
    ↓
Frozen threshold
    ↓
Fraud / legitimate decision
```

## Current status

The notebook-based research stage is close to completion.

The next stage of the project is to convert the experiment into a reproducible Python pipeline.

Planned improvements:

1. Move feature engineering into reusable Python modules.
2. Move temporal splitting logic outside notebooks.
3. Separate Optuna tuning from regular model training.
4. Create a reproducible end-to-end training command.
5. Save experiment metadata and metrics alongside the model.
6. Add automated tests for temporal splitting and feature generation.
7. Add lightweight CI.
8. Introduce leakage-safe historical and velocity features based only on past transaction activity.

A possible future project structure:

```text
.
├── configs/
│   └── catboost.yaml
│
├── data/
│
├── notebooks/
│
├── src/
│   └── fraud_detection/
│       ├── data.py
│       ├── split.py
│       ├── features.py
│       ├── models.py
│       ├── tuning.py
│       ├── calibration.py
│       ├── threshold.py
│       ├── evaluation.py
│       └── train.py
│
├── tests/
│
├── models/
│
└── README.md
```

The goal is to make the entire workflow reproducible from raw transaction data to a calibrated fraud score and final binary decision.
