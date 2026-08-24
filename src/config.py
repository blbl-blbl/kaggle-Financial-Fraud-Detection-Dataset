# config.py

from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]

BASE_FEATURES = [
    'step', 'type',
    'amount', 'nameOrig',
    'oldbalanceOrg', 'newbalanceOrig',
    'nameDest', 'oldbalanceDest',
    'newbalanceDest', 'isFlaggedFraud'
]

FINAL_FEATURES = [
    'step', 'type',
    'amount', 'log_amount',
    'day', 'hour_sin',
    'hour_cos'
]

TARGET = 'isFraud'
RANDOM_SEED = 42

CATEGORICAL_FEATURES = ['type']
NUMERIC_FEATURES = ['amount', 'log_amount']
OTHER_FEATURES = ['step', 'day', 'hour_sin', 'hour_cos']


TEST_POINT = 550
THRESHOLD_POINT = 500
CALIBRATION_POINT = 450
VALIDATION_POINT = 400

TEMPORAL_FOLDS = [
    (250, 300),
    (300, 350),
    (350, 400),
    (400, 450),
]

