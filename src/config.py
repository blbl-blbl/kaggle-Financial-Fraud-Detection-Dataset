# config.py

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

CATEGORICAL_FEATURES = ['type']

TARGET = 'isFraud'

