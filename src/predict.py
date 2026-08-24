# predict.py

from src.config import PROJECT_PATH
import joblib
import pandas as pd


def load_artifact(
        path = PROJECT_PATH / "models" / "catboost_fraud_model.joblib"
):
    return joblib.load(path)

def predict(X, artifact = load_artifact()):
    model = artifact['model']
    threshold = artifact['threshold']

    probabilities = model.predict_proba(X)[:, -1]
    predictions = (probabilities >= threshold).astype(int)

    result = pd.DataFrame({
        "probabilities": probabilities,
        "predictions": predictions
    })

    return result
