# predict.py

from src.config import PROJECT_PATH
import joblib
import pandas as pd


def predict(X, artifact):

    model = artifact['model']
    threshold = artifact['threshold']

    probabilities = model.predict_proba(X)[:, -1]
    predictions = (probabilities >= threshold).astype(int)

    result = pd.DataFrame({
        "probabilities": probabilities,
        "predictions": predictions
    })

    return result
