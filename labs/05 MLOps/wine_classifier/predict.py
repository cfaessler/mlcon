"""Load a registered Wine model from MLflow and score a single sample."""

import os
from pathlib import Path

import mlflow
from sklearn.datasets import load_wine

LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{(LAB_DIR / 'mlflow.db').as_posix()}"
REGISTERED_MODEL_NAME = "wine-classifier"


def predict(features: list[float], model_stage: str = "Production") -> str:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    wine = load_wine()
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{model_stage}")
    prediction = model.predict([features])[0]
    return wine.target_names[prediction]


if __name__ == "__main__":
    wine = load_wine()
    sample = wine.data[0].tolist()
    cultivar = predict(sample)
    print("Input features:")
    for name, value in zip(wine.feature_names, sample):
        print(f"  {name}: {value}")
    print(f"Predicted cultivar: {cultivar}")
