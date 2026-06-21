"""Load a registered Wine model from MLflow and score a single sample.

Production inference loads by registry *stage*, not by run ID or file path.
That way promoting v2 to Production automatically changes what this script loads.
"""

import os
from pathlib import Path

import mlflow
from sklearn.datasets import load_wine

LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{(LAB_DIR / 'mlflow.db').as_posix()}"
REGISTERED_MODEL_NAME = "wine-classifier"


def predict(features: list[float], model_stage: str = "Production") -> str:
    """Return the cultivar name for a single 13-feature wine sample.

    Args:
        features: 13 chemical measurements in sklearn Wine feature order.
        model_stage: Registry stage to load — typically "Production" or "Staging".
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    # Feature names only needed to map integer predictions → human-readable labels.
    wine = load_wine()

    # URI format: models:/<registered-name>/<stage>
    # Decouples serving code from whichever training run produced the model.
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{model_stage}")

    prediction = model.predict([features])[0]
    return wine.target_names[prediction]


if __name__ == "__main__":
    wine = load_wine()

    # Demo: score the first sample from the built-in dataset.
    sample = wine.data[0].tolist()
    cultivar = predict(sample)

    print("Input features:")
    for name, value in zip(wine.feature_names, sample):
        print(f"  {name}: {value}")
    print(f"Predicted cultivar: {cultivar}")
