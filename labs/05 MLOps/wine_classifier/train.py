"""Train a Wine cultivar classifier and log the run to MLflow.

This script is the entry point for the MLflow Project in this folder.
Students can run it directly or via:

    mlflow run . --env-manager local -P max_depth=5 -P min_samples_split=2
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
LAB_DIR = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{(LAB_DIR / 'mlflow.db').as_posix()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Wine decision tree with MLflow logging.")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--experiment-name", type=str, default="wine-classifier-project")
    return parser.parse_args()


def train_and_log(args: argparse.Namespace) -> None:
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=wine.target,
    )

    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="wine-classifier",
    )

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1:.4f}")


def main() -> None:
    args = parse_args()
    under_mlflow_project = "MLFLOW_RUN_ID" in os.environ

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(TRACKING_URI)

    if under_mlflow_project:
        # `mlflow run` already created the run — just log into it.
        train_and_log(args)
    else:
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name="project-run"):
            train_and_log(args)


if __name__ == "__main__":
    main()
