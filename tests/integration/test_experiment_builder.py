"""Integration tests for the experiment builder prototype."""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from prototypes.experiment_builder.config import ExperimentSpec
from prototypes.experiment_builder.generator import build_project_zip


def make_dataset(task: str) -> pd.DataFrame:
    indices = np.arange(60)
    frame = pd.DataFrame(
        {
            "signal": indices.astype(float),
            "group": ["north", "south", "west"] * 20,
        },
    )
    if task == "regression":
        frame["outcome"] = indices.astype(float) * 1.7 + 4
    else:
        frame["outcome"] = np.where(indices % 2, "case", "control")
    return frame


def run_generated_project(spec: ExperimentSpec, frame: pd.DataFrame, tmp_path: Path) -> Path:
    archive = build_project_zip(spec, frame.to_csv(index=False).encode("utf-8"))
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        zipped.extractall(tmp_path)
    project = next(tmp_path.iterdir())
    environment = os.environ.copy()
    subprocess.run([sys.executable, "train.py"], cwd=project, check=True, env=environment, timeout=90)
    subprocess.run([sys.executable, "evaluate.py"], cwd=project, check=True, env=environment, timeout=90)
    return project


@pytest.mark.integration
def test_experiment_spec_yaml_round_trip():
    spec = ExperimentSpec(
        dataset_filename="study.csv",
        target="outcome",
        features=["signal"],
        task="regression",
        models=[{"name": "random_forest_regressor", "parameters": {"n_estimators": 20}}],
    )

    restored = ExperimentSpec.from_yaml(spec.to_yaml())

    assert restored == spec
    assert restored.metrics == ["mae", "rmse", "r2"]


@pytest.mark.integration
def test_spec_rejects_task_incompatible_model():
    with pytest.raises(ValidationError, match="does not support"):
        ExperimentSpec(
            dataset_filename="study.csv",
            target="outcome",
            features=["signal"],
            task="classification",
            models=[{"name": "linear_regression"}],
        )


@pytest.mark.integration
def test_dataset_validation_allows_mixed_features_and_stratification():
    frame = pd.DataFrame(
        {
            "signal": [0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "group": ["a", "b"] * 5,
            "label": ["low", "high"] * 5,
        },
    )
    spec = ExperimentSpec(
        dataset_filename="study.csv",
        target="label",
        features=["signal", "group"],
        task="classification",
        models=[{"name": "logistic_regression"}],
    )

    spec.validate_dataset(frame)


@pytest.mark.integration
def test_dataset_validation_rejects_missing_classes_for_split():
    frame = pd.DataFrame({"signal": range(8), "label": ["a"] * 7 + ["b"]})
    spec = ExperimentSpec(
        dataset_filename="study.csv",
        target="label",
        features=["signal"],
        task="classification",
        models=[{"name": "logistic_regression"}],
    )

    with pytest.raises(ValueError, match="Each classification class needs at least two rows"):
        spec.validate_dataset(frame)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("task", "model_name"),
    [
        ("regression", "random_forest_regressor"),
        ("classification", "logistic_regression"),
    ],
)
def test_generated_project_executes(task, model_name, tmp_path):
    frame = make_dataset(task)
    model_parameters = {"n_estimators": 8} if "random_forest" in model_name else {}
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"] if task == "classification" else None
    visualizations = (
        ["predicted_vs_actual", "residuals"]
        if task == "regression"
        else ["confusion_matrix", "roc_curve"]
    )
    spec = ExperimentSpec(
        project_name="Test Study",
        dataset_filename="study.csv",
        target="outcome",
        features=["signal", "group"],
        task=task,
        models=[{"name": model_name, "parameters": model_parameters}],
        metrics=metrics,
        visualizations=visualizations,
    )

    project = run_generated_project(spec, frame, tmp_path)

    scores = json.loads((project / "outputs/metrics" / f"{model_name}.json").read_text())
    assert scores
    assert all(np.isfinite(value) for value in scores.values())
    if task == "regression":
        assert (project / "outputs/figures" / f"{model_name}_residuals.png").exists()
    else:
        assert (project / "outputs/figures" / f"{model_name}_confusion_matrix.png").exists()


@pytest.mark.integration
def test_generated_project_zip_contains_configuration_and_upload():
    frame = make_dataset("regression")
    spec = ExperimentSpec(
        dataset_filename="study.csv",
        target="outcome",
        features=["signal", "group"],
        task="regression",
        models=[{"name": "linear_regression"}],
    )

    archive = build_project_zip(spec, frame.to_csv(index=False).encode("utf-8"))

    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = set(zipped.namelist())
        assert "tabular_experiment/experiment.yaml" in names
        assert "tabular_experiment/data/dataset.csv" in names
        assert "tabular_experiment/train.py" in names
        assert "tabular_experiment/README.md" in names
        assert b"outcome" in zipped.read("tabular_experiment/experiment.yaml")
        assert len(pd.read_csv(io.BytesIO(zipped.read("tabular_experiment/data/dataset.csv")))) == len(frame)