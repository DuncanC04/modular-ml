"""Validated, framework-neutral experiment configuration."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.model_selection import train_test_split

Task = Literal["regression", "classification"]

MODEL_TASKS: dict[str, set[str]] = {
    "linear_regression": {"regression"},
    "random_forest_regressor": {"regression"},
    "logistic_regression": {"classification"},
    "random_forest_classifier": {"classification"},
}

MODEL_PARAMETERS: dict[str, set[str]] = {
    "linear_regression": {"fit_intercept"},
    "random_forest_regressor": {"n_estimators", "max_depth", "min_samples_leaf"},
    "logistic_regression": {"C"},
    "random_forest_classifier": {"n_estimators", "max_depth", "min_samples_leaf"},
}

DEFAULT_METRICS: dict[str, list[str]] = {
    "regression": ["mae", "rmse", "r2"],
    "classification": ["accuracy", "precision", "recall", "f1"],
}


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> ModelChoice:
        allowed = MODEL_PARAMETERS.get(self.name)
        if allowed is None:
            raise ValueError(f"Unsupported model: {self.name}")
        unsupported = set(self.parameters).difference(allowed)
        if unsupported:
            raise ValueError(f"Unsupported parameters for {self.name}: {sorted(unsupported)}")
        for key, value in self.parameters.items():
            if key == "fit_intercept" and not isinstance(value, bool):
                raise ValueError("fit_intercept must be true or false")
            if key in {"n_estimators", "min_samples_leaf"} and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{key} must be a positive integer")
            if key == "max_depth" and value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError("max_depth must be a positive integer or null")
            if key == "C" and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError("C must be greater than zero")
        return self


class ExperimentSpec(BaseModel):
    """One GUI configuration, serialized into the generated project's YAML."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "tabular_experiment"
    dataset_filename: str
    target: str
    features: list[str] = Field(min_length=1)
    task: Task
    models: list[ModelChoice] = Field(min_length=1)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = 42
    metrics: list[str] | None = None
    visualizations: list[str] = Field(default_factory=list)
    generate_summary: bool = True

    @model_validator(mode="after")
    def validate_choices(self) -> ExperimentSpec:
        if not self.dataset_filename or "/" in self.dataset_filename or "\\" in self.dataset_filename:
            raise ValueError("dataset_filename must be a plain filename")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Feature selections must be unique")
        if self.target in self.features:
            raise ValueError("The target cannot also be selected as a feature")
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Model selections must be unique")
        for model in self.models:
            if model.name not in MODEL_TASKS:
                raise ValueError(f"Unsupported model: {model.name}")
            if self.task not in MODEL_TASKS[model.name]:
                raise ValueError(f"{model.name} does not support {self.task}")
        if self.test_size < 0.1 or self.test_size > 0.5:
            raise ValueError("test_size must be between 0.1 and 0.5")

        defaults = DEFAULT_METRICS[self.task]
        if self.metrics is None:
            self.metrics = defaults.copy()
        elif not self.metrics:
            raise ValueError("Select at least one evaluation metric")
        elif not set(self.metrics).issubset(defaults + (["roc_auc"] if self.task == "classification" else [])):
            raise ValueError(f"Unsupported {self.task} metric selection")
        if self.task == "regression":
            allowed_visualizations = {"predicted_vs_actual", "residuals"}
        else:
            allowed_visualizations = {"confusion_matrix", "roc_curve"}
        if not set(self.visualizations).issubset(allowed_visualizations):
            raise ValueError(f"Unsupported {self.task} visualization selection")
        return self

    def validate_dataset(self, frame: pd.DataFrame) -> None:
        """Check selected columns and whether the requested split is feasible."""
        missing = set([self.target, *self.features]).difference(frame.columns)
        if missing:
            raise ValueError(f"Selected columns are missing from the CSV: {sorted(missing)}")

        target = frame[self.target]
        if target.isna().any():
            raise ValueError("The target column contains missing values")
        if self.task == "regression":
            if not pd.api.types.is_numeric_dtype(target):
                raise ValueError("Regression requires a numeric target column")
        else:
            counts = target.value_counts(dropna=False)
            if len(counts) < 2:
                raise ValueError("Classification requires at least two target classes")
            if (counts < 2).any():
                raise ValueError("Each classification class needs at least two rows")
            if ("roc_auc" in self.metrics or "roc_curve" in self.visualizations) and len(counts) != 2:
                raise ValueError("ROC outputs require a binary classification target")

        for feature in self.features:
            dtype = frame[feature].dtype
            supported = (
                pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            )
            if not supported:
                raise ValueError(f"Feature '{feature}' has an unsupported data type")

        try:
            train_test_split(
                frame.index,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=target if self.task == "classification" else None,
            )
        except ValueError as exc:
            raise ValueError(f"The selected train/test split is not feasible: {exc}") from exc

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, content: str) -> ExperimentSpec:
        raw = yaml.safe_load(content)
        if not isinstance(raw, dict):
            raise ValueError("Experiment YAML must contain a mapping")
        return cls.model_validate(raw)