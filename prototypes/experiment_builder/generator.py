"""Render deterministic experiment projects and package them as ZIP archives."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from prototypes.experiment_builder.config import ExperimentSpec

BASE_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "templates" / "base"
SCIKIT_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "templates" / "scikit_learn"
REQUIREMENT_LINES = ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib", "pyyaml")


def project_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_-").lower()
    return slug or "tabular_experiment"


def build_project_zip(spec: ExperimentSpec, dataset: bytes) -> bytes:
    """Return a ZIP containing a complete, configured project."""
    frame = pd.read_csv(io.BytesIO(dataset))
    spec.validate_dataset(frame)
    env = Environment(
        loader=FileSystemLoader([str(BASE_TEMPLATE_DIRECTORY), str(SCIKIT_TEMPLATE_DIRECTORY)]),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    context = {
        "spec": spec.model_dump(mode="json"),
        "project_name": project_slug(spec.project_name),
        "requirement_lines": REQUIREMENT_LINES,
        "models": [model.model_dump(mode="json") for model in spec.models],
        "features": spec.features,
        "target": spec.target,
        "task": spec.task,
        "metrics": spec.metrics,
        "visualizations": spec.visualizations,
        "generate_summary": spec.generate_summary,
        "has_numeric_features": any(
            pd.api.types.is_numeric_dtype(frame[column].dtype) for column in spec.features
        ),
        "has_categorical_features": any(
            not pd.api.types.is_numeric_dtype(frame[column].dtype) for column in spec.features
        ),
    }

    entries: dict[str, bytes] = {
        "experiment.yaml": spec.to_yaml().encode("utf-8"),
        "data/dataset.csv": dataset,
        "requirements.txt": env.get_template("requirements.txt.j2").render(**context).encode("utf-8"),
        "README.md": env.get_template("README.md.j2").render(**context).encode("utf-8"),
    }
    for template_name in ("models.py.j2", "train.py.j2", "evaluate.py.j2"):
        output_name = template_name.removesuffix(".j2")
        entries[output_name] = env.get_template(template_name).render(**context).encode("utf-8")

    for directory in (
        "outputs/figures/.gitkeep",
        "outputs/metrics/.gitkeep",
        "outputs/models/.gitkeep",
        "outputs/reports/.gitkeep",
    ):
        entries[directory] = b""

    archive = io.BytesIO()
    root = project_slug(spec.project_name)
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for relative_path, content in entries.items():
            zipped.writestr(f"{root}/{relative_path}", content)
    return archive.getvalue()