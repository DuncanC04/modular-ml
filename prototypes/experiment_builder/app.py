"""Streamlit interface for configuring and exporting tabular experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from prototypes.experiment_builder.config import DEFAULT_METRICS, ExperimentSpec
from prototypes.experiment_builder.generator import build_project_zip, project_slug

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest_regressor": "Random Forest Regressor",
    "logistic_regression": "Logistic Regression",
    "random_forest_classifier": "Random Forest Classifier",
}
MODEL_OPTIONS = {
    "regression": ["linear_regression", "random_forest_regressor"],
    "classification": ["logistic_regression", "random_forest_classifier"],
}
BRAND_BANNER_PATH = Path(__file__).resolve().parents[2] / "docs" / "_static" / "logos" / "modularml_logo_banner.png"


def section_heading(number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="section-number">{number}</span>
            <div>
                <h2>{title}</h2>
                <p>{description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def restore_yaml(uploaded_yaml) -> None:
    if uploaded_yaml is None:
        return
    contents = uploaded_yaml.getvalue()
    content_hash = hashlib.sha256(contents).hexdigest()
    if st.session_state.get("_restored_yaml_hash") == content_hash:
        return
    try:
        restored = ExperimentSpec.from_yaml(contents.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, ValidationError) as exc:
        st.sidebar.error(f"Could not restore YAML: {exc}")
        return

    for key, value in {
        "project_name": restored.project_name,
        "task": restored.task,
        "target": restored.target,
        "features": restored.features,
        "models": [model.name for model in restored.models],
        "test_size": restored.test_size,
        "random_seed": restored.random_seed,
        "metrics": restored.metrics,
        "visualizations": restored.visualizations,
    }.items():
        st.session_state[key] = value
    st.session_state["_restored_yaml_hash"] = content_hash
    st.sidebar.success("Experiment settings restored. Upload the matching dataset to continue.")


def model_parameters(model_name: str) -> dict:
    if model_name == "linear_regression":
        return {"fit_intercept": st.checkbox("Fit intercept", value=True, key="fit_intercept")}
    if model_name == "logistic_regression":
        return {"C": st.number_input("Regularization strength C", 0.01, 100.0, 1.0, 0.1, key="logistic_c")}
    parameters = {
        "n_estimators": st.slider("Number of trees", 10, 300, 100, 10, key=f"{model_name}_trees"),
    }
    max_depth = st.selectbox(
        "Maximum tree depth",
        options=[None, 5, 10, 20, 40],
        format_func=lambda value: "Unlimited" if value is None else str(value),
        key=f"{model_name}_depth",
    )
    if max_depth is not None:
        parameters["max_depth"] = max_depth
    parameters["min_samples_leaf"] = st.slider(
        "Minimum samples per leaf",
        1,
        10,
        1,
        key=f"{model_name}_leaf",
    )
    return parameters


def main() -> None:
    st.set_page_config(page_title="ModularML Experiment Builder", page_icon="M", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --ink: #111318;
            --muted: #596273;
            --blue: #075b9a;
            --blue-soft: #edf5fc;
            --purple: #6600a8;
            --purple-soft: #f5effa;
            --line: #dce1e8;
            --primary-color: #075b9a;
            --text-color: #111318;
            --background-color: #ffffff;
            --secondary-background-color: #f8faff;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #ffffff !important;
            color: var(--ink) !important;
            font-family: "Aptos", "Segoe UI Variable", sans-serif;
        }
        [data-testid="stHeader"] { background: #ffffff; }
        [data-testid="stToolbar"] { right: 1rem; }
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1160px;
            padding: 2rem 2.4rem 4rem;
        }
        [data-testid="stSidebar"] {
            background: #f8faff !important;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color: var(--ink); }
        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
        h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; letter-spacing: 0; }
        [data-testid="stMarkdownContainer"] .brand-title {
            margin: .15rem 0 0;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.9rem;
            font-weight: 600;
            line-height: 1.2;
        }
        .brand-subtitle { margin: .35rem 0 0; color: var(--muted); font-size: .95rem; }
        .brand-kicker { color: var(--purple); font-weight: 700; font-size: .72rem; text-transform: uppercase; }
        .section-heading {
            display: flex;
            align-items: flex-start;
            gap: .85rem;
            border-top: 1px solid var(--line);
            padding-top: 1.35rem;
            margin: 2.2rem 0 1.15rem;
        }
        .section-number {
            color: var(--purple);
            font: 600 .95rem Georgia, "Times New Roman", serif;
            padding-top: .18rem;
        }
        .section-heading h2 { margin: 0; font-size: 1.35rem; }
        .section-heading p { color: var(--muted); margin: .25rem 0 0; font-size: .88rem; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-top: 2px solid var(--blue);
            padding: .7rem .85rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: var(--ink) !important; }
        [data-testid="stFileUploaderDropzone"] {
            background: #fbfcff;
            border: 1px dashed #8a65ae;
            border-radius: 4px;
        }
        [data-testid="stFileUploaderDropzone"] button {
            color: var(--blue);
            background: #ffffff;
            border: 1px solid var(--blue);
        }
        [data-testid="stFileUploaderDropzone"] small { color: var(--muted); }
        input, textarea, [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #b8c0cc !important;
            border-radius: 5px !important;
            box-shadow: none !important;
        }
        [data-baseweb="select"] > div:focus-within,
        input:focus, textarea:focus, [role="combobox"]:focus {
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 2px rgba(7, 91, 154, .15) !important;
            outline-color: var(--purple) !important;
        }
        [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"],
        [data-testid="stMultiSelect"] .react-aria-ComboBox > [role="group"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #b8c0cc !important;
            border-radius: 5px !important;
            box-shadow: none !important;
        }
        [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"]:focus-within,
        [data-testid="stMultiSelect"] .react-aria-ComboBox > [role="group"]:focus-within {
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 2px rgba(7, 91, 154, .15) !important;
        }
        [data-testid="stMultiSelect"] [data-tag] {
            background: var(--blue) !important;
            color: #ffffff !important;
            border: 1px solid var(--blue) !important;
            border-radius: 4px !important;
        }
        [data-testid="stMultiSelect"] [data-tag] [data-testid="stMarkdownContainer"],
        [data-testid="stMultiSelect"] [data-tag] button,
        [data-testid="stMultiSelect"] [data-tag] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stMultiSelect"] [data-tag] button:hover {
            background: var(--purple) !important;
        }
        [data-baseweb="popover"], [data-baseweb="popover"] > div,
        [data-baseweb="menu"], [role="listbox"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: var(--line) !important;
        }
        [role="option"] {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        [role="option"]:hover, [role="option"][aria-selected="true"] {
            background: var(--blue-soft) !important;
            color: var(--blue) !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background: var(--blue) !important;
            border: 1px solid var(--blue) !important;
            color: #ffffff !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] button:hover {
            background: var(--purple) !important;
        }
        [data-testid="stSegmentedControl"] button,
        [data-testid="stSegmentedControl"] [role="radio"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: var(--line) !important;
        }
        [role="radiogroup"] [role="radio"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: var(--line) !important;
        }
        [data-testid="stSegmentedControl"] button[aria-pressed="true"],
        [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"],
        [role="radiogroup"] [role="radio"][aria-checked="true"],
        [role="radiogroup"] [role="radio"][data-selected="true"] {
            background: var(--blue-soft) !important;
            color: var(--blue) !important;
            border-color: var(--blue) !important;
            box-shadow: inset 0 -2px 0 var(--purple) !important;
        }
        [data-testid="stSegmentedControl"] button:hover,
        [data-testid="stSegmentedControl"] [role="radio"]:hover,
        [role="radiogroup"] [role="radio"]:hover {
            border-color: var(--purple) !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: var(--blue) !important;
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 2px #ffffff, 0 0 0 4px var(--blue-soft) !important;
        }
        [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div:first-child {
            background: var(--blue) !important;
        }
        [data-testid="stSlider"] [data-testid="stSliderThumbValue"],
        [data-testid="stSlider"] [data-testid="stSliderThumbValue"] * {
            color: var(--blue) !important;
        }
        [data-testid="stSlider"] [data-testid="stSliderTickBar"],
        [data-testid="stSlider"] [data-testid="stSliderTickBar"] * {
            color: var(--muted) !important;
        }
        input[type="range"], input[type="checkbox"], input[type="radio"] {
            accent-color: var(--blue) !important;
        }
        button:focus-visible, [role="button"]:focus-visible {
            outline: 2px solid var(--purple) !important;
            outline-offset: 2px;
        }
        [data-testid="stCheckbox"] label, [data-testid="stRadio"] label,
        [data-testid="stMultiSelect"] label, [data-testid="stSelectbox"] label { color: var(--ink) !important; }
        div.stButton > button[kind="primary"],
        div.stDownloadButton > button {
            background: var(--blue) !important;
            color: #ffffff !important;
            border: 1px solid var(--blue) !important;
            border-radius: 4px;
            font-weight: 650;
        }
        div.stButton > button[kind="primary"]:hover,
        div.stDownloadButton > button:hover { background: var(--purple) !important; border-color: var(--purple) !important; }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 4px; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); }
        [data-testid="stAlert"] { background: var(--blue-soft); color: var(--ink); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.image(str(BRAND_BANNER_PATH), width=980)
    st.markdown(
        "<h1 class='brand-title'>Experiment Builder</h1><p class='brand-subtitle'>Configure a tabular study and export an editable Python project.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Restore settings")
        yaml_upload = st.file_uploader("Upload experiment.yaml", type=["yaml", "yml"], key="yaml_upload")
        restore_yaml(yaml_upload)
        st.divider()
        st.caption("The app generates project files only. Model training runs locally from the downloaded project.")

    uploaded_csv = st.file_uploader("Dataset CSV", type=["csv"], key="csv_upload")
    if uploaded_csv is None:
        st.info("Upload a CSV to inspect its columns and configure an experiment.")
        st.stop()

    try:
        frame = pd.read_csv(uploaded_csv)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    if frame.empty or not len(frame.columns):
        st.error("The uploaded CSV has no rows or columns.")
        st.stop()

    st.session_state.setdefault("project_name", Path(uploaded_csv.name).stem)
    st.session_state.setdefault("task", "regression")
    st.session_state.setdefault("test_size", 0.2)
    st.session_state.setdefault("random_seed", 42)
    st.session_state.setdefault("metrics", DEFAULT_METRICS[st.session_state["task"]])
    st.session_state.setdefault("visualizations", ["predicted_vs_actual", "residuals"])

    overview = st.columns(3)
    overview[0].metric("Rows", f"{len(frame):,}")
    overview[1].metric("Columns", len(frame.columns))
    overview[2].metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    with st.expander("Preview and column types", expanded=True):
        st.dataframe(frame.head(8), use_container_width=True)
        types = pd.DataFrame(
            {
                "Column": frame.columns,
                "Type": [str(dtype) for dtype in frame.dtypes],
                "Missing": [int(frame[column].isna().sum()) for column in frame.columns],
            },
        )
        st.dataframe(types, hide_index=True, use_container_width=True)

    section_heading("01", "Experiment design", "Choose the prediction task, response, predictors, and estimators.")
    project_name = st.text_input("Project name", key="project_name")
    task = st.segmented_control("Task", options=["regression", "classification"], key="task")
    if task is None:
        task = "regression"
    previous_task = st.session_state.get("_last_task", task)
    if previous_task != task:
        st.session_state["metrics"] = DEFAULT_METRICS[task].copy()
        st.session_state["visualizations"] = (
            ["predicted_vs_actual", "residuals"]
            if task == "regression"
            else ["confusion_matrix"]
        )
    st.session_state["_last_task"] = task
    available_targets = list(frame.columns)
    if st.session_state.get("target") not in available_targets:
        st.session_state["target"] = available_targets[-1]
    target = st.selectbox("Target variable", available_targets, key="target")
    available_features = [column for column in frame.columns if column != target]
    saved_features = st.session_state.get("features")
    if not saved_features or not set(saved_features).issubset(available_features):
        st.session_state["features"] = available_features
    features = st.multiselect("Input features", available_features, key="features")

    model_names = MODEL_OPTIONS[task]
    saved_models = [name for name in st.session_state.get("models", []) if name in model_names]
    st.session_state["models"] = saved_models or model_names[:1]
    selected_models = st.multiselect(
        "Models (each is trained independently on the same split)",
        options=model_names,
        format_func=lambda name: MODEL_LABELS[name],
        key="models",
    )
    model_configs = []
    if selected_models:
        with st.expander("Model settings", expanded=False):
            for model_name in selected_models:
                st.markdown(f"**{MODEL_LABELS[model_name]}**")
                model_configs.append({"name": model_name, "parameters": model_parameters(model_name)})

    section_heading("02", "Validation plan", "Set a reproducible holdout and select the outputs to generate.")
    split_columns = st.columns(2)
    test_size = split_columns[0].slider("Test set proportion", 0.1, 0.5, 0.2, 0.05, key="test_size")
    random_seed = split_columns[1].number_input("Random seed", min_value=0, max_value=2_147_483_647, step=1, key="random_seed")
    metric_options = DEFAULT_METRICS[task].copy()
    binary_target = task == "classification" and frame[target].nunique(dropna=True) == 2
    st.caption("Generated projects use scikit-learn with pandas-based data preparation.")
    if task == "classification" and binary_target:
        metric_options.append("roc_auc")
    st.session_state["metrics"] = [
        metric for metric in st.session_state.get("metrics", []) if metric in metric_options
    ]
    selected_metrics = st.multiselect(
        "Evaluation metrics",
        options=metric_options,
        key="metrics",
    )
    if task == "regression":
        visualization_options = ["predicted_vs_actual", "residuals"]
        visualization_labels = {"predicted_vs_actual": "Predicted vs. actual", "residuals": "Residual distribution"}
    else:
        visualization_options = ["confusion_matrix"]
        visualization_labels = {"confusion_matrix": "Confusion matrix"}
        if binary_target:
            visualization_options.append("roc_curve")
            visualization_labels["roc_curve"] = "ROC curve"
    st.session_state["visualizations"] = [
        value for value in st.session_state.get("visualizations", []) if value in visualization_options
    ]
    selected_visualizations = st.multiselect(
        "Visualizations",
        options=visualization_options,
        format_func=lambda value: visualization_labels[value],
        key="visualizations",
    )
    generate_summary = st.checkbox("Generate experiment summary in the README", value=True)

    spec = None
    if features and selected_models:
        try:
            spec = ExperimentSpec(
                project_name=project_slug(project_name),
                dataset_filename=uploaded_csv.name,
                target=target,
                features=features,
                task=task,
                models=model_configs,
                test_size=float(test_size),
                random_seed=int(random_seed),
                metrics=selected_metrics,
                visualizations=selected_visualizations,
                generate_summary=generate_summary,
            )
            spec.validate_dataset(frame)
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))
            spec = None
    else:
        st.warning("Select at least one feature and one model to continue.")

    section_heading("03", "Review and export", "Check the specification, then package the runnable project.")
    if spec is not None:
        st.dataframe(
            pd.DataFrame(
                [
                    ("Task", spec.task),
                    ("Target", spec.target),
                    ("Features", ", ".join(spec.features)),
                    ("Models", ", ".join(MODEL_LABELS[model.name] for model in spec.models)),
                    ("Framework", "scikit-learn"),
                    ("Train/test", f"{1 - spec.test_size:.0%}/{spec.test_size:.0%}"),
                    ("Seed", spec.random_seed),
                    ("Metrics", ", ".join(spec.metrics or [])),
                ],
                columns=["Setting", "Selection"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Experiment YAML"):
            st.code(spec.to_yaml(), language="yaml")
        try:
            project_zip = build_project_zip(spec, uploaded_csv.getvalue())
        except (ValidationError, ValueError, KeyError) as exc:
            st.error(f"Project generation failed: {exc}")
        else:
            st.download_button(
                "Download project ZIP",
                data=project_zip,
                file_name=f"{project_slug(spec.project_name)}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()