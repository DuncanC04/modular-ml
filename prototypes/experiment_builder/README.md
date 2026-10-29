# ModularML Experiment Builder Prototype

An isolated Streamlit prototype for configuring a tabular ML experiment and exporting
an editable Python project ZIP. Project files are generated from deterministic local
templates and a validated YAML specification; no AI code generation or Cookiecutter
dependency is used. Training happens in the downloaded project, not inside the GUI.

## Run the GUI

From the repository root, install the prototype requirements into an environment that
can import this checkout of ModularML:

```powershell
python -m pip install -r prototypes/experiment_builder/requirements.txt
python -m streamlit run prototypes/experiment_builder/app.py
```

Upload a CSV, select the target and one or more features, choose a task and compatible
models, configure the seeded holdout and outputs, then download the ZIP. Existing
`experiment.yaml` files can be uploaded in the sidebar to restore settings; upload the
matching CSV as well.

## Current Scope

- Regression: Linear Regression and Random Forest Regressor.
- Classification: Logistic Regression and Random Forest Classifier, with stratified
  train/test splitting.
- Numeric features use median imputation. Categorical features use most-frequent
  imputation followed by one-hot encoding, fit on training rows only.
- Regression metrics: MAE, RMSE, and R2. Classification metrics: accuracy, precision,
  recall, and F1. ROC outputs are available for binary scikit-learn projects only.
- Each selected model is trained independently on the same seeded split.
- Generated projects use scikit-learn estimators and pandas-based data preparation.

Grouped splitting, cross-validation, broad hyperparameter search, and training in the
GUI are intentionally deferred. This prototype does not currently generate ModularML
projects and is not added to the ModularML wheel.

## Generated Project

The ZIP contains `experiment.yaml`, the uploaded CSV, model/training/evaluation scripts,
backend-specific requirements, a generated README, and output folders. Run
`python train.py` followed by `python evaluate.py` in the extracted project directory.
Metrics and figures are computed from held-out predictions, not placeholders.

Automated prototype tests are in `tests/integration/test_experiment_builder.py` and
can be run with `python -m pytest --noconftest tests/integration/test_experiment_builder.py`.