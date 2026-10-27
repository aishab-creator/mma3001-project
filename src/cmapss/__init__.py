"""Remaining-useful-life (RUL) prediction on NASA C-MAPSS turbofan data.

This package holds the reusable, tested building blocks behind the project
notebooks (see ``notebooks/09_training_cap_comparison.ipynb``). The functions
here are the same logic as the notebook cells, with explicit parameters in
place of notebook globals so that they can be unit tested.

Modules
-------
data
    Reading and validating the NASA text files; computing training RUL labels.
features
    Leakage-safe temporal features and operating-regime features.
metrics
    Training-target capping and the evaluation metrics (MAE, RMSE, R2, bias,
    near-failure and common-100 diagnostics).
validation
    Engine-separated train/validation splits and truncated validation snapshots.
models
    The Random Forest pipeline used throughout the project.

Conventions
-----------
* RUL is measured in engine operating **cycles**.
* Signed error is ``predicted - actual``; a positive value means the model
  overestimated the remaining life (the optimistic, risky direction).
* ``RUL_raw`` is the true (uncapped) RUL label; a training *cap* only changes
  the target a model is fitted to, never this column.
"""

__all__ = ["data", "features", "metrics", "validation", "models"]
