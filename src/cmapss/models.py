"""The Random Forest pipeline used for every cap and dataset."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_model(n_estimators=150, random_state=42):
    """Build the Random Forest regression pipeline.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees (default 150, as in the cap experiment).
    random_state : int, optional
        Seed, so repeated fits on the same data give identical models.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``SimpleImputer(median)`` followed by ``RandomForestRegressor``.
        Medians are learned from whatever data the pipeline is fitted on, so
        validation/test rows never influence them.

    Notes
    -----
    Settings are held constant across all cap candidates so that the
    comparison isolates the effect of the training target: unlimited depth,
    ``min_samples_leaf=4``, ``max_features="sqrt"``, all CPU cores.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=None,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
