"""Shared evaluation helpers for the modelling notebooks (week 3 onwards).

Conventions used everywhere:
- Target is log10(cycle life). Errors are reported in cycles as mean absolute
  percentage error, because a 100-cycle miss matters more for a 300-cycle cell
  than a 2,000-cycle one.
- Signed bias is mean (predicted - true) / true, in %. Positive = over-predicted life.
- Cross-validation uses GroupKFold grouped by charge policy, so cells on the same
  policy never sit on both sides of a fold.
- The final test sets are scored once per model, after model choice is made by CV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score

SPLITS = ["train", "test", "secondary_test"]
BATCHES = ["b1", "b2", "b3"]


def pct_error(y_true_log, y_pred_log) -> np.ndarray:
    """Absolute % error in cycles, per cell."""
    true, pred = 10 ** np.asarray(y_true_log), 10 ** np.asarray(y_pred_log)
    return 100 * np.abs(pred - true) / true


def signed_pct(y_true_log, y_pred_log) -> np.ndarray:
    """Signed % error in cycles, per cell (+ = over-predicted life)."""
    true, pred = 10 ** np.asarray(y_true_log), 10 ** np.asarray(y_pred_log)
    return 100 * (pred - true) / true


def grouped_cv(n_splits: int = 5) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)


def cv_log_mae(model, X, y, groups, n_splits: int = 5) -> tuple[float, float]:
    """Mean and std of grouped-CV MAE in log10 units (training cells only)."""
    scores = cross_val_score(clone(model), X, y, groups=groups, cv=grouped_cv(n_splits),
                             scoring="neg_mean_absolute_error")
    return float(-scores.mean()), float(scores.std())


def evaluate(name: str, model, data: pd.DataFrame, features: list[str],
             note: str = "", cv: bool = True) -> tuple[dict, object]:
    """Fit on the training split, score every split, return a metrics row.

    ``data`` is the features table (index = cell) with columns split, batch,
    charge_policy, cycle_life and the feature columns.
    """
    y = np.log10(data["cycle_life"])
    tr = data["split"] == "train"
    X_tr, y_tr = data.loc[tr, features], y[tr]

    fitted = clone(model).fit(X_tr, y_tr)
    pred = pd.Series(fitted.predict(data[features]), index=data.index)

    row = {"model": name, "n_features": len(features), "note": note}
    if cv:
        row["cv_log_mae"], row["cv_log_mae_std"] = cv_log_mae(
            model, X_tr, y_tr, data.loc[tr, "charge_policy"])
    for split in SPLITS:
        m = data["split"] == split
        row[f"{split}_pct"] = float(pct_error(y[m], pred[m]).mean())
    eval_cells = data["split"] != "train"
    for batch in BATCHES:
        m = eval_cells & (data["batch"] == batch)
        if m.any():
            row[f"{batch}_bias_pct"] = float(signed_pct(y[m], pred[m]).mean())
    return row, fitted


def save_metrics(rows: list[dict], path: Path | str) -> pd.DataFrame:
    """Merge rows into metrics.csv and return the whole table.

    A row whose model name is already in the file replaces the old row, so
    re-running a notebook never duplicates; rows from other notebooks are kept
    (03 writes the linear models, 04 adds the tree models).
    """
    path = Path(path)
    table = pd.DataFrame(rows).set_index("model")
    if path.exists():
        old = pd.read_csv(path, index_col="model")
        table = pd.concat([old.drop(index=table.index, errors="ignore"), table])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return table
