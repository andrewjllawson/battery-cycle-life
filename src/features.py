"""Early-life features for cycle-life prediction (after Severson et al. 2019).

Week 1: the ΔQ(V) curve. Week 2: the full 12-feature table (``features.csv``).

Indexing conventions (verified in week 1):
- Within-cycle data (``cell["cycles"]``) is keyed by position: key "0" is
  cycle 1 (nearly empty, as cycle 1 was excluded), so cycle N is key N-1.
- Summary arrays (``cell["summary"]``) are also by position: index 0 is
  cycle 1, so ``[1:100]`` is cycles 2-100 (same as the authors' code).

``Qdlin`` is each cycle's discharge capacity interpolated onto a fixed voltage
grid of 1000 points from 3.6 V down to 2.0 V.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

V_GRID = np.linspace(3.6, 2.0, 1000)

FEATURE_COLUMNS = [
    "log_var_dq",
    "log_abs_min_dq",
    "log_abs_skew_dq",
    "log_abs_kurt_dq",
    "qd_cycle2",
    "qd_max_minus_cycle2",
    "fade_slope",
    "fade_intercept",
    "charge_time_2to6",
    "temp_max",
    "ir_min",
    "ir_change_2to100",
]


# =============================================================================
# ΔQ(V)
# =============================================================================
def cycle_key(cycle_number: int) -> str:
    """Within-cycle data is stored by position: key "0" is cycle 1 (nearly
    empty, as cycle 1 was excluded), so cycle N is key N-1. Verified by
    matching summary QD[j] with max(cycles[str(j)]["Qd"])."""
    return str(cycle_number - 1)


def delta_q(cell: dict, early: int = 10, late: int = 100) -> np.ndarray:
    """ΔQ_late-early(V): Qdlin at cycle ``late`` minus Qdlin at cycle ``early``.

    ``early`` and ``late`` are true cycle numbers (as in the paper), not keys.
    Negative values mean the cell delivers less charge at that voltage than it
    did early in life. The paper's key feature is the variance of this curve.
    """
    q_early = np.asarray(cell["cycles"][cycle_key(early)]["Qdlin"], dtype=float)
    q_late = np.asarray(cell["cycles"][cycle_key(late)]["Qdlin"], dtype=float)
    return q_late - q_early


def log_var_delta_q(cell: dict, early: int = 10, late: int = 100) -> float:
    """log10 of the variance of ΔQ(V), the paper's single strongest feature."""
    return float(np.log10(np.var(delta_q(cell, early, late))))


# =============================================================================
# Full feature set (week 2)
# =============================================================================
def despike(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Rolling median: removes one-cycle logging spikes, keeps real trends.

    Without it, logging spikes in QD put qd_max_minus_cycle2 886 training
    standard deviations out for b1c18 (226 for b1c0), and linear regression
    extrapolated them to predicted lives of 1e19 and 3e7 cycles. Grouped CV on
    the training cells did not catch this, because the spiky cells were not all
    in training.
    """
    return pd.Series(x).rolling(window, center=True, min_periods=1).median().to_numpy()


def early_features(cell: dict) -> dict:
    """The 12 early-life features for one cell, using only cycles 1-100."""
    dq = delta_q(cell)  # Q(cycle 100) - Q(cycle 10) on the voltage grid
    s = cell["summary"]
    qd = despike(s["QD"][1:100])  # discharge capacity, cycles 2-100, spikes removed
    cycles = np.arange(2, 101)
    fade_slope, fade_intercept = np.polyfit(cycles, qd, 1)
    ir = despike(s["IR"][1:100])
    ir_nonzero = ir[ir > 0]  # a few IR readings are logged as 0
    return {
        # ΔQ(V) curve shape: the paper's strongest group
        "log_var_dq": np.log10(np.var(dq)),
        "log_abs_min_dq": np.log10(np.abs(dq.min())),
        "log_abs_skew_dq": np.log10(np.abs(skew(dq))),
        "log_abs_kurt_dq": np.log10(np.abs(kurtosis(dq))),
        # Capacity fade over cycles 2-100
        "qd_cycle2": qd[0],
        "qd_max_minus_cycle2": qd.max() - qd[0],
        "fade_slope": fade_slope,
        "fade_intercept": fade_intercept,
        # Charging, temperature, internal resistance
        "charge_time_2to6": np.mean(s["chargetime"][1:6]),
        "temp_max": np.max(s["Tmax"][1:100]),
        "ir_min": ir_nonzero.min() if ir_nonzero.size else np.nan,
        "ir_change_2to100": ir[-1] - ir[0],
    }


def build_feature_table(cells: dict, splits: dict) -> pd.DataFrame:
    """One row per cell: split, batch, charge policy, 12 features, cycle life.

    Same layout as the week 2 ``features.csv`` (index name "cell").
    """
    split_of = {k: name for name, ks in splits.items() for k in ks}
    X = pd.DataFrame.from_dict(
        {key: early_features(cell) for key, cell in cells.items()}, orient="index"
    )[FEATURE_COLUMNS]
    out = X.copy()
    out.insert(0, "split", [split_of.get(k, "unused") for k in out.index])
    out.insert(1, "batch", [k[:2] for k in out.index])
    out.insert(2, "charge_policy", [cells[k]["charge_policy"] for k in out.index])
    out["cycle_life"] = [cells[k]["cycle_life"] for k in out.index]
    out.index.name = "cell"
    return out


def load_feature_table(path: Path | str) -> pd.DataFrame:
    """Read ``features.csv`` back with the cell names as the index."""
    return pd.read_csv(path, index_col="cell")
