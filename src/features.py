"""Feature helpers. Week 1 needs only the ΔQ(V) curve; week 2 adds the rest.

``Qdlin`` is each cycle's discharge capacity interpolated onto a fixed voltage
grid of 1000 points from 3.6 V down to 2.0 V (see the paper's Methods; check
this against your data with ``len(cell["cycles"]["10"]["Qdlin"])``).
"""

from __future__ import annotations

import numpy as np

V_GRID = np.linspace(3.6, 2.0, 1000)


def delta_q(cell: dict, early: int = 10, late: int = 100) -> np.ndarray:
    """ΔQ_late-early(V): Qdlin at cycle ``late`` minus Qdlin at cycle ``early``.

    Negative values mean the cell delivers less charge at that voltage than it
    did early in life. The paper's key feature is the variance of this curve.
    """
    q_early = np.asarray(cell["cycles"][str(early)]["Qdlin"], dtype=float)
    q_late = np.asarray(cell["cycles"][str(late)]["Qdlin"], dtype=float)
    return q_late - q_early


def log_var_delta_q(cell: dict, early: int = 10, late: int = 100) -> float:
    """log10 of the variance of ΔQ(V), the paper's single strongest feature."""
    return float(np.log10(np.var(delta_q(cell, early, late))))


def cycle_key(cycle_number: int) -> str:
    """Cycle data is stored by position: key "0" is cycle 1, so cycle N is key N-1."""
    return str(cycle_number - 1)


def delta_q(cell: dict, early: int = 10, late: int = 100) -> np.ndarray:
    q_early = np.asarray(cell["cycles"][cycle_key(early)]["Qdlin"], dtype=float)
    q_late = np.asarray(cell["cycles"][cycle_key(late)]["Qdlin"], dtype=float)
    return q_late - q_early
