"""Load the Severson et al. (2019) battery cycle-life dataset.

Two stages:

1. ``convert_batches`` reads the three MATLAB v7.3 batch files (HDF5 under the
   hood) and writes one Python pickle per batch. This replaces the authors'
   ``BuildPkl_Batch*.ipynb`` notebooks, which use ``dataset.value`` - an h5py
   feature removed in h5py 3.0. Here we use ``dataset[()]`` instead.

2. ``load_cells`` reads those pickles and applies the authors' cleaning from
   their ``Load Data.ipynb``: drop cells that never reach 80% capacity, splice
   the five batch-1 cells that continued in batch 2, and drop noisy batch-3
   channels. The result is the 124 cells used in the paper.

Usage from the repo root:

    python -m src.load_data --raw data/raw --out data/processed

Source of the cleaning rules:
https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# --- The three main batch files from https://data.matr.io/1/ -----------------
BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

SUMMARY_FIELDS = {  # pickle key -> field name in the .mat struct
    "IR": "IR",
    "QC": "QCharge",
    "QD": "QDischarge",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "Tmax": "Tmax",
    "chargetime": "chargetime",
    "cycle": "cycle",
}

CYCLE_FIELDS = {  # pickle key -> field name in the .mat struct
    "I": "I",
    "Qc": "Qc",
    "Qd": "Qd",
    "Qdlin": "Qdlin",
    "T": "T",
    "Tdlin": "Tdlin",
    "V": "V",
    "dQdV": "discharge_dQdV",
    "t": "t",
}

# --- Cleaning rules, copied from the authors' Load Data notebook -------------
B1_NOT_REACHING_80PCT = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
# Five batch-1 cells kept cycling in batch 2 under new keys.
B1_CONTINUED = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
B2_CONTINUATIONS = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
B2_EXTRA_LIFE = [662, 981, 1060, 208, 482]
B3_NOISY = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]

EXPECTED_COUNTS = {"b1": 41, "b2": 43, "b3": 40}  # after cleaning, 124 total


# =============================================================================
# Stage 1: .mat -> .pkl
# =============================================================================
def _read(f, ref) -> np.ndarray:
    """Dereference an HDF5 object reference and return its data as an array."""
    return np.asarray(f[ref][()])


def convert_mat(mat_path: Path, prefix: str, max_cycle: int | None = None) -> dict:
    """Convert one batch .mat file into the authors' nested-dict format.

    ``max_cycle`` keeps only within-cycle data for cycles 0..max_cycle, which
    cuts memory and pickle size a lot. Per-cycle summary data is always kept
    in full. Use ``max_cycle=None`` to keep every cycle.
    """
    import h5py  # imported here so the rest of the module works without it

    cells = {}
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        n_cells = batch["summary"].shape[0]
        for i in range(n_cells):
            cycle_life = float(np.ravel(_read(f, batch["cycle_life"][i, 0]))[0])
            # MATLAB stores char arrays as uint16; take every other byte.
            policy = _read(f, batch["policy_readable"][i, 0]).tobytes()[::2].decode()

            summ_grp = f[batch["summary"][i, 0]]
            summary = {
                key: np.hstack(summ_grp[field][0, :].tolist())
                for key, field in SUMMARY_FIELDS.items()
            }

            cyc_grp = f[batch["cycles"][i, 0]]
            n_cycles = cyc_grp["I"].shape[0]
            last = n_cycles if max_cycle is None else min(n_cycles, max_cycle + 1)
            cycles = {}
            for j in range(last):
                cycles[str(j)] = {
                    key: np.hstack(_read(f, cyc_grp[field][j, 0]))
                    for key, field in CYCLE_FIELDS.items()
                }

            cells[f"{prefix}c{i}"] = {
                "cycle_life": cycle_life,
                "charge_policy": policy,
                "summary": summary,
                "cycles": cycles,
                "cycles_truncated": max_cycle is not None,
            }
            print(f"  {prefix}c{i}: life={cycle_life:.0f}, {last} cycles kept")
    return cells


def convert_batches(raw_dir: Path, out_dir: Path, max_cycle: int | None = 100) -> None:
    """Convert all three batch files to data/processed/batch{1,2,3}.pkl."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for prefix, filename in BATCH_FILES.items():
        mat_path = raw_dir / filename
        if not mat_path.exists():
            raise FileNotFoundError(
                f"Missing {mat_path}. Download the three main batch files from "
                "https://data.matr.io/1/ (not the low-rate Figure 4 files)."
            )
        print(f"Converting {filename} ...")
        cells = convert_mat(mat_path, prefix, max_cycle=max_cycle)
        out_path = out_dir / f"batch{prefix[1]}.pkl"
        with open(out_path, "wb") as fp:
            pickle.dump(cells, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(cells)} cells to {out_path}")


# =============================================================================
# Stage 2: .pkl -> cleaned 124 cells
# =============================================================================
def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def load_cells(processed_dir: Path) -> dict:
    """Load the three pickles and apply the authors' cleaning.

    Returns an ordered dict of 124 cells: batch 1, then batch 2, then batch 3.
    The order matters, because the paper's train/test split is index-based.
    """
    processed_dir = Path(processed_dir)
    b1 = _load_pickle(processed_dir / "batch1.pkl")
    b2 = _load_pickle(processed_dir / "batch2.pkl")
    b3 = _load_pickle(processed_dir / "batch3.pkl")

    for key in B1_NOT_REACHING_80PCT:
        del b1[key]

    # Splice the batch-2 continuations onto their batch-1 cells.
    for b1_key, b2_key, extra in zip(B1_CONTINUED, B2_CONTINUATIONS, B2_EXTRA_LIFE):
        cell, cont = b1[b1_key], b2[b2_key]
        cell["cycle_life"] = float(np.ravel(cell["cycle_life"])[0]) + extra
        n_prev = len(cell["summary"]["cycle"])
        for field, values in cell["summary"].items():
            add = cont["summary"][field]
            if field == "cycle":
                add = add + n_prev
            cell["summary"][field] = np.hstack((values, add))
        # Within-cycle data: only splice when the batch-1 cell was kept whole,
        # otherwise the cycle numbering would be wrong. Features only use
        # early cycles, so a truncated conversion loses nothing we need.
        if not cell.get("cycles_truncated", False):
            n_cyc = len(cell["cycles"])
            for j, cyc_key in enumerate(cont["cycles"]):
                cell["cycles"][str(n_cyc + j)] = cont["cycles"][cyc_key]
        del b2[b2_key]

    for key in B3_NOISY:
        del b3[key]

    for name, batch in (("b1", b1), ("b2", b2), ("b3", b3)):
        if len(batch) != EXPECTED_COUNTS[name]:
            print(
                f"Warning: {name} has {len(batch)} cells after cleaning, "
                f"expected {EXPECTED_COUNTS[name]}."
            )

    cells = {**b1, **b2, **b3}
    for cell in cells.values():
        cell["cycle_life"] = float(np.ravel(cell["cycle_life"])[0])
    return cells


def split_keys(cells: dict) -> dict:
    """The paper's train / primary test / secondary test split (41 / 43 / 40).

    Copied from the authors' Load Data notebook: alternate batch 1+2 cells
    between test (even indices, plus index 83) and train (odd indices), and
    hold out all of batch 3 as the secondary test set.
    """
    keys = list(cells)
    n12 = sum(k.startswith(("b1", "b2")) for k in keys)
    test_idx = np.hstack((np.arange(0, n12, 2), 83))
    train_idx = np.arange(1, n12 - 1, 2)
    secondary_idx = np.arange(n12, len(keys))
    return {
        "train": [keys[i] for i in train_idx],
        "test": [keys[i] for i in test_idx],
        "secondary_test": [keys[i] for i in secondary_idx],
    }


def cell_table(cells: dict) -> pd.DataFrame:
    """One row per cell: batch, charge policy, cycle life and split."""
    split_of = {k: s for s, ks in split_keys(cells).items() for k in ks}
    rows = [
        {
            "cell": key,
            "batch": key[:2],
            "charge_policy": cell["charge_policy"],
            "cycle_life": cell["cycle_life"],
            "split": split_of.get(key, "unused"),
        }
        for key, cell in cells.items()
    ]
    return pd.DataFrame(rows).set_index("cell")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default="data/raw", help="folder with the .mat files")
    parser.add_argument("--out", default="data/processed", help="folder for the .pkl files")
    parser.add_argument(
        "--max-cycle",
        type=int,
        default=100,
        help="keep within-cycle data up to this cycle (default 100); -1 keeps all",
    )
    args = parser.parse_args()
    max_cycle = None if args.max_cycle < 0 else args.max_cycle
    convert_batches(Path(args.raw), Path(args.out), max_cycle=max_cycle)

    cells = load_cells(Path(args.out))
    table = cell_table(cells)
    table.to_csv(Path(args.out) / "cells.csv")
    print(f"\n{len(cells)} cells after cleaning (expected 124)")
    print(table["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
