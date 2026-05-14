"""Train/val/test splits for the uplift pipeline.

Stratified by binarized treatment so every split has the same treated:control
ratio. This isolates split variance from treatment-distribution variance,
which makes cross-model comparisons cleaner downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from uplift.data import PROJECT_ROOT, load_raw
from uplift.treatment import make_binary_treatment

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def make_splits(
    df: pd.DataFrame | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits stratified by binarized treatment.

    Parameters
    ----------
    df
        DataFrame to split. Loads raw Hillstrom if None.
    train_frac, val_frac, test_frac
        Must sum to 1.0.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict with keys 'train', 'val', 'test' mapping to DataFrames.
    """
    if df is None:
        df = load_raw()

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    T = make_binary_treatment(df)

    # Two-step split: (train) vs (val + test), then split (val + test) into val and test.
    df_train, df_rest = train_test_split(
        df,
        test_size=val_frac + test_frac,
        stratify=T,
        random_state=seed,
    )
    T_rest = make_binary_treatment(df_rest)
    df_val, df_test = train_test_split(
        df_rest,
        test_size=test_frac / (val_frac + test_frac),
        stratify=T_rest,
        random_state=seed,
    )

    return {"train": df_train, "val": df_val, "test": df_test}


def save_splits(
    splits: dict[str, pd.DataFrame],
    out_dir: Path | str | None = None,
) -> None:
    """Persist splits to parquet in data/processed/."""
    out = Path(out_dir) if out_dir is not None else PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(out / f"{name}.parquet")


def load_splits(in_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load previously-saved splits from parquet."""
    src = Path(in_dir) if in_dir is not None else PROCESSED_DIR
    return {name: pd.read_parquet(src / f"{name}.parquet") for name in ("train", "val", "test")}
