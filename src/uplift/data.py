"""Loading and validation of the raw Hillstrom dataset.

The Hillstrom (MineThatData) dataset is a randomized 3-arm email experiment
from 2008 with 64,000 customers. We use it as the basis for causal uplift
modeling: estimating which customers should be sent a promotional email,
given that sending is costly and some customers would purchase anyway.

This module is intentionally narrow: it loads the raw CSV, checks the file
hash, validates the schema, and returns a typed DataFrame. Feature
engineering, treatment binarization, and splitting live elsewhere.

The raw data is preserved as-is. Note that the source file contains the
spelling "Surburban" (sic) in the `zip_code` column; this is not corrected
here so that hashes and row counts match the original. Downstream
processing modules may rename it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve the project root from this file's location, so the module works
# regardless of the current working directory. __file__ -> src/uplift/data.py,
# so three .parent calls put us at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hillstrom.csv"

# SHA256 of the original CSV from Hillstrom's blog. If this ever changes,
# either the source file moved, the download is corrupt, or someone edited
# the raw data. Any of these should fail loudly.
EXPECTED_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

EXPECTED_ROW_COUNT = 64_000

EXPECTED_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]

# Explicit dtype map. pandas will infer mostly-correctly, but pinning dtypes
# makes the behavior reproducible across pandas versions and machines.
COLUMN_DTYPES = {
    "recency": "int16",  # months since last purchase, 1..12
    "history_segment": "category",
    "history": "float64",  # dollars
    "mens": "int8",  # 0/1 flag
    "womens": "int8",  # 0/1 flag
    "zip_code": "category",  # Urban / Surburban / Rural
    "newbie": "int8",  # 0/1 flag
    "channel": "category",  # Phone / Web / Multichannel
    "segment": "category",  # No E-Mail / Mens E-Mail / Womens E-Mail
    "visit": "int8",  # 0/1 outcome
    "conversion": "int8",  # 0/1 outcome
    "spend": "float64",  # dollars in the 2-week window
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw(path: Path | str | None = None, *, verify_hash: bool = True) -> pd.DataFrame:
    """Load the raw Hillstrom CSV with schema validation.

    Parameters
    ----------
    path
        Override for the raw data path. Defaults to ``data/raw/hillstrom.csv``
        relative to the project root.
    verify_hash
        If True, check that the file's SHA256 matches ``EXPECTED_SHA256``.
        Set False only when intentionally working with a modified file.

    Returns
    -------
    pd.DataFrame
        64,000 rows × 12 columns with the declared dtypes.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing. Run the download command in the README.
    ValueError
        If row count, columns, or hash don't match expectations.
    """
    csv_path = Path(path) if path is not None else RAW_DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. "
            "Download it with the Invoke-WebRequest command in README.md."
        )

    if (
        verify_hash
        and EXPECTED_SHA256 != "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
    ):
        actual = _sha256(csv_path)
        if actual.lower() != EXPECTED_SHA256.lower():
            raise ValueError(
                f"Hash mismatch for {csv_path}.\n"
                f"  expected: {EXPECTED_SHA256}\n"
                f"  actual:   {actual}\n"
                "The raw file has changed or is corrupt."
            )

    df = pd.read_csv(csv_path, dtype=COLUMN_DTYPES)

    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Internal: enforce the expected shape and column set."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Column mismatch.\n  expected: {EXPECTED_COLUMNS}\n  got: {list(df.columns)}"
        )
    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Row count mismatch. Expected {EXPECTED_ROW_COUNT}, got {len(df)}.")


def _sha256(path: Path) -> str:
    """Compute the SHA256 of a file by streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
