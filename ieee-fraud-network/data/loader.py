"""
data/loader.py

Loads and joins the IEEE-CIS Fraud Detection transaction + identity tables,
and produces a TIME-BASED train/test split.

IMPORTANT: TransactionDT is a timedelta from an arbitrary reference point,
NOT a real calendar timestamp. It's still monotonically increasing with real
transaction order though, so it's valid to sort/split on - we just can't map
it to an actual date.

Why time-based, not random:
The competition documentation explicitly notes that train and test are
separated by time, with a gap between them. Transactions are not i.i.d.
across the collection window - fraud patterns, issuers, and even feature
distributions (e.g. the C1-C14 running counts) drift over time. A random
split lets rows that are "in the future" relative to other training rows
leak into training (e.g. near-duplicate transactions on the same card
landing on both sides of the split), which inflates offline metrics in a
way that won't survive a real, forward-only deployment. Splitting on
TransactionDT keeps all training data strictly earlier than all evaluation
data, mirroring how the model is actually used in production.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"


def load_raw(
    transaction_path: Path = TRANSACTION_FILE,
    identity_path: Path = IDENTITY_FILE,
) -> pd.DataFrame:
    """
    Load and left-join transaction + identity tables on TransactionID.

    Transaction is the base table: most transactions do NOT have a matching
    identity row (roughly 76% in the public dataset), and that's expected,
    not a bug - identity data (device, browser, connection info) is only
    captured for a subset of sessions. An inner join here would silently
    drop the majority of transactions and bias the dataset toward whichever
    users happen to have identity data, distorting both the fraud rate and
    anything trained on it.
    """
    if not transaction_path.exists():
        raise FileNotFoundError(
            f"Missing {transaction_path}. Expected train_transaction.csv in data/raw/."
        )
    if not identity_path.exists():
        raise FileNotFoundError(
            f"Missing {identity_path}. Expected train_identity.csv in data/raw/."
        )

    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    df = transactions.merge(identity, on="TransactionID", how="left")
    return df


def time_based_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    time_col: str = "TransactionDT",
):
    """
    Split by TransactionDT ascending: the earliest (1 - test_frac) of rows
    go to train, the latest test_frac go to test. This is NOT a random
    split - see module docstring for why that matters here.

    test_frac=0.2 is a reasonable default held-out fraction for this
    dataset's size; tune as needed once you see how fraud rate/volume
    behave near the split boundary.
    """
    if time_col not in df.columns:
        raise KeyError(f"Expected time column '{time_col}' not found in dataframe.")

    df_sorted = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))

    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()

    # Sanity check: hard-fail if any time leakage snuck across the boundary
    assert train[time_col].max() <= test[time_col].min(), (
        "Time leakage detected: train contains a TransactionDT later than "
        "the earliest test row. Check for duplicate TransactionDT values "
        "sitting right at the split boundary."
    )

    return train, test


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} joined transactions "
          f"({df['TransactionID'].nunique():,} unique TransactionIDs)")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")
    print(f"Identity match rate: {df['DeviceType'].notna().mean():.2%}")

    train, test = time_based_split(df)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"Train TransactionDT range: {train['TransactionDT'].min()} - {train['TransactionDT'].max()}")
    print(f"Test  TransactionDT range: {test['TransactionDT'].min()} - {test['TransactionDT'].max()}")