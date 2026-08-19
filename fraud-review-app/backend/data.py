"""
backend/data.py

Loads the three-file export contract (see ieee-fraud-network's
export/to_app.py, the pipeline half of this project).

load_data() is a pure function - an explicit data_dir goes in, an AppData
bundle comes out - specifically so it's trivially testable without any
module-reimport tricks. Tests either call it directly with a tmp_path, or
bypass it entirely via FastAPI's dependency_overrides (see backend/main.py,
which is where the real testing story lives).

scored_transactions.csv is required: raises loudly if missing, rather than
failing confusingly on the first request. shap_detail.csv and
entity_clusters.csv are optional - missing means an empty DataFrame and a
logged warning, not a crash, so the case queue still works without the
explanation/network overlays.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_review_app")


@dataclass
class AppData:
    scored_transactions: pd.DataFrame
    shap_detail: pd.DataFrame
    entity_clusters: pd.DataFrame
    metadata: Optional[dict]

    @property
    def has_ground_truth(self) -> bool:
        # The real pipeline export deliberately omits isFraud (a production
        # deployment wouldn't have ground truth at scoring time either -
        # see the pipeline README). The bundled sample data MAY include it
        # for demo purposes.
        return "isFraud" in self.scored_transactions.columns


def _load_scored_transactions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "scored_transactions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"scored_transactions.csv not found at {path}. This file is "
            f"required - the app cannot function without it. Set the "
            f"DATA_DIR environment variable to a directory containing the "
            f"pipeline's exported files, or use the bundled data/sample/."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} scored transactions from {path}")
    return df


def _load_optional(path: Path, columns: list) -> pd.DataFrame:
    if not path.exists():
        logger.warning(
            f"{path.name} not found at {path} - continuing with an empty "
            f"DataFrame. The app will still run, but the feature that "
            f"depends on this file will be unavailable."
        )
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def _load_metadata(path: Path) -> Optional[dict]:
    if not path.exists():
        logger.warning(f"metadata.json not found at {path} - /api/summary will omit pr_auc.")
        return None
    with open(path) as f:
        return json.load(f)


def load_data(data_dir) -> AppData:
    """Pure loading function: given a directory, returns an AppData bundle.
    No module-level side effects, no environment reads inside this
    function - callers (real startup code below, or tests) decide where
    data_dir comes from."""
    data_dir = Path(data_dir)
    return AppData(
        scored_transactions=_load_scored_transactions(data_dir),
        shap_detail=_load_optional(
            data_dir / "shap_detail.csv",
            ["TransactionID", "feature_name", "shap_value", "feature_value"],
        ),
        entity_clusters=_load_optional(
            data_dir / "entity_clusters.csv",
            ["cluster_id", "TransactionID", "shared_attribute", "cluster_fraud_rate"],
        ),
        metadata=_load_metadata(data_dir / "metadata.json"),
    )


# Real app startup: load once from DATA_DIR (or the bundled sample), at
# import time - this is what a genuine `uvicorn backend.main:app` process
# actually uses. Defaults to the bundled synthetic sample so the app is
# runnable with zero setup; point DATA_DIR at the real pipeline's
# export/output/ to use real data - one-line config change, not a code
# change. This is the ONLY module-level side effect left in this file -
# everything else is a pure function tests can call directly.
DATA_DIR = Path(os.environ.get("DATA_DIR", "data/sample"))
app_data = load_data(DATA_DIR)


class CaseStatusStore:
    """
    In-memory case-status tracker (Reviewed / Escalated / Dismissed),
    deliberately kept separate from AppData: AppData is a read-only bundle
    of what the pipeline exported, this is mutable state the app itself
    creates. Lives for the process's lifetime - resets on restart, and
    isn't shared across multiple worker processes if this were ever run
    with more than one (uvicorn --workers > 1). That's an acceptable
    tradeoff for a single-reviewer portfolio demo, not something a real
    multi-user deployment could rely on as-is - a real version would need
    this backed by a database or shared cache instead.
    """

    def __init__(self):
        self._statuses: dict = {}

    def get(self, transaction_id: int) -> Optional[str]:
        return self._statuses.get(transaction_id)

    def set(self, transaction_id: int, status: Optional[str]) -> None:
        if status is None:
            self._statuses.pop(transaction_id, None)
        else:
            self._statuses[transaction_id] = status

    def all(self) -> dict:
        return dict(self._statuses)


case_status_store = CaseStatusStore()