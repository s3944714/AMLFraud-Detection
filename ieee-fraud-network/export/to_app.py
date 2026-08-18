"""
export/to_app.py

Assembles the three-file export contract that fraud-review-app reads (and
never touches the model/SHAP/entity graph directly - see that project's own
README). This script's job is ONLY to repackage what the earlier pipeline
stages already computed into the exact schema the app expects; it does not
recompute risk scores, SHAP values, or entity clusters itself.

Files written to export/output/ (point fraud-review-app's DATA_DIR at this
directory once populated):
- scored_transactions.csv: TransactionID, TransactionAmt, risk_score,
  top_reason, cluster_id (nullable)
- shap_detail.csv: TransactionID, feature_name, shap_value, feature_value
- entity_clusters.csv: cluster_id, TransactionID, shared_attribute,
  cluster_fraud_rate
- metadata.json: summary stats for the app's /api/summary endpoint

DELIBERATE SCOPE DECISIONS:
- Only transactions with a SHAP-derived reason code (the top-risk slice
  explainability/shap_report.py already computed, by default the top 5% by
  risk score) are exported. A transaction without a top_reason isn't
  something an investigator could act on anyway, and the schema treats
  top_reason as always-present, unlike the explicitly nullable cluster_id.
- Only CANDIDATE-RING clusters (network/entity_graph.py's
  is_candidate_ring flag) get exported with a real cluster_id. Non-ring
  clusters exist in the pipeline's own entity_clusters.csv as an
  intermediate artifact for transparency, but surfacing them to a review
  app as if they were "rings" would misrepresent ordinary shared-attribute
  noise (e.g. two unrelated people who happen to share a card+address
  combination) as an actual finding.
- isFraud is deliberately NOT included, even though it exists in this
  labeled historical dataset - the schema this export follows doesn't call
  for it, and a real production deployment wouldn't have ground truth
  available at scoring time either. fraud-review-app's own build plan
  explicitly anticipates this and is expected to degrade gracefully (its
  budget-simulation endpoint needs a documented fallback when isFraud isn't
  present in the exported file).
"""

import json
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = MODULE_DIR.parent / "data" / "processed"
EXPORT_DIR = MODULE_DIR / "output"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

REASON_CODES_SRC = PROCESSED_DIR / "reason_codes.csv"
SHAP_DETAIL_SRC = PROCESSED_DIR / "shap_detail.csv"
ENTITY_CLUSTERS_SRC = PROCESSED_DIR / "entity_clusters.csv"
BACKTEST_REPORT_SRC = PROCESSED_DIR / "backtest_report.json"

SCORED_TRANSACTIONS_OUT = EXPORT_DIR / "scored_transactions.csv"
SHAP_DETAIL_OUT = EXPORT_DIR / "shap_detail.csv"
ENTITY_CLUSTERS_OUT = EXPORT_DIR / "entity_clusters.csv"
METADATA_OUT = EXPORT_DIR / "metadata.json"


def _require(path: Path, produced_by: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m {produced_by}` first."
        )


def build_scored_transactions() -> pd.DataFrame:
    _require(REASON_CODES_SRC, "explainability.shap_report")
    reason_df = pd.read_csv(REASON_CODES_SRC)

    if ENTITY_CLUSTERS_SRC.exists():
        clusters_df = pd.read_csv(ENTITY_CLUSTERS_SRC)
        ring_df = clusters_df.loc[clusters_df["is_candidate_ring"], ["TransactionID", "cluster_id"]]
        # Each TransactionID belongs to exactly one connected component (a
        # transaction can't be in two different graph components at once),
        # so this left join can't duplicate rows regardless of how many
        # candidate rings exist.
        scored = reason_df.merge(ring_df, on="TransactionID", how="left")
    else:
        print(f"WARNING: {ENTITY_CLUSTERS_SRC} not found - exporting without cluster_id. "
              f"Run `python -m network.entity_graph` first to include ring membership.")
        scored = reason_df.copy()
        scored["cluster_id"] = pd.NA

    return scored[["TransactionID", "TransactionAmt", "risk_score", "top_reason", "cluster_id"]]


def build_shap_detail() -> pd.DataFrame:
    _require(SHAP_DETAIL_SRC, "explainability.shap_report")
    return pd.read_csv(SHAP_DETAIL_SRC)


def build_entity_clusters() -> pd.DataFrame:
    if not ENTITY_CLUSTERS_SRC.exists():
        print(f"WARNING: {ENTITY_CLUSTERS_SRC} not found - exporting an empty entity_clusters.csv. "
              f"Run `python -m network.entity_graph` first to include ring data.")
        return pd.DataFrame(columns=["cluster_id", "TransactionID", "shared_attribute", "cluster_fraud_rate"])

    clusters_df = pd.read_csv(ENTITY_CLUSTERS_SRC)
    ring_only = clusters_df[clusters_df["is_candidate_ring"]]
    return ring_only[["cluster_id", "TransactionID", "shared_attribute", "cluster_fraud_rate"]]


def build_metadata(scored_df: pd.DataFrame, clusters_df: pd.DataFrame) -> dict:
    pr_auc = None
    if BACKTEST_REPORT_SRC.exists():
        with open(BACKTEST_REPORT_SRC) as f:
            pr_auc = json.load(f).get("pr_auc")
    else:
        print(f"WARNING: {BACKTEST_REPORT_SRC} not found - metadata.json will omit pr_auc. "
              f"Run `python -m evals.backtest` first to include it.")

    return {
        "n_cases": int(len(scored_df)),
        "n_cluster_members": int(clusters_df["TransactionID"].nunique()) if len(clusters_df) else 0,
        "n_candidate_rings": int(clusters_df["cluster_id"].nunique()) if len(clusters_df) else 0,
        "pr_auc": pr_auc,
    }


def run_export():
    scored_df = build_scored_transactions()
    shap_df = build_shap_detail()
    clusters_df = build_entity_clusters()
    metadata = build_metadata(scored_df, clusters_df)

    scored_df.to_csv(SCORED_TRANSACTIONS_OUT, index=False)
    shap_df.to_csv(SHAP_DETAIL_OUT, index=False)
    clusters_df.to_csv(ENTITY_CLUSTERS_OUT, index=False)
    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"scored_transactions.csv -> {SCORED_TRANSACTIONS_OUT} ({len(scored_df):,} rows)")
    print(f"shap_detail.csv         -> {SHAP_DETAIL_OUT} ({len(shap_df):,} rows)")
    print(f"entity_clusters.csv     -> {ENTITY_CLUSTERS_OUT} ({len(clusters_df):,} rows, "
          f"{metadata['n_candidate_rings']:,} candidate rings)")
    print(f"metadata.json           -> {METADATA_OUT}")
    print(f"\n{EXPORT_DIR} is ready to point fraud-review-app's DATA_DIR at.")

    return scored_df, shap_df, clusters_df, metadata


if __name__ == "__main__":
    run_export()