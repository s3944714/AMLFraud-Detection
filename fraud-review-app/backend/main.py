"""
backend/main.py

FastAPI backend for fraud-review-app. Reads the three-file export contract
(loaded once at startup - see backend/data.py) - this app never runs the
model, SHAP, or the entity graph live, it only serves what the pipeline
already computed. Run with: uvicorn backend.main:app --reload

Route handlers receive the loaded data via FastAPI's dependency injection
(Depends(get_app_data)) rather than importing a module-level global
directly. This is deliberate: it's FastAPI's own recommended pattern for
testability - tests override get_app_data via
app.dependency_overrides[get_app_data] to inject arbitrary test data
directly, with no need to touch environment variables, temp files, or
reimport any modules.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from backend.data import AppData, app_data
from backend.models import (
    BudgetSimulationResponse,
    CaseDetail,
    CaseSummary,
    ClusterInfo,
    PaginatedCases,
    ShapFeature,
    SummaryResponse,
)

logger = logging.getLogger("fraud_review_app")

app = FastAPI(title="Fraud Review App API")

# Seeded rather than freshly randomised per call: a reviewer dragging the
# budget slider back and forth should see a STABLE random-order comparison
# number, not a different one on every request - and it makes this
# endpoint's tests deterministic. See get_budget_simulation below.
_RANDOM_SEED = 42


def get_app_data() -> AppData:
    """FastAPI dependency provider. Route handlers depend on this rather
    than importing the loaded data directly, specifically so tests can
    swap in arbitrary data via app.dependency_overrides[get_app_data]."""
    return app_data


def _cluster_id_or_none(value) -> Optional[int]:
    return int(value) if pd.notna(value) else None


def _row_to_case_summary(row: pd.Series) -> CaseSummary:
    return CaseSummary(
        TransactionID=int(row["TransactionID"]),
        TransactionAmt=float(row["TransactionAmt"]),
        risk_score=float(row["risk_score"]),
        top_reason=str(row["top_reason"]),
        cluster_id=_cluster_id_or_none(row.get("cluster_id")),
    )


@app.get("/api/cases", response_model=PaginatedCases)
def get_cases(
    min_risk_score: float = Query(0.0, ge=0.0, le=1.0),
    cluster_only: bool = Query(False),
    search: Optional[str] = Query(None, description="Substring match against TransactionID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    data: AppData = Depends(get_app_data),
):
    df = data.scored_transactions
    filtered = df[df["risk_score"] >= min_risk_score]
    if cluster_only:
        filtered = filtered[filtered["cluster_id"].notna()]
    if search:
        # Server-side, not client-side: the queue is paginated, so a
        # client-side search could only ever search whatever page happens
        # to be loaded - a search that silently misses matches sitting on
        # other pages would be worse than no search at all. Substring
        # match on the string form of TransactionID, not exact match, so
        # typing partial digits narrows results as-you-type.
        search_digits = search.strip()
        filtered = filtered[
            filtered["TransactionID"].astype(str).str.contains(search_digits, na=False, regex=False)
        ]
    filtered = filtered.sort_values("risk_score", ascending=False)

    total = len(filtered)
    page = filtered.iloc[offset : offset + limit]

    return PaginatedCases(
        items=[_row_to_case_summary(row) for _, row in page.iterrows()],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/cases/{transaction_id}", response_model=CaseDetail)
def get_case_detail(transaction_id: int, data: AppData = Depends(get_app_data)):
    df = data.scored_transactions
    match = df[df["TransactionID"] == transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No case found for TransactionID {transaction_id}")

    row = match.iloc[0]
    summary = _row_to_case_summary(row)

    shap_rows = data.shap_detail[data.shap_detail["TransactionID"] == transaction_id]
    if len(shap_rows):
        shap_rows = shap_rows.reindex(shap_rows["shap_value"].abs().sort_values(ascending=False).index)
    shap_features = [
        ShapFeature(
            feature_name=str(r["feature_name"]),
            shap_value=float(r["shap_value"]),
            feature_value=str(r["feature_value"]) if pd.notna(r["feature_value"]) else None,
        )
        for _, r in shap_rows.iterrows()
    ]

    cluster_info = None
    if summary.cluster_id is not None:
        cluster_rows = data.entity_clusters[data.entity_clusters["cluster_id"] == summary.cluster_id]
        own_row = cluster_rows[cluster_rows["TransactionID"] == transaction_id]
        if not own_row.empty:
            other_ids = (
                cluster_rows.loc[cluster_rows["TransactionID"] != transaction_id, "TransactionID"]
                .astype(int)
                .tolist()
            )
            cluster_info = ClusterInfo(
                cluster_id=summary.cluster_id,
                shared_attribute=str(own_row.iloc[0]["shared_attribute"]),
                cluster_fraud_rate=float(own_row.iloc[0]["cluster_fraud_rate"]),
                member_transaction_ids=other_ids,
            )

    return CaseDetail(**summary.model_dump(), shap_features=shap_features, cluster_info=cluster_info)


@app.get("/api/budget-simulation", response_model=BudgetSimulationResponse)
def get_budget_simulation(
    budget_pct: float = Query(..., gt=0.0, le=1.0),
    data: AppData = Depends(get_app_data),
):
    df = data.scored_transactions
    n = len(df)
    n_reviewed = max(1, int(np.ceil(n * budget_pct))) if n else 0

    ranked = df.sort_values("risk_score", ascending=False)
    top_k = ranked.iloc[:n_reviewed]

    rng = np.random.default_rng(_RANDOM_SEED)
    random_idx = rng.choice(n, size=min(n_reviewed, n), replace=False) if n else np.array([], dtype=int)
    random_sample = df.iloc[random_idx]

    if data.has_ground_truth:
        total_positive = int(df["isFraud"].sum())
        model_count = int(top_k["isFraud"].sum())
        random_count = int(random_sample["isFraud"].sum())
        model_metric = model_count / total_positive if total_positive else 0.0
        random_metric = random_count / total_positive if total_positive else 0.0
        metric_name = "recall"
        note = "Computed against actual isFraud ground truth present in the loaded data."
    else:
        # No ground truth in the real pipeline export by design (see
        # AppData.has_ground_truth's comment). Falls back to a documented
        # PROXY: candidate-ring membership as a stand-in "known risky"
        # signal, since ring membership correlates strongly with fraud
        # (10x+ baseline fraud rate in the source pipeline's own
        # evaluation) even though it's not a confirmed label.
        total_positive = int(df["cluster_id"].notna().sum())
        model_count = int(top_k["cluster_id"].notna().sum())
        random_count = int(random_sample["cluster_id"].notna().sum())
        model_metric = model_count / total_positive if total_positive else 0.0
        random_metric = random_count / total_positive if total_positive else 0.0
        metric_name = "candidate_ring_capture_rate"
        note = (
            "No isFraud ground truth in the loaded data, so this is a PROXY metric: "
            "the fraction of candidate-ring-linked transactions captured, not true recall. "
            "Candidate-ring membership correlates strongly with fraud but is not a "
            "confirmed label - treat this as directional, not exact."
        )

    return BudgetSimulationResponse(
        budget_pct=budget_pct,
        n_transactions=n,
        n_reviewed=n_reviewed,
        metric_name=metric_name,
        is_ground_truth_available=data.has_ground_truth,
        model_metric=model_metric,
        random_metric=random_metric,
        model_count=model_count,
        random_count=random_count,
        note=note,
    )


@app.get("/api/summary", response_model=SummaryResponse)
def get_summary(data: AppData = Depends(get_app_data)):
    n_cases = len(data.scored_transactions)
    n_cluster_members = (
        int(data.entity_clusters["TransactionID"].nunique()) if len(data.entity_clusters) else 0
    )
    pr_auc = data.metadata.get("pr_auc") if data.metadata else None
    return SummaryResponse(n_cases=n_cases, n_cluster_members=n_cluster_members, pr_auc=pr_auc)


# Mounted LAST, after every /api route is registered - StaticFiles is a
# catch-all for "/", so mounting it first would shadow the API routes
# entirely.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")