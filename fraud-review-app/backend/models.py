"""
backend/models.py

Pydantic response models for the API - this is a real contract the
frontend depends on, not ad-hoc dicts. Every endpoint in main.py returns
one of these.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel

CaseStatus = Literal["reviewed", "escalated", "dismissed"]


class CaseSummary(BaseModel):
    TransactionID: int
    TransactionAmt: float
    risk_score: float
    top_reason: str
    cluster_id: Optional[int] = None
    status: Optional[CaseStatus] = None


class ShapFeature(BaseModel):
    feature_name: str
    shap_value: float
    # feature_value can be numeric or categorical in the source data
    # (e.g. a transaction amount vs a device string) - stringified here so
    # the API has one consistent shape rather than a union type the
    # frontend would need to branch on.
    feature_value: Optional[str] = None


class ClusterInfo(BaseModel):
    cluster_id: int
    shared_attribute: str
    cluster_fraud_rate: float
    member_transaction_ids: List[int]  # OTHER transactions in the cluster, not including this one


class CaseDetail(CaseSummary):
    shap_features: List[ShapFeature] = []
    cluster_info: Optional[ClusterInfo] = None


class CaseStatusUpdate(BaseModel):
    status: Optional[CaseStatus] = None  # null clears the status back to unreviewed


class BudgetSimulationResponse(BaseModel):
    budget_pct: float
    n_transactions: int
    n_reviewed: int
    # "recall" when isFraud ground truth is present in the loaded data,
    # otherwise a documented proxy metric - see main.py's
    # get_budget_simulation for why, and note below for the specific
    # caveat on whichever metric this response actually contains.
    metric_name: str
    is_ground_truth_available: bool
    model_metric: float
    random_metric: float
    model_count: int
    random_count: int
    note: str


class PaginatedCases(BaseModel):
    items: List[CaseSummary]
    total: int
    limit: int
    offset: int


class SummaryResponse(BaseModel):
    n_cases: int
    n_cluster_members: int
    pr_auc: Optional[float] = None


class RiskBucket(BaseModel):
    label: str
    min_score: float
    max_score: float
    count: int


class RiskDistributionResponse(BaseModel):
    buckets: List[RiskBucket]
    total: int