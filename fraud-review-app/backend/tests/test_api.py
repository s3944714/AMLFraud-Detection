"""
backend/tests/test_api.py

Tests for the FastAPI backend, using FastAPI's TestClient plus
app.dependency_overrides to inject test data directly - this is FastAPI's
own recommended testing pattern. An earlier version of this suite tried
forcing module reimports via sys.modules manipulation to swap DATA_DIR
between tests, and hit real, environment-specific Python import-caching
behavior that made it unreliable across machines. Dependency overrides
sidestep that whole problem category: no environment variables, no temp
files for the "normal" tests, no reimporting anything - just handing the
app a different AppData object per test.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.data import AppData, load_data
from backend.main import app, get_app_data


def _make_app_data(
    include_isfraud=True,
    include_shap=True,
    include_clusters=True,
    include_metadata=True,
) -> AppData:
    """
    3 transactions: two in a cluster (one fraud, one not), one standalone
    high-confidence non-fraud - enough to exercise filtering, cluster
    lookups, and the ground-truth/proxy budget-simulation split without
    the noise of a larger fixture.
    """
    rows = [
        {"TransactionID": 1, "TransactionAmt": 100.0, "risk_score": 0.9, "top_reason": "reason A", "cluster_id": 1},
        {"TransactionID": 2, "TransactionAmt": 50.0, "risk_score": 0.8, "top_reason": "reason B", "cluster_id": 1},
        {"TransactionID": 3, "TransactionAmt": 10.0, "risk_score": 0.2, "top_reason": "reason C", "cluster_id": None},
    ]
    if include_isfraud:
        for row, is_fraud in zip(rows, [1, 0, 0]):
            row["isFraud"] = is_fraud
    scored = pd.DataFrame(rows)

    if include_shap:
        shap = pd.DataFrame([
            {"TransactionID": 1, "feature_name": "f_small", "shap_value": 0.2, "feature_value": "x"},
            {"TransactionID": 1, "feature_name": "f_big", "shap_value": -0.9, "feature_value": "y"},
        ])
    else:
        shap = pd.DataFrame(columns=["TransactionID", "feature_name", "shap_value", "feature_value"])

    if include_clusters:
        clusters = pd.DataFrame([
            {"cluster_id": 1, "TransactionID": 1, "shared_attribute": "device", "cluster_fraud_rate": 0.5},
            {"cluster_id": 1, "TransactionID": 2, "shared_attribute": "device", "cluster_fraud_rate": 0.5},
        ])
    else:
        clusters = pd.DataFrame(columns=["cluster_id", "TransactionID", "shared_attribute", "cluster_fraud_rate"])

    metadata = {"pr_auc": 0.55} if include_metadata else None

    return AppData(scored_transactions=scored, shap_detail=shap, entity_clusters=clusters, metadata=metadata)


@pytest.fixture
def client_factory():
    """Yields a function that builds a TestClient backed by whatever
    AppData you ask for - overrides the app's data dependency directly."""
    def _factory(**file_flags):
        data = _make_app_data(**file_flags)
        app.dependency_overrides[get_app_data] = lambda: data
        return TestClient(app)

    yield _factory
    app.dependency_overrides.clear()  # don't leak overrides into other tests


# ---------------------------------------------------------------------------
# /api/cases filtering
# ---------------------------------------------------------------------------

def test_cases_returns_all_sorted_by_risk_score_descending(client_factory):
    client = client_factory()
    resp = client.get("/api/cases")
    assert resp.status_code == 200
    body = resp.json()
    scores = [c["risk_score"] for c in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert len(body["items"]) == 3
    assert body["total"] == 3


def test_cases_min_risk_score_filter(client_factory):
    client = client_factory()
    resp = client.get("/api/cases?min_risk_score=0.5")
    ids = {c["TransactionID"] for c in resp.json()["items"]}
    assert ids == {1, 2}  # txn 3 (risk_score=0.2) excluded


def test_cases_cluster_only_filter(client_factory):
    client = client_factory()
    resp = client.get("/api/cases?cluster_only=true")
    ids = {c["TransactionID"] for c in resp.json()["items"]}
    assert ids == {1, 2}  # txn 3 has no cluster_id


def test_cases_pagination_limit_and_offset(client_factory):
    client = client_factory()
    page1 = client.get("/api/cases?limit=1&offset=0").json()
    page2 = client.get("/api/cases?limit=1&offset=1").json()
    assert len(page1["items"]) == 1
    assert len(page2["items"]) == 1
    assert page1["items"][0]["TransactionID"] != page2["items"][0]["TransactionID"]
    # total reflects the full filtered count, not just this page's size
    assert page1["total"] == 3
    assert page1["limit"] == 1
    assert page2["offset"] == 1


def test_cases_pagination_offset_past_end_returns_empty(client_factory):
    client = client_factory()
    resp = client.get("/api/cases?limit=10&offset=100")
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 3  # total still reflects the real count, not zero


def test_cases_search_substring_match(client_factory):
    client = client_factory()
    resp = client.get("/api/cases?search=1")
    ids = {c["TransactionID"] for c in resp.json()["items"]}
    assert ids == {1}  # fixture IDs are 1, 2, 3 - "1" only substring-matches ID 1


def test_cases_search_no_match_returns_empty_not_error(client_factory):
    client = client_factory()
    resp = client.get("/api/cases?search=99999")
    assert resp.status_code == 200
    assert resp.json()["items"] == []
    assert resp.json()["total"] == 0


def test_cases_search_special_characters_treated_literally(client_factory):
    # Regex metacharacters in the search box must not error out or be
    # interpreted as a pattern - a search box is not a regex box.
    client = client_factory()
    resp = client.get("/api/cases?search=" + "(" + "[1]")
    assert resp.status_code == 200
    assert resp.json()["items"] == []  # no TransactionID literally contains "([1]"


def test_cluster_id_serializes_as_int_not_float(client_factory):
    # cluster_id comes out of pandas as float64 (NaN forces the whole
    # column to float) - the API should still hand back a clean int for
    # transactions that DO have one, not "1.0".
    client = client_factory()
    resp = client.get("/api/cases")
    case_1 = next(c for c in resp.json()["items"] if c["TransactionID"] == 1)
    case_3 = next(c for c in resp.json()["items"] if c["TransactionID"] == 3)
    assert case_1["cluster_id"] == 1
    assert isinstance(case_1["cluster_id"], int)
    assert case_3["cluster_id"] is None


# ---------------------------------------------------------------------------
# /api/cases/{transaction_id}
# ---------------------------------------------------------------------------

def test_case_detail_includes_cluster_mates_not_self(client_factory):
    client = client_factory()
    resp = client.get("/api/cases/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cluster_info"]["member_transaction_ids"] == [2]  # not [1, 2]


def test_case_detail_shap_features_sorted_by_magnitude(client_factory):
    client = client_factory()
    resp = client.get("/api/cases/1")
    features = resp.json()["shap_features"]
    assert features[0]["feature_name"] == "f_big"  # |-0.9| > |0.2|


def test_case_detail_404_for_unknown_transaction(client_factory):
    client = client_factory()
    resp = client.get("/api/cases/99999")
    assert resp.status_code == 404


def test_case_detail_no_cluster_info_for_unclustered_transaction(client_factory):
    client = client_factory()
    resp = client.get("/api/cases/3")
    assert resp.json()["cluster_info"] is None


# ---------------------------------------------------------------------------
# Missing-optional-file graceful degradation
# ---------------------------------------------------------------------------

def test_missing_shap_detail_degrades_gracefully(client_factory):
    client = client_factory(include_shap=False)
    resp = client.get("/api/cases/1")
    assert resp.status_code == 200
    assert resp.json()["shap_features"] == []


def test_missing_entity_clusters_degrades_gracefully(client_factory):
    # Transaction 1 still has cluster_id=1 in scored_transactions, but with
    # no entity_clusters data to look up member details from, the app
    # should return cluster_info=None rather than erroring.
    client = client_factory(include_clusters=False)
    resp = client.get("/api/cases/1")
    assert resp.status_code == 200
    assert resp.json()["cluster_info"] is None


def test_missing_metadata_degrades_gracefully(client_factory):
    client = client_factory(include_metadata=False)
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    assert resp.json()["pr_auc"] is None


def test_missing_required_scored_transactions_raises_at_load(tmp_path):
    # No files written at all - scored_transactions.csv is required, so
    # this should fail loudly rather than the app silently starting broken.
    # Tested directly against load_data(), a pure function - no app or
    # dependency-injection machinery needed for this one.
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path)


# ---------------------------------------------------------------------------
# /api/budget-simulation
# ---------------------------------------------------------------------------

def test_budget_simulation_uses_real_recall_when_ground_truth_present(client_factory):
    client = client_factory(include_isfraud=True)
    resp = client.get("/api/budget-simulation?budget_pct=0.3")  # ceil(3*0.3)=1 -> top-1 only
    body = resp.json()
    assert body["metric_name"] == "recall"
    assert body["is_ground_truth_available"] is True
    assert body["n_reviewed"] == 1
    # top-1 by risk_score is txn 1 (isFraud=1); total fraud in fixture = 1
    assert body["model_count"] == 1
    assert body["model_metric"] == pytest.approx(1.0)


def test_budget_simulation_falls_back_to_proxy_without_ground_truth(client_factory):
    client = client_factory(include_isfraud=False)
    resp = client.get("/api/budget-simulation?budget_pct=0.3")  # ceil(3*0.3)=1 -> top-1 only
    body = resp.json()
    assert body["metric_name"] == "candidate_ring_capture_rate"
    assert body["is_ground_truth_available"] is False
    assert "PROXY" in body["note"]
    assert body["n_reviewed"] == 1
    # top-1 by risk_score is txn 1, which IS a cluster member
    assert body["model_count"] == 1


def test_budget_simulation_random_baseline_is_reproducible(client_factory):
    # Seeded, not freshly randomised per call - see main.py's _RANDOM_SEED
    # comment. Two calls with the same budget must return the same
    # random-order comparison number.
    client = client_factory()
    resp1 = client.get("/api/budget-simulation?budget_pct=0.67")
    resp2 = client.get("/api/budget-simulation?budget_pct=0.67")
    assert resp1.json()["random_count"] == resp2.json()["random_count"]


def test_budget_simulation_n_reviewed_scales_with_budget(client_factory):
    client = client_factory()
    resp_small = client.get("/api/budget-simulation?budget_pct=0.1")
    resp_large = client.get("/api/budget-simulation?budget_pct=0.9")
    assert resp_small.json()["n_reviewed"] <= resp_large.json()["n_reviewed"]


# ---------------------------------------------------------------------------
# /api/summary
# ---------------------------------------------------------------------------

def test_summary_counts_and_pr_auc(client_factory):
    client = client_factory()
    resp = client.get("/api/summary")
    body = resp.json()
    assert body["n_cases"] == 3
    assert body["n_cluster_members"] == 2  # txns 1 and 2
    assert body["pr_auc"] == pytest.approx(0.55)