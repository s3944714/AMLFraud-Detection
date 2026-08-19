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

from backend.data import AppData, CaseStatusStore, load_data
from backend.main import app, get_app_data, get_case_status_store


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
    AppData you ask for - overrides the app's data dependency directly.
    Also overrides the case-status dependency with a FRESH store per
    test, so a status set in one test can never leak into another - the
    same class of cross-test pollution bug already hit once in this
    project's history with a different mechanism (module reimports), so
    it's worth being deliberate about avoiding it here too."""
    def _factory(**file_flags):
        data = _make_app_data(**file_flags)
        status_store = CaseStatusStore()  # created ONCE per client, not per-request -
        # a lambda that constructs CaseStatusStore() fresh on every call would
        # silently discard status between a PUT and the following GET, since
        # FastAPI re-invokes dependency override callables on every request,
        # not once per test (confirmed directly: this was a real bug here
        # until fixed - status appeared to write successfully then vanish).
        app.dependency_overrides[get_app_data] = lambda: data
        app.dependency_overrides[get_case_status_store] = lambda: status_store
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


# ---------------------------------------------------------------------------
# /api/cases sorting
# ---------------------------------------------------------------------------

def test_cases_sort_by_amount_ascending(client_factory):
    client = client_factory()
    resp = client.get("/api/cases?sort_by=TransactionAmt&sort_dir=asc")
    amounts = [c["TransactionAmt"] for c in resp.json()["items"]]
    assert amounts == sorted(amounts)  # fixture: 10.0, 50.0, 100.0


def test_cases_sort_by_transaction_id_descending(client_factory):
    client = client_factory()
    resp = client.get("/api/cases?sort_by=TransactionID&sort_dir=desc")
    ids = [c["TransactionID"] for c in resp.json()["items"]]
    assert ids == [3, 2, 1]


def test_cases_default_sort_unchanged_from_before_sorting_existed(client_factory):
    # No sort params at all should behave exactly as it always has:
    # risk_score descending - existing frontend code and this endpoint's
    # documented default both depend on this not silently changing.
    client = client_factory()
    resp = client.get("/api/cases")
    scores = [c["risk_score"] for c in resp.json()["items"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# /api/cases/export
# ---------------------------------------------------------------------------

def test_export_returns_csv_with_correct_headers(client_factory):
    client = client_factory()
    resp = client.get("/api/cases/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]


def test_export_respects_filters_not_just_current_page(client_factory):
    # Export must reflect the SAME filtered set /api/cases would show,
    # not be capped to a page size - this is the whole point of a
    # separate export endpoint rather than reusing the paginated one.
    client = client_factory()
    resp = client.get("/api/cases/export?min_risk_score=0.5")
    lines = resp.text.strip().split("\n")
    # header + 2 matching rows (fixture: risk_score 0.9 and 0.8 both >= 0.5)
    assert len(lines) == 1 + 2


def test_export_includes_status_column(client_factory):
    client = client_factory()
    client.put("/api/cases/1/status", json={"status": "reviewed"})
    resp = client.get("/api/cases/export")
    assert "status" in resp.text.split("\n")[0]
    assert "reviewed" in resp.text


# ---------------------------------------------------------------------------
# /api/cases/{transaction_id}/status
# ---------------------------------------------------------------------------

def test_set_case_status_reflected_in_case_list(client_factory):
    client = client_factory()
    resp = client.put("/api/cases/1/status", json={"status": "escalated"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "escalated"

    list_resp = client.get("/api/cases")
    case_1 = next(c for c in list_resp.json()["items"] if c["TransactionID"] == 1)
    assert case_1["status"] == "escalated"


def test_set_case_status_reflected_in_case_detail(client_factory):
    client = client_factory()
    client.put("/api/cases/1/status", json={"status": "dismissed"})
    detail = client.get("/api/cases/1").json()
    assert detail["status"] == "dismissed"


def test_clearing_status_with_null_removes_it(client_factory):
    client = client_factory()
    client.put("/api/cases/1/status", json={"status": "reviewed"})
    client.put("/api/cases/1/status", json={"status": None})
    detail = client.get("/api/cases/1").json()
    assert detail["status"] is None


def test_status_defaults_to_null_when_never_set(client_factory):
    client = client_factory()
    detail = client.get("/api/cases/2").json()
    assert detail["status"] is None


def test_set_status_on_unknown_transaction_404s(client_factory):
    client = client_factory()
    resp = client.put("/api/cases/99999/status", json={"status": "reviewed"})
    assert resp.status_code == 404


def test_invalid_status_value_rejected(client_factory):
    client = client_factory()
    resp = client.put("/api/cases/1/status", json={"status": "not_a_real_status"})
    assert resp.status_code == 422  # Pydantic Literal validation, not silently accepted


def test_status_store_does_not_leak_across_client_factory_calls(client_factory):
    # Each client_factory() call gets a FRESH status store (see the
    # fixture) - this is the regression test for that guarantee.
    client_a = client_factory()
    client_a.put("/api/cases/1/status", json={"status": "reviewed"})

    client_b = client_factory()
    detail = client_b.get("/api/cases/1").json()
    assert detail["status"] is None


# ---------------------------------------------------------------------------
# /api/risk-distribution
# ---------------------------------------------------------------------------

def test_risk_distribution_buckets_sum_to_total(client_factory):
    client = client_factory()
    resp = client.get("/api/risk-distribution")
    body = resp.json()
    assert sum(b["count"] for b in body["buckets"]) == body["total"] == 3


def test_risk_distribution_places_scores_in_correct_tier(client_factory):
    # fixture scores: 0.9 (Critical), 0.8 (Critical, since 0.8 >= the 0.75
    # Critical threshold), 0.2 (Low)
    client = client_factory()
    body = client.get("/api/risk-distribution").json()
    by_label = {b["label"]: b["count"] for b in body["buckets"]}
    assert by_label == {"Critical": 2, "High": 0, "Medium": 0, "Low": 1}


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