"""
explainability/shap_report.py

Per-transaction SHAP explanations ("reason codes") for the top-risk
transactions flagged by the baseline model. models/baseline.py's docstring
explains why accuracy-style metrics aren't the target here; this module is
the flip side of that concern: a flagged transaction needs a justification
an investigator (or eventually a regulator) can actually read, not just a
bare score.

ON THE HONESTY OF THESE EXPLANATIONS: most of this dataset's columns
(V1-V339, C1-C14, D1-D15, and most id_ columns) are anonymized, Vesta-
engineered features - Kaggle's own documentation does not disclose what they
represent. We do NOT invent plausible-sounding business meaning for these
(e.g. claiming "V127 measures device fingerprint mismatches" would be a
fabrication). Reason codes for anonymized columns are phrased honestly as
"an anonymized engineered feature" with its raw value and SHAP contribution
- still useful to an investigator (it tells them which signal drove the
score, how strongly, and in which direction), just without a false
narrative bolted onto it. Named columns with real-world meaning
(TransactionAmt, ProductCD, card network/type, email domain, device type,
etc.) get a proper plain-language template instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from models.baseline import get_train_test_features, load_baseline_model

MODULE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = MODULE_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SHAP_DETAIL_PATH = PROCESSED_DIR / "shap_detail.csv"
REASON_CODES_PATH = PROCESSED_DIR / "reason_codes.csv"

# How many top-risk test transactions to explain, expressed as an alert
# budget - matches the same "top N% by risk score" framing used for
# recall-at-budget in models/baseline.py / evals/backtest.py. These are the
# transactions a review team would actually see, so they're the ones that
# need reason codes; computing SHAP for every non-flagged transaction too
# would be expensive and pointless.
DEFAULT_BUDGET_PCT = 0.05
DETAIL_TOP_K = 10   # SHAP features kept per transaction in the detail table (bar-chart source)
REASON_TOP_K = 3    # features summarized in the plain-language top_reason string


def _fmt_amt(value):
    return f"${value:,.2f}"


# Plain-language templates for columns whose real-world meaning is known.
# Each takes (feature_value, direction) -> a short phrase, where direction
# is "higher" (positive SHAP - pushed toward fraud) or "lower" (negative
# SHAP - pushed toward legitimate).
def _template_transaction_amt(value, direction):
    tag = "unusually high" if direction == "higher" else "unusually low"
    return f"{tag} transaction amount ({_fmt_amt(value)})"


def _template_hour_of_day(value, direction):
    tag = "unusual for" if direction == "higher" else "typical for"
    return f"transaction timing {tag} this hour ({int(value):02d}:00)"


def _template_product_cd(value, direction):
    tag = "elevated" if direction == "higher" else "lower"
    return f"product category '{value}' associated with {tag} risk in this data"


def _template_card_network(value, direction):
    tag = "elevated" if direction == "higher" else "lower"
    return f"card network '{value}' associated with {tag} risk in this data"


def _template_card_type(value, direction):
    tag = "elevated" if direction == "higher" else "lower"
    return f"card type '{value}' associated with {tag} risk in this data"


def _template_email_domain(value, direction):
    label = value if pd.notna(value) else "missing/no email domain"
    tag = "elevated" if direction == "higher" else "lower"
    return f"email domain '{label}' associated with {tag} risk in this data"


def _template_device_type(value, direction):
    label = value if pd.notna(value) else "no device/identity match"
    tag = "elevated" if direction == "higher" else "lower"
    return f"device type '{label}' associated with {tag} risk in this data"


def _template_distance(value, direction):
    tag = "unusually large" if direction == "higher" else "unusually small"
    return f"{tag} distance field ({value:.1f})"


def _template_match_flag(feature_name, value, direction):
    label = value if pd.notna(value) else "missing"
    tag = "elevated" if direction == "higher" else "lower"
    return f"identity match flag {feature_name}='{label}' associated with {tag} risk in this data"


REASON_TEMPLATES = {
    "TransactionAmt": _template_transaction_amt,
    "hour_of_day": _template_hour_of_day,
    "ProductCD": _template_product_cd,
    "card4": _template_card_network,
    "card6": _template_card_type,
    "P_emaildomain": _template_email_domain,
    "R_emaildomain": _template_email_domain,
    "DeviceType": _template_device_type,
    "dist1": _template_distance,
    "dist2": _template_distance,
}
for _m in [f"M{i}" for i in range(1, 10)]:
    REASON_TEMPLATES[_m] = lambda value, direction, _m=_m: _template_match_flag(_m, value, direction)


def describe_feature(feature_name: str, feature_value, shap_value: float) -> str:
    """
    Return a short, honest phrase describing one feature's contribution to
    a transaction's risk score. Known/named columns get a real
    plain-language template; anonymized engineered columns (V*, C*, D*,
    id_*, card1/2/3/5, addr1/2, DeviceInfo) get an honest generic
    description rather than an invented one - see module docstring.
    """
    direction = "higher" if shap_value > 0 else "lower"

    template = REASON_TEMPLATES.get(feature_name)
    if template is not None:
        return template(feature_value, direction)

    is_numeric = isinstance(feature_value, (int, float, np.floating)) and pd.notna(feature_value)
    value_str = f"{feature_value:.2f}" if is_numeric else str(feature_value)
    push = "pushed risk up" if direction == "higher" else "pushed risk down"
    return (f"anonymized engineered feature {feature_name} (value={value_str}) - {push}; "
            f"exact business meaning not disclosed by the data provider")


def generate_reason_codes(budget_pct: float = DEFAULT_BUDGET_PCT):
    """
    Compute SHAP values for the top `budget_pct` fraction of test
    transactions by risk score (the ones a review team would actually see),
    and write two files to data/processed/:
    - shap_detail.csv: long format (TransactionID, feature_name,
      shap_value, feature_value), top DETAIL_TOP_K features per
      transaction - source data for a SHAP bar chart in the review app
    - reason_codes.csv: (TransactionID, TransactionAmt, risk_score,
      top_reason) - a REASON_TOP_K-feature plain-language justification
      per transaction
    """
    booster = load_baseline_model()
    X_train, y_train, X_test, y_test, cat_cols, train_df, test_df = get_train_test_features()

    risk_scores = booster.predict(X_test)

    n = len(risk_scores)
    k = max(1, int(np.ceil(n * budget_pct)))
    flagged_idx = np.argsort(-risk_scores)[:k]

    X_flagged = X_test.iloc[flagged_idx].reset_index(drop=True)
    scores_flagged = risk_scores[flagged_idx]
    txn_ids_flagged = test_df["TransactionID"].values[flagged_idx]
    amt_flagged = test_df["TransactionAmt"].values[flagged_idx]

    print(f"Computing SHAP values for {len(X_flagged):,} flagged transactions "
          f"(top {budget_pct:.0%} of {n:,} test transactions by risk score)...")

    # tree_path_dependent feature_perturbation needs no background dataset
    # and is fast for tree ensembles. Values come out in log-odds (margin)
    # space, not probability space - we use them for ranking and direction
    # of each feature's contribution, not as exact probability deltas,
    # which is standard practice for tree SHAP on a sigmoid-linked binary
    # objective.
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_flagged)

    # Depending on shap/lightgbm version, a binary Booster can return either
    # a single (n, n_features) array (positive-class contributions) or a
    # list of two arrays ([class0, class1]) - normalize to positive-class
    # (fraud) contributions either way.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = X_flagged.columns.tolist()

    detail_rows = []
    reason_rows = []

    for i in range(len(X_flagged)):
        row_shap = shap_values[i]
        row_vals = X_flagged.iloc[i]

        top_idx = np.argsort(-np.abs(row_shap))[:DETAIL_TOP_K]

        for j in top_idx:
            fname = feature_names[j]
            fval = row_vals[fname]
            detail_rows.append({
                "TransactionID": txn_ids_flagged[i],
                "feature_name": fname,
                "shap_value": float(row_shap[j]),
                "feature_value": fval.item() if hasattr(fval, "item") else fval,
            })

        top_reason_idx = top_idx[:REASON_TOP_K]
        phrases = [
            describe_feature(feature_names[j], row_vals[feature_names[j]], row_shap[j])
            for j in top_reason_idx
        ]
        reason_rows.append({
            "TransactionID": txn_ids_flagged[i],
            "TransactionAmt": amt_flagged[i],
            "risk_score": float(scores_flagged[i]),
            "top_reason": "; ".join(phrases),
        })

    detail_df = pd.DataFrame(detail_rows)
    reason_df = pd.DataFrame(reason_rows).sort_values("risk_score", ascending=False)

    detail_df.to_csv(SHAP_DETAIL_PATH, index=False)
    reason_df.to_csv(REASON_CODES_PATH, index=False)

    print(f"SHAP detail  -> {SHAP_DETAIL_PATH} ({len(detail_df):,} rows)")
    print(f"Reason codes -> {REASON_CODES_PATH} ({len(reason_df):,} transactions)")
    print("\nExample reason codes:")
    for _, row in reason_df.head(3).iterrows():
        print(f"  TransactionID {row['TransactionID']} (risk={row['risk_score']:.3f}): {row['top_reason']}")

    return detail_df, reason_df


if __name__ == "__main__":
    generate_reason_codes()