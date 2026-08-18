"""
evals/backtest.py

Formal backtest report on the held-out time-based test split: PR-AUC,
recall at fixed alert budgets, and - the number a non-technical reviewer
actually cares about - how many MORE fraud cases the model catches compared
to reviewing the same volume of transactions in random order.

This is deliberately separate from models/baseline.py's own training-time
evaluation (which exists to pick the best model iteration and sanity-check
training as it happens - see that module's docstring). This script re-loads
the SAVED model and re-derives features identically, via models.baseline's
shared entrypoints, rather than retraining. It's meant to be rerun any time
you want a clean, reportable evaluation snapshot, and it's the canonical
source for the numbers that belong in the README and in the exported app's
summary metadata (data/processed/backtest_report.json is what
export/to_app.py should read from for that, rather than recomputing this
itself).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from models.baseline import (
    ALERT_BUDGETS,
    get_train_test_features,
    load_baseline_model,
    random_order_recall_at_budget,
    recall_at_budget,
)

MODULE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = MODULE_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_REPORT_PATH = PROCESSED_DIR / "backtest_report.json"


def run_backtest():
    booster = load_baseline_model()
    X_train, y_train, X_test, y_test, cat_cols, train_df, test_df = get_train_test_features()

    risk_scores = booster.predict(X_test)
    pr_auc = average_precision_score(y_test, risk_scores)

    n_test = len(y_test)
    total_fraud = int(y_test.sum())

    budget_results = []
    for budget in ALERT_BUDGETS:
        model_recall = recall_at_budget(y_test, risk_scores, budget)
        random_recall = random_order_recall_at_budget(y_test, budget)

        n_reviewed = max(1, int(np.ceil(n_test * budget)))
        # Recall fractions converted back to actual case counts - a
        # non-technical reviewer cares about "83 more real fraud cases
        # caught," not "an 8-point recall improvement."
        model_caught = round(model_recall * total_fraud)
        random_caught = round(random_recall * total_fraud)

        budget_results.append({
            "budget_pct": budget,
            "n_transactions_reviewed": n_reviewed,
            "model_recall": model_recall,
            "random_recall": random_recall,
            "model_fraud_caught": model_caught,
            "random_fraud_caught": random_caught,
            "additional_fraud_caught_vs_random": model_caught - random_caught,
            "lift_over_random": (model_recall / random_recall) if random_recall > 0 else None,
        })

    report = {
        "pr_auc": pr_auc,
        "n_test_transactions": n_test,
        "total_fraud_in_test": total_fraud,
        "test_fraud_rate": float(y_test.mean()),
        "budgets": budget_results,
    }

    with open(BACKTEST_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"PR-AUC on held-out test split: {pr_auc:.4f}")
    print(f"Test set: {n_test:,} transactions, {total_fraud:,} actual fraud ({y_test.mean():.4%})\n")
    header = f"{'Budget':<8}{'Reviewed':<12}{'Model catches':<16}{'Random catches':<17}{'Extra caught':<15}{'Lift'}"
    print(header)
    print("-" * len(header))
    for r in budget_results:
        lift_str = f"{r['lift_over_random']:.1f}x" if r["lift_over_random"] else "n/a"
        print(
            f"{r['budget_pct']:<8.0%}"
            f"{r['n_transactions_reviewed']:<12,}"
            f"{r['model_fraud_caught']:<16,}"
            f"{r['random_fraud_caught']:<17,}"
            f"{r['additional_fraud_caught_vs_random']:<15,}"
            f"{lift_str}"
        )

    print(f"\nReport -> {BACKTEST_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_backtest()