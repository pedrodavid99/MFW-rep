"""
Metrics module for model evaluation.

This file contains functions to compute:
- Brier score decomposition
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Standard classification metrics

Used in:
"Lapse and Mid-Term Prediction Models"
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score,
    brier_score_loss
)


# -------------------------------------------------------------------
# Brier score decomposition
# -------------------------------------------------------------------
def brier_decomposition(y_true, y_proba, n_bins=10):
    """
    Computes Brier score decomposition into:
    - Reliability (calibration)
    - Resolution
    - Uncertainty
    """

    y_true = np.array(y_true)
    bins = np.linspace(0, 1, n_bins + 1)

    n = len(y_true)
    o_bar = y_true.mean()

    REL, RES = 0, 0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        else:
            mask = (y_proba >= bins[i]) & (y_proba <= bins[i+1])

        if mask.sum() == 0:
            continue

        n_b = mask.sum()
        f_bar = y_proba[mask].mean()
        o_b = y_true[mask].mean()

        REL += (n_b / n) * (f_bar - o_b) ** 2
        RES += (n_b / n) * (o_b - o_bar) ** 2

    UNC = o_bar * (1 - o_bar)
    BS = REL - RES + UNC

    return {
        'Brier Score': BS,
        'REL (calibration)': REL,
        'RES (resolution)': RES,
        'UNC (uncertainty)': UNC
    }


# -------------------------------------------------------------------
# Expected Calibration Error (ECE)
# -------------------------------------------------------------------
def compute_ece(y_true, y_proba, n_bins=10):
    """
    Computes Expected Calibration Error (ECE)
    """

    y_true = np.array(y_true)
    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        else:
            mask = (y_proba >= bins[i]) & (y_proba <= bins[i+1])

        if mask.sum() == 0:
            continue

        ece += (mask.sum() / len(y_true)) * abs(
            y_proba[mask].mean() - y_true[mask].mean()
        )

    return ece


# -------------------------------------------------------------------
# Maximum Calibration Error (MCE)
# -------------------------------------------------------------------
def compute_mce(y_true, y_proba, n_bins=10):
    """
    Computes Maximum Calibration Error (MCE)
    """

    y_true = np.array(y_true)
    bins = np.linspace(0, 1, n_bins + 1)

    gaps = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        else:
            mask = (y_proba >= bins[i]) & (y_proba <= bins[i+1])

        if mask.sum() == 0:
            continue

        gaps.append(abs(
            y_proba[mask].mean() - y_true[mask].mean()
        ))

    return max(gaps) if gaps else 0


# -------------------------------------------------------------------
# Full metrics calculation
# -------------------------------------------------------------------
def metrics_calculator(clf, X_test, y_test, model_name, n_bins=10):
    """
    Computes performance metrics for a trained classifier.

    Includes:
    - Classification metrics (accuracy, precision, recall, F1)
    - ROC-AUC and PR-AUC
    - Brier score and decomposition
    - ECE and MCE
    """

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    y_test = np.array(y_test).astype(float)

    # Calibration metrics
    brier = brier_decomposition(y_test, y_proba, n_bins=n_bins)
    ece = compute_ece(y_test, y_proba, n_bins=n_bins)
    mce = compute_mce(y_test, y_proba, n_bins=n_bins)

    # Main metrics
    result = pd.DataFrame(
        data=[
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_proba),
            average_precision_score(y_test, y_proba),
            brier_score_loss(y_test, y_proba),
            brier['REL (calibration)'],
            brier['RES (resolution)'],
            brier['UNC (uncertainty)'],
            ece,
            mce
        ],
        index=[
            'Accuracy', 'Precision', 'Recall', 'F1-score',
            'ROC-AUC', 'PR-AUC',
            'Brier Score',
            'REL (calibration)', 'RES (resolution)', 'UNC (uncertainty)',
            'ECE', 'MCE'
        ],
        columns=[model_name]
    )

    # Format output
    pct_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'PR-AUC']
    raw_metrics = ['Brier Score', 'REL (calibration)', 'RES (resolution)',
                   'UNC (uncertainty)', 'ECE', 'MCE']

    result_display = result.copy().astype(object)

    # Percentage metrics
    result_display.loc[pct_metrics] = (
        (result.loc[pct_metrics] * 100)
        .round(2)
        .astype(str) + '%'
    )

    # Raw metrics
    result_display.loc[raw_metrics] = (
        result.loc[raw_metrics]
        .round(4)
        .astype(str)
    )

    return result_display

