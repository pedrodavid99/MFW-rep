"""
Evaluation module for model assessment.

This file contains functions for:
- Model evaluation plots
- Threshold analysis
- Calibration analysis
- Calibration curve comparison
- SHAP cumulative importance plots

Used in:
"Lapse and Mid-Term Prediction Models"

Note:
The dataset is not included in the repository due to confidentiality constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import shap

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from matplotlib.colors import LinearSegmentedColormap

from metrics import (
    brier_decomposition,
    compute_ece,
    compute_mce,
    metrics_calculator
)


# -------------------------------------------------------------------
# Model evaluation: classification report, ROC, PR curve, confusion matrix
# -------------------------------------------------------------------
def model_evaluation(clf, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluates a fitted classifier using:
    - Classification reports
    - Probability metrics
    - Confusion matrix
    - ROC curve
    - Precision-recall curve
    - Summary metrics table
    """

    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)

    sns.set(font_scale=1.2)

    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]

    def side_by_side(left_title, left_text, right_title, right_text, pad=60):
        """
        Prints two text blocks side by side.
        """
        left_lines = (left_title + "\n" + "-" * 55 + "\n" + left_text).splitlines()
        right_lines = (right_title + "\n" + "-" * 35 + "\n" + right_text).splitlines()

        max_lines = max(len(left_lines), len(right_lines))
        left_lines += [""] * (max_lines - len(left_lines))
        right_lines += [""] * (max_lines - len(right_lines))

        for left, right in zip(left_lines, right_lines):
            print(f"{left:<{pad}}{right}")

        print()

    def prob_metrics_str(split_name, y_true, y_prob):
        """
        Returns probability-based metrics as formatted text.
        """
        bs = brier_score_loss(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
        ece = compute_ece(y_true, y_prob)

        return (
            f"  {split_name} Probability Metrics\n"
            f"  {'Brier Score':<18} {bs:.4f}\n"
            f"  {'ROC-AUC':<18} {auc:.4f}\n"
            f"  {'PR-AUC':<18} {pr:.4f}\n"
            f"  {'ECE':<18} {ece:.4f}\n"
        )

    # Training set
    y_pred_train = clf.predict(X_train)

    side_by_side(
        "\t Classification report — TRAIN",
        classification_report(y_train, y_pred_train),
        "\t Probability Metrics — TRAIN",
        prob_metrics_str("Train", y_train, prob_train)
    )

    # Test set
    y_pred_test = clf.predict(X_test)

    brier_gap = round(
        brier_score_loss(y_test, prob_test) -
        brier_score_loss(y_train, prob_train),
        4
    )

    auc_gap = round(
        roc_auc_score(y_test, prob_test) -
        roc_auc_score(y_train, prob_train),
        4
    )

    gap_str = (
        f"  Overfitting Check (Test - Train)\n"
        f"  {'Brier gap':<18} {brier_gap:+.4f}\n"
        f"  {'ROC-AUC gap':<18} {auc_gap:+.4f}\n"
    )

    side_by_side(
        "\t Classification report — TEST",
        classification_report(y_test, y_pred_test),
        "\t Probability Metrics — TEST",
        prob_metrics_str("Test", y_test, prob_test) + gap_str
    )

    # Plots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1,
        4,
        figsize=(22, 7),
        dpi=100,
        gridspec_kw={"width_ratios": [2, 2, 2, 1.5]}
    )

    royalblue = LinearSegmentedColormap.from_list(
        "royalblue",
        [(0, (1, 1, 1)), (1, (0.25, 0.41, 0.88))]
    )
    royalblue_r = royalblue.reversed()

    ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        colorbar=False,
        cmap=royalblue_r,
        ax=ax1
    )
    ax1.set_title("Confusion Matrix for Test Data")
    ax1.grid(False)

    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax2)
    ax2.set_title("ROC Curve for Test Data")

    PrecisionRecallDisplay.from_estimator(clf, X_test, y_test, ax=ax3)
    ax3.set_title("Precision-Recall Curve for Test Data")

    result = metrics_calculator(clf, X_test, y_test, model_name)

    table = ax4.table(
        cellText=result.values,
        colLabels=result.columns,
        rowLabels=result.index,
        loc="center"
    )
    table.scale(0.7, 1.5)
    table.set_fontsize(10)

    ax4.axis("tight")
    ax4.axis("off")

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_color("royalblue")

    plt.suptitle(model_name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Interactive threshold analysis using model predictions
# -------------------------------------------------------------------
def plot_metrics_vs_threshold_interactive(
    model,
    X,
    y_true,
    thresholds=np.arange(0, 1.01, 0.01),
    title="Metrics vs Threshold"
):
    """
    Creates an interactive Plotly plot showing:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - True positives
    - False positives

    across different probability thresholds.
    """

    y_proba = model.predict_proba(X)[:, 1]

    return plot_metrics_vs_threshold_interactive_probs(
        y_proba=y_proba,
        y_true=y_true,
        thresholds=thresholds,
        title=title
    )


# -------------------------------------------------------------------
# Interactive threshold analysis using probability scores
# -------------------------------------------------------------------
def plot_metrics_vs_threshold_interactive_probs(
    y_proba,
    y_true,
    thresholds=np.arange(0, 1.01, 0.01),
    title="Metrics vs Threshold"
):
    """
    Creates an interactive threshold plot directly from probability scores.
    Useful after probability calibration.
    """

    y_true = np.array(y_true)

    accs = []
    precs = []
    recs = []
    f1s = []
    tps = []
    fps = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        accs.append(accuracy_score(y_true, y_pred))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tp = cm[1, 1]
            fp = cm[0, 1]
        else:
            tp = 0
            fp = 0

        tps.append(tp)
        fps.append(fp)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=accs,
            mode="lines",
            name="Accuracy",
            line=dict(color="royalblue")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precs,
            mode="lines",
            name="Precision",
            line=dict(color="orange")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recs,
            mode="lines",
            name="Recall",
            line=dict(color="green")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1s,
            mode="lines",
            name="F1-score",
            line=dict(color="red")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=tps,
            mode="lines",
            name="True Positives",
            line=dict(dash="dot", color="purple")
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=fps,
            mode="lines",
            name="False Positives",
            line=dict(dash="dot", color="brown")
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        xaxis_title="Threshold",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        hovermode="x unified",
        template="plotly_white",
        width=950,
        height=500
    )

    fig.update_yaxes(
        title_text="Performance Metrics",
        range=[0, 1],
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Count (TP/FP)",
        secondary_y=True
    )

    fig.show()

    return fig


# -------------------------------------------------------------------
# Isotonic calibration results
# -------------------------------------------------------------------
def calibration_results(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Fits isotonic calibration using a held-out calibration split.

    Returns:
    - Calibration metrics table
    - A dictionary containing the fitted model and isotonic calibrator
    """

    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)

    # Split training data into model-training and calibration subsets
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # Fit model on reduced training data
    fitted_model = clone(model)
    fitted_model.fit(X_tr, y_tr)

    # Fit isotonic regression on calibration set
    prob_cal_data = fitted_model.predict_proba(X_cal)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_cal_data, y_cal)

    # Apply calibration to test probabilities
    prob_raw = fitted_model.predict_proba(X_test)[:, 1]
    prob_cal = iso.predict(prob_raw)

    # Brier scores
    brier_raw = brier_score_loss(y_test, prob_raw)
    brier_cal = brier_score_loss(y_test, prob_cal)

    print(f"Brier Score — Uncalibrated: {brier_raw:.4f}")
    print(f"Brier Score — Calibrated:   {brier_cal:.4f}")

    # Brier decomposition
    decomp_raw = brier_decomposition(y_test, prob_raw)
    decomp_cal = brier_decomposition(y_test, prob_cal)

    results = pd.DataFrame({
        "Uncalibrated": decomp_raw,
        "Isotonic Calibrated": decomp_cal
    }).round(4)

    results.loc["Brier Score", "Uncalibrated"] = round(brier_raw, 4)
    results.loc["Brier Score", "Isotonic Calibrated"] = round(brier_cal, 4)

    results["Difference"] = (
        results["Isotonic Calibrated"] -
        results["Uncalibrated"]
    ).round(4)

    results["Difference percent"] = (
        (
            results["Isotonic Calibrated"] -
            results["Uncalibrated"]
        ) / results["Uncalibrated"] * 100
    ).round(2).astype(str) + "%"

    print(results.to_string())

    # ECE and MCE
    ece_raw = compute_ece(y_test, prob_raw)
    ece_cal = compute_ece(y_test, prob_cal)

    mce_raw = compute_mce(y_test, prob_raw)
    mce_cal = compute_mce(y_test, prob_cal)

    ece_table = pd.DataFrame({
        "Uncalibrated": {"ECE": ece_raw, "MCE": mce_raw},
        "Isotonic Calibrated": {"ECE": ece_cal, "MCE": mce_cal}
    }).round(4)

    ece_table["Difference"] = (
        ece_table["Isotonic Calibrated"] -
        ece_table["Uncalibrated"]
    ).round(4)

    ece_table["Difference percent"] = (
        (
            ece_table["Isotonic Calibrated"] -
            ece_table["Uncalibrated"]
        ) / ece_table["Uncalibrated"] * 100
    ).round(2).astype(str) + "%"

    print(ece_table.to_string())

    # Static calibration plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    CalibrationDisplay.from_predictions(
        y_test,
        prob_raw,
        n_bins=10,
        ax=ax1,
        name=f"Uncalibrated (BS={brier_raw:.4f})",
        color="tomato"
    )

    CalibrationDisplay.from_predictions(
        y_test,
        prob_cal,
        n_bins=10,
        ax=ax1,
        name=f"Isotonic (BS={brier_cal:.4f})",
        color="royalblue"
    )

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax1.set_title(f"Calibration Curve — {model_name}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.hist(
        prob_raw,
        bins=30,
        alpha=0.5,
        color="tomato",
        label="Uncalibrated"
    )

    ax2.hist(
        prob_cal,
        bins=30,
        alpha=0.5,
        color="royalblue",
        label="Calibrated"
    )

    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Interactive threshold plot for calibrated probabilities
    plot_metrics_vs_threshold_interactive_probs(
        prob_cal,
        y_test,
        title=f"Metrics vs Threshold — {model_name} Calibrated"
    )

    calib_result = pd.DataFrame({
        model_name: {
            "Brier Score": brier_cal,
            "REL (calibration)": decomp_cal["REL (calibration)"],
            "RES (resolution)": decomp_cal["RES (resolution)"],
            "UNC (uncertainty)": decomp_cal["UNC (uncertainty)"],
            "ECE": ece_cal,
            "MCE": mce_cal
        }
    }).round(4)

    calibrated_bundle = {
        "model": fitted_model,
        "isotonic": iso
    }

    return calib_result, calibrated_bundle


# -------------------------------------------------------------------
# Interactive calibration curve
# -------------------------------------------------------------------
def plot_calibration_curve(model, X_test, y_test, model_name="Model", n_bins=10):
    """
    Creates interactive calibration curves using:
    - Uniform bins
    - Quantile bins
    """

    probs = model.predict_proba(X_test)[:, 1]
    y_test = np.array(y_test)

    def get_bins(strategy):
        if strategy == "uniform":
            bins = np.linspace(0, 1, n_bins + 1)
            bin_ids = np.digitize(probs, bins[1:-1])
        else:
            bin_ids, _ = pd.qcut(
                probs,
                q=n_bins,
                retbins=True,
                labels=False,
                duplicates="drop"
            )

        bin_centers = []
        frac_pos = []
        counts = []
        bin_labels = []

        for bin_id in np.unique(bin_ids):
            mask = bin_ids == bin_id

            if mask.sum() > 0:
                lower = float(probs[mask].min())
                upper = float(probs[mask].max())

                bin_centers.append(float(probs[mask].mean()))
                frac_pos.append(float(y_test[mask].mean()))
                counts.append(int(mask.sum()))
                bin_labels.append(f"[{lower:.3f},{upper:.3f}]")

        return bin_centers, frac_pos, counts, bin_labels

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{model_name} -- Uniform Bins",
            f"{model_name} -- Quantile Bins",
            "Score Distribution (Uniform)",
            "Score Distribution (Quantile)"
        )
    )

    for col, strategy in enumerate(["uniform", "quantile"], start=1):
        centers, frac, counts, labels = get_bins(strategy)

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect calibration",
                showlegend=(col == 1)
            ),
            row=1,
            col=col
        )

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=frac,
                mode="lines+markers",
                marker=dict(size=8),
                line=dict(color="steelblue"),
                name=model_name,
                showlegend=(col == 1),
                customdata=[[count] for count in counts],
                hovertemplate=(
                    "<b>Mean predicted prob:</b> %{x:.3f}<br>"
                    "<b>Fraction positive:</b> %{y:.3f}<br>"
                    "<b>Count in bin:</b> %{customdata[0]}"
                    "<extra></extra>"
                )
            ),
            row=1,
            col=col
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=counts,
                marker_color="steelblue",
                opacity=0.6,
                showlegend=False,
                hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>"
            ),
            row=2,
            col=col
        )

    fig.update_xaxes(title_text="Mean predicted probability", row=1)
    fig.update_yaxes(title_text="Fraction of positives", row=1)
    fig.update_xaxes(title_text="Bin", row=2, tickangle=45)
    fig.update_yaxes(title_text="Count", row=2)

    fig.update_layout(
        height=750,
        title_text=f"Calibration Curves -- {model_name}",
        template="plotly_white"
    )

    fig.show()

    return fig


# -------------------------------------------------------------------
# Compare calibration curves for multiple models
# -------------------------------------------------------------------
def compare_calibration_curves(models_dict, y_test, n_bins=10):
    """
    Compares calibration curves for multiple models.

    Parameters
    ----------
    models_dict : dict
        Dictionary in the form:
        {
            "Model name": (model, X_test)
        }

    y_test : array-like
        True test labels.
    """

    colors = [
        "tomato",
        "royalblue",
        "seagreen",
        "darkorange",
        "purple",
        "brown"
    ]

    y_test = np.array(y_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    ax1.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Perfect calibration",
        linewidth=1.5
    )

    for (label, (model, X_test)), color in zip(models_dict.items(), colors):
        probs = model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, probs)

        CalibrationDisplay.from_predictions(
            y_test,
            probs,
            n_bins=n_bins,
            ax=ax1,
            name=f"{label} (BS={brier:.4f})",
            color=color
        )

        ax2.hist(
            probs,
            bins=40,
            alpha=0.4,
            color=color,
            label=label,
            edgecolor="white"
        )

    ax1.set_title("Calibration Curves — Comparison")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Score Distributions")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Evaluate effect of calibration
# -------------------------------------------------------------------
def evaluate_calibration_effect(raw_model, cal_model, X_test, y_test,
                                model_name="Model", n_bins=10):
    """
    Compares an uncalibrated model against a calibrated model.

    raw_model:
        Fitted uncalibrated sklearn-compatible model.

    cal_model:
        Fitted calibrated sklearn-compatible model or wrapper with predict_proba.
    """

    y_test = np.array(y_test).astype(float)

    prob_raw = raw_model.predict_proba(X_test)[:, 1]
    prob_cal = cal_model.predict_proba(X_test)[:, 1]

    brier_raw = brier_score_loss(y_test, prob_raw)
    brier_cal = brier_score_loss(y_test, prob_cal)

    print(f"Brier Score — Uncalibrated: {brier_raw:.4f}")
    print(f"Brier Score — Calibrated:   {brier_cal:.4f}")

    decomp_raw = brier_decomposition(y_test, prob_raw)
    decomp_cal = brier_decomposition(y_test, prob_cal)

    results = pd.DataFrame({
        "Uncalibrated": decomp_raw,
        "Calibrated": decomp_cal
    }).round(4)

    results.loc["Brier Score", "Uncalibrated"] = round(brier_raw, 4)
    results.loc["Brier Score", "Calibrated"] = round(brier_cal, 4)

    results["Difference"] = (
        results["Calibrated"] -
        results["Uncalibrated"]
    ).round(4)

    results["Difference percent"] = results.apply(
        lambda row: "0.0%" if row["Uncalibrated"] == 0
        else f"{round((row['Calibrated'] - row['Uncalibrated']) / row['Uncalibrated'] * 100, 2)}%",
        axis=1
    )

    print(results.to_string())

    ece_table = pd.DataFrame({
        "Uncalibrated": {
            "ECE": compute_ece(y_test, prob_raw),
            "MCE": compute_mce(y_test, prob_raw)
        },
        "Calibrated": {
            "ECE": compute_ece(y_test, prob_cal),
            "MCE": compute_mce(y_test, prob_cal)
        }
    }).round(4)

    ece_table["Difference"] = (
        ece_table["Calibrated"] -
        ece_table["Uncalibrated"]
    ).round(4)

    ece_table["Difference percent"] = ece_table.apply(
        lambda row: "0.0%" if row["Uncalibrated"] == 0
        else f"{round((row['Calibrated'] - row['Uncalibrated']) / row['Uncalibrated'] * 100, 2)}%",
        axis=1
    )

    print(ece_table.to_string())

    # Interactive calibration effect plot
    def get_quantile_bins(probs, n_bins):
        bin_ids, _ = pd.qcut(
            probs,
            q=n_bins,
            retbins=True,
            labels=False,
            duplicates="drop"
        )

        centers = []
        fracs = []
        counts = []
        labels = []

        for bin_id in np.unique(bin_ids):
            mask = bin_ids == bin_id

            if mask.sum() > 0:
                centers.append(float(probs[mask].mean()))
                fracs.append(float(y_test[mask].mean()))
                counts.append(int(mask.sum()))
                labels.append(
                    f"[{probs[mask].min():.3f},{probs[mask].max():.3f}]"
                )

        return centers, fracs, counts, labels

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{model_name} — Calibration Curve (Quantile Bins)",
            f"{model_name} — Score Distribution"
        )
    )

    for probs, name, color in [
        (prob_raw, f"Uncalibrated (BS={brier_raw:.4f})", "tomato"),
        (prob_cal, f"Calibrated (BS={brier_cal:.4f})", "royalblue")
    ]:
        centers, fracs, counts, labels = get_quantile_bins(probs, n_bins)

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=fracs,
                mode="lines+markers",
                marker=dict(size=8),
                line=dict(color=color),
                name=name,
                customdata=[[count] for count in counts],
                hovertemplate=(
                    "<b>Mean prob:</b> %{x:.3f}<br>"
                    "<b>Frac positive:</b> %{y:.3f}<br>"
                    "<b>Count:</b> %{customdata[0]}"
                    "<extra></extra>"
                )
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Histogram(
                x=probs,
                nbinsx=40,
                name=name,
                marker_color=color,
                opacity=0.5,
                showlegend=False,
                hovertemplate="Prob: %{x:.3f}<br>Count: %{y}<extra></extra>"
            ),
            row=1,
            col=2
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect calibration"
        ),
        row=1,
        col=1
    )

    fig.update_xaxes(title_text="Mean predicted probability", row=1, col=1)
    fig.update_yaxes(title_text="Fraction of positives", row=1, col=1)

    fig.update_xaxes(title_text="Predicted probability", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        height=500,
        template="plotly_white",
        title_text=f"Calibration Effect — {model_name}"
    )

    fig.show()

    return fig


# -------------------------------------------------------------------
# Cumulative SHAP importance plot
# -------------------------------------------------------------------
def plot_cumulative_shap(
    model,
    X_train: pd.DataFrame,
    model_name: str = "Model",
    thresholds: list = [80, 90, 95]
):
    """
    Plots an interactive cumulative SHAP feature importance curve.

    Parameters
    ----------
    model:
        Fitted tree-based model.

    X_train:
        Encoded training features as a pandas DataFrame.

    model_name:
        Name used in the plot title.

    thresholds:
        Cumulative importance thresholds to highlight.
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs_shap = pd.Series(
        np.abs(shap_arr).mean(axis=0),
        index=X_train.columns
    )

    shap_sorted = mean_abs_shap.sort_values(ascending=False)

    shap_pct = shap_sorted / shap_sorted.sum() * 100
    shap_cumulative = shap_pct.cumsum()

    n_features = list(range(1, len(shap_cumulative) + 1))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=n_features,
            y=shap_cumulative.values,
            mode="lines",
            name="Cumulative SHAP",
            line=dict(color="#1f77b4", width=2),
            hovertemplate=(
                "Top %{x} features → %{y:.1f}%"
                "<extra></extra>"
            )
        )
    )

    colors = {
        80: "green",
        90: "orange",
        95: "red"
    }

    for threshold in thresholds:
        n_at_threshold = int((shap_cumulative <= threshold).sum()) + 1
        color = colors.get(threshold, "gray")

        fig.add_hline(
            y=threshold,
            line=dict(dash="dash", color=color, width=1),
            annotation_text=f"{threshold}% ({n_at_threshold} features)",
            annotation_position="right"
        )

        fig.add_vline(
            x=n_at_threshold,
            line=dict(dash="dot", color=color, width=1)
        )

    fig.update_layout(
        title=f"Cumulative SHAP Importance — {model_name} (training set)",
        xaxis_title="Number of features ranked by mean absolute SHAP value",
        yaxis_title="Cumulative SHAP importance (%)",
        yaxis=dict(range=[0, 101]),
        xaxis=dict(range=[1, len(shap_cumulative)]),
        hovermode="x unified",
        template="plotly_white",
        width=900,
        height=450
    )

    fig.show()

    return shap_sorted, shap_cumulative, fig

