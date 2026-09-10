#!/usr/bin/env python3
"""
Evaluation and Statistical Hypothesis Testing Engine.
Computes standard reliability and screening metrics across identical test partitions:
- Defect Detection Recall (Sensitivity)
- False Negative Escape Rate (FNR)
- Precision & F1-Score
- False Positive Rate (Yield Loss)
- PR-AUC & ROC-AUC
- Early Lead Time
- Exact McNemar Test & 2,000 Paired Bootstrap Confidence Intervals
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_screening_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    failure_times: Optional[List[Optional[float]]] = None,
    t_screen: float = 47.0
) -> Dict[str, float]:
    """
    Computes standard screening and reliability evaluation metrics.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    total = len(y_t)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # PR-AUC and ROC-AUC
    # If continuous scores provided, calculate exact AUC; otherwise use binary predictions
    score_vector = np.array(scores, dtype=float) if scores is not None else y_p.astype(float)
    try:
        # Check if true labels have both classes
        if len(np.unique(y_t)) > 1:
            pr_auc = float(average_precision_score(y_t, score_vector))
            roc_auc = float(roc_auc_score(y_t, score_vector))
        else:
            pr_auc = 1.0 if y_t[0] == 1 else 0.0
            roc_auc = 0.5
    except Exception:
        pr_auc = recall
        roc_auc = 0.5

    # Early Lead Time Analysis
    lead_times = []
    if failure_times is not None:
        for i in range(total):
            if y_t[i] == 1 and y_p[i] == 1 and failure_times[i] is not None:
                ft = float(failure_times[i])
                if ft > t_screen:
                    lead_times.append(ft - t_screen)

    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    min_lead_time = float(np.min(lead_times)) if lead_times else 0.0
    max_lead_time = float(np.max(lead_times)) if lead_times else 0.0

    return {
        "total_samples": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "false_negative_rate": round(fnr, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "yield_loss_pct": round(fpr * 100.0, 2),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "early_detected_count": tp,
        "total_latent_defects": int(np.sum(y_t == 1)),
        "mean_lead_time_hours": round(mean_lead_time, 2),
        "min_lead_time_hours": round(min_lead_time, 2),
        "max_lead_time_hours": round(max_lead_time, 2)
    }


def mcnemar_exact_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict:
    """
    Computes exact two-tailed McNemar test for paired binary classification.
    Tests H0: Method A and Method B have equal marginal error rates.
    """
    y_t = np.array(y_true, dtype=int)
    y_a = np.array(y_pred_a, dtype=int)
    y_b = np.array(y_pred_b, dtype=int)

    # Correct classifications
    corr_a = (y_a == y_t)
    corr_b = (y_b == y_t)

    # Discordant pairs:
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    b = int(np.sum(corr_a & (~corr_b)))
    c = int(np.sum((~corr_a) & corr_b))
    n = b + c

    if n == 0:
        p_val = 1.0
    else:
        # Exact two-tailed binomial test under H0: p = 0.5
        p_val = float(stats.binomtest(b, n, 0.5, alternative="two-sided").pvalue)

    return {
        "b_discordant (A correct, B incorrect)": b,
        "c_discordant (A incorrect, B correct)": c,
        "total_discordant": n,
        "exact_p_value": round(p_val, 6),
        "statistically_significant_005": p_val < 0.05
    }


def paired_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_trad: np.ndarray,
    y_pred_candidate: np.ndarray,
    n_iterations: int = 2000,
    seed: int = 42
) -> Dict:
    """
    Computes 95% paired bootstrap confidence intervals for Delta Recall and Delta F1.
    """
    rng = np.random.RandomState(seed)
    y_t = np.array(y_true, dtype=int)
    y_trad = np.array(y_pred_trad, dtype=int)
    y_cand = np.array(y_pred_candidate, dtype=int)
    N = len(y_t)

    delta_recalls = []
    delta_f1s = []

    for _ in range(n_iterations):
        idx = rng.choice(N, size=N, replace=True)
        yt_boot = y_t[idx]
        ytrad_boot = y_trad[idx]
        ycand_boot = y_cand[idx]

        # Calculate metrics for bootstrap sample
        m_trad = calculate_screening_metrics(yt_boot, ytrad_boot)
        m_cand = calculate_screening_metrics(yt_boot, ycand_boot)

        delta_recalls.append(m_cand["recall"] - m_trad["recall"])
        delta_f1s.append(m_cand["f1_score"] - m_trad["f1_score"])

    # 95% Percentile Confidence Intervals
    r_ci_lower = float(np.percentile(delta_recalls, 2.5))
    r_ci_upper = float(np.percentile(delta_recalls, 97.5))
    f_ci_lower = float(np.percentile(delta_f1s, 2.5))
    f_ci_upper = float(np.percentile(delta_f1s, 97.5))

    return {
        "delta_recall_mean": round(float(np.mean(delta_recalls)), 4),
        "delta_recall_ci_95": [round(r_ci_lower, 4), round(r_ci_upper, 4)],
        "delta_f1_mean": round(float(np.mean(delta_f1s)), 4),
        "delta_f1_ci_95": [round(f_ci_lower, 4), round(f_ci_upper, 4)],
        "bootstrap_iterations": n_iterations
    }


def build_master_comparison_table(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    scores_dict: Optional[Dict[str, np.ndarray]] = None,
    failure_times: Optional[List[Optional[float]]] = None,
    t_screen: float = 47.0
) -> pd.DataFrame:
    """
    Builds the unified comparative evaluation matrix across all screening paradigms.
    """
    rows = []
    for model_name, y_pred in predictions_dict.items():
        scores = scores_dict.get(model_name) if scores_dict else None
        metrics = calculate_screening_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            failure_times=failure_times,
            t_screen=t_screen
        )
        row = {"method": model_name}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)
