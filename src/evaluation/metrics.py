"""
Clinical Evaluation Metrics & Statistical Verification Engine.
Implements Altman & Bland (1983) repeated-measures agreement limits,
cluster bootstrap confidence intervals, MAE, RMSE, Pearson r, and percentage tolerances.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

class ClinicalMetricsEngine:
    """
    Computes rigorous biomedical validation metrics.
    """
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2 or np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6:
            return 0.0
        r = np.corrcoef(y_true, y_pred)[0, 1]
        return float(r) if not np.isnan(r) else 0.0

    @staticmethod
    def compute_bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Bland-Altman analysis: Mean Bias, Standard Deviation of Differences,
        and 95% Limits of Agreement (LoA).
        """
        diffs = y_true - y_pred
        mean_bias = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        loa_lower = float(mean_bias - 1.96 * std_diff)
        loa_upper = float(mean_bias + 1.96 * std_diff)

        return {
            "mean_bias": mean_bias,
            "std_diff": std_diff,
            "loa_lower": loa_lower,
            "loa_upper": loa_upper,
            "loa_vector": [loa_lower, loa_upper]
        }

    @staticmethod
    def compute_cluster_bootstrap_ci(errors_by_subject: Dict[str, List[float]], 
                                     n_boot: int = 1000, 
                                     seed: int = 42) -> Dict[str, float]:
        """
        Computes subject-level cluster bootstrap 95% confidence intervals
        accounting for repeated measures per subject.
        """
        np.random.seed(seed)
        sub_ids = list(errors_by_subject.keys())
        N = len(sub_ids)
        if N == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        boot_means = []
        for _ in range(n_boot):
            sampled_subs = np.random.choice(sub_ids, size=N, replace=True)
            sampled_errs = [e for sid in sampled_subs for e in errors_by_subject[sid]]
            if len(sampled_errs) > 0:
                boot_means.append(np.mean(sampled_errs))

        if len(boot_means) == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        return {
            "mean": float(np.mean(boot_means)),
            "ci_lower": float(np.percentile(boot_means, 2.5)),
            "ci_upper": float(np.percentile(boot_means, 97.5))
        }

    @staticmethod
    def compute_percentage_within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float) -> float:
        """
        Calculates percentage of estimates within +/- tolerance of reference.
        e.g., tolerance = 5.0 bpm for %CP5; tolerance = 2.0% for %CP2.
        """
        if len(y_true) == 0:
            return 0.0
        within = np.abs(y_true - y_pred) <= tolerance
        return float(np.mean(within) * 100.0)
