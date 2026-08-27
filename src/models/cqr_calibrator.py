"""
[EXPERIMENTAL - Canlı pipeline'da kullanılmıyor]
Conformalized Quantile Regression (CQR) Calibration Module

Provides finite-sample validity calibration for LightGBM/Neural Network prediction intervals
(P10 and P90 bounds) using historical out-of-sample calibration non-conformity scores.
lgb_cqr_v2 modeli bu modülü kullanır ve gold.ptf_predictions_experimental tablosunda
mevcuttur (2024-08-11 → 2026-08-12).
"""

import numpy as np
import pandas as pd

class CQRCalibrator:
    def __init__(self, target_coverage=0.80, cal_window_days=30):
        """
        :param target_coverage: Target coverage rate (default 0.80 for 80% confidence interval)
        :param cal_window_days: Lookback calibration window in days (default 30 days = 720 hours)
        """
        self.target_coverage = target_coverage
        self.cal_window_hours = cal_window_days * 24
        
    def calibrate(self, p10_hist, p90_hist, y_hist, p10_future, p90_future):
        """
        Calibrates future P10 and P90 prediction bounds based on recent historical non-conformity scores.
        
        :param p10_hist: Historical predicted P10 bounds (recent 30 days)
        :param p90_hist: Historical predicted P90 bounds (recent 30 days)
        :param y_hist: Historical actual values (recent 30 days)
        :param p10_future: Tomorrow's raw predicted P10 bounds (24 hours)
        :param p90_future: Tomorrow's raw predicted P90 bounds (24 hours)
        :return: (p10_cqr, p90_cqr) CQR-calibrated bounds
        """
        if len(y_hist) == 0 or len(p10_hist) == 0 or len(p90_hist) == 0:
            return p10_future, p90_future
            
        # Take the most recent calibration window
        y_cal = np.array(y_hist[-self.cal_window_hours:])
        p10_cal = np.array(p10_hist[-self.cal_window_hours:])
        p90_cal = np.array(p90_hist[-self.cal_window_hours:])
        
        # Non-conformity score: distance to nearest bound if out of bounds, or negative interior distance
        scores = np.maximum(p10_cal - y_cal, y_cal - p90_cal)
        
        # Compute (1 - alpha) quantile of non-conformity scores where alpha = 1.0 - target_coverage
        # For 80% coverage, quantile is 80th percentile
        n = len(scores)
        quantile_val = min(1.0, self.target_coverage * (1.0 + 1.0 / n))
        q_hat = np.quantile(scores, quantile_val)
        
        # Adjust future prediction bounds
        p10_cqr = p10_future - q_hat
        p90_cqr = p90_future + q_hat
        
        # Ensure monotonic bounds (P10 <= P90)
        p10_cqr = np.minimum(p10_cqr, p90_cqr)
        p90_cqr = np.maximum(p90_cqr, p10_cqr)
        
        return p10_cqr, p90_cqr
