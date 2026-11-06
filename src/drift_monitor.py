import numpy as np
import pandas as pd
from typing import List, Dict, Any

class DriftMonitor:
    """Module kiểm soát Data & Model Drift cho hệ thống HCMC Logistics"""

    @staticmethod
    def calculate_psi(expected: List[float], actual: List[float], num_buckets: int = 10) -> Dict[str, Any]:
        """
        Tính chỉ số Population Stability Index (PSI) cho các đặc trưng (VD: khoảng cách, tốc độ H3)
        - PSI < 0.1: Tập dữ liệu ổn định (No Drift)
        - 0.1 <= PSI <= 0.2: Biến động nhẹ (Moderate Drift)
        - PSI > 0.2: Biến động mạnh (Severe Drift) -> Cần retrain mô hình
        """
        exp_arr = np.array(expected)
        act_arr = np.array(actual)

        if len(exp_arr) == 0 or len(act_arr) == 0:
            return {"psi": 0.0, "status": "INSUFFICIENT_DATA"}

        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(exp_arr, percentiles)
        buckets[0] -= 1e-5
        buckets[-1] += 1e-5

        expected_counts = np.histogram(exp_arr, buckets)[0]
        actual_counts = np.histogram(act_arr, buckets)[0]

        expected_pct = expected_counts / len(exp_arr)
        actual_pct = actual_counts / len(act_arr)

        # Tránh lỗi chia 0 hoặc log(0)
        expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
        actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

        psi_val = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        
        if psi_val < 0.1:
            status = "NO_DRIFT"
        elif psi_val <= 0.2:
            status = "MODERATE_DRIFT"
        else:
            status = "SEVERE_DRIFT_ALERT"

        return {
            "psi_score": round(psi_val, 4),
            "status": status,
            "action_required": psi_val > 0.2
        }

    @staticmethod
    def calculate_mae(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Tính sai số MAE giữa thời gian giao thực tế và ETA dự báo"""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return {"mae_minutes": 0.0, "sample_count": 0}
            
        mae_val = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
        return {
            "mae_minutes": round(mae_val, 2),
            "sample_count": len(y_true)
        }