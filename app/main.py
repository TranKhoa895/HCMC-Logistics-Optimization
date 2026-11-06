from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import time
import os
import sys
import builtins
import pandas as pd 
import numpy as np 
import shap 

# --- 1. UNPICKLE CUSTOM LOSS FUNCTION ---
def asymmetric_objective(y_true, y_pred):
    alpha = 2.5
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * alpha * residual, -2.0 * residual)
    hess = np.where(residual > 0, 2.0 * alpha, 2.0)
    return grad, hess

asymetric_objective = asymmetric_objective
setattr(sys.modules['__main__'], 'asymetric_objective', asymmetric_objective)
setattr(sys.modules['__main__'], 'asymmetric_objective', asymmetric_objective)
setattr(builtins, 'asymetric_objective', asymmetric_objective)
setattr(builtins, 'asymmetric_objective', asymmetric_objective)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.route_optimization import RouteOptimizationEngine
from src.drift_monitor import DriftMonitor

# --- 2. IMPORT SCHEMAS ---
from app.schemas import (
    PredictETARequest, PredictETAResponse,
    OptimizeRouteRequest, OptimizeRouteResponse,
    DriftCheckRequest, FeedbackRequest
)

app = FastAPI(
    title="HCMC Logistics Serving & Application API", 
    description="Production API providing asymmetric ETA predictions, CVRP route optimization, and Prometheus/Drift Monitoring.",
    version="1.0.0"
)

# --- 3. EXPOSE PROMETHEUS METRICS METRICS ENDPOINT ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# --- 4. KHỞI TẠO ENGINE & SHAP ---
try: 
    engine = RouteOptimizationEngine(model_path="data/model_eta.pkl")
    explainer = shap.TreeExplainer(engine.model)
    print("✅ Đã khởi tạo thành công Route Optimization Engine & SHAP Explainer!")
except Exception as e:
    print(f"❌ Lỗi nạp mô hình ETA: {e}")
    sys.exit(1)

# --- 5. API ENDPOINTS ---
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "HEALTHY",
        "service": "HCMC-Logistics-Serving-Layer",
        "timestamp": time.time()
    }

@app.post("/api/v1/predict-eta", response_model=PredictETAResponse)
def predict_eta(req: PredictETARequest):
    start_t = time.time()
    try:
        R = 6371.0
        dlat = np.radians(req.dropoff_lat - req.pickup_lat)
        dlon = np.radians(req.dropoff_lon - req.pickup_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(req.pickup_lat)) * np.cos(np.radians(req.dropoff_lat)) * np.sin(dlon/2)**2
        dist_km = max(2 * R * np.arcsin(np.sqrt(a)) * 1.35, 0.5)

        base_speed = 26.0 if req.vehicle_type == 'MOTORBIKE' else 20.0
        base_duration = (dist_km / base_speed) * 60.0
        is_peak = 1 if (7 <= req.departure_hour <= 9 or 17 <= req.departure_hour <= 19) else 0

        df_feat = pd.DataFrame([{
            'pickup_lat': req.pickup_lat, 'pickup_lon': req.pickup_lon,
            'dropoff_lat': req.dropoff_lat, 'dropoff_lon': req.dropoff_lon,
            'osrm_distance_km': dist_km,
            'base_duration_min': base_duration,
            'manhattan_dist_km': dist_km * 1.15,
            'bearing_angle': 120.0,
            'hour': req.departure_hour, 'dayofweek': 2, 'is_weekend': 0, 'is_peak_hour': is_peak,
            'sin_hour': np.sin(2*np.pi*req.departure_hour/24), 'cos_hour': np.cos(2*np.pi*req.departure_hour/24),
            'sin_dayofweek': 0.5, 'cos_dayofweek': 0.866,
            'pickup_h3_avg_speed': 18.5,
            'vehicle_type_encoded': 0 if req.vehicle_type == 'MOTORBIKE' else 1,
            'is_rainy': 0, 'is_same_h3_r8': 0
        }])[engine.feature_columns]

        eta_min = float(engine.model.predict(df_feat)[0])

        shap_vals = explainer.shap_values(df_feat)[0]
        feature_names = engine.feature_columns
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:4]
        shap_dict = {feature_names[i]: round(float(shap_vals[i]), 3) for i in top_indices}

        latency = (time.time() - start_t) * 1000.0
        return {
            "status": "SUCCESS",
            "predicted_eta_minutes": round(eta_min, 2),
            "predicted_eta_seconds": round(eta_min * 60.0, 1),
            "distance_km": round(dist_km, 2),
            "shapley_values": shap_dict,
            "latency_ms": round(latency, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/optimize-route", response_model=OptimizeRouteResponse)
def optimize_route(req: OptimizeRouteRequest):
    start_t = time.time()
    if len(req.stops) < 2:
        raise HTTPException(status_code=400, detail="Cần tối thiểu 2 điểm dừng (1 Depot + 1 Đơn hàng).")

    stops_data = [stop.dict() for stop in req.stops]

    result = engine.solve_cvrp(
        stops=stops_data,
        vehicle_capacity=req.vehicle_capacity,
        num_vehicles=1,
        vehicle_type=req.vehicle_type,
        hour=req.departure_hour
    )

    latency = (time.time() - start_t) * 1000.0

    if result['status'] == 'SUCCESS':
        total_min = result['total_travel_time_min']
        sla_warning = True if total_min > 120.0 else False

        return {
            "status": "SUCCESS",
            "driver_id": req.driver_id,
            "optimized_route_sequence": result['optimized_route_sequence'],
            "stop_order_ids": result['stop_order_ids'],
            "total_travel_time_min": total_min,
            "total_travel_time_sec": round(total_min * 60.0, 1),
            "sla_risk_warning": sla_warning,
            "latency_ms": round(latency, 2)
        }
    else:
        raise HTTPException(status_code=422, detail=result.get('message', 'Không tìm thấy tuyến đường khả thi'))

# --- MONITORING & DRIFT ENDPOINTS ---
@app.post("/api/v1/monitor/psi")
def check_data_drift(req: DriftCheckRequest):
    """API kiểm tra Data Drift (PSI Index) giữa dữ liệu Baseline và Current"""
    result = DriftMonitor.calculate_psi(req.baseline_distances, req.current_distances)
    return {"status": "SUCCESS", "drift_analysis": result}

@app.post("/api/v1/monitor/feedback")
def track_model_accuracy(req: FeedbackRequest):
    """API Feedback Loop tính MAE giữa thời gian giao thực tế và ETA dự báo"""
    result = DriftMonitor.calculate_mae(req.actual_delivery_times, req.predicted_delivery_times)
    return {"status": "SUCCESS", "accuracy_metrics": result}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)