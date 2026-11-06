from pydantic import BaseModel, Field
from typing import List, Dict

# ==================== ETA PREDICTION SCHEMAS ====================
class PredictETARequest(BaseModel):
    pickup_lat: float = Field(..., json_schema_extra={"example": 10.7760}, description="Vĩ độ điểm lấy hàng")
    pickup_lon: float = Field(..., json_schema_extra={"example": 106.7000}, description="Kinh độ điểm lấy hàng")
    dropoff_lat: float = Field(..., json_schema_extra={"example": 10.7740}, description="Vĩ độ điểm giao hàng")
    dropoff_lon: float = Field(..., json_schema_extra={"example": 106.7030}, description="Kinh độ điểm giao hàng")
    vehicle_type: str = Field("MOTORBIKE", json_schema_extra={"example": "MOTORBIKE"}, description="MOTORBIKE hoặc TRUCK")
    departure_hour: int = Field(14, ge=0, le=23, description="Giờ xuất phát (0-23)")

class PredictETAResponse(BaseModel):
    status: str
    predicted_eta_minutes: float
    predicted_eta_seconds: float
    distance_km: float
    shapley_values: Dict[str, float]
    latency_ms: float


# ==================== ROUTE OPTIMIZATION SCHEMAS ====================
class StopLocation(BaseModel):
    order_id: str = Field(..., json_schema_extra={"example": "ORD_001_NGUYEN_HUE"})
    lat: float = Field(..., json_schema_extra={"example": 10.7740})
    lon: float = Field(..., json_schema_extra={"example": 106.7030})

class OptimizeRouteRequest(BaseModel):
    driver_id: str = Field(..., json_schema_extra={"example": "DRV_HCMC_99"}, description="Mã tài xế")
    vehicle_type: str = Field("MOTORBIKE", json_schema_extra={"example": "MOTORBIKE"})
    vehicle_capacity: int = Field(15, ge=1, le=100)
    departure_hour: int = Field(14, ge=0, le=23)
    stops: List[StopLocation]

class OptimizeRouteResponse(BaseModel):
    status: str
    driver_id: str
    optimized_route_sequence: List[int]
    stop_order_ids: List[str]
    total_travel_time_min: float
    total_travel_time_sec: float
    sla_risk_warning: bool
    latency_ms: float


# ==================== MONITORING & FEEDBACK SCHEMAS ====================
class DriftCheckRequest(BaseModel):
    baseline_distances: List[float] = Field(..., json_schema_extra={"example": [0.5, 1.2, 2.3, 4.1, 5.0]})
    current_distances: List[float] = Field(..., json_schema_extra={"example": [0.6, 1.4, 2.8, 4.5, 6.2]})

class FeedbackRequest(BaseModel):
    actual_delivery_times: List[float] = Field(..., json_schema_extra={"example": [12.5, 18.0, 25.4]})
    predicted_delivery_times: List[float] = Field(..., json_schema_extra={"example": [10.0, 19.5, 22.0]})