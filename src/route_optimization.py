import pandas as pd 
import numpy as np 
import joblib 
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# 1. 1. UNPICKLE CUSTOM LOSS
def asymetric_objective(y_true, y_pred):
    alpha = 2.5
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * alpha * residual, -2.0 * residual)
    hess = np.where(residual > 0, 2.0 * alpha, 2.0)
    return grad, hess

# 2. OPTIMIZATION ENGINE
class RouteOptimizationEngine:
    def __init__(self, model_path="model_eta.pkl"):
        print(f"Đang tải mô hình ETA từ '{model_path}'...")
        self.model = joblib.load(model_path)

        self.feature_columns = [
            'pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon',
            'osrm_distance_km', 'base_duration_min', 'manhattan_dist_km', 'bearing_angle',
            'hour', 'dayofweek', 'is_weekend', 'is_peak_hour',
            'sin_hour', 'cos_hour', 'sin_dayofweek', 'cos_dayofweek',
            'pickup_h3_avg_speed', 'vehicle_type_encoded', 'is_rainy', 'is_same_h3_r8'
        ]

    def _predict_single_eta(self, p_lat, p_lon, d_lat, d_lon, vehicle_type='MOTORBIKE', hour=14):
        R = 6371.0
        dlat, dlon = np.radians(d_lat - p_lat), np.radians(d_lon - p_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(p_lat)) * np.cos(np.radians(d_lat)) * np.sin(dlon/2)**2
        dist_km = max(2 * R * np.arcsin(np.sqrt(a)) * 1.35, 0.5)

        base_speed = 26.0 if vehicle_type == 'MOTORBIKE' else 20.0
        base_duration = (dist_km / base_speed) * 60.0
        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        # Tạo DataFrame chứa 20 features chuẩn
        df_feat = pd.DataFrame([{
            'pickup_lat': p_lat, 'pickup_lon': p_lon,
            'dropoff_lat': d_lat, 'dropoff_lon': d_lon,
            'osrm_distance_km': dist_km,
            'base_duration_min': base_duration,
            'manhattan_dist_km': dist_km * 1.15,
            'bearing_angle': 120.0,
            'hour': hour, 'dayofweek': 2, 'is_weekend': 0, 'is_peak_hour': is_peak,
            'sin_hour': np.sin(2*np.pi*hour/24), 'cos_hour': np.cos(2*np.pi*hour/24),
            'sin_dayofweek': 0.5, 'cos_dayofweek': 0.866,
            'pickup_h3_avg_speed': 18.5,
            'vehicle_type_encoded': 0 if vehicle_type == 'MOTORBIKE' else 1,
            'is_rainy': 0, 'is_same_h3_r8': 0
        }])[self.feature_columns]

        pred = self.model.predict(df_feat)[0]
        return float(pred)

    def build_time_matrix(self, stops, vehicle_type='MOTORBIKE', hour=14):
        N = len(stops)
        matrix = np.zeros((N, N), dtype=int)
        pairs = []
        indices = []

        # Tạo tập hợp các điểm cần tính 
        for i in range(N):
            for j in range(N):
                if i != j:
                    p_lat, p_lon = stops[i]['lat'], stops[i]['lon']
                    d_lat, d_lon = stops[j]['lat'], stops[j]['lon']

                    R = 6371.0
                    dlat, dlon = np.radians(d_lat - p_lat), np.radians(d_lon - p_lon)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(p_lat)) * np.cos(np.radians(d_lat)) * np.sin(dlon/2)**2
                    dist_km = max(2 * R * np.arcsin(np.sqrt(a)) * 1.35, 0.5)

                    base_speed = 26.0 if vehicle_type == 'MOTORBIKE' else 20.0
                    base_duration = (dist_km / base_speed) * 60.0

                    pairs.append({
                        'pickup_lat': p_lat, 'pickup_lon': p_lon,
                        'dropoff_lat': d_lat, 'dropoff_lon': d_lon,
                        'osrm_distance_km': dist_km,
                        'base_duration_min': base_duration,
                        'manhattan_dist_km': dist_km * 1.15,
                        'bearing_angle': 120.0,
                        'hour': hour, 'dayofweek': 2, 'is_weekend': 0, 
                        'is_peak_hour': 1 if (7<=hour<=9 or 17<=hour<=19) else 0,
                        'sin_hour': np.sin(2*np.pi*hour/24), 'cos_hour': np.cos(2*np.pi*hour/24),
                        'sin_dayofweek': 0.5, 'cos_dayofweek': 0.866,
                        'pickup_h3_avg_speed': 18.5,
                        'vehicle_type_encoded': 0 if vehicle_type == 'MOTORBIKE' else 1,
                        'is_rainy': 0, 'is_same_h3_r8': 0
                    })
                    indices.append((i, j))
        
        if pairs:
            # Predict Siêu tốc trong 1 lần gọi duy nhất (Batch Prediction)
            df_batch = pd.DataFrame(pairs)[self.feature_columns]
            preds = self.model.predict(df_batch)
            
            for idx, (i, j) in enumerate(indices):
                matrix[i][j] = int(preds[idx] * 60)
        
        return matrix

    def solve_cvrp(self, stops, vehicle_capacity=15, num_vehicles=1, vehicle_type='MOTORBIKE', hour=14):
        print(f"Đang giải thuật toán VRP cho {len(stops)-1} đơn hàng...")
        time_matrix = self.build_time_matrix(stops, vehicle_type=vehicle_type, hour=hour)

        data = {
            'time_matrix': time_matrix.tolist(),
            'demands': [0] + [1] * (len(stops) - 1),
            'vehicle_capacities': [vehicle_capacity] * num_vehicles,
            'num_vehicles': num_vehicles,
            'depot': 0 
        }

        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def timer_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(timer_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = []
            total_time_sec = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                total_time_sec += routing.GetArcCostForVehicle(previous_index, index, 0)

            route.append(manager.IndexToNode(index))

            return {
                'status': 'SUCCESS',
                'optimized_route_sequence': route,
                'stop_order_ids': [stops[i]['order_id'] for i in route],
                'total_travel_time_min': round(total_time_sec / 60.0, 2)
            }
        else:
            return {'status': 'FAILED', 'message': 'Không tìm thấy tuyến đường khả thi'}

if __name__ == "__main__":
    sample_stops = [
        {'order_id': 'DEPOT_HUB_Q1', 'lat': 10.7760, 'lon': 106.7000},
        {'order_id': 'ORD_001_NGUYEN_HUE', 'lat': 10.7740, 'lon': 106.7030},
        {'order_id': 'ORD_002_LE_LOI',     'lat': 10.7710, 'lon': 106.6980},
        {'order_id': 'ORD_003_VO_VAN_TAN',  'lat': 10.7800, 'lon': 106.6880},
        {'order_id': 'ORD_004_PASTEUR',     'lat': 10.7830, 'lon': 106.6950},
        {'order_id': 'ORD_005_BUI_VIEN',    'lat': 10.7670, 'lon': 106.6930}
    ]

    engine = RouteOptimizationEngine(model_path="model_eta.pkl")
    result = engine.solve_cvrp(sample_stops, vehicle_capacity=10)

    print("KẾT QUẢ TỐI ƯU TUYẾN ĐƯỜNG GIAO HÀNG (VRP)") 
    print(f"Trạng thái:                    {result['status']}")
    print(f"Thứ tự Điểm dừng tối ưu:       {' -> '.join(map(str, result['optimized_route_sequence']))}")
    print(f"Danh sách Order IDs chạy:      {' -> '.join(result['stop_order_ids'])}")
    print(f"Tổng Thời gian Di chuyển ETA:  {result['total_travel_time_min']} phút")