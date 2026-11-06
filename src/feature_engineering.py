import pandas as pd 
import numpy as np 
import h3
import math

def get_h3_cell(lat, lon, res):
    try:
        # H3 v4 API 
        return h3.latlng_to_cell(lat, lon, res)
    except AttributeError:
        return h3.geo_to_h3(lat, lon, res)

# Tính góc hướng di chuyển giữa điểm Pickup và Dropoff
def calculate_bearing(lat1, lon1, lat2, lon2):
    d_lon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

# Ước tính khoảng cách đường thẳng kiểu mạng lưới ô bàn cờ (Manhattan distance in km)
def calculate_manhattan_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat_dist = abs(math.radians(lat2-lat1)) * R
    lon_dist = abs(math.radians(lon2-lon1)) * R * math.cos(math.radians((lat1+lat2) / 2.0))
    return lat_dist + lon_dist

def feature_engineering():
    print("Đang tải dữ liệu từ STAGE 1...")

    # Đọc dữ liệu từ STAGE 1
    input_file  = "hcmc_delivery_trips.csv"
    df = pd.read_csv(input_file)
    df['departure_time'] = pd.to_datetime(df['departure_time'])

    print(f"Đã nạp {len(df)} dòng dữ liệu.")

    # 1. SPATIAL INDEXING WITH UBER H3
    print("1. Đang tạo ô lưới lục giác Uber H3 (Res 8 & Res 9)")
    df['pickup_h3_r8'] = df.apply(lambda r: get_h3_cell(r['pickup_lat'], r['pickup_lon'], 8), axis=1)
    df['dropoff_h3_r8'] = df.apply(lambda r: get_h3_cell(r['dropoff_lat'], r['dropoff_lon'], 8), axis=1)

    df['pickup_h3_r9'] = df.apply(lambda r: get_h3_cell(r['pickup_lat'], r['pickup_lon'], 9), axis=1)
    df['dropoff_h3_r9'] = df.apply(lambda r: get_h3_cell(r['dropoff_lat'], r['dropoff_lon'], 9), axis=1)

    # Đánh dấu chuyến đi trong cùng 1 ô hoặc liên ô 
    df['is_same_h3_r8'] = (df['pickup_h3_r8'] == df['dropoff_h3_r8']).astype(int)

    # 2. ADVANCED DISTANCE & METRICS
    print("2. Tính toán khoảng cách và góc hướng")
    df['manhattan_dist_km'] = df.apply(
        lambda r: calculate_manhattan_distance(r['pickup_lat'], r['pickup_lon'], r['dropoff_lat'], r['dropoff_lon']), axis=1
    )

    df['bearing_angle'] = df.apply(
        lambda r: calculate_bearing(r['pickup_lat'], r['pickup_lon'], r['dropoff_lat'], r['dropoff_lon']), axis=1
    )

    # 3. TEMPORAL & CYCLING ENCODING
    print("3. Trích xuất đặc trưng chuỗi thời gian & Mã hóa chu kỳ Sóng (sin/cosin)")
    df['hour'] = df['departure_time'].dt.hour
    df['dayofweek'] = df['departure_time'].dt.day_of_week
    df['is_weekend'] =df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Đánh dấu giờ cao điểm sáng/chiều
    df['is_peak_hour'] = df.apply(
        lambda r: 1 if (r['dayofweek'] < 5 and ((7 <= r['hour'] <= 9) or (17 <= r['hour'] <= 19))) else 0, axis=1
    )

    # Continuous Cyclical Encoding (Sin/Cosin Transformation)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    # 4. AGGREGATED HISTORICAL SPEED BY H3 REGION
    print("4. TÍnh toán vận tốc trung bình lịch sử theo ô H3 Res 8")
    # Tốc độ chuyến đi ước tính (km/h)
    df['trip_speed_km'] = (df['osrm_distance_km'] / (df['actual_travel_time_min'] / 60.0))

    # Tính tốc độ trung bình theo từng ô Pickup H3 Res 8
    h3_speed_map = df.groupby('pickup_h3_r8')['trip_speed_km'].mean().to_dict()
    global_avg_speed = df['trip_speed_km'].mean()

    df['pickup_h3_avg_speed'] = df['pickup_h3_r8'].map(h3_speed_map).fillna(global_avg_speed)

    # 5. CATEGORICAL ENCODING 
    print("5. Mã hóa các biến phân loại")
    df['vehicle_type_encoded'] = df['vehicle_type'].map({'MOTORBIKE': 0, 'TRUCK': 1})

    # 6. COMPLETE DATASET
    output_filename = "hcmc_processed_features.csv"
    df.to_csv(output_filename, index=False)
    df.to_parquet("hcmc_processed_features.parquet", index=False)

    print(f"Đã tạo file '{output_filename}' ({df.shape[1]} cột đặc trưng).")
    return df 

if __name__ == "__main__":
    df_features = feature_engineering()

    print("\nXEM TRƯỚC H3 CELL VÀ CYCLIC FEATURES")
    cols_to_show = ['trip_id', 'pickup_h3_r8', 'bearing_angle', 'sin_hour', 'cos_hour', 'pickup_h3_avg_speed', 'actual_travel_time_min']
    print(df_features[cols_to_show].head(10))
