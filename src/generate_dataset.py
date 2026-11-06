import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta

# Thiết lập seed cố định
np.random.seed(42)

# Tọa độ thực tế đất liền các Quận TP.HCM (Không rơi xuống sông/biển)
DISTRICT_BOUNDS = {
    'District_1':    {'lat': (10.7680, 10.7880), 'lon': (106.6900, 106.7080)},
    'District_3':    {'lat': (10.7750, 10.7920), 'lon': (106.6750, 106.6900)},
    'District_5':    {'lat': (10.7500, 10.7650), 'lon': (106.6600, 106.6800)},
    'District_7':    {'lat': (10.7200, 10.7450), 'lon': (106.7000, 106.7300)},
    'District_10':   {'lat': (10.7650, 10.7820), 'lon': (106.6600, 106.6780)},
    'Binh_Thanh':   {'lat': (10.7950, 10.8250), 'lon': (106.6950, 106.7250)},
    'Tan_Binh':     {'lat': (10.7900, 10.8200), 'lon': (106.6400, 106.6700)},
    'Thu_Duc':      {'lat': (10.8300, 10.8600), 'lon': (106.7500, 106.7800)}
}

def generate_hcmc_dataset(num_samples=5000):
    print(f"Đang khởi tạo {num_samples} chuyến đi và tiêm nhiễu thực tế TP.HCM...")
    district_list = list(DISTRICT_BOUNDS.keys())
    
    data = []
    start_date = datetime(2026, 10, 1, 6, 0, 0)

    for i in range(num_samples):
        # Chọn quận đi và quận đến ngẫu nhiên
        p_dist = np.random.choice(district_list)
        d_dist = np.random.choice(district_list)

        p_lat = np.random.uniform(*DISTRICT_BOUNDS[p_dist]['lat'])
        p_lon = np.random.uniform(*DISTRICT_BOUNDS[p_dist]['lon'])
        d_lat = np.random.uniform(*DISTRICT_BOUNDS[d_dist]['lat'])
        d_lon = np.random.uniform(*DISTRICT_BOUNDS[d_dist]['lon'])

        # Tính khoảng cách Harvesine
        R = 6371.0
        dlat = np.radians(d_lat - p_lat)
        dlon = np.radians(d_lon - p_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(p_lat)) * np.cos(np.radians(d_lat)) * np.sin(dlon/2)**2
        direct_dist_km = 2 * R * np.arcsin(np.sqrt(a))

        # Hệ số uốn lượn đường bộ TP.HCM (Circuity Factor: 1.25-1.45)
        circuity = np.random.uniform(1.25, 1.45)
        osrm_distance_km = round(max(direct_dist_km * circuity, 0.8), 3)

        # Sinh mốc thời gian xuất phát (Trong tháng 10/2026)
        random_seconds = np.random.randint(0, 30*24*3600)
        dep_time = start_date + timedelta(minutes=random_seconds)
        hour = dep_time.hour
        weekday = dep_time.weekday()

        # Thuộc tính loại xe và thời tiết 
        v_type = np.random.choice(['MOTORBIKE', 'TRUCK'], p=[0.85, 0.15])
        is_rain = np.random.choice([0, 1], p=[0.75, 0.25])

        # Vận tốc cơ sở nội thành chuẩn TP.HCM
        # Xe máy linh hoạt chạy 26km/h, Xe tải lớn vướng kẹt xe/hẻm nhỏ chỉ đạt 18km/h
        base_speed = 26.0 if v_type == 'MOTORBIKE' else 20.0
        base_duration_min = round((osrm_distance_km / base_speed) * 60.0, 2)

        # Tiên nhiễu thực tế Tp.HCM
        # A. Hệ số giờ cao điểm  (7h-9h và 17h-19h T2-T6)
        rush_hour_factor = 1.0
        if weekday < 5:
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                rush_hour_factor = np.random.uniform(1.45, 1.90)
            elif 11 <= hour <= 13:
                rush_hour_factor = np.random.uniform(1.10, 1.25)

        # B. Hệ số thời tiết mưa ngập 
        weather_factor = np.random.uniform(1.30, 1.60) if is_rain == 1 else 1.0

        # C. Hệ số phương tiện 
        vehicle_factor = 0.88 if v_type == 'MOTORBIKE' else 1.25

        # D. Thời gian bốc xếp/phục vụ (Service Time)
        service_time_min = np.random.uniform(3.0, 7.0) if v_type == 'MOTORBIKE' else np.random.uniform(10.0, 20.0)

        # E. Nhiễu Gauss ngẫu nhiên 
        random_noise = np.random.normal(loc=0.0, scale=1.2)

        # Actual ETA Target
        actual_travel_time = (base_duration_min * rush_hour_factor * weather_factor * vehicle_factor) + service_time_min + random_noise
        actual_travel_time = max(round(actual_travel_time, 2), 2.5)

        data.append({
            'trip_id': f"TRIP_HCMC_{i+1:06d}",
            'pickup_lat': round(p_lat, 6),
            'pickup_lon': round(p_lon, 6),
            'dropoff_lat': round(d_lat, 6),
            'dropoff_lon': round(d_lon, 6),
            'departure_time': dep_time.strftime('%Y-%m-%d %H:%M:%S'),
            'osrm_distance_km': osrm_distance_km,
            'base_duration_min': base_duration_min,
            'vehicle_type': v_type,
            'is_rainy': is_rain,
            'actual_travel_time_min': actual_travel_time
        })  

    df = pd.DataFrame(data)
    output_filename = "hcmc_delivery_trips.csv"    
    df.to_csv(output_filename, index=False)
    print(f"Đã tạo xong file '{output_filename}' hoàn chỉnh với {len(df)} dòng dữ liệu!")
    return df

if __name__ == "__main__":
    df_hcm = generate_hcmc_dataset(5000)

    print(df_hcm.head(10))
