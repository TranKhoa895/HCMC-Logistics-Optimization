import streamlit as st
import requests
import pandas as pd
import json
import pydeck as pdk
import folium
import os
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# Lấy API_URL linh hoạt (tự động nhận IP Docker hoặc Local)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")

st.set_page_config(
    page_title="HCMC Logistics Optimization Dashboard",
    page_icon="🚚",
    layout="wide"
)

# Khởi tạo Nominatim Geocoder cho Reverse Geocoding
geolocator = Nominatim(user_agent="hcmc_logistics_dashboard_v5")

def fetch_address(lat, lon):
    """Chuyển Tọa độ (Lat, Lon) -> Tên đường/Địa chỉ ngắn khi click bản đồ"""
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=4, language='vi')
        if location and location.raw.get('address'):
            addr = location.raw['address']
            road = addr.get('road') or addr.get('suburb') or addr.get('amenity') or addr.get('building') or 'Điểm dừng'
            sub_district = addr.get('quarter') or addr.get('suburb') or ''
            res_str = f"{road}, {sub_district}".strip(", ")
            return res_str if res_str else f"Loc_{lat:.4f}_{lon:.4f}"
    except Exception:
        pass
    return f"Loc_{lat:.4f}_{lon:.4f}"

st.title("🚚 HCMC Logistics Serving & Application Dashboard")
st.markdown("Hệ thống dự báo thời gian giao hàng (ETA) & Tối ưu hóa thứ tự tuyến đường (CVRP)")

tab1, tab2 = st.tabs(["⏱️ Dự báo ETA (Single Trip)", "🗺️ Tối ưu tuyến đường (CVRP Routing)"])

# ==================== TAB 1: DỰ BÁO ETA ====================
with tab1:
    st.header("Dự báo thời gian giao hàng (ETA)")
    
    col1, col2 = st.columns(2)
    with col1:
        pickup_lat = st.number_input("Pickup Lat (Vĩ độ lấy)", value=10.7760, format="%.4f")
        pickup_lon = st.number_input("Pickup Lon (Kinh độ lấy)", value=106.7000, format="%.4f")
        vehicle_type = st.selectbox("Loại phương tiện", ["MOTORBIKE", "TRUCK"])
    with col2:
        dropoff_lat = st.number_input("Dropoff Lat (Vĩ độ giao)", value=10.7740, format="%.4f")
        dropoff_lon = st.number_input("Dropoff Lon (Kinh độ giao)", value=106.7030, format="%.4f")
        departure_hour = st.slider("Giờ xuất phát (0 - 23h)", 0, 23, 14)

    if st.button("🚀 Dự báo ETA", key="predict_btn"):
        payload = {
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "dropoff_lat": dropoff_lat,
            "dropoff_lon": dropoff_lon,
            "vehicle_type": vehicle_type,
            "departure_hour": departure_hour
        }
        
        try:
            res = requests.post(f"{API_URL}/predict-eta", json=payload)
            if res.status_code == 200:
                data = res.json()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("ETA Dự báo", f"{data['predicted_eta_minutes']} phút", f"{data['predicted_eta_seconds']} giây")
                m2.metric("Khoảng cách", f"{data['distance_km']} km")
                m3.metric("Độ trễ API (Latency)", f"{data['latency_ms']} ms")
                
                st.subheader("📊 Giải thích dự báo (SHAP Explainability)")
                shap_df = pd.DataFrame(
                    list(data['shapley_values'].items()), 
                    columns=['Đặc trưng (Feature)', 'Mức độ tác động (SHAP Value)']
                ).sort_values(by='Mức độ tác động (SHAP Value)', ascending=True)
                
                st.bar_chart(shap_df.set_index('Đặc trưng (Feature)'))
            else:
                st.error(f"Lỗi API: {res.text}")
        except Exception as e:
            st.error(f"Không thể kết nối tới FastAPI Backend: {e}")

# ==================== TAB 2: TỐI ƯU TUYẾN ĐƯỜNG ====================
with tab2:
    st.header("Tối ưu hóa thứ tự giao hàng cho Tài xế")
    
    if "stops" not in st.session_state:
        st.session_state.stops = [
            {"order_id": "DEPOT_HUB_Q1", "lat": 10.7760, "lon": 106.7000},
            {"order_id": "ORD_001_BEN_THANH", "lat": 10.7725, "lon": 106.6980},
            {"order_id": "ORD_002_NGUYEN_HUE", "lat": 10.7742, "lon": 106.7032}
        ]

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        driver_id = st.text_input("Mã Tài xế (Driver ID)", value="DRV_HCMC_99")
    with col_d2:
        capacity = st.number_input("Sức chứa tối đa (Đơn)", value=15, min_value=1)
    with col_d3:
        route_hour = st.slider("Khung giờ xuất phát", 0, 23, 14, key="route_hour")

    # 🟢 TẦNG 1: KHUNG NHẬP THỦ CÔNG TỌA ĐỘ
    st.subheader("📝 1. Nhập thủ công Tọa độ & Tên địa điểm (Từ Google Maps)")
    c_m1, c_m2, c_m3, c_m4 = st.columns([3, 2, 2, 2])
    with c_m1:
        manual_name = st.text_input("Tên địa điểm / Mã đơn", value="ORD_TON_DUC_THANG", key="m_name")
    with c_m2:
        manual_lat = st.number_input("Vĩ độ (Lat)", value=10.7326, format="%.4f", key="m_lat")
    with c_m3:
        manual_lon = st.number_input("Kinh độ (Lon)", value=106.6997, format="%.4f", key="m_lon")
    with c_m4:
        st.write(" ") # Căn lề nút bấm
        if st.button("➕ Thêm địa điểm", key="add_manual_btn", use_container_width=True):
            st.session_state.stops.append({
                "order_id": manual_name,
                "lat": manual_lat,
                "lon": manual_lon
            })
            st.toast(f"✅ Đã thêm điểm thành công: {manual_name}", icon="📍")
            st.rerun()

    st.markdown("---")
    
    # 🟢 TẦNG 2: CLICK TRỰC TIẾP TRÊN BẢN ĐỒ HCMC
    st.subheader("📍 2. Click trực tiếp điểm trên Bản đồ HCMC")

    # Lọc các bản ghi tọa độ hợp lệ
    valid_stops = [
        s for s in st.session_state.stops 
        if isinstance(s, dict) and pd.notna(s.get('lat')) and pd.notna(s.get('lon'))
    ]

    map_center = [10.7760, 106.7000]
    if valid_stops:
        try:
            map_center = [float(valid_stops[-1]['lat']), float(valid_stops[-1]['lon'])]
        except (ValueError, TypeError):
            map_center = [10.7760, 106.7000]
        
    m = folium.Map(location=map_center, zoom_start=13)

    for idx, stop in enumerate(valid_stops):
        color = 'red' if idx == 0 else 'blue'
        icon_type = 'home' if idx == 0 else 'info-sign'
        folium.Marker(
            [float(stop['lat']), float(stop['lon'])],
            popup=f"<b>{stop['order_id']}</b>",
            tooltip=f"{idx + 1}. {stop['order_id']}",
            icon=folium.Icon(color=color, icon=icon_type)
        ).add_to(m)

    map_data = st_folium(m, height=350, width="100%", key="interactive_picker_map")

    # Xử lý sự kiện khi click trên bản đồ
    if map_data and map_data.get("last_clicked"):
        click_lat = round(map_data["last_clicked"]["lat"], 4)
        click_lon = round(map_data["last_clicked"]["lng"], 4)
        
        last_processed = st.session_state.get("last_processed_click")
        if last_processed != (click_lat, click_lon):
            st.session_state.last_processed_click = (click_lat, click_lon)
            point_count = len(st.session_state.stops)
            
            addr_name = fetch_address(click_lat, click_lon)
            clean_addr = addr_name.upper().replace(' ', '_')[:15]
            
            new_id = "DEPOT_HUB" if point_count == 0 else f"ORD_{point_count:03d}_{clean_addr}"
            st.session_state.stops.append({
                "order_id": new_id,
                "lat": click_lat,
                "lon": click_lon
            })
            st.rerun()

    # Các nút thao tác xóa điểm
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        if st.button("🗑️ Xóa điểm vừa thêm"):
            if len(st.session_state.stops) > 0:
                st.session_state.stops.pop()
                st.rerun()
    with c_btn2:
        if st.button("🧹 Xóa tất cả điểm"):
            st.session_state.stops = []
            st.rerun()

    # Bảng danh sách điểm dừng (cho phép trực tiếp xem và sửa dữ liệu)
    st.subheader("📋 Danh sách các điểm dừng")
    df_current = pd.DataFrame(st.session_state.stops)
    if not df_current.empty:
        edited_df = st.data_editor(
            df_current,
            num_rows="dynamic",
            column_config={
                "order_id": st.column_config.TextColumn("Mã Đơn / Tên Địa điểm", required=True),
                "lat": st.column_config.NumberColumn("Vĩ độ (Lat)", format="%.4f"),
                "lon": st.column_config.NumberColumn("Kinh độ (Lon)", format="%.4f")
            },
            use_container_width=True
        )
        
        cleaned_stops = [
            r for r in edited_df.to_dict(orient="records")
            if pd.notna(r.get("lat")) and pd.notna(r.get("lon")) and r.get("order_id") is not None
        ]
        st.session_state.stops = cleaned_stops

    # Nút bấm tính toán tối ưu VRP
    if st.button("🧩 Tối ưu tuyến đường VRP", key="optimize_btn"):
        if len(st.session_state.stops) < 2:
            st.error("⚠️ Cần tối thiểu 2 điểm dừng (1 Depot + 1 Đơn hàng) để thực hiện tối ưu!")
        else:
            try:
                payload = {
                    "driver_id": driver_id,
                    "vehicle_type": "MOTORBIKE",
                    "vehicle_capacity": capacity,
                    "departure_hour": route_hour,
                    "stops": st.session_state.stops
                }
                
                res = requests.post(f"{API_URL}/optimize-route", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    
                    if data['sla_risk_warning']:
                        st.warning("⚠️ CẢNH BÁO NGUY CƠ TRỄ SLA: Tổng thời gian giao hàng vượt quá 120 phút!")
                    else:
                        st.success("✅ Tuyến đường tối ưu đạt tiêu chuẩn SLA!")
                    
                    # Đã sửa lại định dạng f-string không bị lỗi cú pháp
                    st.write(f"**Tổng thời gian di chuyển dự kiến:** {data['total_travel_time_min']} phút ({data['total_travel_time_sec']} giây)")
                    st.write(f"**Độ trễ giải thuật (Latency):** {data['latency_ms']} ms")
                    
                    st.subheader("📌 Thứ tự di chuyển tối ưu (Stop Sequence)")
                    st.info(" ➔ ".join(data['stop_order_ids']))
                    
                    col_tbl, col_map = st.columns([1, 1])
                    
                    with col_tbl:
                        st.subheader("📋 Bảng thứ tự điểm dừng")
                        df_route = pd.DataFrame([
                            {"Thứ tự": idx + 1, "Mã Đơn / Hub": order_id}
                            for idx, order_id in enumerate(data['stop_order_ids'])
                        ])
                        st.table(df_route)
                    
                    with col_map:
                        st.subheader("🗺️ Bản đồ kết quả lộ trình")
                        stops_dict = {s['order_id']: s for s in st.session_state.stops}
                        route_points = []
                        path_coords = []
                        
                        for idx, order_id in enumerate(data['stop_order_ids']):
                            if order_id in stops_dict:
                                stop_info = stops_dict[order_id]
                                lat = stop_info['lat']
                                lon = stop_info['lon']
                                path_coords.append([lon, lat])
                                
                                is_depot = (idx == 0 or idx == len(data['stop_order_ids']) - 1)
                                route_points.append({
                                    "seq": idx + 1,
                                    "label": str(idx + 1),
                                    "order_id": order_id,
                                    "lat": lat,
                                    "lon": lon,
                                    "color": [239, 68, 68] if is_depot else [59, 130, 246]
                                })
                        
                        df_route_map = pd.DataFrame(route_points)
                        
                        path_layer = pdk.Layer(
                            "PathLayer",
                            data=[{"path": path_coords}],
                            get_path="path",
                            get_color=[245, 158, 11, 220],
                            width_scale=5,
                            width_min_pixels=3,
                        )

                        scatter_layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=df_route_map,
                            get_position=["lon", "lat"],
                            get_fill_color="color",
                            get_radius=70,
                            radius_min_pixels=14,
                            radius_max_pixels=22,
                            pickable=True,
                        )

                        text_layer = pdk.Layer(
                            "TextLayer",
                            data=df_route_map,
                            get_position=["lon", "lat"],
                            get_text="label",
                            get_size=13,
                            get_color=[255, 255, 255],
                            get_text_anchor="'middle'",
                            get_alignment_baseline="'center'",
                        )

                        avg_lat = df_route_map['lat'].mean()
                        avg_lon = df_route_map['lon'].mean()
                        view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=13)

                        st.pydeck_chart(
                            pdk.Deck(
                                layers=[path_layer, scatter_layer, text_layer],
                                initial_view_state=view_state,
                                tooltip={"html": "<b>STT: {seq}</b><br/>Mã: {order_id}"}
                            )
                        )
                else:
                    st.error(f"Lỗi tối ưu: {res.text}")
            except Exception as e:
                st.error(f"Lỗi dữ liệu hoặc kết nối Backend: {e}")