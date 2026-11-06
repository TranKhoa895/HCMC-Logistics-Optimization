# 🚚 HCMC Urban Logistics & Route Optimization System

Hệ thống Dự báo Thời gian Giao hàng (ETA) & Tối ưu hóa Lộ trình Giao hàng đa điểm (CVRP) thời gian thực cho TP.HCM: Tích hợp mô hình Machine Learning (LightGBM + SHAP), Thuật toán Tối ưu hóa (Google OR-Tools), Định vị Địa lý 2 Lớp (Manual Coordinates & Reverse Geocoding), REST API Backend (FastAPI), Prometheus Monitoring và Giao diện Dashboard Quản trị (Streamlit).
---

## 📌 1. Bối Cảnh Kinh Doanh (Business Context) & Mục Tiêu

### 🛵 Bài toán 1: Dự báo ETA Chính xác & Đảm bảo SLA Giao Hàng Chặng Cuối (Last-Mile Delivery)
* **Thực trạng**: 
  * Tình trạng ùn tắc giao thông đặc thù theo giờ cao điểm tại TP.HCM và sự khác biệt giữa các phương tiện (Xe máy vs Xe tải) khiến dự báo ETA thủ công bị lệch từ 25–40%.
  * Tỷ lệ trễ hạn cam kết SLA (Service Level Agreement < 120 phút) cao, gây ảnh hưởng trực tiếp đến trải nghiệm khách hàng và phát sinh chi phí đền bù đơn hàng.
* **Mục tiêu**:
  * Xây dựng mô hình Machine Learning LightGBM Regressor kết hợp không gian (H3 Hexagonal Indexing) và thời gian (Khung giờ xuất phát, loại phương tiện).
  * Đạt thời gian phản hồi (Latency) sub-second (< 100ms), kiểm soát sai số dự báo MAE < 5 phút và tích hợp SHAP Explainability để minh bạch hóa yếu tố ảnh hưởng.

### 🗺️ Bài toán 2: Tối ưu hóa Lộ trình Giao hàng Đa điểm dừng (Capacitated Vehicle Routing Problem - CVRP)
* **Thực trạng**:
  * Tài xế giao hàng chọn lộ trình theo cảm tính hoặc thói quen, dẫn đến quãng đường di chuyển trùng lặp, lãng phí nhiên liệu và quá tải sức chứa của phương tiện.
  * Thiếu công cụ cảnh báo rủi ro vi phạm SLA cho toàn bộ chuỗi lộ trình giao hàng.
* **Mục tiêu**:
  * Triển khai thuật toán tối ưu hóa Google OR-Tools CVRP Solver để sắp xếp thứ tự dừng (Stop Sequence) tối ưu nhất cho tài xế.
  * Tích hợp tự động cảnh báo nguy cơ trễ SLA và trực quan hóa lộ trình di chuyển 3D/2D trực quan trên bản đồ tương tác PyDeck & Folium.

---

## 🏗️ 2. Kiến Trúc Dữ Liệu & Hệ Thống (Data Architecture)

Hệ thống được thiết kế theo mô hình 5 lớp rời rạc (Decoupled 5-Layer MLOps Architecture):

```text
[ Data Sources / HCMC Trips CSV ]
           │
           ▼
[ 1. Pipeline & Feature Engineering (src/generate_dataset.py & src/feature_engineering.py) ]
           │
           ▼
[ 2. Offline Training & Model Artifacts (src/train_model.py) ]
           │
           ├───> [ Model Store: data/model_eta.pkl ]
           │
           ▼
[ 3. Serving & Optimization Engine (app/main.py) ]
           ├── FastAPI REST Endpoints (/predict-eta, /optimize-route)
           ├── Google OR-Tools Solver (src/route_optimization.py)
           └── Prometheus Metrics Exporter (/metrics)
           │
           ├───────────────────────────────┐
           ▼                               ▼
[ 4. Streamlit UI Dashboard ]    [ 5. MLOps & Monitoring Layer ]
(streamlit/app.py)                (src/drift_monitor.py)
- Tab 1: Single-Trip ETA & SHAP   - Continuous PSI & MAE Drift Analysis
- Tab 2: CVRP Route Optimization   - System Latency & Prometheus Integration

```

---

## 📁 3. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
Project_03/
├── app/                           # FastAPI Backend Serving Layer
│   ├── __init__.py
│   ├── main.py                    # API Routes, Lifecycle & Prometheus Instrumentator
│   └── schemas.py                 # Pydantic Schemas (Input/Output Data Validation)
├── src/                           # Machine Learning & Optimization Core Engine
│   ├── __init__.py
│   ├── drift_monitor.py           # Drift Detection Engine (PSI & MAE Analysis)
│   ├── feature_engineering.py     # Spatial (H3 Hexagon) & Temporal Feature Processing
│   ├── generate_dataset.py        # HCMC Synthetic & Clean Delivery Trip Data Generator
│   ├── route_optimization.py      # Google OR-Tools CVRP Solver Core Logic
│   └── train_model.py             # LightGBM Regressor Training & Artifact Saver
├── data/                          # Thư mục chứa dữ liệu và mô hình đã huấn luyện
│   ├── hcmc_delivery_trips.csv    # Dữ liệu gốc chuyến giao hàng TP.HCM
│   ├── hcmc_processed_features.csv# Dữ liệu đặc trưng đã xử lý
│   └── model_eta.pkl              # File trọng số mô hình LightGBM ETA
├── streamlit/                     # Streamlit Frontend UI Layer
│   └── app.py                     # Dynamic Interactive Dashboard (PyDeck & Folium)
├── Dockerfile.backend             # Docker Build Recipe cho FastAPI Backend
├── Dockerfile.frontend            # Docker Build Recipe cho Streamlit Dashboard
├── docker-compose.yml             # Cấu hình triển khai Multi-container Orchestration
├── requirements.txt               # Danh sách thư viện và phiên bản hệ thống
├── .gitignore                     # Cấu hình bỏ qua file rác / môi trường ảo
└── README.md                      # Tài liệu hướng dẫn vận hành dự án

```

---

## ⚙️ 4. Luồng Xử Lý & Nghiệp Vụ Logics (ETL & Business Rules)

### A. Engine Dự Báo ETA (LightGBM + SHAP Explainability)

1. **Feature Engineering:** Tính toán khoảng cách Haversine/Manhattan giữa điểm lấy và điểm giao, mã hóa thuộc tính thời gian (giờ xuất phát, ngày trong tuần) và chỉ số địa lý H3 Spatial Indexing tại TP.HCM.
2. **Model Training:** Huấn luyện mô hình LightGBM Regressor với hàm tổn thất tối ưu hóa cho độ trễ giao hàng.
3. **SHAP Interpretability:** Trích xuất giá trị SHAPley để giải thích trực quan mức độ tác động của từng đặc trưng (Khoảng cách, Giờ cao điểm, Phương tiện) đến kết quả ETA.

### B. Engine Tối Ưu Lộ Trình CVRP (Google OR-Tools)

1. **Distance/Time Matrix:** Xây dựng ma trận thời gian di chuyển giữa tất cả các cặp điểm dừng (Depot Hub + Điểm giao).
2. **Constraint Satisfaction:** Thiết lập các ràng buộc tối ưu hóa bao gồm sức chứa tối đa của phương tiện (Capacity), thời gian phục vụ tại từng điểm dừng và điểm xuất phát/trở về Depot.
3. **SLA Risk Warning:** Kiểm tra tổng thời gian lộ trình; nếu vượt quá ngưỡng 120 phút, hệ thống tự động kích hoạt cảnh báo nguy cơ vi phạm SLA.

### C. Cơ Chế Nhập Vị Trí 2 Lớp (2-Tier Data Entry System)

* **Lớp 1 (Manual Coordinate Entry - Ưu tiên hàng đầu):** Cho phép nhập trực tiếp Vĩ độ (Lat) và Kinh độ (Lon) cùng Tên/Mã đơn hàng (Copy dễ dàng từ Google Maps), đảm bảo 100% độ chính xác và không phụ thuộc API bên ngoài.
* **Lớp 2 (Interactive Map Click):** Click chọn điểm trực tiếp trên bản đồ Folium tương tác để tự động chuyển tọa độ điểm nhấp thành tên đường/phường xã (Reverse Geocoding).

### D. Hệ Thống Giám Sát MLOps (Prometheus & Drift Detection)

* **Prometheus Metrics:** Tự động đo lường độ trễ API (Latency), số lượng request và trạng thái HTTP Response Code qua /metrics.

* **Drift Monitor Engine:** Tính toán chỉ số PSI (Population Stability Index) cho biến đầu vào và MAE Drift cho biến dự báo nhằm phát hiện hiện tượng sụt giảm độ chính xác mô hình theo thời gian.

---

## 🛠️ 5. Cài Đặt Môi Trường (Installation)

### Bước 1: Clone Repository & Chuẩn bị thư mục

```powershell
git clone https://github.com/your-username/hcmc-logistics-optimization.git
cd hcmc-logistics-optimization

```

### Bước 2: Tạo và Kích hoạt Môi trường ảo (.venv)

* **Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate

```


* **macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate

```



### Bước 3: Cài đặt Dependencies

```powershell
pip install -r requirements.txt

```

---

## 🚀 6. Hướng Dẫn Vận Hành Hệ Thống (Usage)

### Cách 1: Vận hành thủ công (Local Development)

Vận hành dự án theo 3 bước chuẩn:

1️⃣ Bước 1: Chạy Data Pipeline, Huấn Luyện Mô Hình & Batch Service
Khởi tạo Feature Store SQLite, huấn luyện mô hình và xuất file kết quả dự báo mới nhất:
```powershell
python src/1_data_processing.py
python src/2_feature_store.py
python src/3a_train_rfm_kmeans.py
python src/3b_train_forecasting.py
python src/3c_train_recommender.py
python src/4_batch_forecast.py
```

2️⃣ Bước 2: Khởi chạy FastAPI Backend (Terminal 1)
```powershell
uvicorn app.main:app --reload --port 8000
```

* Truy cập Swagger UI kiểm thử API tại: http://127.0.0.1:8000/docs

3️⃣ Bước 3: Khởi chạy Streamlit Frontend (Terminal 2)
Mở một cửa sổ Terminal mới, kích hoạt `.venv` và chạy:
```powershell
streamlit run streamlit_app/app.py
```

* Truy cập Dashboard quản trị tại: http://localhost:8501

### Cách 2: Triển khai Đóng gói bằng Docker Compose (Production Standard)

Chạy 1 lệnh duy nhất để build và khởi chạy đồng thời cả Backend FastAPI và Frontend Streamlit:

```powershell
docker-compose up --build -d
```

* Streamlit Dashboard UI: http://localhost:8501

* FastAPI Swagger API: http://localhost:8000/docs

* Prometheus Metrics: http://localhost:8000/metrics

Tắt toàn bộ hệ thống container:

```powershell
docker-compose down
```

---

## 📌 7. Giả Định & Hạn Chế (Assumptions & Limitations)

* **Reverse Geocoding Rates:** Dịch vụ định vị ngược miễn phí OpenStreetMap Nominatim có giới hạn tần suất truy vấn khi nhấp trên bản đồ. Trong môi trường Production lớn, nên tích hợp dịch vụ nội bộ OSRM/Nominatim hoặc Google Maps Geocoding API.

* **Ma trận khoảng cách (Distance Matrix)**: Engine VRP hiện tính toán thời gian dựa trên khoảng cách Haversine/Manhattan điều chỉnh theo hệ số giao thông. Đã sẵn sàng giao tiếp để nâng cấp lên ma trận khoảng cách thực tế thời gian thực từ OSRM Engine.

* **Quy mô bài toán VRP:** Thuật toán Google OR-Tools cho hiệu năng tối ưu Sub-second (< 100ms) với các lộ trình từ 5–30 điểm dừng/tài xế. Với các bài toán cực lớn (> 500 điểm dừng), cần phân cụm địa lý (Cluster-First Route-Second) trước khi đưa vào solver.

---

## 📈 8. Kết Quả Kỹ Thuật & Giá Trị Doanh Nghiệp (Business Impact)

| Thành phần hệ thống | Thuật toán / Kỹ thuật | Giải pháp cho bài toán | Chỉ số mục tiêu (Target KPI) |
| --- | --- | --- | --- |
| **ETA Predictor** | LightGBM Regressor + H3 Spatial Indexing | Dự báo thời gian giao hàng chặng cuối cho đa dạng phương tiện (Xe máy, Xe tải) | $\text{MAE} < 5$ phút, Latency $\text{API} < 100\text{ms}$ |
| **Route Optimizer** | Google OR-Tools CVRP Solver | Tối ưu chuỗi thứ tự dừng giao hàng, loại bỏ đường trùng lặp, kiểm soát sức chứa | Giảm 12–18% tổng quãng đường di chuyển |
| **SLA Risk Control** | Constraint Checking Logic | Cảnh báo tự động khi tổng lộ trình vượt quá 120 phút | Giảm 15% tỷ lệ trễ hạn đơn hàng SLA |
| **2-Tier Data Entry** | Direct Coordinate Input + Reverse Geocoding Map | Loại bỏ phụ thuộc tìm kiếm địa danh thất bại, đảm bảo 100% nhập điểm thành công | 100% Reliability & Accuracy |
| **MLOps & Monitorin** | Prometheus Exporter + PSI/MAE Drift Monitor | Phát hiện sụt giảm độ chính xác mô hình & giám sát hiệu năng hạ tầng API | Cảnh báo Drift tự động khi $\text{PSI} > 0.25$ |

---

## 👤 9. Tác Giả & Bản Quyền (Contributors & Data Source)

* **Tác giả:** Tran Phuoc Khoa - Data Scientist / Machine Learning Engineer
* **Nguồn dữ liệu:** Dữ liệu vận chuyển thực tế & mô phỏng giao thông TP.HCM.