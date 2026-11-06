import os
import pandas as pd 
import numpy as np 
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap 
import joblib
import warnings 
warnings.filterwarnings('ignore')

# 1. ĐỊNH NGHĨA CUSTOM ASYMMETRIC LOSS FUNCTION
# Custom Loss Function phạt nặng hơn khi Under-prediction (Dự báo sớm hơn thực tế -> Trễ SLA)
def asymetric_objective(y_true, y_pred):
    alpha = 2.5
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0 * alpha * residual, -2.0 * residual)
    hess = np.where(residual > 0, 2.0 * alpha, 2.0)
    return grad, hess

# Evaluation Metric tương ứng cho LightGBM
def asymmetric_eval_metric(y_true, y_pred):
    alpha = 2.5
    residual = y_true - y_pred
    loss = np.where(residual > 0, alpha*(residual**2), residual**2)
    return 'asymmetric_mse', np.mean(loss), False

# 2. TRAIN PIPELINE & TIME-BASED SPLIT 
def training_pipeline():
    print("Đang nạp dataset 'hcmc_processed_features.parquet'...")
    try:
        df = pd.read_parquet("hcmc_processed_features.parquet")
    except Exception:
        df = pd.read_csv("hcmc_processed_features.csv")

    df['departure_time'] = pd.to_datetime(df['departure_time'])
    df = df.sort_values('departure_time').reset_index(drop=True)

    # Định nghĩa danh sách Feature đưa vào Model (Loại bỏ ID, Timestamp và Target)
    features = [
        'pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon',
        'osrm_distance_km', 'base_duration_min', 'manhattan_dist_km', 'bearing_angle',
        'hour', 'dayofweek', 'is_weekend', 'is_peak_hour',
        'sin_hour', 'cos_hour', 'sin_dayofweek', 'cos_dayofweek',
        'pickup_h3_avg_speed', 'vehicle_type_encoded', 'is_rainy', 'is_same_h3_r8'
    ]
    target = 'actual_travel_time_min'

    # Time-based Split
    n = len(df)
    train_end = int(n*0.8)  
    val_end = int(n*0.9)    

    # Train: 80%
    X_train, y_train = df[features].iloc[:train_end], df[target].iloc[:train_end]
    # Valudation: 10%
    X_val, y_val = df[features].iloc[train_end:val_end], df[target].iloc[train_end:val_end]
    # Test: 10%
    X_test, y_test = df[features].iloc[val_end:], df[target].iloc[val_end:]

    print(f"Phân chia dữ liệu: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 3. HUẤN LUYỆN MODEL LIGHTGBM
    print("Đang huấn luyện LightGBM với Custom Asymetric Loss...")

    model = lgb.LGBMRegressor(
        n_estimators=500, 
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    # Set custom objective 
    model.set_params(objective=asymetric_objective)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=asymmetric_eval_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    # 4. ĐÁNH GIÁ CHỈ SỐ KỸ THUẬT VÀ BUSINESS METRICS
    y_pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Business Metric: OTD Rate (Tỷ lệ giao đúng hẹn trong khoảng [-5 min;+10 min])
    diff = y_test - y_pred_test # Positive = Actual > Pred (Giao trễ hơn ETA yêu cầu)
    otd_success = (diff >= -5.0) & (diff <= 10.0)
    otd_rate = (otd_success.sum() / len(y_test)) * 100.0

    under_pred_rate = (diff > 10.0).sum() / len(y_test) * 100.0 # Tỷ lệ vi phạm SLA nặng

    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print(f"-> MAE (Mean Absolute Error): {mae:.2f} phút")
    print(f"-> RMSE:                      {rmse:.2f} phút")
    print(f"-> Business OTD Rate (%):     {otd_rate:.2f}% (Mục tiêu >= 92%)")
    print(f"-> Tỷ lệ vi phạm SLA (Trễ):   {under_pred_rate:.2f}%")

    # 5. EXPLAINABLE AI (XAI) VỚI SHAP
    print("\nĐang tính toán giá trị SHAP (Explainable AI)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:200])

    # Trích xuất top 3 feature quan trọng nhất hệ thống 
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = [features[i] for i in top_features_idx]
    print(f"Top 3 Feature ảnh hưởng nhất đến ETA: {top_features}")

    # 6. EXPORT MÔ HÌNH SANG NATIVE FORMAT
    print("\nĐang chuyển đổi mô hình LightGBM sang Native Joblib Format...")

    model_filename = "model_eta.pkl"
    joblib.dump(model, model_filename)
    print(f"Đã lưu thành công mô hình tại: '{model_filename}'")

    # Kiểm thử nạp và suy luận
    loaded_model = joblib.load(model_filename)
    test_pred = loaded_model.predict(X_test.iloc[:1])
    print(f"Kiểm thử Inference Speed: {test_pred[0]:.2f} phút.")

if __name__ == "__main__":
    training_pipeline()
