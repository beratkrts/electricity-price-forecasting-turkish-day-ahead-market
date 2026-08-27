import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from db.connection import get_db_engine
from src.features.holidays import add_holiday_features
engine = get_db_engine()

# Fetch Load Forecast and Weather Data
query = '''
    SELECT 
        l.ts,
        l.load_forecast_mw,
        w.turkey_weighted_temperature_c as temp_c
    FROM raw_load_forecast_hourly l
    LEFT JOIN raw_weather_hourly w ON l.ts = w.ts
    WHERE l.ts >= CURRENT_DATE - INTERVAL '12 months'
    ORDER BY l.ts ASC
'''
df = pd.read_sql(query, engine)
df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert('Europe/Istanbul')
df = df.set_index('ts')

# Create Features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# 1. Lags (The weights we learned earlier)
df['lag_1d'] = df['load_forecast_mw'].shift(24)
df['lag_7d'] = df['load_forecast_mw'].shift(24 * 7)
df['lag_14d'] = df['load_forecast_mw'].shift(24 * 14)

# 2. Weather Features (Cooling & Heating Degree Hours)
# Assuming 18C is the comfort zone
df['cdh'] = np.maximum(df['temp_c'] - 18.0, 0.0)
df['hdh'] = np.maximum(18.0 - df['temp_c'], 0.0)

# Drop NaNs due to shifting
df = df.dropna()

# Train-Test Split (Last 60 days)
test_days = 60
train_size = len(df) - (test_days * 24)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ['hour', 'dayofweek', 'is_weekend', 'lag_1d', 'lag_7d', 'lag_14d', 'temp_c', 'cdh', 'hdh']
target = 'load_forecast_mw'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

def wape(y_t, y_p):
    return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100

# Model 1: Ridge (MWA + Weather Linear)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# Model 2: LightGBM (Non-linear MWA + Weather + Calendar)
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

print("=== Yük Tahmini (Load Forecast) + Hava Durumu (Open-Meteo) ===")
print(f"Önceki MWA (Sadece Geçmiş Veri) WAPE: %2.66")
print(f"Ridge (Geçmiş + Sıcaklık) WAPE: {wape(y_test, ridge_pred):.2f}%")
print(f"LightGBM (Geçmiş + Sıcaklık + Takvim) WAPE: {wape(y_test, lgb_pred):.2f}%")

# Feature Importance for LightGBM
importance = pd.DataFrame({'Feature': features, 'Importance': lgb_model.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=False)
print("\nLightGBM Modelinin En Çok Önem Verdiği Değişkenler:")
print(importance.to_string(index=False))
