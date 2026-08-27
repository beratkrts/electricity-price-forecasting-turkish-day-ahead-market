import sys
import httpx
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from db.connection import get_db_engine
from sqlalchemy import text

print("1. Veritabanından KGÜP (Planlanan Rüzgar) Çekiliyor...")
engine = get_db_engine()
query = '''
    SELECT 
        ts,
        wind_mw as kgup_wind
    FROM raw_kgup_hourly
    WHERE ts >= '2024-01-01' AND ts < CURRENT_DATE
    ORDER BY ts ASC
'''
df_db = pd.read_sql(text(query), engine)
df_db['ts'] = pd.to_datetime(df_db['ts']).dt.tz_convert('Europe/Istanbul')
df_db = df_db.set_index('ts')

print("2. Open-Meteo'dan Rüzgar Verisi Çekiliyor (İzmir, Balıkesir, Çanakkale Ağırlıklı)...")
# İzmir, Çanakkale, Balıkesir (Türkiye'nin en büyük rüzgar santralleri burada)
cities = {
    'izmir': {'lat': 38.41, 'lon': 27.14},
    'canakkale': {'lat': 40.15, 'lon': 26.40},
    'balikesir': {'lat': 39.64, 'lon': 27.88}
}

df_weather = None
for city, coords in cities.items():
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['lat']}&longitude={coords['lon']}&start_date=2024-01-01&end_date=2026-08-09&hourly=wind_speed_100m&timezone=Europe%2FIstanbul"
    response = httpx.get(url, timeout=30.0)
    data = response.json()
    
    times = pd.to_datetime(data['hourly']['time'])
    # Convert naive timestamps to Europe/Istanbul to match DB
    times = times.tz_localize('Europe/Istanbul', ambiguous='infer', nonexistent='shift_forward')
    
    wind_speeds = data['hourly']['wind_speed_100m']
    
    temp_df = pd.DataFrame({'ts': times, f'wind_{city}': wind_speeds}).set_index('ts')
    if df_weather is None:
        df_weather = temp_df
    else:
        df_weather = df_weather.join(temp_df)

print("3. Veriler Birleştiriliyor ve Özellik (Feature) Üretiliyor...")
df = df_db.join(df_weather).dropna()

# Rüzgar türbinleri kübik güç üretir (Power ~ v^3). Bu yüzden rüzgarın küpünü alarak fiziksel bir feature oluşturuyoruz!
for city in cities.keys():
    df[f'wind_power_{city}'] = df[f'wind_{city}'] ** 3

df['hour'] = df.index.hour
df['month'] = df.index.month
for i in range(1, 8):
    df[f'kgup_wind_lag_{i}d'] = df['kgup_wind'].shift(24 * i)

df = df.dropna()

print("4. Model Eğitiliyor ve 1 Yıllık Test Yapılıyor...")
test_days = 365
train_size = len(df) - (test_days * 24)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ['hour', 'month'] + [f'wind_power_{c}' for c in cities.keys()] + [f'wind_{c}' for c in cities.keys()] + [f'kgup_wind_lag_{i}d' for i in range(1, 8)]
target = 'kgup_wind'

lgb_wind = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
lgb_wind.fit(train[features], train[target])
pred = lgb_wind.predict(test[features])
pred = np.maximum(pred, 0) # Negatif rüzgar olamaz

def wape(y_t, y_p): return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100

error = wape(test[target], pred)
print(f"\n🚀 SADECE LAG MODELİ (Eski) WAPE: %36.22")
print(f"🚀 OPEN-METEO RÜZGAR HIZI EKLENMİŞ YENİ MODEL WAPE: {error:.2f}%")

imp = pd.DataFrame({'Feature': features, 'Importance': lgb_wind.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\nLightGBM Modelinin En Çok Önem Verdiği Değişkenler:")
print(imp.head(7).to_string(index=False))
