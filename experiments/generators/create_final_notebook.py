import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# Tüketim ve Üretim Tahmin Modelleri (Pre-KGÜP Forecasters) - GERÇEK WALK-FORWARD BACKTEST
Bu notebook, EPİAŞ'ın saat 14:00'te açıkladığı Yük Tahmini ve KGÜP verilerini, sabah 04:00'te (henüz yayınlanmamışken) tahmin etmek için tasarlanmıştır.

**ÖNEMLİ:** Bu testte statik bir eğitim kullanılmamıştır. Model her gün uyanıp, o güne kadarki TÜM geçmiş veriyle **kendini yeniden eğiterek** (Walk-Forward) ertesi günü tahmin eder. Bu, modelin üretim (production) ortamındaki gerçek performansını simüle eder.

**Kullanılan Metodolojiler:**
1. **Yük Tahmini (Load):** LightGBM (Geçmiş Lags + Open-Meteo Türkiye Sıcaklık)
2. **Güneş Üretimi (Solar):** Ridge Regression MWA (Geçmiş Lags)
3. **Rüzgar Üretimi (Wind):** LightGBM (Geçmiş Lags + Open-Meteo Rüzgar Gücü)
"""

code_setup = """import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
import lightgbm as lgb
import httpx
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from db.connection import get_db_engine
engine = get_db_engine()"""

code_db_data = """# 1. Veritabanından Hedef (Target) ve Sıcaklık Verilerini Çek
query = '''
    SELECT 
        l.ts,
        l.load_forecast_mw,
        k.wind_mw as kgup_wind,
        k.solar_mw as kgup_solar,
        w.turkey_weighted_temperature_c as temp_c
    FROM raw_load_forecast_hourly l
    JOIN raw_kgup_hourly k ON l.ts = k.ts
    LEFT JOIN raw_weather_hourly w ON l.ts = w.ts
    WHERE l.ts >= CURRENT_DATE - INTERVAL '24 months'
    ORDER BY l.ts ASC
'''
df = pd.read_sql(query, engine)
df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert('Europe/Istanbul')
df = df.set_index('ts')
display(df.head(3))"""

code_wind_data = """# 2. Open-Meteo API'den Rüzgar Hızı Çek
cities = {
    'izmir': {'lat': 38.41, 'lon': 27.14},
    'canakkale': {'lat': 40.15, 'lon': 26.40},
    'balikesir': {'lat': 39.64, 'lon': 27.88}
}

df_weather = None
print("Open-Meteo'dan veriler çekiliyor...")
for city, coords in cities.items():
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['lat']}&longitude={coords['lon']}&start_date=2024-01-01&end_date=2026-08-10&hourly=wind_speed_100m&timezone=Europe%2FIstanbul"
    response = httpx.get(url, timeout=30.0)
    data = response.json()
    
    times = pd.to_datetime(data['hourly']['time'])
    times = times.tz_localize('Europe/Istanbul', ambiguous='infer', nonexistent='shift_forward')
    wind_speeds = data['hourly']['wind_speed_100m']
    
    temp_df = pd.DataFrame({'ts': times, f'wind_{city}': wind_speeds}).set_index('ts')
    if df_weather is None:
        df_weather = temp_df
    else:
        df_weather = df_weather.join(temp_df)
print("Çekim tamamlandı!")"""

code_feature_eng = """# 3. Feature Engineering
df = df.join(df_weather).dropna()

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

df['cdh'] = np.maximum(df['temp_c'] - 18.0, 0.0)
df['hdh'] = np.maximum(18.0 - df['temp_c'], 0.0)

for city in cities.keys():
    df[f'wind_power_{city}'] = df[f'wind_{city}'] ** 3

targets = ['load_forecast_mw', 'kgup_wind', 'kgup_solar']
for t in targets:
    for i in range(1, 15):
        df[f'{t}_lag_{i}d'] = df[t].shift(24 * i)

df = df.dropna()
print(f"Toplam Veri Seti Boyutu: {df.shape}")"""

code_walk_forward = """# 4. WALK-FORWARD BACKTEST FONKSİYONU
def walk_forward_validation(df_data, target, features, model_obj, test_days=365):
    total_hours = len(df_data)
    test_hours = test_days * 24
    start_test = total_hours - test_hours
    
    predictions = []
    actuals = []
    
    print(f"Walk-Forward Backtest Başlıyor: {test_days} Gün (Her gün model yeniden eğitilecek)")
    
    # 7 günde bir eğitim (Hızlandırmak için haftalık eğitim de yapılabilir ama biz günlük yapacağız)
    for day in tqdm(range(test_days)):
        train_end = start_test + (day * 24)
        test_end = train_end + 24
        
        train_df = df_data.iloc[:train_end]
        test_df = df_data.iloc[train_end:test_end]
        
        # Model eğitimi (Her iterasyonda geçmiş güncel veriyle baştan eğitilir)
        model_obj.fit(train_df[features], train_df[target])
        
        # Tahmin (Sadece ertesi 24 saat)
        pred = model_obj.predict(test_df[features])
        
        predictions.extend(pred)
        actuals.extend(test_df[target].values)
        
    actuals = np.array(actuals)
    predictions = np.maximum(np.array(predictions), 0) # Negatif tahminleri engelle
    wape_err = (np.abs(actuals - predictions).sum() / actuals.sum()) * 100
    
    return actuals, predictions, wape_err"""

code_model_load = """# MODEL 1: Yük Tahmini (Load Forecast)
features_load = ['hour', 'dayofweek', 'month', 'is_weekend', 'temp_c', 'cdh', 'hdh'] + [f'load_forecast_mw_lag_{i}d' for i in range(1, 15)]
lgb_load = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)

actuals_load, preds_load, err_load = walk_forward_validation(df, 'load_forecast_mw', features_load, lgb_load, test_days=365)
print(f"✅ Yük Tahmini Gerçek Walk-Forward 1 Yıllık WAPE: {err_load:.2f}%")"""

code_model_solar = """# MODEL 2: Güneş Üretimi (Solar KGÜP)
features_solar = [f'kgup_solar_lag_{i}d' for i in range(1, 15)]
ridge_solar = Ridge(alpha=100.0)

actuals_solar, preds_solar, err_solar = walk_forward_validation(df, 'kgup_solar', features_solar, ridge_solar, test_days=365)
print(f"✅ Güneş KGÜP Gerçek Walk-Forward 1 Yıllık WAPE: {err_solar:.2f}%")"""

code_model_wind = """# MODEL 3: Rüzgar Üretimi (Wind KGÜP)
features_wind = ['hour', 'month'] + [f'wind_power_{c}' for c in cities.keys()] + [f'wind_{c}' for c in cities.keys()] + [f'kgup_wind_lag_{i}d' for i in range(1, 15)]
lgb_wind = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)

actuals_wind, preds_wind, err_wind = walk_forward_validation(df, 'kgup_wind', features_wind, lgb_wind, test_days=365)
print(f"✅ Rüzgar KGÜP Gerçek Walk-Forward 1 Yıllık WAPE: {err_wind:.2f}%")"""

nb.cells = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_db_data),
    nbf.v4.new_code_cell(code_wind_data),
    nbf.v4.new_code_cell(code_feature_eng),
    nbf.v4.new_code_cell(code_walk_forward),
    nbf.v4.new_code_cell(code_model_load),
    nbf.v4.new_code_cell(code_model_solar),
    nbf.v4.new_code_cell(code_model_wind)
]

with open('eda/advanced_kgup_load_forecaster.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook 'eda/advanced_kgup_load_forecaster.ipynb' Walk-Forward destekli olarak başarıyla oluşturuldu!")
