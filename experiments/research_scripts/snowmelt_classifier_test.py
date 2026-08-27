import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import httpx
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from scripts.predict_daily_pipeline import load_all_historical_data

print("1. DB'den veriler çekiliyor...")
df = load_all_historical_data()
df['ts_idx'] = df.index
min_date = df.index.min().strftime('%Y-%m-%d')
max_date = df.index.max().strftime('%Y-%m-%d')

print("2. Open-Meteo Uydusundan Kar Kalınlığı (Snow Depth) çekiliyor...")
cities = {
    'artvin': {'lat': 41.18, 'lon': 41.81},
    'erzincan': {'lat': 39.75, 'lon': 39.50},
    'tunceli': {'lat': 39.10, 'lon': 39.54}
}
df_snow = None
for city, coords in cities.items():
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['lat']}&longitude={coords['lon']}&start_date={min_date}&end_date={max_date}&hourly=snow_depth&timezone=Europe%2FIstanbul"
    resp = httpx.get(url, timeout=30.0)
    data = resp.json()
    times = pd.to_datetime(data['hourly']['time']).tz_localize('Europe/Istanbul', ambiguous='infer', nonexistent='shift_forward')
    depths = data['hourly']['snow_depth']
    temp_df = pd.DataFrame({'ts': times, f'snow_{city}': depths}).set_index('ts')
    if df_snow is None: df_snow = temp_df
    else: df_snow = df_snow.join(temp_df)

df = df.join(df_snow)

# Feature Engineering
df['dayofyear'] = df.index.dayofyear
df['hour'] = df.index.hour
df['month'] = df.index.month
df['mcp_usd_lag_24'] = df['mcp_price_usd'].shift(24)
df['mcp_usd_lag_168'] = df['mcp_price_usd'].shift(168)

from db.connection import get_db_engine
from sqlalchemy import text
query = text("""
    SELECT 
        m.ts,
        k.river_hydro_mw AS kgup_river_hydro_mw,
        k.dammed_hydro_mw AS kgup_dammed_hydro_mw,
        pf.predicted_load_lag0,
        pf.predicted_solar_lag0,
        pf.predicted_wind_lag0
    FROM raw_mcp_hourly m
    JOIN raw_kgup_hourly k ON m.ts = k.ts
    JOIN gold.kgup_load_pre_forecasts pf ON m.ts = pf.target_ts
    ORDER BY m.ts ASC;
""")
df_sql = pd.read_sql(query, get_db_engine())
df_sql['ts'] = pd.to_datetime(df_sql['ts']).dt.tz_convert('Europe/Istanbul')
df_sql = df_sql.set_index('ts')
df_sql = df_sql[['kgup_river_hydro_mw', 'kgup_dammed_hydro_mw']]
df = df.join(df_sql)

df['kgup_river_hydro_lag_24'] = df['kgup_river_hydro_mw'].shift(24)
df['kgup_dammed_hydro_lag_24'] = df['kgup_dammed_hydro_mw'].shift(24)

# Akarsu (River Hydro) Tahmin Modeli
hydro_feats = ['month', 'dayofyear', 'kgup_river_hydro_lag_24', 'snow_artvin', 'snow_erzincan', 'snow_tunceli']
df_hydro = df.dropna(subset=hydro_feats + ['kgup_river_hydro_mw'])
lgb_hydro = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
lgb_hydro.fit(df_hydro[hydro_feats], df_hydro['kgup_river_hydro_mw'])
df['predicted_river_hydro_lag0'] = lgb_hydro.predict(df[hydro_feats].fillna(0))

# ZPPI Formülü
df['zppi_ratio'] = (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0']) / df['predicted_load_lag0']

# Classifier Eğitimi
features = [
    'predicted_load_lag0', 'predicted_solar_lag0', 'predicted_wind_lag0', 
    'predicted_river_hydro_lag0', 'zppi_ratio', 'kgup_dammed_hydro_lag_24',
    'mcp_usd_lag_24', 'mcp_usd_lag_168', 'hour', 'month', 'dayofyear',
    'snow_artvin', 'snow_erzincan', 'snow_tunceli'
]
df = df.dropna(subset=features + ['mcp_price_usd'])
df['is_zero_price'] = (df['mcp_price_usd'] <= 5.0).astype(int)

train_size = len(df) - (180 * 24)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:].copy()

clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)
clf.fit(train_df[features], train_df['is_zero_price'])

test_df['pred_is_zero_prob'] = clf.predict_proba(test_df[features])[:, 1]
# Threshold'u 0.30'a çekerek recall'u maksimize ediyoruz
test_df['pred_is_zero_class'] = (test_df['pred_is_zero_prob'] > 0.30).astype(int)

print("\n--- YENİ GELİŞMİŞ CLASSIFIER (ZPPI + GERÇEK KAR UYDU VERİSİ) BAŞARISI ---")
print(classification_report(test_df['is_zero_price'], test_df['pred_is_zero_class']))

imp = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\n--- CLASSIFIER FEATURE IMPORTANCE ---")
print(imp.head(7).to_string(index=False))
