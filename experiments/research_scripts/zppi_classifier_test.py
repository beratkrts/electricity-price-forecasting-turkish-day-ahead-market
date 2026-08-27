import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from scripts.predict_daily_pipeline import load_all_historical_data

print("Gelişmiş Veriler DB'den çekiliyor...")
df = load_all_historical_data()

# Akarsu (River) ve Barajlı (Dammed) özelliklerini SQL'den çekmedim (tek hydro kolonumuz var), bu yüzden River'ı basit bir yaklaşımla ayıralım.
# Aslında en doğrusu SQL'i güncellemektir ama test için "River = %30, Dammed = %70" gibi bir kural kullanamayız. 
# Zaten SQL query'de k.river_hydro_mw vardı. load_all_historical_data'da yoktu.
# Bu yüzden SQL sorgusunu kendim yazıyorum:

from db.connection import get_db_engine
from sqlalchemy import text
query = text("""
    SELECT 
        m.ts,
        m.price_usd AS mcp_price_usd,
        l.load_forecast_mw,
        k.wind_mw AS kgup_wind_mw,
        k.solar_mw AS kgup_solar_mw,
        k.river_hydro_mw AS kgup_river_hydro_mw,
        k.dammed_hydro_mw AS kgup_dammed_hydro_mw,
        w.turkey_weighted_temperature_c AS temperature_c,
        pf.predicted_load_lag0,
        pf.predicted_solar_lag0,
        pf.predicted_wind_lag0
    FROM raw_mcp_hourly m
    JOIN raw_load_forecast_hourly l ON m.ts = l.ts
    JOIN raw_kgup_hourly k ON m.ts = k.ts
    LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
    JOIN gold.kgup_load_pre_forecasts pf ON m.ts = pf.target_ts
    ORDER BY m.ts ASC;
""")
engine = get_db_engine()
df = pd.read_sql(query, engine)
df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert('Europe/Istanbul')
df = df.set_index('ts')

df['dayofyear'] = df.index.dayofyear
df['hour'] = df.index.hour
df['month'] = df.index.month
df['mcp_usd_lag_24'] = df['mcp_price_usd'].shift(24)
df['mcp_usd_lag_168'] = df['mcp_price_usd'].shift(168)

df['kgup_river_hydro_lag_24'] = df['kgup_river_hydro_mw'].shift(24)
df['kgup_dammed_hydro_lag_24'] = df['kgup_dammed_hydro_mw'].shift(24)

# Kar Erime Risk Faktörü
df['temp_c'] = df['temperature_c']
df['temp_lag_24'] = df['temp_c'].shift(24)
df['temp_diff'] = df['temp_c'] - df['temp_lag_24']
df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
df['snowmelt_risk'] = (df['is_spring'] * df['temp_diff']).clip(lower=0)

# River Hydro LightGBM Modeli
hydro_feats = ['month', 'dayofyear', 'kgup_river_hydro_lag_24', 'temp_c', 'snowmelt_risk']
df_hydro = df.dropna(subset=hydro_feats + ['kgup_river_hydro_mw'])
lgb_hydro = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
lgb_hydro.fit(df_hydro[hydro_feats], df_hydro['kgup_river_hydro_mw'])
df['predicted_river_hydro_lag0'] = lgb_hydro.predict(df[hydro_feats].fillna(0))

# ZPPI Formülü
df['zppi_ratio'] = (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0']) / df['predicted_load_lag0']

features = [
    'predicted_load_lag0', 'predicted_solar_lag0', 'predicted_wind_lag0', 
    'predicted_river_hydro_lag0', 'zppi_ratio', 'kgup_dammed_hydro_lag_24',
    'mcp_usd_lag_24', 'mcp_usd_lag_168', 'hour', 'month', 'dayofyear', 'snowmelt_risk'
]
df = df.dropna(subset=features + ['mcp_price_usd'])

df['is_zero_price'] = (df['mcp_price_usd'] <= 5.0).astype(int)

train_size = len(df) - (180 * 24)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:].copy()

clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, class_weight='balanced', random_state=42, verbose=-1)
clf.fit(train_df[features], train_df['is_zero_price'])

test_df['pred_is_zero_prob'] = clf.predict_proba(test_df[features])[:, 1]
test_df['pred_is_zero_class'] = (test_df['pred_is_zero_prob'] > 0.35).astype(int)

print("\n--- YENİ GELİŞMİŞ CLASSIFIER (ZPPI + AKARSU + SNOWMELT) BAŞARISI ---")
print(classification_report(test_df['is_zero_price'], test_df['pred_is_zero_class']))

imp = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\n--- CLASSIFIER FEATURE IMPORTANCE ---")
print(imp.head(7).to_string(index=False))
