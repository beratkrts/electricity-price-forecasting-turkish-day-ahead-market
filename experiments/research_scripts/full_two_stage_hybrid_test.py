import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from db.connection import get_db_engine
from sqlalchemy import text

print("1. Veriler Çekiliyor ve Altın Sinyaller (Özellikler) Üretiliyor...")
query = text("""
    SELECT 
        m.ts,
        m.price_usd AS mcp_price_usd,
        l.load_forecast_mw,
        k.wind_mw AS kgup_wind_mw,
        k.solar_mw AS kgup_solar_mw,
        k.river_hydro_mw AS kgup_river_hydro_mw,
        k.dammed_hydro_mw AS kgup_dammed_hydro_mw,
        pf.predicted_load_lag0,
        pf.predicted_solar_lag0,
        pf.predicted_wind_lag0
    FROM raw_mcp_hourly m
    JOIN raw_load_forecast_hourly l ON m.ts = l.ts
    JOIN raw_kgup_hourly k ON m.ts = k.ts
    JOIN gold.kgup_load_pre_forecasts pf ON m.ts = pf.target_ts
    ORDER BY m.ts ASC;
""")
df = pd.read_sql(query, get_db_engine())
df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert('Europe/Istanbul')
df = df.set_index('ts')

df['predicted_river_hydro_lag0'] = df['kgup_river_hydro_mw'].shift(24)

# ALTIN SİNYALLER (FEATURE ENGINEERING)
df['Net_Load'] = df['predicted_load_lag0'] - (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0'])
df['Renewable_Ratio'] = (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0']) / df['predicted_load_lag0']
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Is_Spring_Month'] = df.index.month.isin([3, 4, 5]).astype(int)

df['PTF_lag_24'] = df['mcp_price_usd'].shift(24)
df['PTF_lag_48'] = df['mcp_price_usd'].shift(48)
df['PTF_lag_168'] = df['mcp_price_usd'].shift(168)
df['Net_Load_lag_24'] = df['Net_Load'].shift(24)

df['Is_Sunday_Noon'] = ((df['DayOfWeek'] == 6) & (df['Hour'].isin([11, 12, 13, 14]))).astype(int)
df['Duck_Curve_Risk'] = df['Is_Sunday_Noon'] * df['predicted_solar_lag0']
df['Is_Must_Run_Violation'] = (df['Net_Load'] < 16000).astype(int)

features = [
    'Net_Load', 'Renewable_Ratio', 'Hour', 'DayOfWeek', 'Is_Weekend', 'Is_Spring_Month',
    'PTF_lag_24', 'PTF_lag_48', 'PTF_lag_168', 'Net_Load_lag_24',
    'Duck_Curve_Risk', 'Is_Must_Run_Violation', 'predicted_wind_lag0', 'predicted_solar_lag0'
]
df = df.dropna(subset=features + ['mcp_price_usd'])

ZERO_PRICE_THRESHOLD = 10.0
df['Target_Crash'] = (df['mcp_price_usd'] <= ZERO_PRICE_THRESHOLD).astype(int)

# Train/Test (Son 180 Gün Test)
train_size = len(df) - (180 * 24)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:].copy()

print("\n2. ESKİ MODEL EĞİTİLİYOR (Standart Tek Parça Regresyon)...")
old_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
old_model.fit(train_df[features], train_df['mcp_price_usd'])
test_df['Pred_Old'] = old_model.predict(test_df[features])

print("\n3. İKİ AŞAMALI HİBRİT MODEL EĞİTİLİYOR...")
# Aşama 3.1: Classifier (Sinyalleri Oku)
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, is_unbalance=True, random_state=42, verbose=-1)
clf.fit(train_df[features], train_df['Target_Crash'])

# Aşama 3.2: Normal Günler İçin Temiz Regresör (Sadece kriz olmayan günleri öğrenir)
train_normal = train_df[train_df['Target_Crash'] == 0]
reg_normal = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
reg_normal.fit(train_normal[features], train_normal['mcp_price_usd'])

# Aşama 3.3: Kriz Günleri İçin Quantile Regresör (Fiyatı dipte tutmak için alpha=0.1)
train_crash = train_df[train_df['Target_Crash'] == 1]
reg_crash = lgb.LGBMRegressor(objective='quantile', alpha=0.1, n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
if len(train_crash) > 0:
    reg_crash.fit(train_crash[features], train_crash['mcp_price_usd'])

# HİBRİT TAHMİN (TEST AŞAMASI)
test_df['Prob_Crash'] = clf.predict_proba(test_df[features])[:, 1]
test_df['Pred_Hybrid'] = 0.0

for idx in test_df.index:
    row = test_df.loc[[idx]]
    if row['Prob_Crash'].values[0] > 0.50:
        # Kriz Yakalandı! Quantile modelle veya 0 olarak bas
        test_df.loc[idx, 'Pred_Hybrid'] = reg_crash.predict(row[features])[0] if len(train_crash)>0 else 0.0
    else:
        # Kriz Yok! Temiz modelle tahmin et
        test_df.loc[idx, 'Pred_Hybrid'] = reg_normal.predict(row[features])[0]

def wape(y_t, y_p):
    return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100

mae_old = mean_absolute_error(test_df['mcp_price_usd'], test_df['Pred_Old'])
wape_old = wape(test_df['mcp_price_usd'], test_df['Pred_Old'])

mae_hybrid = mean_absolute_error(test_df['mcp_price_usd'], test_df['Pred_Hybrid'])
wape_hybrid = wape(test_df['mcp_price_usd'], test_df['Pred_Hybrid'])

print("\n===========================================")
print("🎯 NİHAİ A/B TEST SONUÇLARI (SON 6 AY)")
print("===========================================")
print(f"❌ ESKİ MODEL (Tek Parça)")
print(f"   MAE  : ${mae_old:.2f}")
print(f"   WAPE : %{wape_old:.2f}")
print("-------------------------------------------")
print(f"✅ YENİ İKİ AŞAMALI HİBRİT MODEL (Classifier + Quantile)")
print(f"   MAE  : ${mae_hybrid:.2f}")
print(f"   WAPE : %{wape_hybrid:.2f}")
print("===========================================")
