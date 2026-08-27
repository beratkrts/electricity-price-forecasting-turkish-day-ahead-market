import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from scripts.predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns

print("1. Canlıdaki Asıl Veriler ve Özellikler Yükleniyor...")
df_raw = load_all_historical_data()
df = build_robust_features(df_raw) # CANLIDAKİ GERÇEK 30+ ÖZELLİK (Robust)

print("2. Yeni Hibrit 'Altın' Özellikler Ekleniyor...")
# Kendi geçici River Hydro Pre-forecast'imiz
from db.connection import get_db_engine
from sqlalchemy import text
query = text("SELECT ts, river_hydro_mw AS kgup_river_hydro_mw FROM raw_kgup_hourly ORDER BY ts ASC;")
df_river = pd.read_sql(query, get_db_engine())
df_river['ts'] = pd.to_datetime(df_river['ts']).dt.tz_convert('Europe/Istanbul')
df = df.join(df_river.set_index('ts'))

df['predicted_river_hydro_lag0'] = df['kgup_river_hydro_mw'].shift(24)

df['Net_Load'] = df['predicted_load_lag0'] - (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0'])
df['Renewable_Ratio'] = (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0']) / df['predicted_load_lag0']
df['Is_Sunday_Noon'] = ((df['dayofweek'] == 6) & (df.index.hour.isin([11, 12, 13, 14]))).astype(int)
df['Duck_Curve_Risk'] = df['Is_Sunday_Noon'] * df['predicted_solar_lag0']
df['Is_Must_Run_Violation'] = (df['Net_Load'] < 16000).astype(int)
df['Net_Load_lag_24'] = df['Net_Load'].shift(24)

# Eski Modelin GERÇEK özellikleri (Canlıdaki model ne kullanıyorsa o!)
old_features = get_feature_columns('robust', df)

# Yeni Hibrit Modelin özellikleri (Canlıdakiler + Yeni Sinyaller)
hybrid_features = old_features + [
    'Net_Load', 'Renewable_Ratio', 'Duck_Curve_Risk', 'Is_Must_Run_Violation', 'Net_Load_lag_24'
]

# NaN Temizliği
df = df.dropna(subset=hybrid_features + ['mcp_price_usd'])

ZERO_PRICE_THRESHOLD = 10.0
df['Target_Crash'] = (df['mcp_price_usd'] <= ZERO_PRICE_THRESHOLD).astype(int)

# Walk-Forward Ayarları
test_days = 365
max_ts = df.index.max()

actuals = []
preds_old = []
preds_hybrid = []
timestamps = []

print("3. Gerçek 1 Yıllık Walk-Forward Testi Başlıyor...")
for day_idx in tqdm(range(test_days - 1, -1, -1)):
    test_end = max_ts - pd.Timedelta(days=day_idx)
    test_start = test_end - pd.Timedelta(hours=23)
    train_end = test_start - pd.Timedelta(hours=1)
    
    tr_df = df.loc[:train_end]
    te_df = df.loc[test_start:test_end]
    
    if len(tr_df) < 1000 or len(te_df) < 12:
        continue
        
    # TAMAMEN ADİL KIYASLAMA: Eski model sadece canlıdaki özellikleri kullanacak
    old_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
    old_model.fit(tr_df[old_features], tr_df['mcp_price_usd'])
    p_old = old_model.predict(te_df[old_features])
    
    # YENİ HİBRİT MİMARİ
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=5, is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(tr_df[hybrid_features], tr_df['Target_Crash'])
    
    tr_normal = tr_df[tr_df['Target_Crash'] == 0]
    reg_normal = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
    if len(tr_normal) > 0: reg_normal.fit(tr_normal[hybrid_features], tr_normal['mcp_price_usd'])
    
    tr_crash = tr_df[tr_df['Target_Crash'] == 1]
    reg_crash = lgb.LGBMRegressor(objective='quantile', alpha=0.1, n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1, n_jobs=-1)
    if len(tr_crash) > 0: reg_crash.fit(tr_crash[hybrid_features], tr_crash['mcp_price_usd'])
    
    # Tahmin
    p_hybrid = []
    prob_crash = clf.predict_proba(te_df[hybrid_features])[:, 1]
    
    for i, (idx, row) in enumerate(te_df.iterrows()):
        feats = row[hybrid_features].to_frame().T
        if prob_crash[i] > 0.50:
            val = reg_crash.predict(feats)[0] if len(tr_crash) > 0 else 0.0
        else:
            val = reg_normal.predict(feats)[0] if len(tr_normal) > 0 else old_model.predict(row[old_features].to_frame().T)[0]
        p_hybrid.append(val)
        
    timestamps.extend(te_df.index)
    actuals.extend(te_df['mcp_price_usd'].values)
    preds_old.extend(p_old)
    preds_hybrid.extend(p_hybrid)

def wape(y_t, y_p):
    return (np.sum(np.abs(np.array(y_t) - np.array(y_p))) / np.sum(y_t)) * 100

df_res = pd.DataFrame({'ts': timestamps, 'actual': actuals, 'pred_old': preds_old, 'pred_hybrid': preds_hybrid})
df_res = df_res.set_index('ts')
df_res['month'] = df_res.index.month

spring_df = df_res[df_res['month'].isin([3, 4, 5])]
winter_df = df_res[~df_res['month'].isin([3, 4, 5])]

print("\n===========================================")
print("🎯 1 YILLIK WALK-FORWARD - MEVSİM ANALİZİ")
print("===========================================")
print(f"❌ ESKİ MODEL (Tek Parça)")
print(f"   TÜM YIL WAPE : %{wape(df_res['actual'], df_res['pred_old']):.2f}")
print(f"   SADECE BAHAR WAPE : %{wape(spring_df['actual'], spring_df['pred_old']):.2f}")
print(f"   KIŞ/YAZ WAPE : %{wape(winter_df['actual'], winter_df['pred_old']):.2f}")
print("-------------------------------------------")
print(f"✅ YENİ HİBRİT MODEL")
print(f"   TÜM YIL WAPE : %{wape(df_res['actual'], df_res['pred_hybrid']):.2f}")
print(f"   SADECE BAHAR WAPE : %{wape(spring_df['actual'], spring_df['pred_hybrid']):.2f}")
print(f"   KIŞ/YAZ WAPE : %{wape(winter_df['actual'], winter_df['pred_hybrid']):.2f}")
print("===========================================")
