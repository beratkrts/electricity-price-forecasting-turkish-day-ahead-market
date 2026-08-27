import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging

logging.basicConfig(level=logging.ERROR) # Sadece hataları göster
sys.path.insert(0, str(Path.cwd()))

from src.features.feature_engineering import build_robust_features, get_feature_columns
from scripts.predict_daily_pipeline import load_all_historical_data

def wape(y_t, y_p):
    return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100

def mae(y_t, y_p):
    return np.abs(y_t - y_p).mean()

print("Veriler Yükleniyor...")
df_raw = load_all_historical_data()
df_feat = build_robust_features(df_raw)

target_col = 'mcp_price_usd'

# 1. Eski Model (Pre-Forecast Özellikleri Olmadan)
old_feat_cols = [c for c in get_feature_columns('robust', df_feat) if c not in ['predicted_load_lag0', 'predicted_solar_lag0', 'predicted_wind_lag0']]

df_old = df_feat.dropna(subset=old_feat_cols + [target_col])
train_old = df_old.iloc[:-90*24]
test_old = df_old.iloc[-90*24:]

lgb_old = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
lgb_old.fit(train_old[old_feat_cols], train_old[target_col].values)
pred_old = lgb_old.predict(test_old[old_feat_cols])

print(f"\n--- ESKİ MODEL (Ön-Tahminler Yok) ---")
print(f"MAE: ${mae(test_old[target_col].values, pred_old):.2f}")
print(f"WAPE: {wape(test_old[target_col].values, pred_old):.2f}%")

# 2. Yeni Model (Pre-Forecast Özellikleri İle)
new_feat_cols = get_feature_columns('robust', df_feat)

# Eğer veritabanında pre-forecasts henüz yoksa script uyarı verir
if 'predicted_load_lag0' not in df_feat.columns:
    print("\nHATA: Ön-tahmin verileri henüz veritabanına kaydedilmemiş. Lütfen backfill işleminin bitmesini bekleyin.")
    sys.exit(1)

df_new = df_feat.dropna(subset=new_feat_cols + [target_col])
train_new = df_new.iloc[:-90*24]
test_new = df_new.iloc[-90*24:]

lgb_new = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
lgb_new.fit(train_new[new_feat_cols], train_new[target_col].values)
pred_new = lgb_new.predict(test_new[new_feat_cols])

print(f"\n--- YENİ MODEL (Ön-Tahminler Dahil) ---")
print(f"MAE: ${mae(test_new[target_col].values, pred_new):.2f}")
print(f"WAPE: {wape(test_new[target_col].values, pred_new):.2f}%")

imp = pd.DataFrame({'Feature': new_feat_cols, 'Importance': lgb_new.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\n--- YENİ MODEL FEATURE IMPORTANCE (İLK 10) ---")
print(imp.head(10).to_string(index=False))
