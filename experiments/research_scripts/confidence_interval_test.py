import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from scripts.predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns

print("1. Veriler Yükleniyor ve Özellikler Çıkarılıyor...")
df_raw = load_all_historical_data()
df = build_robust_features(df_raw)
features = get_feature_columns('robust', df)
df = df.dropna(subset=features + ['mcp_price_usd'])

# Son 30 Günü Test İçin Ayıralım (Hızlı Gösterim)
test_days = 30
train_size = len(df) - (test_days * 24)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:].copy()

print("2. Ana Regresör (Nokta Tahmini) Eğitiliyor...")
model_point = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
model_point.fit(train_df[features], train_df['mcp_price_usd'])
test_df['Pred_Point'] = model_point.predict(test_df[features])

print("3. Alt Sınır (%10 Quantile) Regresörü Eğitiliyor...")
model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.10, n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
model_lower.fit(train_df[features], train_df['mcp_price_usd'])
test_df['Pred_Lower'] = model_lower.predict(test_df[features])

print("4. Üst Sınır (%90 Quantile) Regresörü Eğitiliyor...")
model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
model_upper.fit(train_df[features], train_df['mcp_price_usd'])
test_df['Pred_Upper'] = model_upper.predict(test_df[features])

# Mantık Hatasını Düzeltme (Alt Sınır Üst Sınırdan büyük olamaz)
test_df['Pred_Lower'] = np.minimum(test_df['Pred_Lower'], test_df['Pred_Upper'] - 0.1)

# --- DEĞERLENDİRME ---
# 1. Kapsama Oranı (Coverage Ratio): Gerçek fiyat bu iki sınırın arasına düştü mü?
test_df['Is_Inside_Interval'] = ((test_df['mcp_price_usd'] >= test_df['Pred_Lower']) & 
                                 (test_df['mcp_price_usd'] <= test_df['Pred_Upper'])).astype(int)
coverage_ratio = test_df['Is_Inside_Interval'].mean() * 100

# 2. Ortalama Aralık Genişliği (Bandwidth): Aralık ne kadar geniş? Çok genişse model güvensizdir.
test_df['Interval_Width'] = test_df['Pred_Upper'] - test_df['Pred_Lower']
mean_width = test_df['Interval_Width'].mean()

print("\n===========================================")
print("🎯 GÜVEN ARALIĞI (CONFIDENCE INTERVAL) TESTİ (Son 30 Gün)")
print("===========================================")
print(f"Hedeflenen Güven Aralığı : %80 (P10 - P90)")
print(f"Gerçekleşen Kapsama Oranı: %{coverage_ratio:.1f} (Fiyatların %{coverage_ratio:.1f}'si koridora düştü)")
print(f"Ortalama Koridor Genişliği: ${mean_width:.2f} (Üst Sınır ile Alt Sınır Arasındaki Ortalama Fark)")
print("\nÖrnek 5 Saatlik Tahmin Çıktısı:")
sample_output = test_df[['mcp_price_usd', 'Pred_Lower', 'Pred_Point', 'Pred_Upper', 'Is_Inside_Interval']].tail(5)
for idx, row in sample_output.iterrows():
    status = "✅ İÇİNDE" if row['Is_Inside_Interval'] else "❌ DIŞINDA"
    print(f"Saat: {idx.strftime('%Y-%m-%d %H:%00')} | Gerçek: ${row['mcp_price_usd']:.2f} | Tahmin: ${row['Pred_Point']:.2f} | Aralık: [${row['Pred_Lower']:.2f} - ${row['Pred_Upper']:.2f}] -> {status}")
print("===========================================")
