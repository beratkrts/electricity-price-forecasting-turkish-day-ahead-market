import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from db.connection import get_db_engine
from sqlalchemy import text

print("1. Veri Boru Hattı (Data Pipeline) Başlatılıyor...")
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

print("2. Özellik Mühendisliği (Feature Engineering) Uygulanıyor...")
# Kendi Pre-forecaster'larımız (Bahar volatilitesi için)
# (River Hydro için şimdilik önceki günün aynı saatini kullanıyoruz, Airflow'da SHAP'a bağlanacak)
df['predicted_river_hydro_lag0'] = df['kgup_river_hydro_mw'].shift(24)

# --- GEMINI RAPORUNDAKİ KRİTİK ÖZELLİKLER ---
# Net Yük (En Kritik Sinyal)
df['Net_Load'] = df['predicted_load_lag0'] - (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0'])

# Yenilenebilir Oranı
df['Renewable_Ratio'] = (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['predicted_river_hydro_lag0']) / df['predicted_load_lag0']

# Takvim Özellikleri
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Is_Spring_Month'] = df.index.month.isin([3, 4, 5]).astype(int)

# Gecikmeli Değişkenler (Lags)
df['PTF_lag_24'] = df['mcp_price_usd'].shift(24)
df['PTF_lag_48'] = df['mcp_price_usd'].shift(48)
df['PTF_lag_168'] = df['mcp_price_usd'].shift(168)
df['Net_Load_lag_24'] = df['Net_Load'].shift(24)

# --- BİZİM KEŞFETTİĞİMİZ "ALTIN" ETKİLEŞİM ÖZELLİKLERİ ---
# 1. Pazar Öğleni Ördek Eğrisi (Duck Curve)
df['Is_Sunday_Noon'] = ((df['DayOfWeek'] == 6) & (df['Hour'].isin([11, 12, 13, 14]))).astype(int)
df['Duck_Curve_Risk'] = df['Is_Sunday_Noon'] * df['predicted_solar_lag0']

# 2. Minimum Kapatılamayan Yük İhlali (Must-Run Violation - 15.000 MW eşiği)
df['Is_Must_Run_Violation'] = (df['Net_Load'] < 16000).astype(int)

# 3. Model Hazırlığı
features = [
    'Net_Load', 'Renewable_Ratio', 'Hour', 'DayOfWeek', 'Is_Weekend', 'Is_Spring_Month',
    'PTF_lag_24', 'PTF_lag_48', 'PTF_lag_168', 'Net_Load_lag_24',
    'Duck_Curve_Risk', 'Is_Must_Run_Violation', 'predicted_wind_lag0', 'predicted_solar_lag0'
]
df = df.dropna(subset=features + ['mcp_price_usd'])

# Hedef Değişken (10 Dolar altı Kriz)
ZERO_PRICE_THRESHOLD = 10.0
df['Target_Crash'] = (df['mcp_price_usd'] <= ZERO_PRICE_THRESHOLD).astype(int)

# Train/Test Ayrımı (Son 6 Ay Test)
train_size = len(df) - (180 * 24)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:].copy()

print("3. LightGBM Sınıflandırıcı Eğitiliyor (is_unbalance=True)...")
clf = lgb.LGBMClassifier(
    n_estimators=300, 
    learning_rate=0.03, 
    max_depth=6, 
    is_unbalance=True, # Dengesiz veriyi otomatik yönetir
    random_state=42, 
    verbose=-1
)
clf.fit(train_df[features], train_df['Target_Crash'])

# Tahmin ve Metrikler
test_df['Prob_Crash'] = clf.predict_proba(test_df[features])[:, 1]
# PR-AUC optimizasyonu için eşiği belirliyoruz (Örn: 0.50)
test_df['Pred_Crash'] = (test_df['Prob_Crash'] > 0.50).astype(int)

precision, recall, thresholds = precision_recall_curve(test_df['Target_Crash'], test_df['Prob_Crash'])
pr_auc = auc(recall, precision)

print("\n--- NİHAİ MODEL BAŞARISI ---")
print(f"PR-AUC Skoru: {pr_auc:.3f}")
print("\nKarmaşıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(test_df['Target_Crash'], test_df['Pred_Crash']))
print("\nSınıflandırma Raporu:")
print(classification_report(test_df['Target_Crash'], test_df['Pred_Crash']))

print("4. SHAP (Feature Importance) Analizi Çıkarılıyor...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(test_df[features])

# SHAP Görselini Kaydet
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, test_df[features], show=False)
plt.title("SHAP Feature Importance - Sıfır Fiyat Kök Nedenleri", fontsize=14)
plt.tight_layout()
plt.savefig("eda/shap_summary.png", dpi=300)
print("SHAP analizi başarıyla kaydedildi: eda/shap_summary.png")
