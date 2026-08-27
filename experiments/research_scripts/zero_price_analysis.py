import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))
from scripts.predict_daily_pipeline import load_all_historical_data
from db.connection import get_db_engine
from sqlalchemy import text

print("Veriler Yükleniyor...")
# 1. Ana veriyi çek
df = load_all_historical_data()

# 2. Pre-Forecast verilerini DB'den çekip joinle (ZPPI ve Residual Load için gerekli)
query = text("""
    SELECT 
        m.ts,
        k.river_hydro_mw AS kgup_river_hydro_mw,
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

# Kolon çakışmasını önlemek için sadece gerekli olanları alalım
df_sql = df_sql[['kgup_river_hydro_mw']]
df = df.join(df_sql)

# NaN temizliği
df = df.dropna(subset=['mcp_price_usd', 'predicted_load_lag0', 'predicted_wind_lag0', 'predicted_solar_lag0', 'kgup_river_hydro_mw'])

# 3. Kalan Yük (Residual Load) Hesaplama
# Residual Load = Toplam Tüketim - (Rüzgar + Güneş + Akarsu)
df['residual_load_mw'] = df['predicted_load_lag0'] - (df['predicted_wind_lag0'] + df['predicted_solar_lag0'] + df['kgup_river_hydro_mw'])

# 4. Kriz (Sıfır Fiyat) Etiketlemesi
ZERO_PRICE_THRESHOLD = 5.0
df['is_zero_price'] = (df['mcp_price_usd'] <= ZERO_PRICE_THRESHOLD).astype(int)

# Sadece Kriz ve Normal günleri ayır
kriz_df = df[df['is_zero_price'] == 1]
normal_df = df[df['is_zero_price'] == 0]

print(f"\nToplam Analiz Edilen Saat: {len(df)}")
print(f"Normal Saat Sayısı: {len(normal_df)}")
print(f"Sıfır Fiyat (Kriz) Saat Sayısı: {len(kriz_df)}\n")

# --- ANALİZ 1: KALAN YÜK (RESIDUAL LOAD) ORTALAMALARI ---
print("--- 1. KALAN YÜK (RESIDUAL LOAD) ANALİZİ ---")
print(f"Normal Saatlerde Ortalama Kalan Yük: {normal_df['residual_load_mw'].mean():.0f} MW")
print(f"Sıfır Fiyat Saatlerinde Ortalama Kalan Yük: {kriz_df['residual_load_mw'].mean():.0f} MW")
print(f"Sıfır Fiyat Saatlerinde MINIMUM Kalan Yük: {kriz_df['residual_load_mw'].min():.0f} MW")

# --- ANALİZ 2: SAAT (HOUR) DAĞILIMI ---
print("\n--- 2. KRİZLERİN SAATLERE GÖRE DAĞILIMI ---")
hour_dist = kriz_df.index.hour.value_counts().sort_index()
total_kriz = len(kriz_df)
for h, count in hour_dist.items():
    if count / total_kriz > 0.05: # Sadece krizlerin %5'inden fazlasının yaşandığı saatleri göster
        print(f"Saat {h:02d}:00 -> %{(count/total_kriz)*100:.1f} ({count} saat)")

# --- ANALİZ 3: GÜN (DAY OF WEEK) DAĞILIMI ---
print("\n--- 3. KRİZLERİN GÜNLERE GÖRE DAĞILIMI ---")
days = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
dow_dist = kriz_df.index.dayofweek.value_counts().sort_index()
for d, count in dow_dist.items():
    print(f"{days[d]}: %{(count/total_kriz)*100:.1f} ({count} saat)")

# --- ANALİZ 4: AY (MONTH) DAĞILIMI ---
print("\n--- 4. KRİZLERİN AYLARA GÖRE DAĞILIMI (Bahar Etkisi) ---")
months = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara']
month_dist = kriz_df.index.month.value_counts().sort_index()
for m, count in month_dist.items():
    print(f"{months[m-1]}: %{(count/total_kriz)*100:.1f} ({count} saat)")
