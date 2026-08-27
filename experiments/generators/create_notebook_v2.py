import nbformat as nbf

nb = nbf.v4.new_notebook()

code_setup = """import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from db.connection import get_db_engine
engine = get_db_engine()"""

code_data = """# 2 Yıllık Veri Çekimi (Test için son 1 yıl kullanılacak)
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
df.head()"""

code_features = """# Feature Engineering (Takvim, Hava Durumu ve Lag Özellikleri)
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Hava Durumu (18 derece baz alınarak)
df['cdh'] = np.maximum(df['temp_c'] - 18.0, 0.0)
df['hdh'] = np.maximum(18.0 - df['temp_c'], 0.0)

# Geçmiş 14 günün lagları (Tüm targetler için)
targets = ['load_forecast_mw', 'kgup_wind', 'kgup_solar']
for t in targets:
    for i in range(1, 15):
        df[f'{t}_lag_{i}d'] = df[t].shift(24 * i)

df = df.dropna()
df.shape"""

code_train_test = """# Train-Test Split (Son 365 Gün Test Seti Olarak Ayrılıyor)
test_days = 365
train_size = len(df) - (test_days * 24)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

def wape(y_t, y_p):
    return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100

print(f"Eğitim Seti (Train): {len(train)} saat")
print(f"Test Seti (Test)  : {len(test)} saat (Tam {test_days} Gün)")"""

code_load = """# 1. Yük Tahmini (Load Forecast) - LightGBM (Geçmiş + Takvim + Hava Durumu)
features_load = ['hour', 'dayofweek', 'month', 'is_weekend', 'temp_c', 'cdh', 'hdh'] + [f'load_forecast_mw_lag_{i}d' for i in range(1, 15)]

X_train_load, y_train_load = train[features_load], train['load_forecast_mw']
X_test_load, y_test_load = test[features_load], test['load_forecast_mw']

lgb_load = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
lgb_load.fit(X_train_load, y_train_load)
pred_load = lgb_load.predict(X_test_load)

err_load = wape(y_test_load, pred_load)
print(f"Yük Tahmini (1 Yıllık Test) WAPE: {err_load:.2f}%")

imp = pd.DataFrame({'Feature': features_load, 'Importance': lgb_load.feature_importances_}).sort_values(by='Importance', ascending=False)
display(imp.head(5))"""

code_solar = """# 2. Güneş (KGUP Solar) - Ridge Regression (Trained MWA)
features_solar = [f'kgup_solar_lag_{i}d' for i in range(1, 15)]

X_train_solar, y_train_solar = train[features_solar], train['kgup_solar']
X_test_solar, y_test_solar = test[features_solar], test['kgup_solar']

ridge_solar = Ridge(alpha=100.0)
ridge_solar.fit(X_train_solar, y_train_solar)
pred_solar = ridge_solar.predict(X_test_solar)
# Negatifleri sıfırla
pred_solar = np.maximum(pred_solar, 0)

err_solar = wape(y_test_solar, pred_solar)
print(f"Güneş KGUP Tahmini (1 Yıllık Test) Ridge WAPE: {err_solar:.2f}%")"""

code_wind = """# 3. Rüzgar (KGUP Wind) - LightGBM (Geçmiş + Takvim)
features_wind = ['hour', 'month'] + [f'kgup_wind_lag_{i}d' for i in range(1, 15)]

X_train_wind, y_train_wind = train[features_wind], train['kgup_wind']
X_test_wind, y_test_wind = test[features_wind], test['kgup_wind']

lgb_wind = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
lgb_wind.fit(X_train_wind, y_train_wind)
pred_wind = lgb_wind.predict(X_test_wind)
pred_wind = np.maximum(pred_wind, 0)

err_wind = wape(y_test_wind, pred_wind)
print(f"Rüzgar KGUP Tahmini (1 Yıllık Test) LightGBM WAPE: {err_wind:.2f}%")"""

nb.cells = [
    nbf.v4.new_markdown_cell("# Gelişmiş Tüketim ve Üretim Tahmin Modelleri (1 Yıllık Backtest)\nBu notebook, EPİAŞ'ın KGÜP ve Yük Tahmini verilerini sabah 04:00'te (henüz EPİAŞ yayınlamadan önce) yüksek doğrulukla tahmin etmek için tasarlanmıştır."),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_code_cell(code_features),
    nbf.v4.new_code_cell(code_train_test),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_solar),
    nbf.v4.new_code_cell(code_wind)
]

with open('eda/advanced_kgup_load_forecaster.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook eda/advanced_kgup_load_forecaster.ipynb oluşturuldu.")
