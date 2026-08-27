import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, LinearRegression
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))
from db.connection import get_db_engine
engine = get_db_engine()

# Fetch KGUP and Load Forecast
query = '''
    SELECT 
        k.ts,
        l.load_forecast_mw,
        k.wind_mw as kgup_wind,
        k.solar_mw as kgup_solar
    FROM raw_kgup_hourly k
    JOIN raw_load_forecast_hourly l ON k.ts = l.ts
    WHERE k.ts >= CURRENT_DATE - INTERVAL '12 months'
    ORDER BY k.ts ASC
'''
df = pd.read_sql(query, engine)
df['ts'] = pd.to_datetime(df['ts']).dt.tz_convert('Europe/Istanbul')
df = df.set_index('ts')

def create_lag_features(series, window=14):
    df_lags = pd.DataFrame({'target': series})
    for i in range(1, window + 1):
        df_lags[f'lag_{i}d'] = series.shift(24 * i)
    return df_lags.dropna()

def train_and_evaluate(series_name, series, window=14):
    df_lags = create_lag_features(series, window)
    
    # Train-test split (Last 60 days for testing)
    test_days = 60
    train_size = len(df_lags) - (test_days * 24)
    
    train = df_lags.iloc[:train_size]
    test = df_lags.iloc[train_size:]
    
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    X_test = test.drop(columns=['target'])
    y_test = test['target']
    
    # 1. Naive Mean (Eşit Ağırlıklı Ortalama)
    naive_pred = X_test.mean(axis=1)
    
    # 2. Linear Regression (Positive Weights - Klasik Eğitilmiş MWA)
    lr = LinearRegression(fit_intercept=False, positive=True)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    # 3. Ridge Regression (Regularized)
    ridge = Ridge(alpha=100.0, fit_intercept=True)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    
    def wape(y_t, y_p):
        return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100
        
    print(f"=== {series_name} ===")
    print(f"Naive (Eşit Ağırlıklı) WAPE: {wape(y_test, naive_pred):.2f}%")
    print(f"Linear Regression MWA WAPE: {wape(y_test, lr_pred):.2f}%")
    print(f"Ridge Regression WAPE: {wape(y_test, ridge_pred):.2f}%")
    
    weights = [round(w, 3) for w in lr.coef_]
    print(f"Öğrenilen Ağırlıklar (Son 1 Günden -> 14 Gün Önceye):\n{weights}\n")

train_and_evaluate("EPİAŞ Yük Tahmini (Load Forecast)", df['load_forecast_mw'], window=14)
train_and_evaluate("EPİAŞ Rüzgar (KGUP Wind)", df['kgup_wind'], window=14)
train_and_evaluate("EPİAŞ Güneş (KGUP Solar)", df['kgup_solar'], window=14)
