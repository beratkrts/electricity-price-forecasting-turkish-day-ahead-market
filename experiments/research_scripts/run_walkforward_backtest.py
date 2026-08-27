import sys
import time
import logging
from pathlib import Path

project_root = Path("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini")
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from scripts.predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WalkForwardBacktest")

logger.info("🚀 Starting 365-Day Daily Walk-Forward Backtest (Every single day model re-trained)...")

start_time = time.time()

# 1. Load historical dataset
df_raw = load_all_historical_data()
df_feat = build_robust_features(df_raw)

# 2. Add New Feature Engineering Innovations (Lag 0 Ratios)
load_lag0_safe = df_feat['predicted_load_lag0'].replace(0, np.nan).fillna(df_feat['load_forecast_mw'].replace(0, np.nan))

df_feat['renewable_pressure_ratio_lag0'] = (
    (df_feat['predicted_solar_lag0'].fillna(df_feat['kgup_solar_mw']) + 
     df_feat['predicted_wind_lag0'].fillna(df_feat['kgup_wind_mw']) + 
     df_feat['kgup_hydro_lag_24'].fillna(0)) / load_lag0_safe
).fillna(0)

df_feat['solar_peak_pressure_ratio_lag0'] = (
    df_feat['predicted_solar_lag0'].fillna(df_feat['kgup_solar_mw']) / load_lag0_safe
).fillna(0)

df_feat['zero_price_risk_score'] = (
    df_feat['renewable_pressure_ratio_lag0'] * (1.0 - (df_feat['hour'].isin([17,18,19,20,21])).astype(float))
)

base_cols = get_feature_columns('robust', df_feat)
new_cols = ['renewable_pressure_ratio_lag0', 'solar_peak_pressure_ratio_lag0', 'zero_price_risk_score']
enhanced_cols = base_cols + new_cols
target_col = 'mcp_price_usd'

df_clean = df_feat.dropna(subset=base_cols + [target_col]).copy()

# Setup Walk-Forward loop for 365 days
max_ts = df_clean.index.max()
num_days = 365

m1_preds_list = []
m2_preds_list = []
m3_preds_list = []

logger.info(f"⏳ Walk-Forward Simülasyonu başlatılıyor: {num_days} gün boyunca her gün model sıfırdan eğitilecek...")

for day_idx in range(num_days - 1, -1, -1):
    test_end = max_ts - pd.Timedelta(days=day_idx)
    test_start = test_end - pd.Timedelta(hours=23)
    train_end = test_start - pd.Timedelta(hours=1)
    
    train_subset = df_clean.loc[:train_end]
    test_subset = df_clean.loc[test_start:test_end]
    
    if len(train_subset) < 1000 or len(test_subset) < 12:
        continue

    # Model 1: Baseline (No Lag0 ratios, No Log)
    m1_p50 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_subset[base_cols], train_subset[target_col]).predict(test_subset[base_cols])
    m1_p10 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.10, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_subset[base_cols], train_subset[target_col]).predict(test_subset[base_cols])
    m1_p90 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.90, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_subset[base_cols], train_subset[target_col]).predict(test_subset[base_cols])
    
    # Model 2: Enhanced (Lag 0 Ratios)
    m2_p50 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_subset[enhanced_cols], train_subset[target_col]).predict(test_subset[enhanced_cols])
    m2_p10 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.10, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_subset[enhanced_cols], train_subset[target_col]).predict(test_subset[enhanced_cols])
    m2_p90 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.90, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_subset[enhanced_cols], train_subset[target_col]).predict(test_subset[enhanced_cols])

    # Model 3: Log-Transform
    m3_p50 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=True).fit(train_subset[base_cols], train_subset[target_col]).predict(test_subset[base_cols])
    m3_p10 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.10, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=True).fit(train_subset[base_cols], train_subset[target_col]).predict(test_subset[base_cols])
    m3_p90 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.90, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'n_jobs': -1, 'verbose': -1, 'random_state': 42}, use_log_transform=True).fit(train_subset[base_cols], train_subset[target_col]).predict(test_subset[base_cols])

    for ts_idx, real_price in zip(test_subset.index, test_subset[target_col]):
        i_pos = test_subset.index.get_loc(ts_idx)
        m1_preds_list.append({'ts': ts_idx, 'y_true': real_price, 'p10': m1_p10[i_pos], 'p50': m1_p50[i_pos], 'p90': m1_p90[i_pos]})
        m2_preds_list.append({'ts': ts_idx, 'y_true': real_price, 'p10': m2_p10[i_pos], 'p50': m2_p50[i_pos], 'p90': m2_p90[i_pos]})
        m3_preds_list.append({'ts': ts_idx, 'y_true': real_price, 'p10': m3_p10[i_pos], 'p50': m3_p50[i_pos], 'p90': m3_p90[i_pos]})

    processed_days = num_days - day_idx
    if processed_days % 30 == 0 or processed_days == num_days:
        elapsed = time.time() - start_time
        logger.info(f"⏳ Processed {processed_days}/{num_days} days ({processed_days/num_days*100:.1f}%) in {elapsed:.1f}s...")

df_m1 = pd.DataFrame(m1_preds_list).set_index('ts')
df_m2 = pd.DataFrame(m2_preds_list).set_index('ts')
df_m3 = pd.DataFrame(m3_preds_list).set_index('ts')

def eval_df(name, df_pred):
    y = df_pred['y_true']
    p10 = np.minimum(df_pred['p10'], df_pred['p50'])
    p50 = df_pred['p50']
    p90 = np.maximum(df_pred['p90'], df_pred['p50'])
    
    mae_all = np.mean(np.abs(y - p50))
    rmse_all = np.sqrt(np.mean((y - p50)**2))
    
    low_mask = y <= 15.0
    mae_low = np.mean(np.abs(y[low_mask] - p50[low_mask])) if np.sum(low_mask) > 0 else 0
    below_p10_low = np.mean(y[low_mask] < p10[low_mask]) * 100 if np.sum(low_mask) > 0 else 0
    
    in_interval = np.mean((y >= p10) & (y <= p90)) * 100
    below_p10 = np.mean(y < p10) * 100
    above_p90 = np.mean(y > p90) * 100
    avg_width = np.mean(p90 - p10)
    max_p90 = np.max(p90)
    
    return {
        "Model": name,
        "Overall MAE ($)": round(mae_all, 2),
        "Overall RMSE ($)": round(rmse_all, 2),
        "Low-Price MAE ($<=15)": round(mae_low, 2),
        "Low-Price Below P10 (%)": round(below_p10_low, 1),
        "80% Coverage (%)": round(in_interval, 1),
        "Below P10 All (%)": round(below_p10, 1),
        "Above P90 All (%)": round(above_p90, 1),
        "Avg Band ($)": round(avg_width, 2),
        "Max P90 ($)": round(max_p90, 1)
    }

results = [
    eval_df("1. Baseline (Mevcut Model - WF)", df_m1),
    eval_df("2. Gelişmiş Özellikli (Lag0 Ratios - WF)", df_m2),
    eval_df("3. Log-Transform (Log1p Kuantil - WF)", df_m3),
]

summary_df = pd.DataFrame(results)
print("\n" + "="*80)
print("🏆 365-DAY WALK-FORWARD BACKTEST RESULTS (EVERY SINGLE DAY RE-TRAINED)")
print("="*80)
print(summary_df.to_string(index=False))

# Monthly breakdown comparison
df_m1['month'] = df_m1.index.to_period('M')
df_m2['month'] = df_m2.index.to_period('M')
df_m3['month'] = df_m3.index.to_period('M')

df_m1['mae'] = np.abs(df_m1['y_true'] - df_m1['p50'])
df_m2['mae'] = np.abs(df_m2['y_true'] - df_m2['p50'])
df_m3['mae'] = np.abs(df_m3['y_true'] - df_m3['p50'])

df_m1['cov'] = (df_m1['y_true'] >= df_m1['p10']) & (df_m1['y_true'] <= df_m1['p90'])
df_m2['cov'] = (df_m2['y_true'] >= df_m2['p10']) & (df_m2['y_true'] <= df_m2['p90'])
df_m3['cov'] = (df_m3['y_true'] >= df_m3['p10']) & (df_m3['y_true'] <= df_m3['p90'])

m1_monthly = df_m1.groupby('month').agg(m1_mae=('mae', 'mean'), m1_cov=('cov', lambda x: x.mean()*100))
m2_monthly = df_m2.groupby('month').agg(m2_mae=('mae', 'mean'), m2_cov=('cov', lambda x: x.mean()*100))
m3_monthly = df_m3.groupby('month').agg(m3_mae=('mae', 'mean'), m3_cov=('cov', lambda x: x.mean()*100))

monthly_wf = m1_monthly.join(m2_monthly).join(m3_monthly)
print("\n=== MONTHLY WALK-FORWARD BREAKDOWN ===")
print(monthly_wf.round(2).to_string())

# Save results to CSVs
summary_df.to_csv("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini/eda/zero_price_experiment_results.csv", index=False)
monthly_wf.round(2).to_csv("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini/eda/monthly_1year_backtest_results.csv")
print("\n✅ Results saved to CSVs.")
