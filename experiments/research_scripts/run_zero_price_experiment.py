import sys
from pathlib import Path
project_root = Path("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini")
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from scripts.predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster

print("🚀 Starting 1-Year Full Backtest (Out-of-Sample: Last 365 Days / 8760 Hours)...")

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

# Train / Test split: Last 365 Days as Test set
test_start = df_clean.index.max() - pd.Timedelta(days=365)
train_df = df_clean.loc[:test_start].copy()
test_df = df_clean.loc[test_start:].copy()

print(f"📊 Dataset Total: {len(df_clean)} rows")
print(f"Eğitim Seti (Train): {len(train_df)} saat ({train_df.index.min()} -> {train_df.index.max()})")
print(f"Test Seti (Son 1 Yıl): {len(test_df)} saat ({test_df.index.min()} -> {test_df.index.max()})")

low_price_count = (test_df[target_col] <= 15.0).sum()
print(f"📉 Test setindeki düşük fiyatlı saatler (<= $15/MWh): {low_price_count} saat ({low_price_count/len(test_df)*100:.2f}%)")

def evaluate_predictions(name, y_true, p10, p50, p90):
    mae_all = np.mean(np.abs(y_true - p50))
    rmse_all = np.sqrt(np.mean((y_true - p50)**2))
    
    low_mask = y_true <= 15.0
    mae_low = np.mean(np.abs(y_true[low_mask] - p50[low_mask])) if np.sum(low_mask) > 0 else 0
    below_p10_low = np.mean(y_true[low_mask] < p10[low_mask]) * 100 if np.sum(low_mask) > 0 else 0
    
    in_interval = np.mean((y_true >= p10) & (y_true <= p90)) * 100
    below_p10 = np.mean(y_true < p10) * 100
    above_p90 = np.mean(y_true > p90) * 100
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

results = []

# --- MODEL 1: Baseline ---
print("Training Model 1: Baseline (Son 1 Yıl Backtest)...")
m1_p50 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_df[base_cols], train_df[target_col]).predict(test_df[base_cols])
m1_p10 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.10, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_df[base_cols], train_df[target_col]).predict(test_df[base_cols])
m1_p90 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.90, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_df[base_cols], train_df[target_col]).predict(test_df[base_cols])
m1_p10 = np.minimum(m1_p10, m1_p50)
m1_p90 = np.maximum(m1_p90, m1_p50)
results.append(evaluate_predictions("1. Baseline (Mevcut Model)", test_df[target_col].values, m1_p10, m1_p50, m1_p90))

# --- MODEL 2: Gelişmiş Özellikli (Lag 0 Ratios) ---
print("Training Model 2: Gelişmiş Özellikli (Lag 0 Ratios)...")
m2_p50 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'min_child_samples': 10, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_df[enhanced_cols], train_df[target_col]).predict(test_df[enhanced_cols])
m2_p10 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.10, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'min_child_samples': 10, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_df[enhanced_cols], train_df[target_col]).predict(test_df[enhanced_cols])
m2_p90 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.90, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'min_child_samples': 10, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(train_df[enhanced_cols], train_df[target_col]).predict(test_df[enhanced_cols])
m2_p10 = np.minimum(m2_p10, m2_p50)
m2_p90 = np.maximum(m2_p90, m2_p50)
results.append(evaluate_predictions("2. Gelişmiş Özellikli (Lag0 Ratios)", test_df[target_col].values, m2_p10, m2_p50, m2_p90))

# --- MODEL 3: Log-Transform ---
print("Training Model 3: Log-Transform (Log1p Kuantil)...")
m3_p50 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'verbose': -1, 'random_state': 42}, use_log_transform=True).fit(train_df[base_cols], train_df[target_col]).predict(test_df[base_cols])
m3_p10 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.10, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'verbose': -1, 'random_state': 42}, use_log_transform=True).fit(train_df[base_cols], train_df[target_col]).predict(test_df[base_cols])
m3_p90 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.90, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'verbose': -1, 'random_state': 42}, use_log_transform=True).fit(train_df[base_cols], train_df[target_col]).predict(test_df[base_cols])
m3_p10 = np.minimum(m3_p10, m3_p50)
m3_p90 = np.maximum(m3_p90, m3_p50)
results.append(evaluate_predictions("3. Log-Transform (Log1p Kuantil)", test_df[target_col].values, m3_p10, m3_p50, m3_p90))

results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("🏆 1-YEAR BACKTEST RESULTS COMPARISON (OUT-OF-SAMPLE TEST SET: LAST 365 DAYS / 8760 HOURS)")
print("="*80)
print(results_df.to_string(index=False))

# Monthly breakdown for test_df
test_df['month'] = pd.to_datetime(test_df.index).to_period('M')
test_df['y_true'] = test_df[target_col]
test_df['m1_mae'] = np.abs(test_df['y_true'] - m1_p50)
test_df['m2_mae'] = np.abs(test_df['y_true'] - m2_p50)
test_df['m3_mae'] = np.abs(test_df['y_true'] - m3_p50)

test_df['m1_cov'] = (test_df['y_true'] >= m1_p10) & (test_df['y_true'] <= m1_p90)
test_df['m2_cov'] = (test_df['y_true'] >= m2_p10) & (test_df['y_true'] <= m2_p90)
test_df['m3_cov'] = (test_df['y_true'] >= m3_p10) & (test_df['y_true'] <= m3_p90)

monthly_backtest = test_df.groupby('month').agg(
    hours=('y_true', 'count'),
    m1_mae=('m1_mae', 'mean'),
    m2_mae=('m2_mae', 'mean'),
    m3_mae=('m3_mae', 'mean'),
    m1_cov=('m1_cov', lambda x: x.mean() * 100),
    m2_cov=('m2_cov', lambda x: x.mean() * 100),
    m3_cov=('m3_cov', lambda x: x.mean() * 100)
)

print("\n=== MONTHLY BREAKDOWN OVER THE 1-YEAR BACKTEST ===")
print(monthly_backtest.round(2).to_string())

# Save results to CSVs for notebook visualization
results_df.to_csv("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini/eda/zero_price_experiment_results.csv", index=False)
monthly_backtest.round(2).to_csv("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini/eda/monthly_1year_backtest_results.csv")
print("\n✅ Results saved to eda/zero_price_experiment_results.csv and eda/monthly_1year_backtest_results.csv")
