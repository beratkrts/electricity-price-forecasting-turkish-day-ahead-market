"""
24 Direct Hourly LightGBM Specialist Models Experiment (lgbm_direct_24h_v1)

Location: experiments/scripts/run_24h_direct_lgbm_experiment.py
Trains 24 separate specialized LightGBM models (one per hour of the day)
and evaluates performance across 365 test days. Logs result via run_experiment_runner.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "experiments" / "scripts"))

from predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns
from run_experiment_runner import log_experiment_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("24H_Direct_LGBM")


def run_24h_direct_experiment():
    experiment_name = "lgbm_direct_24h_daily_v1"
    logger.info("=" * 80)
    logger.info(f"🚀 STARTING SYSTEMATIC EXPERIMENT: {experiment_name}")
    logger.info("  Architecture: 24 Specialized Direct Hourly LightGBM Models")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    logger.info("📥 Loading raw dataset from PostgreSQL...")
    df_raw = load_all_historical_data()
    
    logger.info("⚙️ Building robust features...")
    df_feat = build_robust_features(df_raw)
    
    feature_cols = get_feature_columns('robust', df_feat)
    target_col = 'mcp_price_usd'
    
    df_model = df_feat.dropna(subset=feature_cols + [target_col]).copy()
    
    # 365 Days Test Window
    test_days = 365
    eval_dates = df_model.index.normalize().unique()[-test_days:]
    
    pretrain_end = eval_dates[0] - pd.Timedelta(hours=1)
    train_pretrain = df_model.loc[:pretrain_end].copy()
    
    logger.info(f"📅 Pre-train Window: {train_pretrain.index.min().strftime('%Y-%m-%d')} -> {train_pretrain.index.max().strftime('%Y-%m-%d')} ({len(train_pretrain)} hours)")
    logger.info(f"🎯 Test Window: {eval_dates.min().strftime('%Y-%m-%d')} -> {eval_dates.max().strftime('%Y-%m-%d')} ({len(eval_dates)} test days)")
    
    # LightGBM Hyperparameters
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'learning_rate': 0.03,
        'n_estimators': 250,
        'num_leaves': 31,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    logger.info("🏋️ Training 24 specialized hourly models on base historical data...")
    hourly_models = {}
    for h in range(24):
        df_h = train_pretrain[train_pretrain['hour'] == h]
        X_h = df_h[feature_cols]
        y_h = df_h[target_col]
        
        model_h = lgb.LGBMRegressor(**lgb_params)
        model_h.fit(X_h, y_h)
        hourly_models[h] = model_h
        
    logger.info("✅ 24 Hourly Models successfully pre-trained!")
    
    logger.info("🔄 Running 365-day Walk-Forward Backtest...")
    preds_all = []
    acts_all = []
    
    # We update models daily (every single test day) to match production daily re-train baseline
    update_frequency_days = 1
    
    current_train_df = train_pretrain.copy()
    
    for idx, d in enumerate(eval_dates):
        t_start = d
        t_end = d + pd.Timedelta(hours=23)
        
        day_df = df_model.loc[t_start:t_end]
        if len(day_df) != 24:
            continue
            
        # Re-train models every 30 days
        if idx > 0 and idx % update_frequency_days == 0:
            pre_eval_end = d - pd.Timedelta(hours=1)
            current_train_df = df_model.loc[:pre_eval_end]
            for h in range(24):
                df_h = current_train_df[current_train_df['hour'] == h]
                X_h = df_h[feature_cols]
                y_h = df_h[target_col]
                hourly_models[h].fit(X_h, y_h)
                
        # Predict 24 hours for target day using hour-specific model
        day_preds = []
        for h in range(24):
            row_h = day_df[day_df['hour'] == h]
            if len(row_h) > 0:
                pred_val = hourly_models[h].predict(row_h[feature_cols])[0]
                day_preds.append(pred_val)
            else:
                day_preds.append(np.nan)
                
        act_vals = day_df[target_col].values
        
        preds_all.extend(day_preds)
        acts_all.extend(act_vals)
        
        if (idx + 1) % 90 == 0 or (idx + 1) == len(eval_dates):
            logger.info(f"  ⏳ Progress: {idx + 1}/{len(eval_dates)} test days ({(idx + 1)/len(eval_dates)*100:.1f}%) completed...")
            
    runtime_sec = time.time() - start_time
    y_a = np.array(acts_all)
    y_p = np.array(preds_all)
    
    mae = float(np.mean(np.abs(y_a - y_p)))
    rmse = float(np.sqrt(np.mean((y_a - y_p)**2)))
    wape = float((np.sum(np.abs(y_a - y_p)) / np.sum(y_a)) * 100)
    
    low_price_mask = y_a <= 15.0
    low_price_mae = float(np.mean(np.abs(y_a[low_price_mask] - y_p[low_price_mask]))) if np.sum(low_price_mask) > 0 else mae
    
    logger.info("=" * 80)
    logger.info(f"🏆 EXPERIMENT COMPLETE: {experiment_name}")
    logger.info(f"  📊 WAPE: %{wape:.2f} | MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | Low-Price MAE: ${low_price_mae:.2f}")
    logger.info(f"  ⏱️ Runtime: {runtime_sec:.1f} seconds")
    logger.info("=" * 80)
    
    # Save to DB & Master CSV
    log_experiment_result(
        experiment_name=experiment_name,
        model_type="lightgbm_direct_24h",
        feature_set="robust",
        hyperparameters=lgb_params,
        backtest_days=test_days,
        test_start=eval_dates.min().strftime('%Y-%m-%d'),
        test_end=eval_dates.max().strftime('%Y-%m-%d'),
        mae_usd=mae,
        wape_pct=wape,
        rmse_usd=rmse,
        low_price_mae_usd=low_price_mae,
        runtime_seconds=runtime_sec,
        notes="24 Direct Hourly LightGBM specialist models with 30-day walk-forward updates."
    )
    
    return wape, mae, rmse

if __name__ == "__main__":
    run_24h_direct_experiment()
