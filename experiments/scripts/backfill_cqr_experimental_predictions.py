"""
CQR-Calibrated 2-Year Experimental Model Prediction Backfill Script

Runs the Enhanced LightGBM Model (v2) with Conformalized Quantile Regression (CQR)
calibration over 2 YEARS (730 days / 17,520 hours) and writes all calibrated predictions
into `gold.ptf_predictions_experimental` under model_name = 'lgb_cqr_v2'.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sqlalchemy import text
from db.connection import get_db_engine
from scripts.predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster
from src.models.cqr_calibrator import CQRCalibrator
from fetch_epias_data import resolve_usd_try_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CQRBackfill")

MODEL_NAME = "lgb_cqr_v2"
DAYS_TO_BACKFILL = 730  # 2 Years

def run_cqr_backfill():
    engine = get_db_engine()
    
    logger.info("📥 Loading all historical raw data from PostgreSQL...")
    df_raw = load_all_historical_data()
    
    logger.info("⚙️ Building robust features including Net Load & Lag0 Renewable Ratios...")
    df_feat = build_robust_features(df_raw)
    
    feature_cols = get_feature_columns('robust', df_feat)
    target_col = 'mcp_price_usd'
    
    df_model = df_feat.dropna(subset=feature_cols + [target_col]).copy()
    
    max_ts = df_model.index.max()
    start_ts = max_ts - pd.Timedelta(days=DAYS_TO_BACKFILL)
    
    logger.info(f"🚀 Starting CQR-Calibrated Backfill for model '{MODEL_NAME}'")
    logger.info(f"📅 Backfill Range: {start_ts.strftime('%Y-%m-%d')} -> {max_ts.strftime('%Y-%m-%d')}")
    
    unique_dates = df_model.loc[start_ts:max_ts].index.normalize().unique()
    total_dates = len(unique_dates)
    
    calibrator = CQRCalibrator(target_coverage=0.80, cal_window_days=30)
    # Fallback only — the per-day rate below is resolved inside the walk-forward loop.
    # This used to be resolved ONCE here and applied to every historical day, which
    # converted 2024 predictions with a 2026 rate (up to +43% inflation in TRY).
    fallback_usd_try = resolve_usd_try_rate(df=df_raw, engine=engine)
    
    records = []
    
    # Track historical predictions & actuals for calibration
    hist_p10 = []
    hist_p90 = []
    hist_y = []
    
    for idx, d in enumerate(unique_dates):
        train_end = d - pd.Timedelta(days=1) + pd.Timedelta(hours=23)
        train_data = df_model.loc[:train_end]
        
        target_start = d
        target_end = d + pd.Timedelta(hours=23)
        future_data = df_model.loc[target_start:target_end]
        
        if len(future_data) == 0 or len(train_data) < 24 * 60:
            continue
            
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_future = future_data[feature_cols]
        
        # Fit 3 Models (P50, P10, P90)
        m_p50 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'min_child_samples': 10, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(X_train, y_train)
        m_p10 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.10, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'min_child_samples': 10, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(X_train, y_train)
        m_p90 = LightGBMForecaster(params={'objective': 'quantile', 'alpha': 0.90, 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63, 'min_child_samples': 10, 'verbose': -1, 'random_state': 42}, use_log_transform=False).fit(X_train, y_train)
        
        raw_p50 = m_p50.predict(X_future)
        raw_p10 = m_p10.predict(X_future)
        raw_p90 = m_p90.predict(X_future)
        
        raw_p10 = np.minimum(raw_p10, raw_p50)
        raw_p90 = np.maximum(raw_p90, raw_p50)
        
        # Apply CQR Calibration
        p10_cqr, p90_cqr = calibrator.calibrate(
            p10_hist=hist_p10,
            p90_hist=hist_p90,
            y_hist=hist_y,
            p10_future=raw_p10,
            p90_future=raw_p90
        )
        
        act_y = future_data[target_col].values
        hist_p10.extend(raw_p10)
        hist_p90.extend(raw_p90)
        hist_y.extend(act_y)
        
        # FX rate as it was knowable at prediction time: the last rate in the training
        # window, i.e. the day before the target. Same definition the live pipeline uses.
        day_fx = fallback_usd_try
        if 'usd_try' in train_data.columns:
            known_fx = train_data['usd_try'].dropna()
            if len(known_fx) > 0:
                day_fx = float(known_fx.iloc[-1])

        for ts_val, p50, p10, p90 in zip(future_data.index, raw_p50, p10_cqr, p90_cqr):
            records.append({
                'target_ts': ts_val,
                'model_name': MODEL_NAME,
                'predicted_mcp_usd': round(float(p50), 4),
                'predicted_mcp_try': round(float(p50 * day_fx), 4),
                'predicted_mcp_usd_p10': round(float(p10), 4),
                'predicted_mcp_try_p10': round(float(p10 * day_fx), 4),
                'predicted_mcp_usd_p90': round(float(p90), 4),
                'predicted_mcp_try_p90': round(float(p90 * day_fx), 4)
            })
            
        if (idx + 1) % 50 == 0 or (idx + 1) == total_dates:
            logger.info(f"⏳ Processed {idx + 1}/{total_dates} days ({(idx + 1)/total_dates*100:.1f}%) — {len(records)} CQR prediction rows ready.")

    logger.info(f"💾 Saving {len(records)} CQR records to gold.ptf_predictions_experimental under model_name='{MODEL_NAME}'...")
    
    insert_sql = text("""
        INSERT INTO gold.ptf_predictions_experimental (
            target_ts, model_name,
            predicted_mcp_usd, predicted_mcp_try,
            predicted_mcp_usd_p10, predicted_mcp_try_p10,
            predicted_mcp_usd_p90, predicted_mcp_try_p90
        )
        VALUES (
            :target_ts, :model_name,
            :predicted_mcp_usd, :predicted_mcp_try,
            :predicted_mcp_usd_p10, :predicted_mcp_try_p10,
            :predicted_mcp_usd_p90, :predicted_mcp_try_p90
        )
        ON CONFLICT (target_ts, model_name) DO UPDATE SET
            predicted_mcp_usd = EXCLUDED.predicted_mcp_usd,
            predicted_mcp_try = EXCLUDED.predicted_mcp_try,
            predicted_mcp_usd_p10 = EXCLUDED.predicted_mcp_usd_p10,
            predicted_mcp_try_p10 = EXCLUDED.predicted_mcp_try_p10,
            predicted_mcp_usd_p90 = EXCLUDED.predicted_mcp_usd_p90,
            predicted_mcp_try_p90 = EXCLUDED.predicted_mcp_try_p90,
            created_at = CURRENT_TIMESTAMP;
    """)
    
    chunk_size = 1000
    with engine.connect() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            conn.execute(insert_sql, chunk)
        conn.commit()
        
    logger.info(f"🎉 Successfully completed CQR Backfill! {len(records)} rows written to gold.ptf_predictions_experimental under '{MODEL_NAME}'.")

if __name__ == "__main__":
    run_cqr_backfill()
