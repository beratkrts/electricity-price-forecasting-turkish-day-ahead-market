"""
2-Year Experimental Model Prediction Backfill Script

Runs the new Enhanced LightGBM Model (v2 with Lag0 Renewable Ratios, Net Load, and Ramp Rates)
over the last 2 YEARS (730 days / 17,520 hours) day-by-day and writes all predictions into 
the dedicated experimental table `gold.ptf_predictions_experimental`.
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
from fetch_epias_data import resolve_usd_try_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ExpBackfill")

MODEL_NAME = "lgb_lag0_v2"
DAYS_TO_BACKFILL = 730  # 2 Years

def init_exp_table(engine):
    """Creates gold.ptf_predictions_experimental table if it does not exist."""
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.ptf_predictions_experimental (
                target_ts TIMESTAMPTZ NOT NULL,
                model_name VARCHAR(50) NOT NULL,
                predicted_mcp_usd NUMERIC(10, 4),
                predicted_mcp_try NUMERIC(10, 4),
                predicted_mcp_usd_p10 NUMERIC(10, 4),
                predicted_mcp_try_p10 NUMERIC(10, 4),
                predicted_mcp_usd_p90 NUMERIC(10, 4),
                predicted_mcp_try_p90 NUMERIC(10, 4),
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (target_ts, model_name)
            );
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gold_predictions_exp_ts ON gold.ptf_predictions_experimental(target_ts);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gold_predictions_exp_model ON gold.ptf_predictions_experimental(model_name);"))
        conn.commit()
    logger.info("✅ Verified/Created table gold.ptf_predictions_experimental")

def run_experimental_backfill():
    engine = get_db_engine()
    init_exp_table(engine)
    
    logger.info("📥 Loading all historical raw data from PostgreSQL...")
    df_raw = load_all_historical_data()
    
    logger.info("⚙️ Building robust features including Net Load & Lag0 Renewable Ratios...")
    df_feat = build_robust_features(df_raw)
    
    feature_cols = get_feature_columns('robust', df_feat)
    target_col = 'mcp_price_usd'
    
    df_model = df_feat.dropna(subset=feature_cols + [target_col]).copy()
    
    max_ts = df_model.index.max()
    start_ts = max_ts - pd.Timedelta(days=DAYS_TO_BACKFILL)
    
    logger.info(f"🚀 Starting 2-Year ({DAYS_TO_BACKFILL} Days) Experimental Backfill for model '{MODEL_NAME}'")
    logger.info(f"📅 Backfill Range: {start_ts.strftime('%Y-%m-%d')} -> {max_ts.strftime('%Y-%m-%d')}")
    
    # Generate unique evaluation dates
    unique_dates = df_model.loc[start_ts:max_ts].index.normalize().unique()
    
    records = []
    
    total_dates = len(unique_dates)
    logger.info(f"⏳ Total prediction days to process: {total_dates}")
    
    # Fallback only — the per-day rate below is resolved inside the walk-forward loop.
    # This used to be resolved ONCE here and applied to every historical day, which
    # converted 2024 predictions with a 2026 rate (up to +43% inflation in TRY).
    fallback_usd_try = resolve_usd_try_rate(df=df_raw, engine=engine)

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
        
        preds_p50 = m_p50.predict(X_future)
        preds_p10 = m_p10.predict(X_future)
        preds_p90 = m_p90.predict(X_future)
        
        preds_p10 = np.minimum(preds_p10, preds_p50)
        preds_p90 = np.maximum(preds_p90, preds_p50)

        # FX rate as it was knowable at prediction time: the last rate in the training
        # window, i.e. the day before the target. Same definition the live pipeline uses.
        day_fx = fallback_usd_try
        if 'usd_try' in train_data.columns:
            known_fx = train_data['usd_try'].dropna()
            if len(known_fx) > 0:
                day_fx = float(known_fx.iloc[-1])

        for ts_val, p50, p10, p90 in zip(future_data.index, preds_p50, preds_p10, preds_p90):
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
            
        if (idx + 1) % 30 == 0 or (idx + 1) == total_dates:
            logger.info(f"⏳ Processed {idx + 1}/{total_dates} days ({(idx + 1)/total_dates*100:.1f}%) — {len(records)} prediction rows ready.")

    logger.info(f"💾 Saving {len(records)} experimental records to gold.ptf_predictions_experimental...")
    
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
    
    # Write in batch chunks of 1000
    chunk_size = 1000
    with engine.connect() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            conn.execute(insert_sql, chunk)
        conn.commit()
        
    logger.info(f"🎉 Successfully completed 2-Year Experimental Backfill! {len(records)} rows written to gold.ptf_predictions_experimental.")

if __name__ == "__main__":
    run_experimental_backfill()
