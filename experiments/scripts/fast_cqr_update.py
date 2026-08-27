"""
Ultra-Fast CQR Calibration Script (Correct Romano et al. NIPS 2019 Formula)

Applies Conformalized Quantile Regression (CQR) calibration to gold.ptf_predictions_experimental
under model_name='lgb_cqr_v2'.
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sqlalchemy import text
from db.connection import get_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FastCQR")

MODEL_SOURCE = "lgb_lag0_v2"
MODEL_TARGET = "lgb_cqr_v2"
TARGET_COVERAGE = 0.80
CAL_WINDOW_HOURS = 30 * 24  # 30 Days = 720 Hours

def run_fast_cqr():
    engine = get_db_engine()
    
    logger.info(f"📥 Reading existing predictions for '{MODEL_SOURCE}' from PostgreSQL...")
    with engine.connect() as conn:
        df_pred = pd.read_sql(text("""
            SELECT 
                e.target_ts,
                e.predicted_mcp_usd,
                e.predicted_mcp_try,
                e.predicted_mcp_usd_p10,
                e.predicted_mcp_try_p10,
                e.predicted_mcp_usd_p90,
                e.predicted_mcp_try_p90,
                m.price_usd as y_true,
                -- FX rate as knowable at prediction time: the day before the target.
                -- Do NOT derive it back out of predicted_mcp_try / predicted_mcp_usd —
                -- that propagates whatever rate the source backfill happened to use.
                mc.usd_try as live_usd_try
            FROM gold.ptf_predictions_experimental e
            JOIN raw_mcp_hourly m ON e.target_ts = m.ts
            LEFT JOIN raw_macro_daily mc
                   ON mc.entry_date = ((e.target_ts AT TIME ZONE 'Europe/Istanbul')::date - 1)
            WHERE e.model_name = :model_name
            ORDER BY e.target_ts ASC;
        """), conn, params={"model_name": MODEL_SOURCE})
        
    logger.info(f"✅ Loaded {len(df_pred)} prediction rows from database.")
    
    y_all = df_pred['y_true'].values
    p10_all = df_pred['predicted_mcp_usd_p10'].astype(float).values
    p90_all = df_pred['predicted_mcp_usd_p90'].astype(float).values
    p50_all = df_pred['predicted_mcp_usd'].astype(float).values
    # A missing macro row for the day before the target leaves the rate NULL; carry the
    # nearest known rate forward/backward rather than writing NaN into the TRY columns.
    fx_series = pd.to_numeric(df_pred['live_usd_try'], errors='coerce').ffill().bfill()
    if fx_series.isna().all():
        raise RuntimeError("No USD/TRY rate available for any target day — aborting CQR update.")
    fx_all = fx_series.values
    ts_all = df_pred['target_ts'].values
    
    cqr_records = []
    
    for i in range(len(df_pred)):
        cal_start = max(0, i - CAL_WINDOW_HOURS)
        
        if i < 24 * 7:
            q_hat = 0.0
        else:
            y_cal = y_all[cal_start:i]
            p10_cal = p10_all[cal_start:i]
            p90_cal = p90_all[cal_start:i]
            
            # CQR non-conformity score (Romano et al. 2019)
            scores = np.maximum(p10_cal - y_cal, y_cal - p90_cal)
            
            # Non-conformity score quantile
            n = len(scores)
            q_raw = np.quantile(scores, TARGET_COVERAGE * (1.0 + 1.0 / n))
            
            # If quantile is negative (meaning raw bounds covered >80% in cal set), q_hat is 0.0;
            # If quantile is positive (under-coverage), q_hat widens the bounds.
            q_hat = max(0.0, float(q_raw))
            
        p10_cqr_usd = p10_all[i] - q_hat
        p90_cqr_usd = p90_all[i] + q_hat
        
        # Enforce monotonicity and non-negativity if needed
        p10_cqr_usd = min(p10_cqr_usd, p50_all[i])
        p90_cqr_usd = max(p90_cqr_usd, p50_all[i])
        
        fx = fx_all[i]
        
        cqr_records.append({
            'target_ts': pd.Timestamp(ts_all[i]).to_pydatetime(),
            'model_name': MODEL_TARGET,
            'predicted_mcp_usd': round(float(p50_all[i]), 4),
            'predicted_mcp_try': round(float(p50_all[i] * fx), 4),
            'predicted_mcp_usd_p10': round(float(p10_cqr_usd), 4),
            'predicted_mcp_try_p10': round(float(p10_cqr_usd * fx), 4),
            'predicted_mcp_usd_p90': round(float(p90_cqr_usd), 4),
            'predicted_mcp_try_p90': round(float(p90_cqr_usd * fx), 4)
        })
        
    logger.info(f"💾 Saving {len(cqr_records)} CQR-calibrated records to gold.ptf_predictions_experimental...")
    
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
        for i in range(0, len(cqr_records), chunk_size):
            chunk = cqr_records[i:i + chunk_size]
            conn.execute(insert_sql, chunk)
        conn.commit()
        
    logger.info(f"🎉 Fast CQR Update Completed! {len(cqr_records)} rows updated in database under '{MODEL_TARGET}'.")

if __name__ == "__main__":
    run_fast_cqr()
