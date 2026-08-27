"""
Sistematik Deney Runner & Merkezi Raporlama Altyapısı

Location: experiments/scripts/run_experiment_runner.py
Tüm model deneylerinin metriklerini veritabanına (`gold.experiment_results`)
ve merkezi CSV dosyasına (`experiments/experiment_master_log.csv`) standart formatta kaydeder.
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from db.connection import get_db_engine
from sqlalchemy import text

# Logging setup
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "experiment_runner.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Experiment_Runner")


def init_db_results_table():
    """Veritabanında gold.experiment_results tablosunu yoksa oluşturur."""
    engine = get_db_engine()
    sql_create = text("""
        CREATE TABLE IF NOT EXISTS gold.experiment_results (
            experiment_id SERIAL PRIMARY KEY,
            experiment_name VARCHAR(100) NOT NULL UNIQUE,
            model_type VARCHAR(50) NOT NULL,
            feature_set VARCHAR(50) NOT NULL,
            hyperparameters JSONB,
            backtest_type VARCHAR(30) DEFAULT 'walk_forward',
            backtest_days INTEGER NOT NULL,
            train_start DATE,
            test_start DATE,
            test_end DATE,
            mae_usd NUMERIC(10, 4),
            wape_pct NUMERIC(6, 2),
            rmse_usd NUMERIC(10, 4),
            low_price_mae_usd NUMERIC(10, 4),
            runtime_seconds NUMERIC(10, 2),
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql_create)
        logger.info("✅ DB Table 'gold.experiment_results' verified/created successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Could not create DB table (will continue with CSV logging): {e}")


def log_experiment_result(
    experiment_name: str,
    model_type: str,
    feature_set: str,
    hyperparameters: dict,
    backtest_days: int,
    test_start: str,
    test_end: str,
    mae_usd: float,
    wape_pct: float,
    rmse_usd: float,
    low_price_mae_usd: float,
    runtime_seconds: float,
    notes: str = ""
):
    """
    Deney sonucunu hem DB'ye (`gold.experiment_results`) hem de CSV'ye yazar.
    """
    init_db_results_table()
    
    result_record = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "feature_set": feature_set,
        "hyperparameters": json.dumps(hyperparameters, ensure_ascii=False),
        "backtest_type": "walk_forward",
        "backtest_days": backtest_days,
        "test_start": str(test_start),
        "test_end": str(test_end),
        "mae_usd": round(float(mae_usd), 4),
        "wape_pct": round(float(wape_pct), 2),
        "rmse_usd": round(float(rmse_usd), 4),
        "low_price_mae_usd": round(float(low_price_mae_usd), 4),
        "runtime_seconds": round(float(runtime_seconds), 2),
        "notes": notes,
        "created_at": datetime.now().isoformat()
    }
    
    # 1. DB Logging
    try:
        engine = get_db_engine()
        sql_insert = text("""
            INSERT INTO gold.experiment_results (
                experiment_name, model_type, feature_set, hyperparameters,
                backtest_type, backtest_days, test_start, test_end,
                mae_usd, wape_pct, rmse_usd, low_price_mae_usd, runtime_seconds, notes
            ) VALUES (
                :experiment_name, :model_type, :feature_set, :hyperparameters,
                :backtest_type, :backtest_days, :test_start, :test_end,
                :mae_usd, :wape_pct, :rmse_usd, :low_price_mae_usd, :runtime_seconds, :notes
            )
            ON CONFLICT (experiment_name) DO UPDATE SET
                mae_usd = EXCLUDED.mae_usd,
                wape_pct = EXCLUDED.wape_pct,
                rmse_usd = EXCLUDED.rmse_usd,
                low_price_mae_usd = EXCLUDED.low_price_mae_usd,
                runtime_seconds = EXCLUDED.runtime_seconds,
                notes = EXCLUDED.notes,
                created_at = CURRENT_TIMESTAMP;
        """)
        with engine.begin() as conn:
            conn.execute(sql_insert, {
                "experiment_name": experiment_name,
                "model_type": model_type,
                "feature_set": feature_set,
                "hyperparameters": json.dumps(hyperparameters),
                "backtest_type": "walk_forward",
                "backtest_days": backtest_days,
                "test_start": str(test_start),
                "test_end": str(test_end),
                "mae_usd": round(float(mae_usd), 4),
                "wape_pct": round(float(wape_pct), 2),
                "rmse_usd": round(float(rmse_usd), 4),
                "low_price_mae_usd": round(float(low_price_mae_usd), 4),
                "runtime_seconds": round(float(runtime_seconds), 2),
                "notes": notes
            })
        logger.info(f"💾 Saved result to DB table gold.experiment_results: {experiment_name}")
    except Exception as e:
        logger.warning(f"⚠️ Could not insert into DB: {e}")
        
    # 2. Master CSV Logging
    master_csv = project_root / "experiments" / "experiment_master_log.csv"
    df_new = pd.DataFrame([result_record])
    
    if master_csv.exists():
        df_old = pd.read_csv(master_csv)
        # Update if exists, else append
        df_old = df_old[df_old["experiment_name"] != experiment_name]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
        
    df_all.to_csv(master_csv, index=False, encoding="utf-8")
    logger.info(f"📄 Saved result to master CSV: {master_csv}")
    
    return result_record
