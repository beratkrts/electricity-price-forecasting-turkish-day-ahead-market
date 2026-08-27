"""
Standalone LightGBM Model Backtest & Verification Script.
Eğitim: Silver Layer veritabanından veri çeker, LightGBM (log-transform) modelini eğitir/backtest yapar
ve tahmin kalitesini, log çıktılarını kontrol etmemizi sağlar.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from db.connection import get_db_engine
from sqlalchemy import text
from src.features.feature_engineering import build_robust_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster
from src.routing.regime_detector import RegimeDetector
from src.routing.model_router import ModelRouter

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LGBM_Verification")


def load_silver_dataset():
    """PostgreSQL veritabanından master veri setini çeker."""
    engine = get_db_engine()
    master_sql = text("""
        SELECT 
            m.ts,
            m.price_usd AS mcp_price_usd,
            m.price_try AS mcp_price_try,
            s.system_marginal_price_try AS smp_price_try,
            l.load_forecast_mw,
            k.total_mw AS kgup_total_mw,
            k.natural_gas_mw AS kgup_gas_mw,
            k.wind_mw AS kgup_wind_mw,
            k.solar_mw AS kgup_solar_mw,
            k.dammed_hydro_mw + k.river_hydro_mw AS kgup_hydro_mw,
            k.import_coal_mw + k.lignite_mw + k.black_coal_mw AS kgup_coal_mw,
            g.total_mw AS actual_gen_total_mw,
            c.consumption_mw AS actual_cons_mw,
            w.turkey_weighted_temperature_c AS temperature_c,
            mc.usd_try,
            mc.brent_oil_usd,
            ng.gas_reference_price_try AS natural_gas_grf_try,
            wp.hydro_water_energy_mwh
        FROM raw_mcp_hourly m
        LEFT JOIN raw_smp_hourly s ON m.ts = s.ts
        LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
        LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
        LEFT JOIN raw_actual_generation_hourly g ON m.ts = g.ts
        LEFT JOIN raw_actual_consumption_hourly c ON m.ts = c.ts
        LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
        LEFT JOIN raw_macro_daily mc ON DATE(m.ts) = mc.entry_date
        LEFT JOIN raw_natural_gas_daily ng ON DATE(m.ts) = ng.entry_date
        LEFT JOIN (
            SELECT DATE(date_time) AS entry_date, SUM(water_energy_provision_mwh) AS hydro_water_energy_mwh
            FROM raw_master_water_energy_provision
            GROUP BY DATE(date_time)
        ) wp ON DATE(m.ts) = wp.entry_date
        ORDER BY m.ts ASC;
    """)

    with engine.connect() as conn:
        df_raw = pd.read_sql(master_sql, conn)

    df_raw['ts'] = pd.to_datetime(df_raw['ts']).dt.tz_convert('Europe/Istanbul')
    df_raw = df_raw.set_index('ts').sort_index()

    # Forward fill macro
    df_raw['usd_try'] = df_raw['usd_try'].ffill().bfill()
    df_raw['brent_oil_usd'] = df_raw['brent_oil_usd'].ffill().bfill()
    df_raw['natural_gas_grf_try'] = df_raw['natural_gas_grf_try'].ffill().bfill()
    if 'hydro_water_energy_mwh' in df_raw.columns:
        df_raw['hydro_water_energy_mwh'] = df_raw['hydro_water_energy_mwh'].ffill().bfill()

    return df_raw


def run_test(days=30):
    logger.info("📦 Veritabanından veri yükleniyor...")
    df_raw = load_silver_dataset()
    logger.info(f"✅ Toplam ham veri boyutu: {len(df_raw)} satır ({df_raw.index.min()} -> {df_raw.index.max()})")

    logger.info("⚙️ Robust Feature mühendisliği uygulanıyor...")
    df_feat = build_robust_features(df_raw)
    df_model = df_feat.dropna().copy()
    
    feature_cols = get_feature_columns('robust', df_model)
    target_col = 'mcp_price_usd'
    logger.info(f"📊 Kullanılan feature sayısı: {len(feature_cols)}")

    detector = RegimeDetector()
    lgbm = LightGBMForecaster()

    # Son N gün için Walk-forward testi
    max_days = min(days, 60)
    logger.info(f"\n🏃 Son {max_days} gün için LightGBM ve Rejim Router Doğrulaması Başlatılıyor...\n")

    results = []
    
    for day_idx in range(max_days, 0, -1):
        test_end = df_model.index.max() - pd.Timedelta(days=day_idx-1)
        test_start = test_end - pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)
        
        tr_df = df_model.loc[:train_end]
        te_df = df_model.loc[test_start:test_end]
        
        if len(tr_df) < 1000 or len(te_df) < 12:
            continue

        # LightGBM Eğitimi
        lgbm.fit(tr_df, tr_df[target_col], feature_columns=feature_cols)
        lgbm_preds = lgbm.predict(te_df)

        # Rejim Tespiti
        signals = detector.compute_signals(tr_df)
        regime = detector.detect_regime(signals)

        y_true = te_df[target_col].values
        mae = np.mean(np.abs(y_true - lgbm_preds))
        wape = (np.sum(np.abs(y_true - lgbm_preds)) / np.sum(np.abs(y_true))) * 100

        results.append({
            'date': test_start.strftime('%Y-%m-%d'),
            'regime': regime,
            'mae': mae,
            'wape': wape,
            'mean_true_price': np.mean(y_true),
            'mean_pred_price': np.mean(lgbm_preds)
        })

        logger.info(
            f"📅 Tarih: {test_start.strftime('%Y-%m-%d')} | "
            f"Rejim: {regime:8s} | "
            f"Gerçek Fiyat: ${np.mean(y_true):5.2f} | "
            f"LGBM Tahmin: ${np.mean(lgbm_preds):5.2f} | "
            f"MAE: ${mae:5.2f} | WAPE: %{wape:5.2f}"
        )

    res_df = pd.DataFrame(results)
    
    logger.info("\n" + "=" * 80)
    logger.info("🏆 LIGHTGBM TEST SONUÇLARI ÖZETİ")
    logger.info("=" * 80)
    logger.info(f"📊 Toplam Gün: {len(res_df)}")
    logger.info(f"💵 Genel Ortalama MAE : ${res_df['mae'].mean():.2f}/MWh")
    logger.info(f"🎯 Genel Ortalama WAPE: %{res_df['wape'].mean():.2f}")
    
    logger.info("\nRejim Bazlı Performans:")
    for reg, grp in res_df.groupby('regime'):
        logger.info(f"  • {reg:8s} ({len(grp)} gün) -> MAE: ${grp['mae'].mean():.2f} | WAPE: %{grp['wape'].mean():.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="Test edilecek gün sayısı")
    args = parser.parse_args()
    run_test(days=args.days)
