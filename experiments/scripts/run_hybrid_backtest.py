"""
Hibrit (Rejim Duyarlı) Model 365-Günlük Backtest & Performans Analiz Scripti.

EPNet ROBUST_02 ve LightGBM (log-transform) modellerinin 365 günlük backtest sonuçlarını
Rejim Yönlendirici (RegimeDetector) ile birleştirir ve son 1, 3, 6, 9, 12 aylık 
MAE, WAPE ve MAPE metriklerini hesaplar.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from db.connection import get_db_engine
from sqlalchemy import text
from src.features.feature_engineering import build_robust_features, get_feature_columns
from src.routing.regime_detector import RegimeDetector


def load_master_dataset():
    """PostgreSQL veritabanından ham veriyi yükler."""
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

    df_raw['usd_try'] = df_raw['usd_try'].ffill().bfill()
    df_raw['brent_oil_usd'] = df_raw['brent_oil_usd'].ffill().bfill()
    df_raw['natural_gas_grf_try'] = df_raw['natural_gas_grf_try'].ffill().bfill()
    if 'hydro_water_energy_mwh' in df_raw.columns:
        df_raw['hydro_water_energy_mwh'] = df_raw['hydro_water_energy_mwh'].ffill().bfill()

    return df_raw


def run_hybrid_backtest():
    print("📦 Master Veri Seti Yükleniyor...")
    df_raw = load_master_dataset()
    
    print("⚙️ Robust Feature Set Yapılandırılıyor...")
    df_feat = build_robust_features(df_raw)
    df_model = df_feat.dropna().copy()
    
    feature_cols = get_feature_columns('robust', df_model)
    target_col = 'mcp_price_usd'
    
    # 1. EPNet ROBUST_02 Backtest Loglarını Yükle
    epnet_csv_path = project_root / "logs" / "epnet_fast_robust_experiments" / "ROBUST_02_Cyclic_Seasonality_Huber_daily.csv"
    if not epnet_csv_path.exists():
        raise FileNotFoundError(f"EPNet ROBUST_02 günlük backtest verisi bulunamadı: {epnet_csv_path}")
        
    df_epnet_daily = pd.read_csv(epnet_csv_path)
    df_epnet_daily['tarih'] = pd.to_datetime(df_epnet_daily['tarih'])
    
    detector = RegimeDetector()
    
    lgbm_params = {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 8,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': 42
    }
    
    print("\n🏃 365-Günlük LightGBM Walk-Forward & Rejim Yönlendirmeli Backtest Çalıştırılıyor...\n")
    
    hybrid_results = []
    
    # EPNet backtest tarihlerini sırayla dön
    epnet_dates = df_epnet_daily.sort_values('tarih')['tarih'].tolist()
    
    for idx, test_date in enumerate(epnet_dates):
        test_start = pd.Timestamp(test_date).tz_localize('Europe/Istanbul')
        test_end = test_start + pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)
        
        tr_df = df_model.loc[:train_end]
        te_df = df_model.loc[test_start:test_end]
        
        if len(tr_df) < 1000 or len(te_df) < 12:
            continue
            
        # LightGBM Klasik Walk-Forward Eğitimi
        X_tr = tr_df[feature_cols]
        y_tr_log = np.log1p(np.maximum(tr_df[target_col].values, 0.0))
        
        lgbm = lgb.LGBMRegressor(**lgbm_params)
        lgbm.fit(X_tr, y_tr_log)
        
        X_te = te_df[feature_cols]
        lgbm_preds_log = lgbm.predict(X_te)
        lgbm_preds = np.maximum(np.expm1(lgbm_preds_log), 0.0)
        
        # Rejim Tespiti
        signals = detector.compute_signals(tr_df)
        regime = detector.detect_regime(signals)
        
        y_true = te_df[target_col].values
        
        # Metrikler
        lgbm_mae = np.mean(np.abs(y_true - lgbm_preds))
        lgbm_wape = (np.sum(np.abs(y_true - lgbm_preds)) / np.sum(np.abs(y_true))) * 100
        safe_denom = np.maximum(y_true, 1.0)
        lgbm_mape = np.mean(np.abs((y_true - lgbm_preds) / safe_denom)) * 100
        
        epnet_row = df_epnet_daily[df_epnet_daily['tarih'] == pd.to_datetime(test_date).tz_localize(None)].iloc[0]
        epnet_mae = epnet_row['mae']
        epnet_wape = epnet_row['wape']
        epnet_mape = epnet_row['mape']
        
        # Rejime göre Hibrit Tahmin Metriğini Seç
        if regime == 'CRASH':
            hybrid_mae = lgbm_mae
            hybrid_wape = lgbm_wape
            hybrid_mape = lgbm_mape
            selected_model = 'LightGBM'
        elif regime == 'VOLATILE':
            # %40 EPNet + %60 LightGBM Blending
            hybrid_mae = 0.4 * epnet_mae + 0.6 * lgbm_mae
            hybrid_wape = 0.4 * epnet_wape + 0.6 * lgbm_wape
            hybrid_mape = 0.4 * epnet_mape + 0.6 * lgbm_mape
            selected_model = 'Blend(0.4EP+0.6LGBM)'
        else: # NORMAL
            hybrid_mae = epnet_mae
            hybrid_wape = epnet_wape
            hybrid_mape = epnet_mape
            selected_model = 'EPNet'
            
        hybrid_results.append({
            'tarih': test_date.strftime('%Y-%m-%d'),
            'regime': regime,
            'selected_model': selected_model,
            'epnet_mae': epnet_mae,
            'epnet_wape': epnet_wape,
            'epnet_mape': epnet_mape,
            'lgbm_mae': lgbm_mae,
            'lgbm_wape': lgbm_wape,
            'lgbm_mape': lgbm_mape,
            'hybrid_mae': hybrid_mae,
            'hybrid_wape': hybrid_wape,
            'hybrid_mape': hybrid_mape
        })
        
        if (idx + 1) % 35 == 0 or (idx + 1) == len(epnet_dates):
            print(f"⏳ İşlenen: [{idx+1:03d}/{len(epnet_dates):03d}] | Tarih: {test_date.strftime('%Y-%m-%d')} | Rejim: {regime:8s} | Seçilen: {selected_model:20s}")

    res_df = pd.DataFrame(hybrid_results)
    res_df['tarih'] = pd.to_datetime(res_df['tarih'])
    res_df = res_df.sort_values('tarih').reset_index(drop=True)
    
    # -------------------------------------------------------------
    # SON 1, 3, 6, 9, 12 AYLIK PERFORMANS METRİKLERİ RAPORLAMASI
    # -------------------------------------------------------------
    max_dt = res_df['tarih'].max()
    
    periods = {
        'Son 1 Ay (30 Gün)': 30,
        'Son 3 Ay (90 Gün)': 90,
        'Son 6 Ay (180 Gün)': 180,
        'Son 9 Ay (270 Gün)': 270,
        'Son 12 Ay (365 Gün)': 365
    }
    
    summary_rows = []
    
    for label, days in periods.items():
        cutoff = max_dt - pd.Timedelta(days=days)
        sub = res_df[res_df['tarih'] > cutoff]
        
        summary_rows.append({
            'Dönem': label,
            'Gün Sayısı': len(sub),
            'EPNet MAE': f"${sub['epnet_mae'].mean():.2f}",
            'EPNet WAPE': f"%{sub['epnet_wape'].mean():.2f}",
            'EPNet MAPE': f"%{sub['epnet_mape'].mean():.2f}",
            'LGBM MAE': f"${sub['lgbm_mae'].mean():.2f}",
            'LGBM WAPE': f"%{sub['lgbm_wape'].mean():.2f}",
            'LGBM MAPE': f"%{sub['lgbm_mape'].mean():.2f}",
            'HİBRİT MAE': f"${sub['hybrid_mae'].mean():.2f}",
            'HİBRİT WAPE': f"%{sub['hybrid_wape'].mean():.2f}",
            'HİBRİT MAPE': f"%{sub['hybrid_mape'].mean():.2f}"
        })
        
    summary_df = pd.DataFrame(summary_rows)
    
    print("\n" + "=" * 120)
    print("🏆 EPNet ROBUST_02 vs LightGBM vs HİBRİT (REJİM YÖNLENDİRMELİ) BACKTEST SONUÇLARI")
    print("=" * 120)
    print(summary_df.to_string(index=False))
    print("=" * 120)
    
    # Save CSV and MD report
    out_dir = project_root / "logs" / "hybrid_routing_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "hybrid_backtest_daily.csv", index=False)
    
    md_content = f"""# 📊 Hibrit (Rejim Duyarlı) Model 365-Günlük Backtest Raporu

## Dönemsel Performans Özeti (Son 1, 3, 6, 9, 12 Aylık Dönemler)

```
{summary_df.to_string(index=False)}
```

## Rejim Dağılım Özeti

- **NORMAL Rejimi (EPNet):** {len(res_df[res_df['regime']=='NORMAL'])} Gün
- **CRASH Rejimi (LightGBM):** {len(res_df[res_df['regime']=='CRASH'])} Gün
- **VOLATILE Rejimi (Blend):** {len(res_df[res_df['regime']=='VOLATILE'])} Gün
"""
    with open(out_dir / "hybrid_backtest_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\n✅ Sonuçlar diske kaydedildi: {out_dir / 'hybrid_backtest_summary.md'}")


if __name__ == "__main__":
    run_hybrid_backtest()
