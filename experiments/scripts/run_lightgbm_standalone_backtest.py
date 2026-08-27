"""
LightGBM Log-Transform Standalone 365-Day Walk-Forward Backtest Engine.

EPNet deney formatına (logs/lightgbm_experiments) tam uyumlu şekilde:
1. Gün gün MAE, WAPE, MAPE metriklerini içeren CSV oluşturur:
   `logs/lightgbm_experiments/LGBM_logtransform_365d_daily.csv`
2. Kapsamlı özet CSV ve MD rapor dosyalarını üretir.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from db.connection import get_db_engine
from sqlalchemy import text
from src.features.feature_engineering import build_robust_features, get_feature_columns


def load_master_dataset():
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

    df_raw['usd_try'] = df_raw['usd_try'].ffill().bfill()
    df_raw['brent_oil_usd'] = df_raw['brent_oil_usd'].ffill().bfill()
    df_raw['natural_gas_grf_try'] = df_raw['natural_gas_grf_try'].ffill().bfill()
    if 'hydro_water_energy_mwh' in df_raw.columns:
        df_raw['hydro_water_energy_mwh'] = df_raw['hydro_water_energy_mwh'].ffill().bfill()

    return df_raw


def run_lgbm_standalone_backtest(max_days=365):
    print("📦 Master Veri Seti Yükleniyor...")
    df_raw = load_master_dataset()

    print("⚙️ Robust Feature Set Mühendisliği Yapılıyor...")
    df_feat = build_robust_features(df_raw)
    df_model = df_feat.dropna().copy()

    feature_cols = get_feature_columns('robust', df_model)
    target_col = 'mcp_price_usd'

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

    log_dir = project_root / "logs" / "lightgbm_experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    daily_csv_file = log_dir / "LGBM_logtransform_365d_daily.csv"

    print(f"\n🏃 365-Günlük LightGBM Standalone Walk-Forward Backtest Başlatılıyor...\n")

    daily_results = []
    
    # EPNet ile birebir eşleşecek şekilde sondan geriye max_days kadar gün için çalış
    max_ts = df_model.index.max()

    for day_idx in range(max_days - 1, -1, -1):
        test_end = max_ts - pd.Timedelta(days=day_idx)
        test_start = test_end - pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)

        tr_df = df_model.loc[:train_end]
        te_df = df_model.loc[test_start:test_end]

        if len(tr_df) < 1000 or len(te_df) < 12:
            continue

        X_tr = tr_df[feature_cols]
        y_tr_log = np.log1p(np.maximum(tr_df[target_col].values, 0.0))

        lgbm = lgb.LGBMRegressor(**lgbm_params)
        lgbm.fit(X_tr, y_tr_log)

        X_te = te_df[feature_cols]
        lgbm_preds_log = lgbm.predict(X_te)
        preds = np.maximum(np.expm1(lgbm_preds_log), 0.0)

        y_true = te_df[target_col].values
        mae = np.mean(np.abs(y_true - preds))
        wape = (np.sum(np.abs(y_true - preds)) / np.sum(np.abs(y_true))) * 100
        safe_denom = np.maximum(y_true, 1.0)
        mape = np.mean(np.abs((y_true - preds) / safe_denom)) * 100

        daily_results.append({
            'day_offset': day_idx + 1,
            'tarih': test_start.strftime('%Y-%m-%d'),
            'mae': mae,
            'wape': wape,
            'mape': mape
        })

        if (len(daily_results)) % 30 == 0 or len(daily_results) == max_days:
            print(f"⏳ İşlenen [{len(daily_results):03d}/{max_days:03d}] | "
                  f"Tarih: {test_start.strftime('%Y-%m-%d')} | "
                  f"MAE: ${mae:5.2f}/MWh | WAPE: %{wape:5.2f} | MAPE: %{mape:5.2f}")

    res_df = pd.DataFrame(daily_results)
    res_df.to_csv(daily_csv_file, index=False)
    print(f"\n💾 Günlük backtest sonuçları kaydedildi: {daily_csv_file}")

    # Dönemsel Özet
    res_df['tarih_dt'] = pd.to_datetime(res_df['tarih'])
    max_dt = res_df['tarih_dt'].max()

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
        sub = res_df[res_df['tarih_dt'] > cutoff]

        summary_rows.append({
            'Dönem': label,
            'Gün Sayısı': len(sub),
            'MAE ($/MWh)': f"${sub['mae'].mean():.2f}",
            'WAPE (%)': f"%{sub['wape'].mean():.2f}",
            'MAPE (%)': f"%{sub['mape'].mean():.2f}"
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 70)
    print("🏆 LIGHTGBM (LOG-TRANSFORM) 365-GÜNLÜK BACKTEST NİHAİ ÖZETİ")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    # Özet MD ve CSV kaydet
    summary_df.to_csv(log_dir / "lightgbm_summary.csv", index=False)

    md_content = f"""# 🌲 LightGBM (Log-Transform) 365-Günlük Backtest Raporu

## Dönemsel İstatistikler (Son 1, 3, 6, 9, 12 Aylık)

```
{summary_df.to_string(index=False)}
```

- **Günlük detay dosyası:** `logs/lightgbm_experiments/LGBM_logtransform_365d_daily.csv`
- **Tarih Aralığı:** {res_df['tarih'].min()} -> {res_df['tarih'].max()}
"""
    with open(log_dir / "lightgbm_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ Rapor ve özetler kaydedildi: {log_dir / 'lightgbm_summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="Backtest edilecek gün sayısı")
    args = parser.parse_args()
    run_lgbm_standalone_backtest(max_days=args.days)
