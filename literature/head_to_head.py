#!/usr/bin/env python3
"""
Kafa kafaya: Polat & Selçuklu'nun özellik kümesi vs bizimki —
AYNI test seti, AYNI yeniden eğitim kadansı, AYNI hedef, AYNI naive baseline.

Neden bu kurgu: onların yayımladığı MAE 5,981 rastgele bölmeden geliyor
(bkz. replicate_polat_selcuklu.py) ve dönemler örtüşmediği için doğrudan
kıyaslanamıyor. Ortak zemin ham veride: onların CSV'si 2018-2023, bizim
DB'miz 2021-2026. Kesişim 2021-2022.

Hedef serilerin aynı olduğu doğrulandı: 17.500 ortak saatte fark 0,0000.

İKİ HANDİKAP BİZİM ALEYHİMİZE — kasıtlı, sonucu muhafazakâr yapmak için:
  1. `gold.kgup_load_pre_forecasts` 2021-2022 için BOŞ. Yani lag0 ön-tahmin
     özellikleri (renewable_pressure_ratio_lag0, net_load, risk_score) —
     ölçülmüş en etkili özelliklerimiz — bu koşuda NaN.
  2. Onların modeli 2018'den itibaren eğitiliyor (ellerindeki tüm geçmiş),
     bizimki 2021'den (bizdeki tüm geçmiş). Onlara 3 yıl fazla veri.
  `--equal-history` ile ikisi de 2021'den başlatılır.

Kullanım:
    python literature/head_to_head.py --their-csv /yol/EXIST_2018.csv
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import get_db_engine
from src.features.feature_engineering import build_robust_features, get_feature_columns

THEIR_PARAMS = dict(learning_rate=0.2, num_leaves=60, n_estimators=350,
                    random_state=35, verbose=-1, n_jobs=-1)
# Canlı pipeline'ın P50 ayarı (predict_daily_pipeline.py)
OUR_PARAMS = dict(objective="quantile", alpha=0.50, n_estimators=300,
                  learning_rate=0.03, max_depth=8, num_leaves=63,
                  min_child_samples=10, verbose=-1, random_state=42, n_jobs=-1)

# Canlı hattın SQL'i birebir (predict_daily_pipeline.load_all_historical_data),
# tek fark: tarih aralığı parametreli. Kolon adları feature builder'ın beklediği
# adlar olmalı, yoksa özellikler sessizce eksik üretiliyor.
MASTER_SQL = """
SELECT m.ts,
       m.price_usd AS mcp_price_usd, m.price_try AS mcp_price_try,
       s.system_marginal_price_try AS smp_price_try,
       l.load_forecast_mw,
       k.total_mw AS kgup_total_mw, k.natural_gas_mw AS kgup_gas_mw,
       k.wind_mw AS kgup_wind_mw, k.solar_mw AS kgup_solar_mw,
       k.dammed_hydro_mw + k.river_hydro_mw AS kgup_hydro_mw,
       k.import_coal_mw + k.lignite_mw + k.black_coal_mw AS kgup_coal_mw,
       g.total_mw AS actual_gen_total_mw, c.consumption_mw AS actual_cons_mw,
       w.turkey_weighted_temperature_c AS temperature_c,
       wf.turkey_weighted_temperature_forecast_c AS temperature_forecast_c,
       mc.usd_try, mc.brent_oil_usd,
       ng.gas_reference_price_try AS natural_gas_grf_try,
       wp.hydro_water_energy_mwh,
       pf.predicted_load_lag0, pf.predicted_solar_lag0, pf.predicted_wind_lag0
FROM raw_mcp_hourly m
LEFT JOIN raw_smp_hourly s ON m.ts = s.ts
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
LEFT JOIN raw_actual_generation_hourly g ON m.ts = g.ts
LEFT JOIN raw_actual_consumption_hourly c ON m.ts = c.ts
LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
LEFT JOIN raw_weather_forecast_hourly wf ON m.ts = wf.ts
LEFT JOIN raw_macro_daily mc ON DATE(m.ts) = mc.entry_date
LEFT JOIN raw_natural_gas_daily ng ON DATE(m.ts) = ng.entry_date
LEFT JOIN (
    SELECT DATE(date_time) AS entry_date,
           SUM(water_energy_provision_mwh) AS hydro_water_energy_mwh
    FROM raw_master_water_energy_provision GROUP BY DATE(date_time)
) wp ON DATE(m.ts) = wp.entry_date
LEFT JOIN gold.kgup_load_pre_forecasts pf ON m.ts = pf.target_ts
WHERE m.ts >= :start AND m.ts < :end
ORDER BY m.ts;
"""


def load_ours(start, end):
    eng = get_db_engine()
    df = pd.read_sql(text(MASTER_SQL), eng, params={"start": start, "end": end})
    ts = pd.to_datetime(df["ts"])
    df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    for c in ["usd_try", "brent_oil_usd", "natural_gas_grf_try", "hydro_water_energy_mwh"]:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()
    feat = build_robust_features(df)
    cols = [c for c in get_feature_columns("robust", feat) if c in feat.columns]
    return feat, cols


def load_theirs(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.Date + " " + df.Hour)
    df["ts"] = ts.dt.tz_localize("Europe/Istanbul", ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    df["_naive"] = np.where(df.index.dayofweek.isin([0, 5, 6]), df.MCP_168, df.MCP_24)
    return df


def walk_forward(X, y, test_mask, params, step_hours, quantile=False):
    """Genişleyen pencere, `step_hours`'ta bir yeniden eğitim."""
    idx = np.arange(len(X))
    first = idx[test_mask].min()
    out = np.full(len(X), np.nan)
    i = first
    while i < len(X):
        j = min(i + step_hours, len(X))
        tr = idx < i
        if tr.sum() > 500:
            m = lgb.LGBMRegressor(**params).fit(X[tr], y[tr])
            out[i:j] = m.predict(X[i:j])
        i = j
    return out


def report(name, y_true, pred, naive):
    ok = ~np.isnan(pred)
    mae = mean_absolute_error(y_true[ok], pred[ok])
    nv = mean_absolute_error(y_true[ok], naive[ok])
    print(f"  {name:<48}{mae:>9.2f}{nv:>9.2f}{mae / nv:>9.3f}")
    return mae, nv, mae / nv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--their-csv", required=True)
    ap.add_argument("--test-year", type=int, default=2022)
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--equal-history", action="store_true",
                    help="ikisi de 2021'den eğitilsin (varsayılan: her biri kendi tam geçmişiyle)")
    a = ap.parse_args()

    their = load_theirs(a.their_csv)
    our_start = "2021-01-01"
    ours, our_cols = load_ours(our_start, f"{a.test_year + 1}-01-01")

    common = ours.index.intersection(their.index)
    common = common[common < pd.Timestamp(f"{a.test_year + 1}-01-01", tz="Europe/Istanbul")]
    print(f"Ortak saat: {len(common)}  ({common.min()} → {common.max()})")

    y = ours.loc[common, "mcp_price_usd"].to_numpy()
    naive = their.loc[common, "_naive"].to_numpy()
    test_mask = common.year == a.test_year
    print(f"Test yılı {a.test_year}: {test_mask.sum()} saat · ort ${y[test_mask].mean():.2f}")
    print(f"Yeniden eğitim: {a.step_days} günde bir, genişleyen pencere\n")

    step = a.step_days * 24
    print(f"  {'Model':<48}{'MAE':>9}{'naive':>9}{'rMAE':>9}")
    print("  " + "-" * 75)

    # --- ONLAR ---
    their_feats = [c for c in their.columns if c not in ("Date", "Hour", "MCP", "_naive")]
    if a.equal_history:
        Xt = their.loc[common, their_feats].to_numpy()
        yt = their.loc[common, "MCP"].to_numpy()
        pt = walk_forward(Xt, yt, test_mask, THEIR_PARAMS, step)
        lbl = "ONLAR (33 değişken, 2021'den)"
    else:
        full = their.index[their.index < pd.Timestamp(f"{a.test_year + 1}-01-01", tz="Europe/Istanbul")]
        Xt = their.loc[full, their_feats].to_numpy()
        yt = their.loc[full, "MCP"].to_numpy()
        tm_full = full.year == a.test_year
        pf = walk_forward(Xt, yt, tm_full, THEIR_PARAMS, step)
        pt = pd.Series(pf, index=full).reindex(common).to_numpy()
        lbl = "ONLAR (33 değişken, 2018'den — 3 yıl fazla veri)"
    report(lbl, y, np.where(test_mask, pt, np.nan), naive)

    # --- BİZ ---
    Xo = ours.loc[common, our_cols].to_numpy()
    po = walk_forward(Xo, y, test_mask, OUR_PARAMS, step)
    report(f"BİZ ({len(our_cols)} özellik, 2021'den, ön-tahminler YOK)",
           y, np.where(test_mask, po, np.nan), naive)

    print("  " + "-" * 75)
    print("  rMAE < 1 = naive baseline'dan iyi (Lago et al. naive).")
    print("  Not: ön-tahmin lag0 özellikleri bu dönemde üretilmemiş; bizim model")
    print("       en etkili özellikleri olmadan koşuyor.")


if __name__ == "__main__":
    main()
