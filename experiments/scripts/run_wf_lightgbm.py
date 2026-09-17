#!/usr/bin/env python3
"""
Faz C.5.1: walk-forward LightGBM — canlı lgb_lag0_v2 konfigüyle keyfi pencerede.

Canlı `gold.ptf_predictions_daily` kaydı yalnız 2024-08+ ve eğitim 2023-01'den.
Bu script rejim stres testi için:
  · spike (2022) dilimi — canlı kayıt yok
  · çöküş dilimi — eğitim 2021-01 (geniş) vs 2023-01 (canlı politika) kıyası

Canlı model = `predict_daily_pipeline.py`: 3-head quantile (P50 nokta),
`build_robust_features` + `get_feature_columns('robust')`, min_child_samples=10,
log1p YOK.

    .venv/bin/python experiments/scripts/run_wf_lightgbm.py \\
        --start 2022-01-01 --end 2022-12-31 --train-start 2021-01-01 --tag spike_tr2021

Çıktı: experiments/notebooks/07_lago_protocol/wf_lgbm_{tag}.csv  (gün × 24, USD/MWh)
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine
from src.features.feature_engineering import build_robust_features, get_feature_columns

OUT_DIR = ROOT / "experiments/notebooks/07_lago_protocol"

PARAMS = dict(objective="quantile", alpha=0.50, n_estimators=300, learning_rate=0.03,
              max_depth=8, num_leaves=63, min_child_samples=10, n_jobs=1, verbose=-1)

MASTER_SQL = text("""
SELECT m.ts, m.price_usd AS mcp_price_usd, m.price_try AS mcp_price_try,
       s.system_marginal_price_try AS smp_price_try, l.load_forecast_mw,
       k.total_mw AS kgup_total_mw, k.natural_gas_mw AS kgup_gas_mw,
       k.wind_mw AS kgup_wind_mw, k.solar_mw AS kgup_solar_mw,
       k.dammed_hydro_mw + k.river_hydro_mw AS kgup_hydro_mw,
       k.geothermal_mw AS kgup_geo_mw, k.biomass_mw AS kgup_biomass_mw,
       k.import_coal_mw + k.lignite_mw + k.black_coal_mw AS kgup_coal_mw,
       g.total_mw AS actual_gen_total_mw, c.consumption_mw AS actual_cons_mw,
       w.turkey_weighted_temperature_c AS temperature_c,
       wf.turkey_weighted_temperature_forecast_c AS temperature_forecast_c,
       mc.usd_try, mc.brent_oil_usd, ng.gas_reference_price_try AS natural_gas_grf_try,
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
LEFT JOIN gold.kgup_load_pre_forecasts pf ON m.ts = pf.target_ts
WHERE m.ts >= :start ORDER BY m.ts
""")


def predict_day(df_model, feat_cols, day, roll_days=None, recency_tau=None):
    train_end = day - pd.Timedelta(hours=1)
    tr = df_model.loc[:train_end]
    if roll_days:                                    # C.5.3a: yuvarlanan pencere
        tr = tr.loc[tr.index >= day - pd.Timedelta(days=roll_days)]
    te = df_model.loc[day:day + pd.Timedelta(hours=23)]
    if len(tr) < 2000 or len(te) < 24:
        return day, np.full(24, np.nan)
    sw = None
    if recency_tau:                                  # C.5.3b: üstel recency ağırlığı
        age = np.asarray((train_end - tr.index).total_seconds()) / 86400.0
        sw = np.exp(-age / recency_tau)
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(tr[feat_cols], tr["mcp_price_usd"], sample_weight=sw)
    return day, m.predict(te[feat_cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--preforecast-override", default=None,
                    help="CSV (target_ts, predicted_*_lag0) — eşleşen ts'lerde DB değerini EZER "
                         "(fillna değil replace). Bozuk rüzgar ön-tahmini düzeltme testi.")
    ap.add_argument("--proxy-missing", action="store_true",
                    help="Hedef gün KGÜP kırılımı henüz yayınlanmamışsa (kgup_total_mw hep NaN) "
                         "canlı pipeline'ın T->T+1 proxy'sini uygula: yük/KGÜP/üretim/tüketim "
                         "sütunlarını bir önceki günün gerçekleşeninden saat-eşli kopyala, "
                         "makro sütunları ffill. Böylece 30 Ağustos gibi 'yarın' günleri koşulabilir.")
    ap.add_argument("--proxy-from", default=None,
                    help="YYYY-MM-DD. Bu tarihten itibaren HER hedef güne T->T+1 proxy uygula. "
                         "!! ÖNERİLMEZ !! Sızıntıya karşı GEREKSİZ: robust özelliklerin tamamı "
                         "shift(24/48/168) lag'i ya da predicted_*_lag0 ön-tahmini; gün D'nin ham "
                         "yük/KGÜP/SMP değeri gün D'nin hiçbir özelliğine girmiyor. Bu bayrak "
                         "yalnız lag_24 zincirini bir gün geriye kaydırır — yani canlıdan DAHA "
                         "KÖTÜ bir simülasyon. Ön-tahmin sadakati için --preforecast-override "
                         "kullan. Bkz. ENSEMBLE_AKSIYON_PLANI.md §1.")
    ap.add_argument("--c52", action="store_true",
                    help="C.5.2 özellikleri: thermal_req_ratio_lag0 + hydro_net_load_lag0 "
                         "(hidro/jeo/biyo must-run'a, T+1 hidro = proxy). feat_cols'a eklenir.")
    ap.add_argument("--roll-days", type=int, default=None,
                    help="C.5.3a: her gün yalnız son N günle eğit (yuvarlanan pencere).")
    ap.add_argument("--recency-tau", type=float, default=None,
                    help="C.5.3b: eğitim örneklerine exp(-yaş_gün/tau) ağırlığı.")
    a = ap.parse_args()

    eng = get_db_engine()
    df = pd.read_sql(MASTER_SQL, eng, params={"start": a.train_start})
    ts = pd.to_datetime(df["ts"])
    df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
    df = df.set_index("ts").sort_index()

    if a.proxy_missing or a.proxy_from:
        # canlı predict_daily_pipeline.py ADIM 5a ile aynı: T->T+1 proxy
        PROXY_COPY = ["load_forecast_mw", "kgup_total_mw", "kgup_gas_mw", "kgup_wind_mw",
                      "kgup_solar_mw", "kgup_hydro_mw", "kgup_geo_mw", "kgup_biomass_mw",
                      "kgup_coal_mw", "actual_gen_total_mw", "actual_cons_mw", "smp_price_try"]
        PROXY_FFILL = ["usd_try", "brent_oil_usd", "natural_gas_grf_try"]
        pfrom = pd.Timestamp(a.proxy_from, tz="Europe/Istanbul") if a.proxy_from else None
        # KASKAD KORUMASI (31 Ağu 2026): proxy kaynağı döngü boyunca DEĞİŞMEYEN bir kopya
        # olmak zorunda. Kaynak `df`'in kendisi olursa gün d işlenirken d-1 bir önceki
        # iterasyonda çoktan ezilmiştir ve aralıktaki HER gün (start-1)'in tek günlük
        # profiline çöker — domino. `--proxy-from 2024-08-01` ile 757 günün tamamı
        # 2024-07-31'i tekrarlıyordu. Bkz. ../enerji_fiyat_tahmini/ENSEMBLE_AKSIYON_PLANI.md §1.
        src = df.copy(deep=True)
        src_ffill = {c: src[c].ffill() for c in PROXY_FFILL}
        dn = df.index.normalize()
        _proxy_days = []
        for d in pd.date_range(a.start, a.end, freq="D", tz="Europe/Istanbul"):
            m = dn == d
            force = pfrom is not None and d >= pfrom
            if not m.any() or (not force and not src.loc[m, "kgup_total_mw"].isna().all()):
                continue
            prev = src.loc[dn == d - pd.Timedelta(days=1)]        # <-- src, df DEĞİL
            if len(prev) < 24:
                continue
            prev_temp = pd.Series(prev["temperature_c"].tail(24).values, index=df.index[m])
            for c in PROXY_COPY:
                df.loc[m, c] = prev[c].tail(24).values
            for c in PROXY_FFILL:
                df.loc[m, c] = src_ffill[c].loc[m]                # <-- df[c].ffill() DEĞİL
            # canlı ADIM 5b: yarının hava tahmini varsa onu, yoksa bugünün sıcaklığını kullan
            base_temp = src.loc[m, "temperature_c"] if not force else pd.Series(np.nan, index=df.index[m])
            df.loc[m, "temperature_c"] = (base_temp
                                          .fillna(src.loc[m, "temperature_forecast_c"])
                                          .fillna(prev_temp))
            _proxy_days.append(d.date())
        # yavaş-değişen feed'ler bayat (SMP Ağu 28, GRF Ağu 27) → lag_48 çözülmüyor.
        # canlı da makroyu ffill'liyor (ADIM 5a); SMP'nin ~2 saatlik boşluğu ihmal edilebilir.
        for c in ["smp_price_try", "natural_gas_grf_try", "usd_try", "brent_oil_usd"]:
            df[c] = df[c].ffill()
        if _proxy_days:
            print(f"T->T+1 proxy: {len(_proxy_days)} gün ({_proxy_days[0]} -> {_proxy_days[-1]})")
        if a.proxy_from:
            print("UYARI: --proxy-from sızıntıya karşı gereksiz ve lag_24 zincirini bozuyor; "
                  "sonuçlar canlıyı temsil etmez. Bkz. --help.")

    df["load_forecast_mw"] = df["load_forecast_mw"].interpolate(limit=6).ffill().bfill()
    df["kgup_total_mw"] = df["kgup_total_mw"].interpolate(limit=6).ffill().bfill()

    # pre-2024-08 pre-forecast'leri: DB'de NULL → walk-forward backfill CSV'sinden doldur
    # (canlı pipeline'ın yaptığı gibi TAHMİN, gerçekleşen değil). Bkz. run_preforecast_backfill.py
    pfbf = OUT_DIR / "preforecast_backfill.csv"
    if pfbf.exists():
        b = pd.read_csv(pfbf, index_col=0, parse_dates=True)
        b.index = pd.DatetimeIndex(b.index)
        if b.index.tz is None:
            b.index = b.index.tz_localize("Europe/Istanbul")
        for c in ["predicted_load_lag0", "predicted_solar_lag0", "predicted_wind_lag0"]:
            df[c] = df[c].fillna(b[c].reindex(df.index))
        print(f"pre-forecast backfill uygulandı ({pfbf.name}): {b.index.min().date()} -> {b.index.max().date()}")

    if a.preforecast_override:
        ov = pd.read_csv(a.preforecast_override, index_col=0, parse_dates=True)
        ov.index = pd.DatetimeIndex(ov.index)
        if ov.index.tz is None:
            ov.index = ov.index.tz_localize("Europe/Istanbul")
        for c in ["predicted_load_lag0", "predicted_solar_lag0", "predicted_wind_lag0"]:
            df[c] = ov[c].reindex(df.index).fillna(df[c])   # override = EZ
        print(f"pre-forecast OVERRIDE ({Path(a.preforecast_override).name}): "
              f"{ov.index.min().date()} -> {ov.index.max().date()} ({len(ov)} saat ezildi)")

    feat = build_robust_features(df)
    feat_cols = get_feature_columns("robust", feat)

    if a.c52:
        # T+1 must-run: güneş/rüzgar = ön-tahmin (lag0); hidro/jeo/biyo pre-forecaster'ı YOK →
        # dünün gerçekleşeni (shift 24, saat-eşli) — canlının 04:00'te sahip olduğu en iyi.
        # LOW_PRICE_REGIME_ANALYSIS: hidro payı yıllar arası transfer olmuyor ama termal
        # ihtiyaç oranı oluyor; net_load_lag0 hidroyu hiç çıkarmıyor, jeo+biyo hiçbir orana girmiyor.
        hyd = feat["kgup_hydro_mw"].interpolate(limit=6).shift(24)
        geo = feat["kgup_geo_mw"].interpolate(limit=6).shift(24)
        bio = feat["kgup_biomass_mw"].interpolate(limit=6).shift(24)
        load0 = feat["predicted_load_lag0"].replace(0, np.nan)
        mustrun = (feat["predicted_solar_lag0"] + feat["predicted_wind_lag0"] + hyd + geo + bio)
        feat["thermal_req_ratio_lag0"] = ((load0 - mustrun) / load0).clip(-0.5, 1.5).fillna(0.0)
        feat["hydro_net_load_lag0"] = (feat["predicted_load_lag0"] - feat["predicted_solar_lag0"]
                                       - feat["predicted_wind_lag0"] - hyd).fillna(0.0)
        feat_cols = feat_cols + ["thermal_req_ratio_lag0", "hydro_net_load_lag0"]
        print("C.5.2 özellikleri eklendi (hidro/jeo/biyo = shift 24): thermal_req_ratio_lag0, hydro_net_load_lag0")

    df_model = feat.dropna(subset=feat_cols + ["mcp_price_usd"]).copy()
    print(f"veri {df_model.index.min().date()} -> {df_model.index.max().date()}  "
          f"{len(feat_cols)} özellik  ({len(df_model):,} saat)")

    days = pd.date_range(a.start, a.end, freq="D", tz="Europe/Istanbul")
    days = days[days.isin(df_model.index.normalize().unique())]
    print(f"tag={a.tag}  test {days[0].date()} -> {days[-1].date()} ({len(days)} gün)  "
          f"eğitim {a.train_start}'den")

    if a.roll_days or a.recency_tau:
        print(f"C.5.3: roll_days={a.roll_days}  recency_tau={a.recency_tau}")
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(predict_day)(df_model, feat_cols, d, a.roll_days, a.recency_tau) for d in days)
    tbl = pd.DataFrame({d: p for d, p in res}).T.sort_index()
    tbl.columns = [f"h{h:02d}" for h in range(24)]
    tbl.index = tbl.index.tz_localize(None)
    out = OUT_DIR / f"wf_lgbm_{a.tag}.csv"
    tbl.to_csv(out)
    nf = int(tbl.isna().all(axis=1).sum())
    print(f"\n{time.time()-t0:.0f}s  ->  {out.name}  {tbl.shape}  (NaN {nf} gün)")


if __name__ == "__main__":
    main()
