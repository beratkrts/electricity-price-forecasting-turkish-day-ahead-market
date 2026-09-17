#!/usr/bin/env python3
"""
Rüzgar (+güneş, yük) ön-tahminini son günler için DÜZGÜN meteo ile yeniden üret.

Canlı `predicted_wind_lag0` haftalardır düz ~2400 MW — forecast API hiç
çağrılmadığı için `train_and_predict_pre_forecasters` meteo NaN ile koşup
`kgup_wind_lag_*`'a yaslanıyor (bkz. WIND_PREFORECAST_INCIDENT_2026-08.md).

Bu script canlı `src.features.pre_forecasters.train_and_predict_pre_forecasters`'ı
DEĞİŞTİRMEDEN, ama meteo girdisini OpenMeteo arşiv + forecast API birleşimiyle
DOLU vererek walk-forward koşar. Çıktı CSV (canlı tablo immutable).

    .venv/bin/python experiments/scripts/fix_preforecast_recent.py --start 2026-08-10 --end 2026-08-30

Çıktı: experiments/notebooks/07_lago_protocol/preforecast_recent_fixed.csv
"""
import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO.parent / "enerji_fiyat_tahmini"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LIVE))
from db.connection import get_db_engine  # noqa: E402
from src.features.pre_forecasters import (  # noqa: E402
    CITY_COORDS, build_pre_forecast_features, train_and_predict_pre_forecasters,
)

OUT = REPO / "experiments/notebooks/07_lago_protocol/preforecast_recent_fixed.csv"
WIND_CACHE = REPO / "experiments/notebooks/07_lago_protocol/_wind_archive_2023_2026.pkl"
TRAIN_START = "2023-01-01"

SQL = text("""
SELECT m.ts, l.load_forecast_mw, k.wind_mw AS kgup_wind_mw, k.solar_mw AS kgup_solar_mw,
       w.turkey_weighted_temperature_c AS temperature_c,
       wf.turkey_weighted_temperature_forecast_c AS temp_fc
FROM raw_mcp_hourly m
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
LEFT JOIN raw_weather_forecast_hourly wf ON m.ts = wf.ts
WHERE m.ts >= :start ORDER BY m.ts
""")


def _pull(base, city, c, extra):
    url = (f"{base}?latitude={c['lat']}&longitude={c['lon']}"
           f"&hourly=wind_speed_100m&timezone=Europe%2FIstanbul&{extra}")
    d = httpx.get(url, timeout=90.0).json()["hourly"]
    idx = pd.to_datetime(d["time"]).tz_localize(
        "Europe/Istanbul", ambiguous="infer", nonexistent="shift_forward")
    return pd.DataFrame({f"wind_{city}": d["wind_speed_100m"]}, index=idx)


def fetch_wind():
    """Arşiv (reanaliz) + forecast (yarın) birleşik — canlı load_wind_features mantığı."""
    if WIND_CACHE.exists():
        w = pd.read_pickle(WIND_CACHE)
        if w.index.max() >= pd.Timestamp.now(tz="Europe/Istanbul"):
            return w
    arch_end = (pd.Timestamp.now(tz="Europe/Istanbul") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    arch, fc = None, None
    for city, c in CITY_COORDS.items():
        a = _pull("https://archive-api.open-meteo.com/v1/archive", city, c,
                  f"start_date={TRAIN_START}&end_date={arch_end}")
        f = _pull("https://api.open-meteo.com/v1/forecast", city, c,
                  "past_days=14&forecast_days=3")
        arch = a if arch is None else arch.join(a)
        fc = f if fc is None else fc.join(f)
    gap = fc[~fc.index.isin(arch.index)]
    out = pd.concat([arch, gap]).sort_index()
    out.to_pickle(WIND_CACHE)
    return out


def one_day(df_feat, day):
    tr = df_feat.loc[:day - pd.Timedelta(hours=1)]
    te = df_feat.loc[day:day + pd.Timedelta(hours=23)]
    if len(tr) < 2000 or len(te) < 24:
        return None
    try:
        r = train_and_predict_pre_forecasters(tr, te).reindex(te.index)
    except Exception as e:  # noqa: BLE001
        print(f"  {day.date()} hata: {e}", flush=True)
        return None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-10")
    ap.add_argument("--end", default="2026-08-30")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()

    eng = get_db_engine()
    df = pd.read_sql(SQL, eng, params={"start": TRAIN_START})
    ts = pd.to_datetime(df["ts"])
    df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    # hedef gün sıcaklığı: forecast varsa o, yoksa gerçekleşen (temp tahmini zaten çok isabetli)
    df["temperature_c"] = df["temp_fc"].fillna(df["temperature_c"])
    df = df.drop(columns=["temp_fc"])
    print(f"ham veri {df.index.min().date()} -> {df.index.max().date()}  ({len(df):,} saat)")

    wind = fetch_wind()
    print(f"rüzgar (arşiv+forecast) {wind.index.min().date()} -> {wind.index.max().date()}  {wind.shape}")
    df_feat = build_pre_forecast_features(df, wind)
    wcols = [c for c in ["wind_izmir", "wind_canakkale", "wind_balikesir"] if c in df_feat.columns]
    print(f"özellik matrisi {df_feat.shape}  meteo kolonları dolu: "
          f"{df_feat.loc['2026-08-20':'2026-08-30', wcols].notna().mean().round(2).to_dict()}")

    days = pd.date_range(a.start, a.end, freq="D", tz="Europe/Istanbul")
    days = days[days.isin(df_feat.index.normalize().unique())]
    print(f"walk-forward: {days[0].date()} -> {days[-1].date()}  ({len(days)} gün)")

    t0 = time.time()
    res = [r for r in Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(one_day)(df_feat, d) for d in days) if r is not None]
    out = pd.concat(res).sort_index()
    out.index.name = "target_ts"
    out.to_csv(OUT)
    print(f"{time.time()-t0:.0f}s  ->  {OUT.name}  {out.shape}")

    # kıyas: düzeltilmiş vs canlı(bozuk) vs gerçek
    real = pd.read_sql(text(
        "SELECT m.ts, l.load_forecast_mw, k.wind_mw, k.solar_mw FROM raw_mcp_hourly m "
        "LEFT JOIN raw_load_forecast_hourly l ON m.ts=l.ts "
        "LEFT JOIN raw_kgup_hourly k ON m.ts=k.ts "
        f"WHERE m.ts >= '{a.start}'"), eng)
    real["ts"] = pd.to_datetime(real["ts"])
    real["ts"] = real["ts"].dt.tz_convert("Europe/Istanbul") if real["ts"].dt.tz else real["ts"].dt.tz_localize("Europe/Istanbul")
    real = real.set_index("ts")
    dbpf = pd.read_sql(text(
        "SELECT target_ts, predicted_load_lag0, predicted_solar_lag0, predicted_wind_lag0 "
        f"FROM gold.kgup_load_pre_forecasts WHERE target_ts >= '{a.start}'"), eng)
    dbpf["target_ts"] = pd.to_datetime(dbpf["target_ts"])
    dbpf["target_ts"] = dbpf["target_ts"].dt.tz_convert("Europe/Istanbul")
    dbpf = dbpf.set_index("target_ts")

    print("\n=== ön-tahmin kalitesi (MAE MW / BIAS MW), hedef günler ===")
    for pf_col, rc in [("predicted_wind_lag0", "wind_mw"), ("predicted_solar_lag0", "solar_mw"),
                       ("predicted_load_lag0", "load_forecast_mw")]:
        k = out.index.intersection(real.index).intersection(dbpf.index)
        a_ = real.loc[k, rc]
        fx = out.loc[k, pf_col]
        db = dbpf.loc[k, pf_col]
        print(f"{pf_col:22s}  DÜZELTİLMİŞ MAE {(fx-a_).abs().mean():6.0f}  BIAS {(fx-a_).mean():+7.0f}   "
              f"|  CANLI MAE {(db-a_).abs().mean():6.0f}  BIAS {(db-a_).mean():+7.0f}")

    print("\n=== gün gün rüzgar (MW): gerçek | düzeltilmiş | canlı ===")
    g = pd.DataFrame({
        "gercek": real["wind_mw"].groupby(real.index.normalize()).mean(),
        "duzeltilmis": out["predicted_wind_lag0"].groupby(out.index.normalize()).mean(),
        "canli": dbpf["predicted_wind_lag0"].groupby(dbpf.index.normalize()).mean(),
    }).dropna(how="all").round(0)
    print(g.to_string())


if __name__ == "__main__":
    main()
