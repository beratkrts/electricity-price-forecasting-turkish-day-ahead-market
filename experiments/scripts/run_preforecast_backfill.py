#!/usr/bin/env python3
"""
Faz C.5 ön koşulu: pre-forecast'leri 2021-2024 için walk-forward backfill.

`gold.kgup_load_pre_forecasts` yalnız 2024-08-13'ten dolu. Öncesinde
`build_robust_features` lag0 özelliklerini (renewable_pressure_ratio_lag0,
net_load_lag0, zero_price_risk_score, ramp'ler) GERÇEKLEŞEN değerle dolduruyor —
oysa canlı pipeline pre-forecaster'ları taze koşup TAHMİN ediyor (yük WAPE ~%2.9,
güneş ~%10.5, rüzgar ~%18). Bu sızıntı 2022 spike / pre-2024 backtest'leri şişiriyor.

Bu script canlı `src/features/pre_forecasters.py`'yi (yük LightGBM, güneş Ridge,
rüzgar LightGBM + OpenMeteo) walk-forward koşup çıktıyı **CSV'ye** yazar
(`gold.kgup_load_pre_forecasts` canlı tablo — CLAUDE.md yazma kısıtı).

    .venv/bin/python experiments/scripts/run_preforecast_backfill.py \\
        --start 2021-01-01 --end 2024-08-12 --first-target 2021-03-01

Çıktı: experiments/notebooks/07_lago_protocol/preforecast_backfill.csv
       (target_ts, predicted_load_lag0, predicted_solar_lag0, predicted_wind_lag0)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO.parent / "enerji_fiyat_tahmini"
sys.path.insert(0, str(LIVE))  # canlı src.features.pre_forecasters + db.connection

from db.connection import get_db_engine  # noqa: E402
from src.features.pre_forecasters import (  # noqa: E402
    CITY_COORDS, build_pre_forecast_features, train_and_predict_pre_forecasters,
)

OUT = REPO / "experiments/notebooks/07_lago_protocol/preforecast_backfill.csv"
WIND_CACHE = REPO / "experiments/notebooks/07_lago_protocol/_wind_2021_2024.pkl"


def fetch_wind(start, end):
    """OpenMeteo arşiv — 3 şehir 100m rüzgar hızı. DB'ye yazmaz (raw_wind_history_hourly
    canlı repoda 2023+; buradaki gap'i canlı cache'e dokunmadan lokal çekiyoruz)."""
    if WIND_CACHE.exists():
        return pd.read_pickle(WIND_CACHE)
    import httpx
    out = None
    for city, c in CITY_COORDS.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={c['lat']}"
               f"&longitude={c['lon']}&start_date={start}&end_date={end}"
               f"&hourly=wind_speed_100m&timezone=Europe%2FIstanbul")
        d = httpx.get(url, timeout=60.0).json()["hourly"]
        idx = pd.to_datetime(d["time"]).tz_localize("Europe/Istanbul", ambiguous="infer", nonexistent="shift_forward")
        s = pd.DataFrame({f"wind_{city}": d["wind_speed_100m"]}, index=idx)
        out = s if out is None else out.join(s)
    out.to_pickle(WIND_CACHE)
    return out

SQL = text("""
SELECT m.ts, l.load_forecast_mw, k.wind_mw AS kgup_wind_mw, k.solar_mw AS kgup_solar_mw,
       w.turkey_weighted_temperature_c AS temperature_c
FROM raw_mcp_hourly m
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
WHERE m.ts >= :start AND m.ts < :end_excl
ORDER BY m.ts
""")


def one_day(df_feat, day):
    tr = df_feat.loc[:day - pd.Timedelta(hours=1)]
    te = df_feat.loc[day:day + pd.Timedelta(hours=23)]
    if len(tr) < 2000 or len(te) < 24:
        return None
    try:
        r = train_and_predict_pre_forecasters(tr, te)
    except Exception as e:  # noqa: BLE001
        print(f"  {day.date()} hata: {e}", flush=True)
        return None
    r = r.reindex(te.index)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-08-13")
    ap.add_argument("--first-target", default="2021-03-01")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()

    eng = get_db_engine()
    df = pd.read_sql(SQL, eng, params={"start": a.start, "end_excl": a.end})
    ts = pd.to_datetime(df["ts"])
    df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    print(f"ham veri {df.index.min().date()} -> {df.index.max().date()}  ({len(df):,} saat)")

    print("OpenMeteo rüzgar (izmir/çanakkale/balıkesir) çekiliyor...")
    df_wind = fetch_wind(a.start, (pd.Timestamp(a.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    print(f"  rüzgar {df_wind.index.min().date()} -> {df_wind.index.max().date()}  {df_wind.shape}")
    df_feat = build_pre_forecast_features(df, df_wind)
    print(f"özellik matrisi {df_feat.shape}  rüzgar kolonları: "
          f"{[c for c in df_feat.columns if c.startswith('wind_')][:6]}")

    days = pd.date_range(a.first_target, pd.Timestamp(a.end) - pd.Timedelta(days=1),
                         freq="D", tz="Europe/Istanbul")
    days = days[days.isin(df_feat.index.normalize().unique())]
    print(f"walk-forward: {days[0].date()} -> {days[-1].date()}  ({len(days)} gün)  n_jobs={a.n_jobs}")

    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_day)(df_feat, d) for d in days)
    res = [r for r in res if r is not None]
    out = pd.concat(res).sort_index()
    out.index.name = "target_ts"
    out.to_csv(OUT)
    print(f"\n{time.time()-t0:.0f}s  ->  {OUT.name}  {out.shape}")
    print(f"  kolon NaN: {out.isna().sum().to_dict()}")
    # gerçekleşenle kıyas (bu backfill'in sızıntıyı ne kadar kapattığını göster)
    real = pd.read_sql(text("SELECT ts, l.load_forecast_mw, k.wind_mw, k.solar_mw "
                            "FROM raw_mcp_hourly m LEFT JOIN raw_load_forecast_hourly l ON m.ts=l.ts "
                            "LEFT JOIN raw_kgup_hourly k ON m.ts=k.ts "
                            f"WHERE m.ts >= '{a.first_target}' AND m.ts < '{a.end}'"), eng)
    real["ts"] = pd.to_datetime(real["ts"]).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    real = real.set_index("ts")
    o2 = out.copy(); o2.index = o2.index.tz_localize(None)
    j = o2.join(real, how="inner")
    for pf, rc in [("predicted_load_lag0", "load_forecast_mw"), ("predicted_solar_lag0", "solar_mw"),
                   ("predicted_wind_lag0", "wind_mw")]:
        e = (j[pf] - j[rc]).abs()
        print(f"  {pf}: WAPE {100*e.sum()/j[rc].abs().sum():.1f}%  MAE {e.mean():.0f} MW")


if __name__ == "__main__":
    main()
