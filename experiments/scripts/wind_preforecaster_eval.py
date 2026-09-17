#!/usr/bin/env python3
"""
Rüzgar ön-tahmincisinin (canlı `train_and_predict_pre_forecasters` -> lgb_wind)
uzun-dönem walk-forward performansı.

Meteo girdisi = OpenMeteo arşiv reanaliz ("iyi meteo elde olsaydı" senaryosu —
mimarinin tavanını ölçer; gerçek D-1 forecast ~1-2 m/s daha kötü).

    .venv/bin/python experiments/scripts/wind_preforecaster_eval.py --start 2025-01-01 --end 2026-08-29

Çıktı: experiments/notebooks/07_lago_protocol/wind_preforecaster_eval.csv  (gün × {pred, gercek})
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LIVE))
from db.connection import get_db_engine  # noqa: E402
from src.features.pre_forecasters import (  # noqa: E402
    build_pre_forecast_features, train_and_predict_pre_forecasters,
)

OUT = REPO / "experiments/notebooks/07_lago_protocol/wind_preforecaster_eval.csv"
WIND_CACHE = REPO / "experiments/notebooks/07_lago_protocol/_wind_archive_2023_2026.pkl"

SQL = text("""
SELECT m.ts, l.load_forecast_mw, k.wind_mw AS kgup_wind_mw, k.solar_mw AS kgup_solar_mw,
       w.turkey_weighted_temperature_c AS temperature_c
FROM raw_mcp_hourly m
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
WHERE m.ts >= '2023-01-01' ORDER BY m.ts
""")


def one_day(df_feat, day):
    tr = df_feat.loc[:day - pd.Timedelta(hours=1)]
    te = df_feat.loc[day:day + pd.Timedelta(hours=23)]
    if len(tr) < 3000 or len(te) < 24:
        return None
    try:
        r = train_and_predict_pre_forecasters(tr, te).reindex(te.index)
    except Exception as e:  # noqa: BLE001
        print(f"  {day.date()} hata: {e}", flush=True)
        return None
    return pd.DataFrame({"pred": r["predicted_wind_lag0"].values,
                         "gercek": te["kgup_wind_mw"].values}, index=te.index)


def wape(p, a):
    m = p.notna() & a.notna()
    return 100 * (p[m] - a[m]).abs().sum() / a[m].abs().sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-08-29")
    ap.add_argument("--n-jobs", type=int, default=7)
    a = ap.parse_args()

    eng = get_db_engine()
    df = pd.read_sql(SQL, eng)
    ts = pd.to_datetime(df["ts"])
    df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    wind = pd.read_pickle(WIND_CACHE)
    wind = wind[~wind.index.duplicated(keep="first")]
    df_feat = build_pre_forecast_features(df, wind)
    print(f"özellik matrisi {df_feat.shape}")

    days = pd.date_range(a.start, a.end, freq="D", tz="Europe/Istanbul")
    days = days[days.isin(df_feat.index.normalize().unique())]
    print(f"walk-forward: {days[0].date()} -> {days[-1].date()} ({len(days)} gün)  n_jobs={a.n_jobs}")

    t0 = time.time()
    res = [r for r in Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(one_day)(df_feat, d) for d in days) if r is not None]
    hourly = pd.concat(res).sort_index()
    hourly.to_csv(OUT)
    print(f"{time.time()-t0:.0f}s  ->  {OUT.name}  {hourly.shape}\n")

    p, act = hourly["pred"], hourly["gercek"]
    err = p - act
    daily = hourly.resample("D").mean()
    dP, dA = daily["pred"], daily["gercek"]

    # naive-1: dün aynı saat.  naive-7: son 7 günün aynı saati ort.
    n1 = act.shift(24)
    n7 = pd.concat([act.shift(24 * i) for i in range(1, 8)], axis=1).mean(axis=1)

    print("=" * 62)
    print(f"GENEL ({daily.index.min().date()} → {daily.index.max().date()}, {len(daily)} gün, {len(hourly):,} saat)")
    print("=" * 62)
    print(f"  saatlik  WAPE {wape(p,act):5.1f}%   MAE {err.abs().mean():5.0f} MW   BIAS {err.mean():+5.0f} MW")
    print(f"  günlük   WAPE {wape(dP,dA):5.1f}%   MAE {(dP-dA).abs().mean():5.0f} MW")
    print(f"  naive-1  WAPE {wape(n1,act):5.1f}%   |  naive-7  WAPE {wape(n7,act):5.1f}%   "
          f"|  rMAE(n1) {err.abs().mean()/(n1-act).abs().mean():.3f}")

    print("\n--- ay ay ---")
    mo = hourly.copy(); mo["m"] = mo.index.to_period("M")
    for m, g in mo.groupby("m"):
        e = g["pred"] - g["gercek"]
        print(f"  {m}  WAPE {wape(g['pred'],g['gercek']):5.1f}%  MAE {e.abs().mean():5.0f}  BIAS {e.mean():+5.0f}  "
              f"(gerçek ort {g['gercek'].mean():.0f} MW)")

    print("\n--- gerçekleşen rüzgar seviyesine göre (saatlik) ---")
    bins = [0, 2000, 4000, 6000, 8000, 1e9]
    lbl = ["<2000", "2-4k", "4-6k", "6-8k", ">8000"]
    hb = hourly.assign(b=pd.cut(act, bins, labels=lbl))
    for b, g in hb.groupby("b", observed=True):
        e = g["pred"] - g["gercek"]
        print(f"  {b:>7}  n={len(g):5d}  MAE {e.abs().mean():5.0f}  BIAS {e.mean():+6.0f}  "
              f"WAPE {wape(g['pred'],g['gercek']):5.1f}%")

    print("\n--- RAMP günleri: |gün ort - dün gün ort| büyük olanlar ---")
    dchg = dA.diff()
    for lo, hi, name in [(0, 1000, "durağan (<1000 MW/gün)"), (1000, 2500, "orta (1-2.5k)"),
                         (2500, 1e9, "sert ramp (>2.5k)")]:
        sel = daily[(dchg.abs() >= lo) & (dchg.abs() < hi)]
        if len(sel) == 0:
            continue
        e = sel["pred"] - sel["gercek"]
        # ramp yönüne göre bias: yükselişte under mi tahmin ediyor?
        up = sel[dchg.reindex(sel.index) > 0]; dn = sel[dchg.reindex(sel.index) < 0]
        eu = (up["pred"] - up["gercek"]).mean() if len(up) else np.nan
        ed = (dn["pred"] - dn["gercek"]).mean() if len(dn) else np.nan
        print(f"  {name:24s} n={len(sel):3d}  günlük MAE {e.abs().mean():5.0f}  "
              f"BIAS {e.mean():+5.0f}  | yükseliş BIAS {eu:+6.0f}  düşüş BIAS {ed:+6.0f}")

    print("\n--- en kötü 10 gün (günlük |hata|) ---")
    d2 = daily.assign(mae=(daily["pred"] - daily["gercek"]).abs(),
                      bias=daily["pred"] - daily["gercek"],
                      dchg=dchg).nlargest(10, "mae")
    print(d2[["gercek", "pred", "bias", "dchg"]].round(0).to_string())


if __name__ == "__main__":
    main()
