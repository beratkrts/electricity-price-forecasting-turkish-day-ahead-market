#!/usr/bin/env python3
"""
Üç KGÜP ön-tahmincisinin (canlı `train_and_predict_pre_forecasters`) uzun-dönem
walk-forward performansı: rüzgar (lgb), güneş (ridge), yük (lgb).

Meteo girdisi = OpenMeteo arşiv reanaliz ("iyi meteo" — mimarinin tavanı).

    .venv/bin/python experiments/scripts/preforecaster_eval.py --start 2025-01-01 --end 2026-08-29

Çıktı: experiments/notebooks/07_lago_protocol/preforecaster_eval_hourly.csv
       (ts, wind_pred, wind_real, solar_pred, solar_real, load_pred, load_real)
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
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(LIVE))
from db.connection import get_db_engine  # noqa: E402
from src.features.pre_forecasters import (  # noqa: E402
    build_pre_forecast_features, train_and_predict_pre_forecasters,
)

OUT = REPO / "experiments/notebooks/07_lago_protocol/preforecaster_eval_hourly.csv"
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
REAL = {"wind": "kgup_wind_mw", "solar": "kgup_solar_mw", "load": "load_forecast_mw"}
PRED = {"wind": "predicted_wind_lag0", "solar": "predicted_solar_lag0", "load": "predicted_load_lag0"}


def one_day(df_feat, day):
    tr = df_feat.loc[:day - pd.Timedelta(hours=1)]
    te = df_feat.loc[day:day + pd.Timedelta(hours=23)]
    if len(tr) < 3000 or len(te) < 24:
        return None
    try:
        r = train_and_predict_pre_forecasters(tr, te).reindex(te.index)
    except Exception as e:  # noqa: BLE001
        print(f"  {day.date()} hata: {e}", flush=True); return None
    d = {}
    for k in REAL:
        d[f"{k}_pred"] = r[PRED[k]].values if PRED[k] in r else np.nan
        d[f"{k}_real"] = te[REAL[k]].values
    return pd.DataFrame(d, index=te.index)


def wape(p, a):
    m = p.notna() & a.notna()
    return 100 * (p[m] - a[m]).abs().sum() / a[m].abs().sum()


def report(name, h):
    p, a = h[f"{name}_pred"], h[f"{name}_real"]
    e = p - a
    dd = h[[f"{name}_pred", f"{name}_real"]].resample("D").mean()
    dp, da = dd[f"{name}_pred"], dd[f"{name}_real"]
    n1 = a.shift(24)
    print(f"\n{'='*64}\n{name.upper()}   ({len(h):,} saat)\n{'='*64}")
    print(f"  saatlik  WAPE {wape(p,a):5.1f}%   MAE {e.abs().mean():6.0f}   BIAS {e.mean():+6.0f}   "
          f"rMAE(naive-1) {e.abs().mean()/(n1-a).abs().mean():.3f}")
    print(f"  günlük   WAPE {wape(dp,da):5.1f}%   MAE {(dp-da).abs().mean():6.0f}")
    # seviyeye göre
    q = a.quantile([.2, .4, .6, .8]).values
    bins = [-1, *q, 1e12]
    lbl = [f"<{q[0]:.0f}", f"{q[0]:.0f}-{q[1]:.0f}", f"{q[1]:.0f}-{q[2]:.0f}",
           f"{q[2]:.0f}-{q[3]:.0f}", f">{q[3]:.0f}"]
    hb = h.assign(b=pd.cut(a, bins, labels=lbl))
    print("  gerçek seviyesine göre (quintile):")
    for b, g in hb.groupby("b", observed=True):
        ee = g[f"{name}_pred"] - g[f"{name}_real"]
        print(f"    {b:>14}  n={len(g):5d}  MAE {ee.abs().mean():6.0f}  BIAS {ee.mean():+7.0f}  WAPE {wape(g[f'{name}_pred'],g[f'{name}_real']):5.1f}%")
    # ay ay bias trendi (kısalt)
    mo = h.copy(); mo["m"] = mo.index.tz_localize(None).to_period("M")
    mb = mo.groupby("m").apply(lambda g: (g[f"{name}_pred"] - g[f"{name}_real"]).mean())
    print(f"  ay-ay BIAS: min {mb.min():+.0f} ({mb.idxmin()})  max {mb.max():+.0f} ({mb.idxmax()})  "
          f"negatif ay: {(mb<0).sum()}/{len(mb)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-08-29")
    ap.add_argument("--n-jobs", type=int, default=7)
    ap.add_argument("--reuse", action="store_true", help="varsa hourly CSV'yi tekrar kullan")
    a = ap.parse_args()

    if a.reuse and OUT.exists():
        h = pd.read_csv(OUT, index_col=0, parse_dates=True)
        h.index = pd.DatetimeIndex(h.index)
        if h.index.tz is None:
            h.index = h.index.tz_localize("Europe/Istanbul")
        print(f"tekrar kullanıldı: {OUT.name} {h.shape}")
    else:
        eng = get_db_engine()
        df = pd.read_sql(SQL, eng)
        ts = pd.to_datetime(df["ts"])
        df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        wind = pd.read_pickle(WIND_CACHE)
        wind = wind[~wind.index.duplicated(keep="first")]
        df_feat = build_pre_forecast_features(df, wind)

        days = pd.date_range(a.start, a.end, freq="D", tz="Europe/Istanbul")
        days = days[days.isin(df_feat.index.normalize().unique())]
        print(f"walk-forward: {days[0].date()} -> {days[-1].date()} ({len(days)} gün)")
        t0 = time.time()
        res = [r for r in Parallel(n_jobs=a.n_jobs, verbose=5)(
            delayed(one_day)(df_feat, d) for d in days) if r is not None]
        h = pd.concat(res).sort_index()
        h.to_csv(OUT)
        print(f"{time.time()-t0:.0f}s -> {OUT.name} {h.shape}")

    for name in ("wind", "solar", "load"):
        report(name, h)


if __name__ == "__main__":
    main()
