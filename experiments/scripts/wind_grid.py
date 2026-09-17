#!/usr/bin/env python3
"""
Rüzgar ön-tahmincisi fix ızgarası — 9 konfig, aynı walk-forward (2025-01→2026-08).

Canlı lgb_wind'i (n100/lr.05/d5, feats: hour,month,wind_power×3,wind×3,kgup_wind_lag_1..14)
DEĞİŞTİRMEDEN kopyalayıp şu knob'ları yarıştırır:
  target : raw MW | cf (kapasite-faktörü, tr-penceresi 60g rolling max ile)
  model  : base (canlı) | deep (n300/lr.03/d7/mcs5)
  calib  : none | linear | isotonic  (tr'nin son 120g'inde in-sample fit)
  window : full | 180  (recency)
  pc     : cube (wind³, canlı) | curve (normalize türbin güç eğrisi)

    .venv/bin/python experiments/scripts/wind_grid.py --start 2025-01-01 --end 2026-08-29

Çıktı: experiments/notebooks/07_lago_protocol/wind_grid_results.csv  (config × gün × 24h)
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import text

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO.parent / "enerji_fiyat_tahmini"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(LIVE))
from db.connection import get_db_engine  # noqa: E402
from src.features.pre_forecasters import build_pre_forecast_features, CITY_COORDS  # noqa: E402

OUT = REPO / "experiments/notebooks/07_lago_protocol/wind_grid_results.csv"
WIND_CACHE = REPO / "experiments/notebooks/07_lago_protocol/_wind_archive_2023_2026.pkl"
CITIES = list(CITY_COORDS)

SQL = text("""
SELECT m.ts, l.load_forecast_mw, k.wind_mw AS kgup_wind_mw, k.solar_mw AS kgup_solar_mw,
       w.turkey_weighted_temperature_c AS temperature_c
FROM raw_mcp_hourly m
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
WHERE m.ts >= '2023-01-01' ORDER BY m.ts
""")

BASE = dict(n_estimators=100, learning_rate=0.05, max_depth=5,
            random_state=42, n_jobs=1, verbose=-1)
DEEP = dict(n_estimators=300, learning_rate=0.03, max_depth=7, min_child_samples=5,
            num_leaves=63, random_state=42, n_jobs=1, verbose=-1)

def _cfg(target="raw", model="base", calib="none", window="full", pc="cube",
         cap="max90", calwin=120):
    return dict(target=target, model=model, calib=calib, window=window, pc=pc,
                cap=cap, calwin=calwin)

CONFIGS = {
    "0_baseline": _cfg(),
    "lin120":     _cfg(calib="linear",   calwin=120),
    "lin365":     _cfg(calib="linear",   calwin=365),
    "iso365":     _cfg(calib="isotonic", calwin=365),
    "iso180":     _cfg(calib="isotonic", calwin=180),
}


def power_curve(v):
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    ramp = (v >= 3) & (v < 12)
    out[ramp] = ((v[ramp] - 3.0) / 9.0) ** 3
    out[(v >= 12) & (v <= 25)] = 1.0
    return out


def feats_for(pc):
    lags = [f"kgup_wind_lag_{i}d" for i in range(1, 15)]
    if pc == "cube":
        return ["hour", "month"] + [f"wind_power_{c}" for c in CITIES] + [f"wind_{c}" for c in CITIES] + lags
    return ["hour", "month"] + [f"wind_curve_{c}" for c in CITIES] + [f"wind_{c}" for c in CITIES] + lags


def calibrator(kind, p, y):
    if kind == "linear":
        a, b = np.polyfit(p, y, 1)
        return lambda x: a * np.asarray(x) + b
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, y)
    return lambda x: ir.predict(np.asarray(x))


def one(df_feat, day, cfg):
    fc = feats_for(cfg["pc"])
    tr = df_feat.loc[:day - pd.Timedelta(hours=1)].dropna(subset=fc + ["kgup_wind_mw"])
    te = df_feat.loc[day:day + pd.Timedelta(hours=23)]
    if len(tr) < 3000 or len(te) < 24 or te[fc].isna().any(axis=None):
        return None
    if cfg["window"] == "180":
        tr = tr.loc[tr.index >= day - pd.Timedelta(days=180)]
    params = DEEP if cfg["model"] == "deep" else BASE
    cf = cfg["target"] == "cf"

    # kapasite-faktörü: her satırı KENDİ döneminin filo büyüklüğüyle normalize et
    # (wind_cap = nedensel 90g rolling max). Tahmini bugünkü filoyla geri ölçekle.
    capcol = f"wind_cap_{cfg['cap']}"
    if cf:
        y = (tr["kgup_wind_mw"] / tr[capcol]).values
        cap_now = float(df_feat[capcol].asof(day - pd.Timedelta(hours=1)))
    else:
        y = tr["kgup_wind_mw"].values

    m = lgb.LGBMRegressor(**params)
    m.fit(tr[fc], y)
    pred = m.predict(te[fc])
    if cf:
        pred = pred * cap_now

    if cfg["calib"] != "none":
        ctr = tr.tail(int(cfg['calwin']) * 24)
        cp = m.predict(ctr[fc])
        if cf:
            cp = cp * ctr[capcol].values
        g = calibrator(cfg["calib"], cp, ctr["kgup_wind_mw"].values)
        pred = g(pred)

    pred = np.maximum(pred, 0)
    return pd.DataFrame({"pred": pred, "real": te["kgup_wind_mw"].values}, index=te.index)


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
    extra = {f"wind_curve_{c}": power_curve(df_feat[f"wind_{c}"]) for c in CITIES}
    # nedensel filo-kapasite proxy'leri (hepsi geriye-bakan)
    w = df_feat["kgup_wind_mw"]
    r90 = w.rolling(90 * 24, min_periods=30 * 24)
    extra["wind_cap_max90"]   = r90.max()
    extra["wind_cap_q90_90"]  = r90.quantile(0.90)
    extra["wind_cap_q99_90"]  = r90.quantile(0.99)
    extra["wind_cap_q995_90"] = r90.quantile(0.995)
    extra["wind_cap_q99_120"] = w.rolling(120 * 24, min_periods=30 * 24).quantile(0.99)
    extra["wind_cap_emax90"]  = r90.max().ewm(span=30 * 24, min_periods=1).mean()
    df_feat = pd.concat([df_feat, pd.DataFrame(extra, index=df_feat.index)], axis=1)
    for c in [x for x in df_feat.columns if x.startswith("wind_cap_")]:
        df_feat[c] = df_feat[c].ffill().bfill()

    days = pd.date_range(a.start, a.end, freq="D", tz="Europe/Istanbul")
    days = days[days.isin(df_feat.index.normalize().unique())]
    print(f"{len(days)} gün × {len(CONFIGS)} konfig = {len(days)*len(CONFIGS)} görev")

    t0 = time.time()
    tasks = [(name, d) for name in CONFIGS for d in days]
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(one)(df_feat, d, CONFIGS[name]) for name, d in tasks)
    rows = []
    for (name, d), r in zip(tasks, res):
        if r is None:
            continue
        r = r.assign(config=name)
        rows.append(r)
    allh = pd.concat(rows)
    allh.to_csv(OUT)
    print(f"{time.time()-t0:.0f}s -> {OUT.name} {allh.shape}\n")

    # özet tablo
    print(f"{'config':16s} {'WAPE%':>6s} {'MAE':>6s} {'BIAS':>7s}  {'bias<2.4k':>9s} {'bias>7.5k':>9s} {'MAE>7.5k':>8s}")
    base_wape = None
    for name in CONFIGS:
        h = allh[allh.config == name]
        if h.empty:
            print(f"{name:16s}  (boş)"); continue
        p, ar = h["pred"], h["real"]
        e = p - ar
        lo = h[ar < 2444]; hi = h[ar > 7482]
        w = wape(p, ar)
        if name == "0_baseline":
            base_wape = w
        tag = "" if base_wape is None else f"  ({w-base_wape:+.1f})"
        print(f"{name:16s} {w:6.1f} {e.abs().mean():6.0f} {e.mean():+7.0f}  "
              f"{(lo['pred']-lo['real']).mean():+9.0f} {(hi['pred']-hi['real']).mean():+9.0f} "
              f"{(hi['pred']-hi['real']).abs().mean():8.0f}{tag}")
    print("\nay-ay WAPE (baseline -> her konfig), en kötü sapma:")
    bh = allh[allh.config == "0_baseline"].copy(); bh["m"] = bh.index.tz_localize(None).to_period("M")
    bw = bh.groupby("m").apply(lambda g: wape(g["pred"], g["real"]))
    for name in CONFIGS:
        if name == "0_baseline": continue
        ch = allh[allh.config == name].copy(); ch["m"] = ch.index.tz_localize(None).to_period("M")
        cw = ch.groupby("m").apply(lambda g: wape(g["pred"], g["real"]))
        d = (cw - bw)
        print(f"  {name:12s} ort {d.mean():+.2f}  en kötü {d.max():+.2f} ({d.idxmax()})  en iyi {d.min():+.2f} ({d.idxmin()})")


if __name__ == "__main__":
    main()
