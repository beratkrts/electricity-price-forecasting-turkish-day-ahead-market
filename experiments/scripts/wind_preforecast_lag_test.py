#!/usr/bin/env python3
"""
WIND_PREFORECAST_INCIDENT_2026-08 hızlı testi.

Soru: rüzgar ön-tahmincisinin `kgup_wind_lag_1d..14d`'ye aşırı yaslanması
(%57.5 önem) çok-günlü rampada modeli geçmiş-ortalamaya çapalıyor. Lag
sayısını azaltmak / kaldırmak / ağırlığını kısmak açığı kapatıyor mu?

Walk-forward Ağu 18-30 2026, her hedef gün için < D verisiyle eğit, D'nin 24
saatini tahmin et. Meteoroloji girdisi = OpenMeteo ARŞİV (reanaliz) — yani
"iyi meteoroloji elimizde olsaydı" senaryosu; sorunun mimaride mi girdide mi
olduğunu izole eder.

Varyantlar:
  base    : mevcut (hour, month, wind_power x3, wind x3, kgup_wind_lag_1..14)
  lag3    : lag 1-3 gün
  lag1    : sadece lag_1d
  nolag   : lag yok (sadece meteoroloji + hour/month)
  ff05    : base + feature_fraction=0.5, bagging
  mono    : base + wind_power/wind hız özelliklerinde monotone_increasing

    .venv/bin/python experiments/scripts/wind_preforecast_lag_test.py
"""
import sys
import time
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

CITY_COORDS = {
    "izmir": {"lat": 38.41, "lon": 27.14},
    "canakkale": {"lat": 40.15, "lon": 26.40},
    "balikesir": {"lat": 39.64, "lon": 27.88},
}
WIND_CACHE = ROOT / "experiments/notebooks/07_lago_protocol/_wind_archive_2023_2026.pkl"
TRAIN_START = "2023-01-01"
ARCHIVE_END = (pd.Timestamp.now(tz="Europe/Istanbul") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
TEST_DAYS = pd.date_range("2026-08-18", "2026-08-29", freq="D", tz="Europe/Istanbul")


def _pull(base, city, c, extra):
    url = (f"{base}?latitude={c['lat']}&longitude={c['lon']}"
           f"&hourly=wind_speed_100m&timezone=Europe%2FIstanbul&{extra}")
    d = httpx.get(url, timeout=90.0).json()["hourly"]
    idx = pd.to_datetime(d["time"]).tz_localize(
        "Europe/Istanbul", ambiguous="infer", nonexistent="shift_forward")
    return pd.DataFrame({f"wind_{city}": d["wind_speed_100m"]}, index=idx)


def fetch_wind_archive():
    if WIND_CACHE.exists():
        return pd.read_pickle(WIND_CACHE)
    arch, fc = None, None
    for city, c in CITY_COORDS.items():
        a = _pull("https://archive-api.open-meteo.com/v1/archive", city, c,
                  f"start_date={TRAIN_START}&end_date={ARCHIVE_END}")
        f = _pull("https://api.open-meteo.com/v1/forecast", city, c,
                  "past_days=10&forecast_days=3")
        arch = a if arch is None else arch.join(a)
        fc = f if fc is None else fc.join(f)
    gap = fc[~fc.index.isin(arch.index)]
    out = pd.concat([arch, gap]).sort_index()
    out.to_pickle(WIND_CACHE)
    return out


def build_feats(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    for city in CITY_COORDS:
        df[f"wind_power_{city}"] = df[f"wind_{city}"] ** 3
    for i in range(1, 15):
        df[f"kgup_wind_lag_{i}d"] = df["kgup_wind_mw"].shift(24 * i)
    return df


METEO = ([f"wind_power_{c}" for c in CITY_COORDS] + [f"wind_{c}" for c in CITY_COORDS])
LAGS = [f"kgup_wind_lag_{i}d" for i in range(1, 15)]
BASE_PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=5,
                   random_state=42, n_jobs=1, verbose=-1)

VARIANTS = {
    "base":  (["hour", "month"] + METEO + LAGS, {}),
    "lag3":  (["hour", "month"] + METEO + LAGS[:3], {}),
    "lag1":  (["hour", "month"] + METEO + LAGS[:1], {}),
    "nolag": (["hour", "month"] + METEO, {}),
    "lagonly": (["hour", "month"] + LAGS, {}),
    "meteoNaN": (["hour", "month"] + METEO + LAGS, {}),  # meteo sütunları NaN'lanır
    "ff05":  (["hour", "month"] + METEO + LAGS,
              dict(feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1)),
}


def mono_constraints(feats):
    return [1 if f in METEO else 0 for f in feats]


def run_variant(name, feats, extra, df_feat):
    rows = []
    for day in TEST_DAYS:
        tr = df_feat.loc[:day - pd.Timedelta(hours=1)].dropna(subset=feats + ["kgup_wind_mw"])
        te = df_feat.loc[day:day + pd.Timedelta(hours=23)].copy()
        if name == "meteoNaN":
            te[METEO] = np.nan  # canlıda ileri rüzgar hızı gelmeyince ne olur
        if len(tr) < 2000 or len(te) < 24:
            continue
        if name != "meteoNaN" and te[feats].isna().any(axis=None):
            continue
        p = dict(BASE_PARAMS, **extra)
        m = lgb.LGBMRegressor(**p)
        m.fit(tr[feats], tr["kgup_wind_mw"])
        pred = np.maximum(m.predict(te[feats]), 0)
        act = te["kgup_wind_mw"].to_numpy()
        rows.append((day.date(), np.mean(np.abs(pred - act)), np.mean(pred - act),
                     pred.mean(), act.mean()))
    r = pd.DataFrame(rows, columns=["gun", "MAE", "BIAS", "pred_mw", "act_mw"]).set_index("gun")
    return name, r


def run_mono(df_feat):
    feats = ["hour", "month"] + METEO + LAGS
    rows = []
    for day in TEST_DAYS:
        tr = df_feat.loc[:day - pd.Timedelta(hours=1)].dropna(subset=feats + ["kgup_wind_mw"])
        te = df_feat.loc[day:day + pd.Timedelta(hours=23)]
        if len(tr) < 2000 or len(te) < 24 or te[feats].isna().any(axis=None):
            continue
        m = lgb.LGBMRegressor(**BASE_PARAMS,
                              monotone_constraints=mono_constraints(feats))
        m.fit(tr[feats], tr["kgup_wind_mw"])
        pred = np.maximum(m.predict(te[feats]), 0)
        act = te["kgup_wind_mw"].to_numpy()
        rows.append((day.date(), np.mean(np.abs(pred - act)), np.mean(pred - act),
                     pred.mean(), act.mean()))
    return "mono", pd.DataFrame(
        rows, columns=["gun", "MAE", "BIAS", "pred_mw", "act_mw"]).set_index("gun")


def main():
    eng = get_db_engine()
    df = pd.read_sql(text(
        "SELECT m.ts, k.wind_mw AS kgup_wind_mw FROM raw_mcp_hourly m "
        "LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts "
        f"WHERE m.ts >= '{TRAIN_START}' ORDER BY m.ts"), eng)
    ts = pd.to_datetime(df["ts"])
    df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    print(f"kgup {df.index.min().date()} -> {df.index.max().date()}  ({len(df):,} saat)")

    wind = fetch_wind_archive()
    print(f"rüzgar arşiv {wind.index.min().date()} -> {wind.index.max().date()}  {wind.shape}")
    df = df.join(wind)
    df_feat = build_feats(df)

    t0 = time.time()
    results = {}
    for name, (feats, extra) in VARIANTS.items():
        n, r = run_variant(name, feats, extra, df_feat)
        results[n] = r
    n, r = run_mono(df_feat)
    results[n] = r
    print(f"{time.time()-t0:.0f}s\n")

    # özet: rampanın kritik günleri
    focus = [pd.Timestamp("2026-08-25").date(), pd.Timestamp("2026-08-26").date(),
             pd.Timestamp("2026-08-27").date(), pd.Timestamp("2026-08-28").date(),
             pd.Timestamp("2026-08-29").date(), pd.Timestamp("2026-08-30").date()]
    print("=== Ağu 25-30 ortalama (rampanın çekirdeği) ===")
    print(f"{'varyant':8s} {'MAE':>7s} {'BIAS':>8s} {'pred_mw':>9s} {'act_mw':>9s}")
    for name, r in results.items():
        sub = r.loc[r.index.isin(focus)]
        print(f"{name:8s} {sub.MAE.mean():7.0f} {sub.BIAS.mean():8.0f} "
              f"{sub.pred_mw.mean():9.0f} {sub.act_mw.mean():9.0f}")

    print("\n=== gün gün BIAS (tahmin - gerçek, MW) ===")
    bias = pd.DataFrame({n: r["BIAS"] for n, r in results.items()})
    print(bias.round(0).to_string())

    print("\n=== gün gün MAE (MW) ===")
    pmw = pd.DataFrame({n: r["pred_mw"] for n, r in results.items()})
    print(pmw.round(0).to_string())
    print("\n=== gün gün pred_mw ile birlikte act_mw ===")
    print(list(results.values())[0]["act_mw"].round(0).to_string())
    mae = pd.DataFrame({n: r["MAE"] for n, r in results.items()})
    print(mae.round(0).to_string())

    out = ROOT / "experiments/notebooks/07_lago_protocol/wind_lag_test_results.csv"
    pd.concat({n: r for n, r in results.items()}, names=["varyant", "gun"]).to_csv(out)
    print(f"\n-> {out.name}")


if __name__ == "__main__":
    main()
