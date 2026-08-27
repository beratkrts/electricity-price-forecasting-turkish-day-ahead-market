#!/usr/bin/env python3
"""
Pilot: LEAR'a Türkiye-özel zengin özellikler eklemek işe yarıyor mu, ne kadar
zaman alıyor — küçük bir gün alt kümesinde sağlamlık + zamanlama testi.

Eklenen 5 özellik (canlı LightGBM'in ROBUST setinden, hepsi lag-güvenli):
  renewable_pressure_ratio_lag0, net_load_lag0, zero_price_risk_score,
  hydro_pressure_ratio, is_low_price_regime

`build_lear_matrix`'in 247 kolonuna bunlar "gün-içi" (lag0) blok olarak
ekleniyor (x1/x2'nin "current day" bloğuyla aynı desen) — zaten lag'lenmiş
özellikler oldukları için ayrıca lag1/lag7 eklenmiyor, p şişmiyor.

    python experiments/notebooks/07_lago_protocol/run_pilot_rich_lear.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from db.connection import get_db_engine
from src.eval import lago_protocol as lp
from src.features.feature_engineering import build_robust_features

RICH_COLS = [
    "renewable_pressure_ratio_lag0",
    "net_load_lag0",
    "zero_price_risk_score",
    "hydro_pressure_ratio",
    "is_low_price_regime",
]

# head_to_head.py'deki MASTER_SQL'in küçültülmüş hali — sadece RICH_COLS +
# build_lear_matrix için gereken kolonlar
MASTER_SQL = """
SELECT m.ts,
       m.price_usd AS mcp_price_usd,
       l.load_forecast_mw,
       k.total_mw AS kgup_total_mw,
       k.wind_mw AS kgup_wind_mw, k.solar_mw AS kgup_solar_mw,
       k.dammed_hydro_mw + k.river_hydro_mw AS kgup_hydro_mw,
       pf.predicted_load_lag0, pf.predicted_solar_lag0, pf.predicted_wind_lag0
FROM raw_mcp_hourly m
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
LEFT JOIN gold.kgup_load_pre_forecasts pf ON m.ts = pf.target_ts
WHERE m.ts >= '2021-01-01'
ORDER BY m.ts;
"""


def daily_block(s: pd.Series, name: str) -> pd.DataFrame:
    w = s.to_frame(name).copy()
    w["_d"] = w.index.normalize()
    w["_h"] = w.index.hour
    m = w.pivot_table(index="_d", columns="_h", values=name, aggfunc="first")
    m.columns = [f"{name}_h{h:02d}" for h in m.columns]
    return m


def main():
    eng = get_db_engine()
    df = pd.read_sql(text(MASTER_SQL), eng)
    ts = pd.to_datetime(df["ts"])
    df["ts"] = ts.dt.tz_convert("Europe/Istanbul") if ts.dt.tz else ts.dt.tz_localize("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    df["load_forecast_mw"] = df["load_forecast_mw"].interpolate(limit=6).ffill().bfill()
    df["kgup_total_mw"] = df["kgup_total_mw"].interpolate(limit=6).ffill().bfill()
    print(f"veri: {len(df):,} saat  {df.index.min().date()} -> {df.index.max().date()}")

    feat = build_robust_features(df)
    missing = [c for c in RICH_COLS if c not in feat.columns]
    if missing:
        raise RuntimeError(f"beklenen zengin özellikler eksik: {missing}")

    X_base, P = lp.build_lear_matrix(df.mcp_price_usd, df.load_forecast_mw, df.kgup_total_mw)
    rich_blocks = [daily_block(feat[c], c) for c in RICH_COLS]
    X_rich = pd.concat([X_base] + rich_blocks, axis=1)
    print(f"taban p={X_base.shape[1]}  zengin p={X_rich.shape[1]}  (+{X_rich.shape[1] - X_base.shape[1]})")

    last_day = P.index.max()
    pilot_days = P.index[(P.index >= last_day - pd.Timedelta(days=25)) &
                          (P.index <= last_day - pd.Timedelta(days=10))][:15]
    print(f"pilot: {len(pilot_days)} gün, {pilot_days[0].date()} -> {pilot_days[-1].date()}")

    CW = 1095
    t0 = time.time()
    base_preds, rich_preds = {}, {}
    for i, day in enumerate(pilot_days, 1):
        t_day = time.time()
        base_preds[day] = lp.lear_predict_day(X_base, P, day, CW)
        rich_preds[day] = lp.lear_predict_day(X_rich, P, day, CW)
        print(f"  {i}/{len(pilot_days)}  {day.date()}  ({time.time() - t_day:.1f}s)", flush=True)
    dt = time.time() - t0
    print(f"\n{dt:.1f}s toplam  ->  gün başına ~{dt / len(pilot_days):.2f}s (taban+zengin birlikte, pencere={CW})")

    actual = P.reindex(pilot_days)

    def mae_of(preds):
        e = [np.abs(preds[d] - actual.loc[d].to_numpy()) for d in pilot_days]
        return np.nanmean(np.concatenate(e))

    print(f"pilot MAE  taban={mae_of(base_preds):.2f}  zengin={mae_of(rich_preds):.2f}")


if __name__ == "__main__":
    main()
