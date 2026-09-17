#!/usr/bin/env python3
"""
Deney: LEAR'a Türkiye-özel 5 zengin (lag-güvenli) özellik eklemek 1 yıllık
çapraz doğrulamada işe yarıyor mu?

`run_crossval_ours.py`'nin birebir ayarlarıyla (CW=1092, seçici=AIC, son 365
gün) ama TABAN ve ZENGİN LEAR'ı aynı koşuda üretir — böylece fark yalnızca
özellik setinden gelir. 15 günlük pilot (`run_pilot_rich_lear.py`) zengin
setin MAE'yi ~%15 kötüleştirdiğini gösterdi; bu tam yıl testi onu kesin
sayıya çevirir.

Eklenen 5 özellik (canlı LightGBM ROBUST setinden, hepsi lag-güvenli):
  renewable_pressure_ratio_lag0, net_load_lag0, zero_price_risk_score,
  hydro_pressure_ratio, is_low_price_regime
build_lear_matrix'in 247 kolonuna 5×24 = 120 "gün-içi" kolon eklenir (p=367).

    python experiments/notebooks/07_lago_protocol/run_crossval_ours_rich.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from db.connection import get_db_engine                       # noqa: E402
from src.eval import lago_protocol as lp                      # noqa: E402
from src.features.feature_engineering import build_robust_features  # noqa: E402

OUT_BASE = Path(__file__).parent / "crossval_ours_rich_base_1y.csv"
OUT_RICH = Path(__file__).parent / "crossval_ours_rich_1y.csv"
CW = 1092
SELECTOR = "aic"
N_JOBS = 6

RICH_COLS = [
    "renewable_pressure_ratio_lag0",
    "net_load_lag0",
    "zero_price_risk_score",
    "hydro_pressure_ratio",
    "is_low_price_regime",
]

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


def safe_predict_day(X, P, day, cw, selector):
    """lear_predict_day, ama sayısal çöküşleri (LARS degenerate active set,
    sinh overflow) NaN olarak yut — zengin sette kollinear özellikler bazı
    kalibrasyon pencerelerinde LASSO-LARS yolunu bozuyor."""
    try:
        return lp.lear_predict_day(X, P, day, cw, selector)
    except Exception as e:  # noqa: BLE001
        return np.full(24, np.nan)


def mae_rmae(y, yhat, ynaive):
    m = np.abs(y - yhat)
    return float(np.nanmean(m)), float(np.nanmean(m) / np.nanmean(np.abs(y - ynaive)))


def main() -> None:
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
        raise RuntimeError(f"zengin özellikler eksik: {missing}")

    X_base, P = lp.build_lear_matrix(df.mcp_price_usd, df.load_forecast_mw, df.kgup_total_mw)
    X_rich = pd.concat([X_base] + [daily_block(feat[c], c) for c in RICH_COLS], axis=1)
    print(f"taban p={X_base.shape[1]}  zengin p={X_rich.shape[1]}  (+{X_rich.shape[1]-X_base.shape[1]})")

    test_days = P.index[P.index >= P.index[-1] - pd.Timedelta(days=364)]
    print(f"test: {test_days[0].date()} -> {test_days[-1].date()}  ({len(test_days)} gün)")
    print(f"n_jobs={N_JOBS}  pencere={CW}  seçici={SELECTOR}\n")

    for tag, X, out_path in [("TABAN", X_base, OUT_BASE), ("ZENGİN", X_rich, OUT_RICH)]:
        t0 = time.time()
        preds = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(safe_predict_day)(X, P, day, CW, SELECTOR) for day in test_days)
        out = pd.DataFrame(preds, index=test_days, columns=[f"h{h:02d}" for h in range(24)])
        out.to_csv(out_path)
        n_fail = int(out.isna().all(axis=1).sum())
        print(f"{tag}: {time.time()-t0:.0f}s -> {out_path.name}  {out.shape}  "
              f"(sayısal çöküş: {n_fail} gün)\n")

    # --- özet metrikler (temiz küme: |tahmin| > 500 olan günler hariç) ---
    base = pd.read_csv(OUT_BASE, index_col=0, parse_dates=True)
    rich = pd.read_csv(OUT_RICH, index_col=0, parse_dates=True)
    price_h = daily_block(df.mcp_price_usd, "y")
    price_h.columns = [f"h{h:02d}" for h in range(24)]
    y = price_h.reindex(base.index)

    nv2 = lp.naive_forecast(df.mcp_price_usd, 2)
    nv2_h = daily_block(nv2, "n"); nv2_h.columns = [f"h{h:02d}" for h in range(24)]
    n2 = nv2_h.reindex(base.index)

    def blown(d):
        return d.index[(d.abs() > 500).any(axis=1)]

    bad = blown(base).union(blown(rich))
    keep = base.index.difference(bad)
    print(f"patlama günü: taban={len(blown(base))} zengin={len(blown(rich))}  "
          f"birleşik dışlanan={len(bad)}  ->  temiz {len(keep)} gün")

    yv, n2v = y.loc[keep].to_numpy(), n2.loc[keep].to_numpy()
    for tag, d in [("TABAN LEAR", base), ("ZENGİN LEAR", rich), ("naive-2", n2)]:
        pv = d.loc[keep].to_numpy() if tag != "naive-2" else n2v
        m, r = mae_rmae(yv, pv, n2v)
        rmse = float(np.sqrt(np.nanmean((yv - pv) ** 2)))
        print(f"  {tag:<12}  MAE={m:6.3f}  RMSE={rmse:6.2f}  rMAE(naive-2)={r:5.3f}")

    # DM testi: taban vs zengin (çok değişkenli, günlük 24-vektör)
    ea = (yv - base.loc[keep].to_numpy())
    eb = (yv - rich.loc[keep].to_numpy())
    ok = ~(np.isnan(ea).any(axis=1) | np.isnan(eb).any(axis=1))
    p_bz = lp.dm_test(ea[ok], eb[ok], variant="multivariate", p=1)
    p_zb = lp.dm_test(eb[ok], ea[ok], variant="multivariate", p=1)
    print(f"\nDM ({ok.sum()} ortak gün):")
    print(f"  H0: TABAN  <= ZENGİN  p={p_bz:.4g}  (küçük p -> zengin daha iyi)")
    print(f"  H0: ZENGİN <= TABAN   p={p_zb:.4g}  (küçük p -> taban daha iyi)")


if __name__ == "__main__":
    main()
