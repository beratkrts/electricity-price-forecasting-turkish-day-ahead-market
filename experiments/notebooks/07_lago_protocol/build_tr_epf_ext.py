#!/usr/bin/env python3
"""
Faz A (LAGO_BENCHMARK_PLAN.md): epftoolbox LEAR referansı için genişletilmiş
girdi CSV'si.

Kanonik Lago LEAR yalnız 2 dışsal kullanır: yük tahmini + KGÜP toplamı.
Çıktı `tr_epf_ext.csv`, epftoolbox formatında:
    index      : saatlik, tz-naive (Europe/Istanbul duvar saati; TR 2016'dan
                 beri kalıcı +03, DST yok)
    Price      : GÖP takas fiyatı, USD/MWh
    Exogenous 1: yük tahmini (load-plan / raw_load_forecast_hourly), MW
    Exogenous 2: KGÜP toplamı (kgup / raw_kgup_hourly.total_mw), MW

Kaynaklar:
  · 2021-01-01 -> bugün : proje DB'si (`02`/`03` ile birebir aynı seri)
  · 2019-01-01 -> 2020-12-31 : EPİAŞ Şeffaflık (eptr2), DB'ye YAZILMADAN

728 günlük test (2024-08-30 -> 2026-08-27) + 4 yıllık (1456g) kalibrasyon
penceresi ilk test günü için ~2020-08'e kadar veri ister; 2019 tamponuyla
daha uzun pencereler / daha erken test başlangıcı da mümkün olur.

    python experiments/notebooks/07_lago_protocol/build_tr_epf_ext.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine  # noqa: E402

OUT = Path(__file__).parent / "tr_epf_ext.csv"
EPTR_CACHE = Path(__file__).parent / "_tr_epf_ext_eptr_raw.pkl"
EPTR_START = "2019-01-01"
EPTR_END = "2020-12-31"

DB_SQL = text("""
    SELECT m.ts,
           m.price_usd            AS price,
           l.load_forecast_mw     AS exog1,
           k.total_mw             AS exog2
    FROM raw_mcp_hourly m
    LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
    LEFT JOIN raw_kgup_hourly          k ON m.ts = k.ts
    WHERE m.ts >= '2021-01-01'
    ORDER BY m.ts
""")


def _naive_index(s: pd.Series) -> pd.Series:
    """tz-aware -> tz-naive (duvar saati), saatlik."""
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Europe/Istanbul").tz_localize(None)
    s = pd.Series(s.to_numpy(), index=idx)
    return s[~s.index.duplicated(keep="last")].sort_index()


def from_db() -> pd.DataFrame:
    df = pd.read_sql(DB_SQL, get_db_engine())
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"  DB : {len(df):,} saat  {df.index.min()} -> {df.index.max()}")
    return df[["price", "exog1", "exog2"]].astype(float)


def from_eptr() -> pd.DataFrame:
    if EPTR_CACHE.exists():
        out = pd.read_pickle(EPTR_CACHE)
        print(f"  eptr (önbellek): {len(out):,} saat  {out.index.min()} -> {out.index.max()}")
        return out

    from eptr2 import EPTR2

    env = dotenv_values(ROOT / ".env")
    e = EPTR2(username=env["EPIAS_USERNAME"], password=env["EPIAS_PASSWORD"])

    months = pd.date_range(EPTR_START, EPTR_END, freq="MS")
    price, load, kgup = [], [], []
    for m0 in months:
        m1 = (m0 + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        s0 = m0.strftime("%Y-%m-%d")
        for _ in range(3):
            try:
                dp = e.call("mcp", start_date=s0, end_date=m1)
                dl = e.call("load-plan", start_date=s0, end_date=m1)
                dk = e.call("kgup", start_date=s0, end_date=m1)
                break
            except Exception as ex:  # noqa: BLE001
                print(f"    {s0} retry ({ex})")
                time.sleep(5)
        else:
            raise RuntimeError(f"eptr2 {s0} 3 denemede başarısız")

        price.append(pd.Series(dp["priceUsd"].to_numpy(float),
                               index=pd.to_datetime(dp["date"])))
        load.append(pd.Series(pd.to_numeric(dl["lep"], errors="coerce").to_numpy(),
                              index=pd.to_datetime(dl["date"])))
        kgup.append(pd.Series(pd.to_numeric(dk["toplam"], errors="coerce").to_numpy(),
                              index=pd.to_datetime(dk["date"])))
        print(f"  eptr {s0}: mcp {len(dp)}  load {len(dl)}  kgup {len(dk)}", flush=True)

    out = pd.DataFrame({
        "price": _naive_index(pd.concat(price)),
        "exog1": _naive_index(pd.concat(load)),
        "exog2": _naive_index(pd.concat(kgup)),
    }).astype(float)
    out.to_pickle(EPTR_CACHE)
    print(f"  eptr: {len(out):,} saat  {out.index.min()} -> {out.index.max()}  (önbellek: {EPTR_CACHE.name})")
    return out


def main() -> None:
    db = from_db()
    ext = from_eptr()

    # DB önceliklidir; ext yalnız DB'nin başlamadığı dönemi doldurur
    ext = ext[ext.index < db.index.min()]
    df = pd.concat([ext, db]).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    gaps = df.isna()

    # dışsal boşluk doldurma: önce D-7 (aynı hafta günü, aynı saat) — bir
    # tahmincinin elindeki en iyi vekil; sonra kısa interpolasyon; sonra ffill.
    for c in ("exog1", "exog2"):
        d7 = df[c].shift(168)
        df[c] = df[c].fillna(d7)
        df[c] = df[c].interpolate(limit=6).ffill().bfill()
    df["price"] = df["price"].interpolate(limit=3)

    filled = gaps[gaps.any(axis=1)]
    if len(filled):
        print(f"\n  DOLDURULAN {len(filled)} saat (kaynak veride yoktu):")
        for col in ["price", "exog1", "exog2"]:
            ts = gaps.index[gaps[col]]
            if len(ts):
                spans = ts.to_series().groupby((ts.to_series().diff() != pd.Timedelta(hours=1)).cumsum())
                for _, g in spans:
                    print(f"    {col:6s}  {g.iloc[0]} -> {g.iloc[-1]}  ({len(g)}h)")

    df.index.name = None
    df.columns = ["Price", "Exogenous 1", "Exogenous 2"]
    df.to_csv(OUT)

    print(f"\nyazıldı: {OUT}")
    print(f"  {len(df):,} saat  {df.index.min()} -> {df.index.max()}")
    print(f"  reindex sonrası ham boşluk: price={int(gaps['price'].sum())} "
          f"exog1={int(gaps['exog1'].sum())} exog2={int(gaps['exog2'].sum())}")
    print(f"  kalan NaN: {df.isna().sum().to_dict()}")
    print(f"  Price USD/MWh  ort={df['Price'].mean():.1f}  "
          f"min={df['Price'].min():.1f}  max={df['Price'].max():.1f}")


if __name__ == "__main__":
    main()
