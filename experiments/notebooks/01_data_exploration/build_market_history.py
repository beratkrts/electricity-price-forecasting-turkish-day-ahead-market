#!/usr/bin/env python3
"""
2019-2026 Türkiye elektrik piyasası saatlik tarihçesi — `fiyat_hikayesi_2019_2026.ipynb`
için tek veri kaynağı.

  · 2021-01-01 -> bugün : proje DB'si (raw_* tabloları, zaten ingest edilmiş)
  · 2019-01-01 -> 2020-12-31 : EPİAŞ Şeffaflık (eptr2), DB'ye YAZILMADAN
  · kur/Brent : frankfurter + yfinance (DB macro yalnız 2021+)

Çıktı: `market_history_2019_2026.pkl` — saatlik tz-naive (Europe/Istanbul,
kalıcı +03, DST yok) DataFrame.

    python experiments/notebooks/01_data_exploration/build_market_history.py
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

OUT = Path(__file__).parent / "market_history_2019_2026.pkl"
RAW_CACHE = Path(__file__).parent / "_market_history_eptr_raw.pkl"
EPTR_START, EPTR_END = "2019-01-01", "2020-12-31"

# eptr rt-gen camelCase -> ortak isim
GEN_MAP = {
    "naturalGas": "gas", "dammedHydro": "dammed_hydro", "river": "river_hydro",
    "lignite": "lignite", "importCoal": "import_coal", "blackCoal": "black_coal",
    "asphaltiteCoal": "asphaltite", "wind": "wind", "sun": "solar",
    "geothermal": "geothermal", "fueloil": "fuel_oil", "biomass": "biomass",
    "lng": "lng", "naphta": "naphtha", "wasteheat": "waste_heat",
    "importExport": "import_export", "total": "gen_total",
}
DB_GEN_MAP = {
    "natural_gas_mw": "gas", "dammed_hydro_mw": "dammed_hydro",
    "river_hydro_mw": "river_hydro", "lignite_mw": "lignite",
    "import_coal_mw": "import_coal", "black_coal_mw": "black_coal",
    "asphaltite_coal_mw": "asphaltite", "wind_mw": "wind", "solar_mw": "solar",
    "geothermal_mw": "geothermal", "fuel_oil_mw": "fuel_oil", "biomass_mw": "biomass",
    "lng_mw": "lng", "naphtha_mw": "naphtha", "waste_heat_mw": "waste_heat",
    "import_export_mw": "import_export", "total_mw": "gen_total",
}


def _naive(s):
    idx = pd.DatetimeIndex(pd.to_datetime(s))
    return idx.tz_convert("Europe/Istanbul").tz_localize(None) if idx.tz is not None else idx


def from_db() -> pd.DataFrame:
    eng = get_db_engine()
    gcols = ", ".join(f"g.{c}" for c in DB_GEN_MAP)
    sql = text(f"""
        SELECT m.ts,
               m.price_try, m.price_usd,
               s.system_marginal_price_try AS smp_try,
               c.consumption_mw AS cons_mw,
               {gcols}
        FROM raw_mcp_hourly m
        LEFT JOIN raw_smp_hourly s ON m.ts = s.ts
        LEFT JOIN raw_actual_consumption_hourly c ON m.ts = c.ts
        LEFT JOIN raw_actual_generation_hourly g ON m.ts = g.ts
        WHERE m.ts >= '2021-01-01'
        ORDER BY m.ts
    """)
    df = pd.read_sql(sql, eng).rename(columns=DB_GEN_MAP)
    df.index = _naive(df.pop("ts"))
    print(f"  DB : {len(df):,} saat  {df.index.min()} -> {df.index.max()}")
    return df[~df.index.duplicated(keep="last")].sort_index().astype(float)


def from_eptr() -> pd.DataFrame:
    if RAW_CACHE.exists():
        df = pd.read_pickle(RAW_CACHE)
        print(f"  eptr (önbellek): {len(df):,} saat")
        return df

    from eptr2 import EPTR2
    env = dotenv_values(ROOT / ".env")
    e = EPTR2(username=env["EPIAS_USERNAME"], password=env["EPIAS_PASSWORD"])

    mcp, smp, cons, gen = [], [], [], []
    for m0 in pd.date_range(EPTR_START, EPTR_END, freq="MS"):
        s0 = m0.strftime("%Y-%m-%d")
        m1 = (m0 + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        for _ in range(3):
            try:
                dp = e.call("mcp", start_date=s0, end_date=m1)
                ds = e.call("smp", start_date=s0, end_date=m1)
                dc = e.call("rt-cons", start_date=s0, end_date=m1)
                dg = e.call("rt-gen", start_date=s0, end_date=m1)
                break
            except Exception as ex:  # noqa: BLE001
                print(f"    {s0} retry ({ex})"); time.sleep(5)
        else:
            raise RuntimeError(f"eptr2 {s0} başarısız")
        mcp.append(dp.assign(ts=pd.to_datetime(dp["date"]))[["ts", "price", "priceUsd"]])
        smp.append(ds.assign(ts=pd.to_datetime(ds["date"]))[["ts", "systemMarginalPrice"]])
        cons.append(dc.assign(ts=pd.to_datetime(dc["date"]))[["ts", "consumption"]])
        gg = dg.assign(ts=pd.to_datetime(dg["date"]))
        gen.append(gg[["ts"] + list(GEN_MAP)])
        print(f"  eptr {s0}: {len(dp)} saat", flush=True)

    mcp = pd.concat(mcp).rename(columns={"price": "price_try", "priceUsd": "price_usd"})
    smp = pd.concat(smp).rename(columns={"systemMarginalPrice": "smp_try"})
    cons = pd.concat(cons).rename(columns={"consumption": "cons_mw"})
    gen = pd.concat(gen).rename(columns=GEN_MAP)
    df = mcp.merge(smp, on="ts").merge(cons, on="ts").merge(gen, on="ts")
    df.index = _naive(df.pop("ts"))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_pickle(RAW_CACHE)
    print(f"  eptr: {len(df):,} saat  {df.index.min()} -> {df.index.max()}")
    return df.astype(float)


def macro_2019_2020() -> pd.DataFrame:
    idx = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    try:
        import requests
        r = requests.get("https://api.frankfurter.app/2019-01-01..2020-12-31?from=USD&to=TRY", timeout=30).json()
        usd = pd.Series({pd.Timestamp(k): v["TRY"] for k, v in r["rates"].items()}).reindex(idx).ffill().bfill()
    except Exception as ex:  # noqa: BLE001
        print(f"  frankfurter hata ({ex}) — usdtry NaN"); usd = pd.Series(np.nan, index=idx)
    try:
        import yfinance as yf
        br = yf.download("BZ=F", start="2019-01-01", end="2021-01-05", interval="1d", progress=False)["Close"]
        br = br.reindex(idx).ffill().bfill()
        br = br.iloc[:, 0] if br.ndim > 1 else br
    except Exception as ex:  # noqa: BLE001
        print(f"  yfinance hata ({ex}) — brent NaN"); br = pd.Series(np.nan, index=idx)
    return pd.DataFrame({"usdtry": usd.values, "brent": np.asarray(br).ravel()}, index=idx)


def main() -> None:
    db = from_db()
    ext = from_eptr()
    ext = ext[ext.index < db.index.min()]
    df = pd.concat([ext.reindex(columns=db.columns), db]).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    gaps = df.reindex(full).isna().sum()
    df = df.reindex(full)

    # kur/Brent: DB'de 2021+ (raw_macro yok bu sorguda) -> hepsini ayrı çek
    eng = get_db_engine()
    mac_db = pd.read_sql(text("SELECT entry_date, usd_try, brent_oil_usd FROM raw_macro_daily ORDER BY entry_date"), eng)
    mac_db.index = pd.to_datetime(mac_db.pop("entry_date"))
    mac = pd.concat([macro_2019_2020().rename(columns={"usdtry": "usd_try", "brent": "brent_oil_usd"}),
                     mac_db[["usd_try", "brent_oil_usd"]]])
    mac = mac[~mac.index.duplicated(keep="last")].sort_index()
    mac_h = mac.reindex(pd.date_range(mac.index.min(), df.index.max(), freq="D")).ffill()
    df["usdtry"] = mac_h["usd_try"].reindex(df.index, method="ffill").values
    df["brent"] = mac_h["brent_oil_usd"].reindex(df.index, method="ffill").values

    # türetilenler
    gen_cols = [c for c in df.columns if c in set(GEN_MAP.values()) and c != "gen_total"]
    df["gen_renewable"] = df[["wind", "solar", "dammed_hydro", "river_hydro", "geothermal", "biomass"]].sum(axis=1)
    df["gen_thermal"] = df[["gas", "lignite", "import_coal", "black_coal", "asphaltite", "fuel_oil", "lng", "naphtha", "waste_heat"]].sum(axis=1)
    df["gen_hydro"] = df[["dammed_hydro", "river_hydro"]].sum(axis=1)
    df["renewable_share"] = df["gen_renewable"] / df["gen_total"]
    df["gas_share"] = df["gas"] / df["gen_total"]
    df["hydro_share"] = df["gen_hydro"] / df["gen_total"]
    df["net_load"] = df["cons_mw"] - df[["wind", "solar"]].sum(axis=1)
    df["smp_usd"] = df["smp_try"] / df["usdtry"]

    df.to_pickle(OUT)
    print(f"\nyazıldı: {OUT}")
    print(f"  {len(df):,} saat  {df.index.min().date()} -> {df.index.max().date()}  ({len(df.columns)} kolon)")
    print(f"  reindex boşlukları: {gaps[gaps > 0].to_dict()}")
    print(f"  kalan NaN (kolon başına, >0): { {k: int(v) for k, v in df.isna().sum().items() if v} }")
    print(f"  yıllık ort PTF USD: { df['price_usd'].groupby(df.index.year).mean().round(1).to_dict() }")


if __name__ == "__main__":
    main()
