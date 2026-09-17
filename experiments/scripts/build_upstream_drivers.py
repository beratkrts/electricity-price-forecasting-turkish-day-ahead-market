#!/usr/bin/env python3
"""silver.upstream_drivers — dışsal maliyet çıpaları (günlük).

C zincirinin (OLAY_ANALIZI_V2_PLAN.md §5.3) yukarı-akış girdileri:
  TTF   — Avrupa gaz referansı (yfinance TTF=F, EUR/MWh)
  Brent — ham petrol (yfinance BZ=F, USD/bbl)
  usd_try, eur_try — DB'deki raw_macro_daily'den (zaten çekiliyor)

TTF olay DEDEKTÖRÜ değil (Türkiye'ye özgü olaylar TTF'de yok, FAZ0_BULGULAR.md),
dışsal maliyet çıpası. Boşluklar ffill (hafta sonu / tatil).

    .venv/bin/python experiments/scripts/build_upstream_drivers.py [--write]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

START = "2020-06-01"
DDL = """
CREATE SCHEMA IF NOT EXISTS silver;
CREATE TABLE IF NOT EXISTS silver.upstream_drivers (
    d            date PRIMARY KEY,
    ttf_eur_mwh  numeric,
    brent_usd_bbl numeric,
    usd_try      numeric,
    eur_try      numeric,
    ttf_is_ffill  boolean DEFAULT false,
    brent_is_ffill boolean DEFAULT false,
    updated_at   timestamptz DEFAULT now()
);
"""


def fetch_yf(ticker: str, col: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=START, progress=False, auto_adjust=True)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    df = s.rename(col).to_frame()
    df.index.name = "d"
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print("yfinance: TTF=F, BZ=F, EURTRY=X …")
    ttf = fetch_yf("TTF=F", "ttf_eur_mwh")
    brent = fetch_yf("BZ=F", "brent_usd_bbl")
    eurtry = fetch_yf("EURTRY=X", "eur_try")

    eng = get_db_engine()
    macro = pd.read_sql(text(
        "SELECT entry_date AS d, usd_try FROM raw_macro_daily ORDER BY entry_date"), eng)
    macro["d"] = pd.to_datetime(macro["d"]).dt.normalize()
    macro = macro.set_index("d").join(eurtry)

    cal = pd.date_range(ttf.index.min(), max(brent.index.max(), macro.index.max()), freq="D")
    df = pd.DataFrame(index=cal)
    df.index.name = "d"
    for src in (ttf, brent):
        df = df.join(src)
    df = df.join(macro)

    for c in ("ttf_eur_mwh", "brent_usd_bbl"):
        df[f"{c.split('_')[0]}_is_ffill"] = df[c].isna()
        df[c] = df[c].ffill()
    df["usd_try"] = df["usd_try"].ffill()
    df["eur_try"] = df["eur_try"].ffill()
    df = df.dropna(subset=["ttf_eur_mwh", "brent_usd_bbl"])

    print(f"\n{len(df)} gün  {df.index.min().date()} → {df.index.max().date()}")
    print(df.tail(3).to_string())
    print("\nTTF aylık ort. (son 12 ay):")
    print(df["ttf_eur_mwh"].resample("MS").mean().tail(12).round(1).to_string())

    if args.write:
        with eng.begin() as c:
            for stmt in DDL.strip().split(";"):
                if stmt.strip():
                    c.execute(text(stmt))
            for d, r in df.iterrows():
                c.execute(text("""
                    INSERT INTO silver.upstream_drivers
                      (d, ttf_eur_mwh, brent_usd_bbl, usd_try, eur_try, ttf_is_ffill, brent_is_ffill)
                    VALUES (:d,:ttf,:brent,:usd,:eur,:tf,:bf)
                    ON CONFLICT (d) DO UPDATE SET
                      ttf_eur_mwh=EXCLUDED.ttf_eur_mwh, brent_usd_bbl=EXCLUDED.brent_usd_bbl,
                      usd_try=EXCLUDED.usd_try, eur_try=EXCLUDED.eur_try,
                      ttf_is_ffill=EXCLUDED.ttf_is_ffill, brent_is_ffill=EXCLUDED.brent_is_ffill,
                      updated_at=now()"""),
                    {"d": d.date(), "ttf": _n(r.ttf_eur_mwh), "brent": _n(r.brent_usd_bbl),
                     "usd": _n(r.usd_try), "eur": _n(r.eur_try),
                     "tf": bool(r.ttf_is_ffill), "bf": bool(r.brent_is_ffill)})
        print(f"\n→ silver.upstream_drivers ({len(df)} satır)")
    return 0


def _n(v):
    return None if pd.isna(v) else float(v)


if __name__ == "__main__":
    raise SystemExit(main())
