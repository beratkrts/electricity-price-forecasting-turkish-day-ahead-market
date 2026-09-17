#!/usr/bin/env python3
"""Faz 0 — TTF (Avrupa gaz) kapsam ve gecikme sondası.

İki soru:
  A. TTF bilinen olayları tarihliyor mu? Hangilerini KAÇIRIYOR?
  B. BOTAŞ elektrik gaz tarifesi TTF'yi hangi gecikmeyle izliyor?

(B) doğrudan "idari vana" tezinin testi: eşzamanlı korelasyon sıfıra yakınsa ve
gecikmeli korelasyon güçlüyse, Avrupa gaz şoku Türkiye fiyatına anında değil,
idari bir karar aracılığıyla ve gecikmeli geçiyor demektir.

Tarife TL cinsinden ve TL enflasyon taşıyor; TTF ile aynı birimde olması için
USD'ye çevriliyor (raw_macro_daily.usd_try, günlük ffill).

    .venv/bin/python experiments/scripts/probe_ttf_tariff.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

TTF_TICKER = "TTF=F"          # ICE Dutch TTF — Avrupa gaz referansı
START, END = "2021-01-01", "2026-09-02"

EVENTS = [
    ("Rusya-Ukrayna başlangıcı", "2022-02-24"),
    ("Gazprom akışı kesti",      "2022-09-01"),
    ("İran gaz kesintisi (TR)",  "2022-01-18"),
    ("İsrail-İran çatışması",    "2025-06-13"),
]


def load_ttf() -> pd.Series:
    s = yf.Ticker(TTF_TICKER).history(start=START, end=END, auto_adjust=False)["Close"]
    if s.empty:
        raise RuntimeError(f"{TTF_TICKER} boş döndü — sembol değişmiş olabilir")
    s.index = s.index.tz_localize(None).normalize()
    return s


def main() -> int:
    ttf = load_ttf()
    print(f"TTF: {len(ttf)} gün, {ttf.index.min().date()} → {ttf.index.max().date()}\n")

    print("=== A. TTF olayları tarihliyor mu? (±30 gün) ===")
    for name, d in EVENTS:
        d = pd.Timestamp(d)
        pre = ttf.loc[d - pd.Timedelta(days=30):d - pd.Timedelta(days=1)]
        post = ttf.loc[d:d + pd.Timedelta(days=30)]
        if not len(pre) or not len(post):
            continue
        print(f"  {name:26s} önce={pre.mean():7.1f}  sonra={post.mean():7.1f}  "
              f"ort {(post.mean()/pre.mean()-1)*100:+6.1f}%  zirve {(post.max()/pre.mean()-1)*100:+6.1f}%")

    eng = get_db_engine()
    tar = pd.read_sql(text("SELECT effective_from, price_try_1000m3 "
                           "FROM silver.gas_tariff_electricity ORDER BY 1"), eng)
    fx = pd.read_sql(text("SELECT entry_date, usd_try FROM raw_macro_daily "
                          "WHERE usd_try IS NOT NULL ORDER BY 1"), eng)
    tar["effective_from"] = pd.to_datetime(tar.effective_from)
    fx["entry_date"] = pd.to_datetime(fx.entry_date)
    fx_d = fx.set_index("entry_date").usd_try.astype(float).resample("D").ffill()
    tar_d = tar.set_index("effective_from").price_try_1000m3.astype(float).resample("D").ffill()
    idx = tar_d.index.intersection(fx_d.index)
    j = pd.DataFrame({"tarife_usd": tar_d.loc[idx] / fx_d.loc[idx],
                      "ttf": ttf.resample("D").ffill()}).dropna()

    print(f"\n=== B. Tarife(USD/1000m³) ↔ TTF(EUR/MWh) — {len(j)} gün ===")
    m = j.resample("MS").mean().pct_change().dropna()
    cors = {lag: m["ttf"].corr(m["tarife_usd"].shift(-lag)) for lag in range(7)}
    best = max(cors, key=lambda k: abs(cors[k]))
    print(f"  aylık % değişim, {len(m)} ay:")
    for lag, c in cors.items():
        print(f"    TTF_t → tarife_t+{lag} ay   {c:+.3f}" + ("   ← en güçlü" if lag == best else ""))

    lv = j.resample("MS").mean()
    print(f"\n  seviye korelasyonu (eşzamanlı): {lv['ttf'].corr(lv['tarife_usd']):+.3f}")
    for (a, b), lab in [(("2021-01", "2022-12"), "2021-2022 (kriz)"),
                        (("2023-01", "2026-08"), "2023-2026 (normalleşme)")]:
        w = lv.loc[a:b]
        print(f"    {lab:24s} {w['ttf'].corr(w['tarife_usd']):+.3f}   ({len(w)} ay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
