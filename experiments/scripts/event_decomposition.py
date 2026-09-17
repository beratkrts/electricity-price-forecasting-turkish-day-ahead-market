#!/usr/bin/env python3
"""B — olay-çalışması, iki-kontrafaktüel ayrıştırması.

`OLAY_ANALIZI_V2_PLAN.md` §5.2, olay evreni `OLAY_EVRENI.md`.

Her olay için tek-blok "leave-the-event-out" kontrafaktüeli:
  CF2 = standart tahmin (fundamental varyant, gas_cost=tariff)
  CF1 = olayın vurduğu feature grubu olay-ÖNCESİ ort. seviyeye sabitlenmiş tahmin
  toplam etki   = gerçek − CF1     (şok hiç olmasaydı)
  mekanik kanal = CF2 − CF1        (şokun temellere geçmiş kısmı)
  kalıntı kanal = gerçek − CF2     (temellerin ötesi)

Model, olay penceresi + tampon DIŞINDAKİ sansürsüz saatlerle eğitilir; olay
penceresi örneklem dışıdır. Tavan payı yüksek pencerelerde CF alt sınırdır
(ağaç eğitim maksimumunun üstünü tahmin edemez) — çıktıda işaretlenir.

Plasebo: aynı feature grubu, sakin pencerelerde (tavan payı <%5, olaydan uzak)
kendi ön-pencere ortalamasına sabitlenerek → her bileşen için boş dağılım.

Bu turda YALNIZCA yakıt-maliyeti kanalı (gaz tarifesi + Brent). Arz-MW (İran) ve
kur kanalları türetilmiş pay feature'ları yüzünden v2'ye bırakıldı.

    .venv/bin/python experiments/scripts/event_decomposition.py [--placebo N] [--write]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine
from src.crisis.analysis_model import (
    ANALYSIS_PARAMS, TARGET_COL, load_analysis_data, build_analysis_features,
    get_analysis_feature_columns, prepare_model_frame, fit_counterfactual,
    uncensored_mask,
)

TZ = "Europe/Istanbul"
BUFFER_DAYS = 8
MODEL_NAME = "event_decomp_v1"

# Yakıt-maliyeti kanalı: dışsal maliyet girdileri (türetilmiş pay değil).
FUEL_COST_COLS = ["gas_tariff_usd_mwh_lag0", "brent_oil_usd_lag0", "brent_oil_lag_48"]

# Olaylar — OLAY_EVRENI.md §5. channel: hangi temel sabitlenerek CF1 kuruluyor.
#   fuel_cost  — tarife + Brent ön-pencere ort.'ına sabit
#   gas_supply — gaz üretimi "kesinti olmasaydı talep-oranıyla" seviyesine, paylar
#                ve arz-talep türevleri tutarlı yeniden hesap
EVENTS = [
    dict(id="G1", label="2021 Avrupa gaz tırmanışı",     onset="2021-09-01", short=30, long=150, channel="fuel_cost"),
    dict(id="G2", label="Rusya-Ukrayna işgali",           onset="2022-02-24", short=30, long=311, channel="fuel_cost"),
    dict(id="G3", label="Freeport LNG patlaması",          onset="2022-06-08", short=30, long=120, channel="fuel_cost"),
    dict(id="G4", label="Nord Stream akış durması",        onset="2022-09-02", short=30, long=90,  channel="fuel_cost"),
    dict(id="G5", label="2022-23 Avrupa gaz normalleşmesi",onset="2022-10-15", short=45, long=120, channel="fuel_cost"),
    dict(id="G6", label="2026 Hürmüz / hat tehdidi",       onset="2026-02-25", short=30, long=90,  channel="fuel_cost"),
    dict(id="O1", label="ABD-İsrail-İran çatışması",       onset="2025-06-13", short=21, long=60,  channel="fuel_cost"),
    dict(id="S1", label="İran gaz kesintisi",              onset="2022-01-19", short=18, long=45,  channel="gas_supply"),
]

PRE_DAYS = 30  # olay öncesi referans (pin seviyesi buradan)


def _apply_gas_supply_cf(blk: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """CF1: gaz üretimi kesinti öncesi talep-yoğunluğuyla ölçeklenmiş seviyeye
    çekilir (gaz = marjinal/swing kaynak), toplam ve paylar tutarlı güncellenir.
    Yalnızca yukarı düzeltilir — kesinti bastırır, artırmaz."""
    b = blk.copy()
    L0 = float(pre["load_forecast_lag0"].mean())
    rho = float(pre["kgup_gas_lag0"].mean()) / L0 if L0 else 0.0

    load = b["load_forecast_lag0"].astype(float)
    gas_cf = np.maximum(b["kgup_gas_lag0"].astype(float), rho * load)
    delta = gas_cf - b["kgup_gas_lag0"].astype(float)
    total_cf = b["kgup_total_lag0"].astype(float) + delta

    coal = b["kgup_coal_lag0"].astype(float)
    hydro = b["kgup_hydro_lag0"].astype(float)
    wind = b["kgup_wind_lag0"].astype(float)
    solar = b["kgup_solar_lag0"].astype(float)
    tsafe = total_cf.replace(0, np.nan)

    b["kgup_gas_lag0"] = gas_cf
    b["kgup_total_lag0"] = total_cf
    b["gas_share_lag0"] = (gas_cf / tsafe).astype(float)
    b["coal_share_lag0"] = (coal / tsafe).astype(float)
    b["thermal_share_lag0"] = ((gas_cf + coal) / tsafe).astype(float)
    b["renewable_share_lag0"] = ((wind + solar + hydro) / tsafe).astype(float)
    b["hydro_share_lag0"] = (hydro / tsafe).astype(float)
    b["supply_demand_gap_lag0"] = load - total_cf
    lsafe = load.replace(0, np.nan)
    b["reserve_margin_lag0"] = ((total_cf - load) / lsafe).astype(float)

    for L in (24, 48, 168):
        gc, lc = f"kgup_gas_lag_{L}", f"load_lag_{L}"
        kc = f"kgup_lag_{L}"
        if gc in b.columns and lc in b.columns:
            g = np.maximum(b[gc].astype(float), rho * b[lc].astype(float))
            d = g - b[gc].astype(float)
            b[gc] = g
            if kc in b.columns:
                b[kc] = b[kc].astype(float) + d
    return b


def build_frame():
    df = load_analysis_data()
    feat = build_analysis_features(df)
    cols = get_analysis_feature_columns("fundamental", df=feat, fuel_dynamics=None, gas_cost="tariff")
    model_df = prepare_model_frame(feat, cols)
    return model_df, cols


def one_counterfactual(model_df, cols, s, e, channel="fuel_cost"):
    """Tek blok: [s−buf, e+buf] dışındaki sansürsüz saatlerle eğit, [s,e] için tahmin.
    CF2 (düz) ve CF1 (kanala göre sabitlenmiş temel) döndür."""
    idx = model_df.index
    buf = pd.Timedelta(days=BUFFER_DAYS)
    in_block = (idx >= s) & (idx < e)
    outside = ((idx < s - buf) | (idx >= e + buf)) & uncensored_mask(model_df).values
    if outside.sum() < 5000 or in_block.sum() < 24:
        return None

    fc = fit_counterfactual(model_df.loc[outside], cols, params=ANALYSIS_PARAMS)

    blk = model_df.loc[in_block].copy()
    cf2 = fc.predict(blk)

    pre = model_df.loc[(idx >= s - pd.Timedelta(days=PRE_DAYS)) & (idx < s)]
    if channel == "gas_supply":
        blk1 = _apply_gas_supply_cf(blk, pre) if len(pre) else blk.copy()
    else:
        blk1 = blk.copy()
        for c in FUEL_COST_COLS:
            if c in blk1.columns and len(pre):
                blk1[c] = float(pre[c].mean())
    cf1 = fc.predict(blk1)

    out = pd.DataFrame({
        "actual": blk[TARGET_COL].astype(float).values,
        "cf2": cf2, "cf1": cf1,
        "at_cap": model_df.loc[in_block, "at_cap"].astype(bool).values,
    }, index=blk.index)
    return out


def decompose(out: pd.DataFrame) -> dict:
    a, c1, c2 = float(out.actual.mean()), float(out.cf1.mean()), float(out.cf2.mean())
    r = lambda x, n=1: float(round(x, n))
    return dict(
        actual=r(a), cf1=r(c1), cf2=r(c2),
        total=r(a - c1), mechanical=r(c2 - c1), residual=r(a - c2),
        at_cap_share=r(float(out.at_cap.mean()), 3), n_hours=int(len(out)),
        lower_bound=bool(out.at_cap.mean() > 0.40),
    )


def placebo_null(model_df, cols, channel, win_days, n, rng):
    """Sakin pencerelerde ayrıştırma bileşenlerinin boş dağılımı (kanala göre)."""
    idx = model_df.index
    daily_cap = model_df["at_cap"].astype(float).resample("D").mean()
    ev_dates = [pd.Timestamp(x["onset"], tz=TZ) for x in EVENTS]
    lo = idx.min() + pd.Timedelta(days=PRE_DAYS + 10)
    hi = idx.max() - pd.Timedelta(days=win_days + 10)
    cal_ok = []
    for d, cap in daily_cap.items():
        if pd.isna(cap) or cap > 0.05 or d < lo or d > hi:
            continue
        if any(abs((d - ed).days) < 45 for ed in ev_dates):
            continue
        cal_ok.append(d)
    picks = rng.choice(len(cal_ok), size=min(n, len(cal_ok)), replace=False)

    rows = []
    for i in picks:
        s = cal_ok[i]
        e = s + pd.Timedelta(days=win_days)
        out = one_counterfactual(model_df, cols, s, e, channel)
        if out is None:
            continue
        rows.append(decompose(out))
    return pd.DataFrame(rows)


def pct_rank(val, dist):
    dist = np.asarray(dist, float)
    return float(round(100.0 * (dist < val).mean(), 0))


CACHE = Path(__file__).resolve().parents[2] / "experiments/notebooks/05_crisis_analysis/_placebo_nulls.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", type=int, default=80)
    ap.add_argument("--refresh-placebo", action="store_true", help="önbelleği yok say, yeniden hesapla")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    print("veri + feature yükleniyor…")
    model_df, cols = build_frame()
    print(f"  {len(model_df)} saat, {len(cols)} feature, "
          f"{model_df.index.min().date()} → {model_df.index.max().date()}")

    # plasebo boş dağılımları — (kanal, pencere-uzunluğu) çiftine göre. Pahalı
    # (kanal başına ~4×80 fit); önbelleğe alınır, eksik anahtarlar tamamlanır.
    # Eski önbellek {wd: df} formatındaysa fuel_cost'a taşınır.
    BUCKETS = [30, 60, 120, 300]
    NEEDED = [("fuel_cost", wd) for wd in BUCKETS] + [("gas_supply", wd) for wd in (30, 60)]
    nulls = {}
    if CACHE.exists() and not args.refresh_placebo:
        cached = pd.read_pickle(CACHE)
        for k, v in cached.items():
            nulls[("fuel_cost", k) if isinstance(k, int) else k] = v
        print(f"\nplasebo önbellekten: {len(nulls)} anahtar")
    for key in NEEDED:
        if key in nulls:
            continue
        ch, wd = key
        print(f"\nplasebo boş dağılım (n={args.placebo}, {ch}, {wd}g pencere)…")
        nulls[key] = placebo_null(model_df, cols, ch, wd, args.placebo, rng)
    pd.to_pickle(nulls, CACHE)
    for key in NEEDED:
        nd, (ch, wd) = nulls[key], key
        for k in ("total", "mechanical", "residual"):
            v = nd[k]
            print(f"  {ch:10s} {wd:3d}g {k:11s}: p5={v.quantile(.05):+6.1f}  "
                  f"p50={v.median():+6.1f}  p95={v.quantile(.95):+6.1f}  (n={len(nd)})")

    def null_for(channel, wd):
        avail = [b for (c, b) in nulls if c == channel]
        return nulls[(channel, min(avail, key=lambda b: abs(b - wd)))]

    results = []
    for ev in EVENTS:
        s = pd.Timestamp(ev["onset"], tz=TZ)
        for horizon, wd in (("kısa", ev["short"]), ("uzun", ev["long"])):
            e = s + pd.Timedelta(days=wd)
            out = one_counterfactual(model_df, cols, s, e, ev["channel"])
            if out is None:
                print(f"  {ev['id']} {horizon}: atlandı (yetersiz veri)")
                continue
            d = decompose(out)
            nd = null_for(ev["channel"], wd)
            d.update(id=ev["id"], label=ev["label"], horizon=horizon,
                     onset=ev["onset"], win_days=wd, channel=ev["channel"],
                     total_pct=pct_rank(d["total"], nd["total"]),
                     mech_pct=pct_rank(d["mechanical"], nd["mechanical"]),
                     resid_pct=pct_rank(d["residual"], nd["residual"]))
            results.append(d)

    res = pd.DataFrame(results)
    cols_show = ["id", "channel", "horizon", "win_days", "actual", "cf1", "cf2",
                 "total", "mechanical", "residual", "at_cap_share", "lower_bound",
                 "total_pct", "mech_pct", "resid_pct"]
    pd.set_option("display.width", 220)
    print("\n=== AYRIŞTIRMA (USD/MWh; *_pct = plasebo dağılımındaki yüzdelik) ===")
    print(res[cols_show].to_string(index=False))

    print("\nOkuma: total = gerçek−CF1 (şok olmasaydı). mechanical = CF2−CF1 "
          "(temellere/tarifeye geçen). residual = gerçek−CF2 (ötesi).")
    print("lower_bound=True → tavan payı >%40, total ve residual alt sınır.")

    if args.write:
        eng = get_db_engine()
        with eng.begin() as c:
            c.execute(text("""
                CREATE TABLE IF NOT EXISTS gold.event_decomposition_v2 (
                    event_id text, horizon text, onset date, win_days int,
                    actual_usd numeric, cf1_usd numeric, cf2_usd numeric,
                    total_usd numeric, mechanical_usd numeric, residual_usd numeric,
                    at_cap_share numeric, is_lower_bound bool,
                    total_pctile numeric, mech_pctile numeric, resid_pctile numeric,
                    n_hours int, model_name text, created_at timestamptz DEFAULT now(),
                    PRIMARY KEY (event_id, horizon, model_name)
                )"""))
            c.execute(text("ALTER TABLE gold.event_decomposition_v2 ADD COLUMN IF NOT EXISTS channel text"))
            c.execute(text("DELETE FROM gold.event_decomposition_v2 WHERE model_name=:m"),
                      {"m": MODEL_NAME})
            for r in results:
                c.execute(text("""
                    INSERT INTO gold.event_decomposition_v2
                    (event_id,channel,horizon,onset,win_days,actual_usd,cf1_usd,cf2_usd,
                     total_usd,mechanical_usd,residual_usd,at_cap_share,is_lower_bound,
                     total_pctile,mech_pctile,resid_pctile,n_hours,model_name)
                    VALUES (:id,:channel,:horizon,:onset,:win_days,:actual,:cf1,:cf2,
                     :total,:mechanical,:residual,:at_cap_share,:lower_bound,
                     :total_pct,:mech_pct,:resid_pct,:n_hours,:mn)"""),
                    {**r, "mn": MODEL_NAME})
        print(f"\n→ gold.event_decomposition_v2 ({len(results)} satır, {MODEL_NAME})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
