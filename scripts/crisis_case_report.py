#!/usr/bin/env python3
"""
Vaka raporu — `gold.crisis_counterfactual`'ı okur, bir takvim penceresini
yıllar arasında karşılaştırır.

    python scripts/crisis_case_report.py --window 01-15 02-10

Kontrafaktüeli ÜRETMEZ, sadece okur (üreten: scripts/build_analysis_model.py).
Ayrım kasıtlı: üretim 14 dakika, vaka analizi saniyeler — ve adım 7 boyunca
pencere/varyant defalarca değişecek.

Neden aynı takvim penceresi: mevsimsellik sabit tutulmadan "2022 anormaldi"
denemez. CRISIS_CASE_IRAN_2022.md §6 aynı karşılaştırmayı ham fiyatla yapmıştı;
buradaki fark, artık fiyatın yerine KALINTI karşılaştırılıyor — yani hava,
talep ve yakıt maliyeti zaten arındırılmış durumda.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text

from db.connection import get_db_engine

DEFAULT_MODEL = "crisis_cf_v2"


def load_window(years, md_start: str, md_end: str, variant: str, model: str) -> pd.DataFrame:
    """Her yıl için aynı ay-gün penceresini çeker + mekanizma sütunları."""
    engine = get_db_engine()
    sql = text("""
        SELECT cf.ts, cf.actual_usd, cf.counterfactual_usd, cf.residual_usd,
               cf.at_cap, cf.is_lower_bound,
               k.natural_gas_mw AS gas_mw,
               k.dammed_hydro_mw + k.river_hydro_mw AS hydro_mw,
               k.total_mw AS kgup_total_mw,
               l.load_forecast_mw,
               w.turkey_weighted_temperature_c AS temp_c,
               cap.cap_try
        FROM gold.crisis_counterfactual cf
        LEFT JOIN raw_kgup_hourly k ON cf.ts = k.ts
        LEFT JOIN raw_load_forecast_hourly l ON cf.ts = l.ts
        LEFT JOIN raw_weather_hourly w ON cf.ts = w.ts
        LEFT JOIN silver.mcp_with_cap cap ON cf.ts = cap.ts
        WHERE cf.variant = :variant AND cf.model_name = :model
          AND cf.ts >= :start AND cf.ts < :end
        ORDER BY cf.ts;
    """)

    frames = []
    with engine.connect() as conn:
        for year in years:
            start = pd.Timestamp(f"{year}-{md_start} 00:00", tz="Europe/Istanbul")
            end = pd.Timestamp(f"{year}-{md_end} 00:00", tz="Europe/Istanbul") + pd.Timedelta(days=1)
            df = pd.read_sql(sql, conn, params={
                "variant": variant, "model": model,
                "start": start.to_pydatetime(), "end": end.to_pydatetime(),
            })
            if df.empty:
                continue
            df["year"] = year
            frames.append(df)

    if not frames:
        raise SystemExit("Bu pencere için kontrafaktüel yok. Önce build_analysis_model.py --write koş.")
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert("Europe/Istanbul")
    for c in ["actual_usd", "counterfactual_usd", "residual_usd"]:
        out[c] = out[c].astype(float)
    return out


def null_band(variant: str, model: str, q_lo: float = 0.05, q_hi: float = 0.95):
    """Kalıntının boş dağılımı — SADECE sansürsüz saatlerden, tüm seri üzerinden."""
    engine = get_db_engine()
    sql = text("""
        SELECT percentile_cont(:q_lo) WITHIN GROUP (ORDER BY residual_usd) AS lo,
               percentile_cont(:q_hi) WITHIN GROUP (ORDER BY residual_usd) AS hi
        FROM gold.crisis_counterfactual
        WHERE variant = :variant AND model_name = :model AND NOT at_cap;
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"variant": variant, "model": model,
                                 "q_lo": q_lo, "q_hi": q_hi}).fetchone()
    return float(row.lo), float(row.hi)


def summarise(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    g = df.groupby("year")
    out = pd.DataFrame({
        "saat": g.size(),
        "gercek_$": g.actual_usd.mean(),
        "kontraf_$": g.counterfactual_usd.mean(),
        "kalinti_$": g.residual_usd.mean(),
        "kalinti_med": g.residual_usd.median(),
        "tavan_saat": g.at_cap.sum(),
        "gaz_mw": g.gas_mw.mean(),
        "hidro_mw": g.hydro_mw.mean(),
        "yuk_mw": g.load_forecast_mw.mean(),
        "sicaklik_c": g.temp_c.mean(),
    })
    out["tavan_%"] = 100 * out["tavan_saat"] / out["saat"]
    # Bandın dışında kalan saat oranı: "anormal" iddiasının tek dürüst ölçüsü.
    out["band_ustu_%"] = 100 * g.residual_usd.apply(lambda s: (s > hi).mean())
    out["band_alti_%"] = 100 * g.residual_usd.apply(lambda s: (s < lo).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description="Kriz vaka raporu — yıllar arası aynı takvim penceresi")
    ap.add_argument("--window", nargs=2, metavar=("MM-DD", "MM-DD"), default=["01-15", "02-10"])
    ap.add_argument("--years", nargs="*", type=int, default=[2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--variant", default="fundamental", choices=["fundamental", "autoregressive"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--focus", type=int, default=2022, help="Günlük dökümü basılacak yıl")
    ap.add_argument("--against", type=int, default=2024, help="Karşılaştırma yılı")
    args = ap.parse_args()

    md_start, md_end = args.window
    df = load_window(args.years, md_start, md_end, args.variant, args.model)
    lo, hi = null_band(args.variant, args.model)

    print(f"\n{'='*92}")
    print(f"VAKA RAPORU — pencere {md_start} → {md_end}, variant={args.variant}, model={args.model}")
    print(f"{'='*92}")
    print(f"Kalıntının boş bandı (tüm seri, sansürsüz saatler): p5=${lo:+.1f}  p95=${hi:+.1f}")
    print("Bir yılın penceresi 'anormal' sayılabilmesi için band_ustu_% ~%5'in belirgin üstünde olmalı.\n")

    summary = summarise(df, lo, hi)
    cols = ["saat", "gercek_$", "kontraf_$", "kalinti_$", "kalinti_med",
            "tavan_saat", "tavan_%", "band_ustu_%", "band_alti_%",
            "gaz_mw", "hidro_mw", "yuk_mw", "sicaklik_c"]
    print(summary[cols].round(1).to_string())
    print("\n† tavan_saat > 0 olan yıllarda kalıntı ALT SINIRDIR — model tavan üstü tahmin edemez,")
    print("  yani o yılların gerçek anormalliği tablodakinden BÜYÜKTÜR.")

    a, b = args.focus, args.against
    if a in summary.index and b in summary.index:
        ra, rb = summary.loc[a], summary.loc[b]
        print(f"\n--- {a} vs {b}, aynı takvim penceresi ---")
        print(f"Gerçek fiyat    : ${ra['gercek_$']:.1f}  vs  ${rb['gercek_$']:.1f}")
        print(f"Kontrafaktüel   : ${ra['kontraf_$']:.1f}  vs  ${rb['kontraf_$']:.1f}")
        print(f"Kalıntı (ort.)  : ${ra['kalinti_$']:+.1f}  vs  ${rb['kalinti_$']:+.1f}"
              f"   → fark ${ra['kalinti_$'] - rb['kalinti_$']:+.1f}/MWh")
        print(f"Band üstü saat  : %{ra['band_ustu_%']:.1f}  vs  %{rb['band_ustu_%']:.1f}   (boş beklenti %5)")
        print(f"Tavanda         : {int(ra['tavan_saat'])} saat (%{ra['tavan_%']:.1f})  vs  "
              f"{int(rb['tavan_saat'])} saat (%{rb['tavan_%']:.1f})")
        gas_delta = 100 * (ra["gaz_mw"] - rb["gaz_mw"]) / rb["gaz_mw"]
        print(f"Gazdan üretim   : {ra['gaz_mw']:.0f} MW  vs  {rb['gaz_mw']:.0f} MW  ({gas_delta:+.0f}%)")
        print(f"Sıcaklık        : {ra['sicaklik_c']:.1f} °C  vs  {rb['sicaklik_c']:.1f} °C")

    print(f"\n--- {a} günlük dökümü ---")
    d = df[df.year == a].set_index("ts")
    daily = d.groupby(d.index.date).agg(
        gercek=("actual_usd", "mean"), kontraf=("counterfactual_usd", "mean"),
        kalinti=("residual_usd", "mean"), tavan=("at_cap", "sum"),
        gaz_mw=("gas_mw", "mean"), sicaklik=("temp_c", "mean"),
    )
    daily["band_ustu"] = d.groupby(d.index.date).residual_usd.apply(lambda s: int((s > hi).sum()))
    print(daily.round(1).to_string())


if __name__ == "__main__":
    main()
