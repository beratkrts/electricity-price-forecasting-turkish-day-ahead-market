#!/usr/bin/env python3
"""
Analiz modeli koşucusu — bloklu örneklem-dışı kontrafaktüel üretir.

    python scripts/build_analysis_model.py --variant fundamental --write

CANLIYA ASLA GİRMEZ. Yazdığı tek tablo `gold.crisis_counterfactual`;
`gold.ptf_predictions_daily`'ye dokunmaz. Tasarım gerekçeleri
`src/crisis/analysis_model.py` docstring'inde.

Çıktı: her saat için kontrafaktüel fiyat ($/MWh) ve kalıntı
(gerçek − kontrafaktüel). Kalıntı = "bilinen fundamentaller verildiğinde
fiyatın açıklanamayan kısmı". Etki analizinin (plan §8 adım 7) girdisi budur.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text

from db.connection import get_db_engine
from src.crisis.analysis_model import (
    ANALYSIS_DATA_START,
    TARGET_COL,
    blocked_oof_counterfactual,
    build_analysis_features,
    get_analysis_feature_columns,
    load_analysis_data,
    prepare_model_frame,
    uncensored_mask,
)

logger = logging.getLogger("BuildAnalysisModel")

# v1: yakıt maliyeti geçmişi YOK — gaz maliyetinin fiyata anında geçtiğini
#     varsayıyordu, Eki 2022–Mar 2023'te +$20 sahte kalıntı üretti.
# v2: FUEL_DYNAMICS_COLS eklendi. Varsayılan bu.
FUEL_SET_TO_MODEL = {
    None: "crisis_cf_v1",      # yakıt geçmişi yok
    "long": "crisis_cf_v2",    # mutlak seviyeler — İran sinyalini yutuyor
    "short": "crisis_cf_v3",   # sadece 7 günlük pencere
    "ratio": "crisis_cf_v4",   # uzun pencere, birimsiz sütunlar
}
# GRF yerine BOTAŞ elektrik üretim tarifesi
MODEL_NAME_TARIFF = "crisis_cf_v5"

# Plan §2.4'teki İran gaz kesintisi penceresi. Adım 7'nin ilk vakası, bu yüzden
# koşunun sonunda otomatik raporlanıyor.
IRAN_WINDOW = ("2022-01-15", "2022-02-10")


def create_table():
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.crisis_counterfactual (
                ts                 TIMESTAMPTZ    NOT NULL,
                variant            VARCHAR(20)    NOT NULL,
                model_name         VARCHAR(50)    NOT NULL DEFAULT 'crisis_cf_v2',
                actual_usd         NUMERIC(10,4),
                counterfactual_usd NUMERIC(10,4),
                residual_usd       NUMERIC(10,4),
                at_cap             BOOLEAN,
                is_lower_bound     BOOLEAN,
                fold_id            INT,
                created_at         TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (ts, variant, model_name)
            );
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_crisis_cf_ts ON gold.crisis_counterfactual(ts);"
        ))
        conn.commit()


def write_results(oof: pd.DataFrame, variant: str, model_name: str):
    engine = get_db_engine()
    rows = [
        {
            "ts": ts.to_pydatetime(),
            "variant": variant,
            "model_name": model_name,
            "actual_usd": float(r[TARGET_COL]),
            "counterfactual_usd": float(r["counterfactual_usd"]),
            "residual_usd": float(r["residual_usd"]),
            "at_cap": bool(r["at_cap"]),
            "is_lower_bound": bool(r["is_lower_bound"]),
            "fold_id": int(r["fold_id"]),
        }
        for ts, r in oof.iterrows()
    ]

    upsert = text("""
        INSERT INTO gold.crisis_counterfactual
            (ts, variant, model_name, actual_usd, counterfactual_usd, residual_usd,
             at_cap, is_lower_bound, fold_id)
        VALUES (:ts, :variant, :model_name, :actual_usd, :counterfactual_usd, :residual_usd,
                :at_cap, :is_lower_bound, :fold_id)
        ON CONFLICT (ts, variant, model_name) DO UPDATE SET
            actual_usd         = EXCLUDED.actual_usd,
            counterfactual_usd = EXCLUDED.counterfactual_usd,
            residual_usd       = EXCLUDED.residual_usd,
            at_cap             = EXCLUDED.at_cap,
            is_lower_bound     = EXCLUDED.is_lower_bound,
            fold_id            = EXCLUDED.fold_id,
            created_at         = NOW();
    """)

    with engine.connect() as conn:
        for i in range(0, len(rows), 5000):
            conn.execute(upsert, rows[i:i + 5000])
        conn.commit()
    logger.info("💾 %d satır yazıldı → gold.crisis_counterfactual (variant=%s, model=%s)", len(rows), variant, model_name)


def report(oof: pd.DataFrame, variant: str, model_name: str):
    """Koşunun dürüstlük raporu. Tek metriğe bakılmaz (bkz. METRICS.md)."""
    unc = oof[~oof["at_cap"].astype(bool)]
    cen = oof[oof["at_cap"].astype(bool)]

    print(f"\n{'='*78}\nANALİZ MODELİ — variant={variant}, model={model_name}\n{'='*78}")
    print(f"Kapsam       : {oof.index.min()} → {oof.index.max()}  ({len(oof)} saat, {oof.fold_id.nunique()} blok)")
    print(f"Sansürsüz    : {len(unc)} saat   |   Tavanda: {len(cen)} saat (%{100*len(cen)/len(oof):.1f})")

    print(f"\n--- Örneklem-dışı doğruluk (SADECE sansürsüz saatler — model orada eğitildi) ---")
    print(f"MAE   : ${unc.residual_usd.abs().mean():7.2f}")
    print(f"BIAS  : ${unc.residual_usd.mean():+7.2f}   (gerçek − kontrafaktüel)")
    print(f"Medyan: ${unc.residual_usd.median():+7.2f}")

    yearly = unc.groupby(unc.index.year).agg(
        n=("residual_usd", "size"),
        ort_fiyat=(TARGET_COL, "mean"),
        MAE=("residual_usd", lambda s: s.abs().mean()),
        BIAS=("residual_usd", "mean"),
    )
    print("\n--- Yıllık (sansürsüz) ---")
    print(yearly.round(2).to_string())

    q = unc.residual_usd.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("\n--- Kalıntının boş dağılımı (sansürsüz saatler) ---")
    print("  ".join(f"p{int(k*100)}=${v:+.1f}" for k, v in q.items()))
    print("Bir olayın kalıntısı bu bandın DIŞINDA değilse 'anormal' denemez.")

    daily = oof.groupby(oof.index.date).agg(
        ort_fiyat=(TARGET_COL, "mean"),
        ort_cf=("counterfactual_usd", "mean"),
        ort_kalinti=("residual_usd", "mean"),
        tavan_saat=("at_cap", "sum"),
    )
    print("\n--- En büyük 20 açıklanamayan gün (ort. kalıntı, ↑) ---")
    top = daily.sort_values("ort_kalinti", ascending=False).head(20)
    print(top.round(1).to_string())
    print("† tavan_saat > 0 olan günlerde kalıntı ALT SINIRDIR (model tavan üstü tahmin edemez).")

    print("\n--- En büyük 10 açıklanamayan gün (ort. kalıntı, ↓) ---")
    print(daily.sort_values("ort_kalinti").head(10).round(1).to_string())

    iran = daily.loc[
        (pd.to_datetime(daily.index) >= IRAN_WINDOW[0]) & (pd.to_datetime(daily.index) <= IRAN_WINDOW[1])
    ]
    if len(iran):
        print(f"\n--- İran gaz kesintisi penceresi {IRAN_WINDOW[0]} → {IRAN_WINDOW[1]} ---")
        print(iran.round(1).to_string())
        print(f"\nPencere ortalaması: gerçek ${iran.ort_fiyat.mean():.1f}, "
              f"kontrafaktüel ${iran.ort_cf.mean():.1f}, "
              f"kalıntı ${iran.ort_kalinti.mean():+.1f}/MWh, "
              f"tavanda {int(iran.tavan_saat.sum())} saat")


def main():
    ap = argparse.ArgumentParser(description="Kriz analiz modeli — bloklu OOF kontrafaktüel")
    ap.add_argument("--variant", default="fundamental", choices=["fundamental", "autoregressive", "both"])
    ap.add_argument("--start", default=ANALYSIS_DATA_START)
    ap.add_argument("--block-days", type=int, default=28)
    ap.add_argument("--buffer-days", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="Sonuçları gold.crisis_counterfactual'a yaz")
    ap.add_argument("--gas-cost", default="grf", choices=["grf", "tariff"],
                    help="Yakıt maliyeti kaynağı. 'tariff' = silver.gas_tariff_electricity (v5).")
    ap.add_argument("--fuel-dynamics", default=None, choices=["long", "short", "ratio"],
                    help="Yakıt maliyeti geçmişi seti. Belirtilmezse v1 (yok).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    logger.info("📥 Veri yükleniyor (%s'ten)...", args.start)
    df_raw = load_analysis_data(start=args.start)
    logger.info("   %d saat, tavanda %%%.1f", len(df_raw), 100 * df_raw["at_cap"].mean())

    logger.info("🔧 Feature'lar kuruluyor...")
    df_feat = build_analysis_features(df_raw)

    variants = ["fundamental", "autoregressive"] if args.variant == "both" else [args.variant]
    if args.write:
        create_table()

    fuel_dynamics = args.fuel_dynamics
    model_name = MODEL_NAME_TARIFF if args.gas_cost == "tariff" else FUEL_SET_TO_MODEL[fuel_dynamics]

    for variant in variants:
        feature_cols = get_analysis_feature_columns(
            variant, df_feat, fuel_dynamics=fuel_dynamics, gas_cost=args.gas_cost)
        df_model = prepare_model_frame(df_feat, feature_cols)
        logger.info(
            "🌲 variant=%s | %d feature | %d satır (%d sansürsüz eğitim adayı)",
            variant, len(feature_cols), len(df_model), int(uncensored_mask(df_model).sum()),
        )
        oof = blocked_oof_counterfactual(
            df_model, feature_cols,
            block_days=args.block_days, buffer_days=args.buffer_days,
        )
        report(oof, variant, model_name)
        if args.write:
            write_results(oof, variant, model_name)


if __name__ == "__main__":
    main()
