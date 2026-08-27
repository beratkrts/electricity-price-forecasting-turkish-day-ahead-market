#!/usr/bin/env python3
"""
Lago protokolünün ağır hesabı: LEAR'ı günlük yeniden kalibrasyonla koşturur.

~50 dakika sürüyor (729 test günü × 3 kalibrasyon penceresi × 24 saatlik model).
Sonuç diske yazılır, notebook onu okur — böylece defter saniyeler içinde açılır.

    python experiments/notebooks/07_lago_protocol/run_protocol.py
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from db.connection import get_db_engine          # noqa: E402
from src.eval import lago_protocol as lp         # noqa: E402

CACHE = Path(__file__).parent / "lear_forecasts.csv"
PARTIAL = Path(__file__).parent / "lear_forecasts_partial.csv"
CHECKPOINT_EVERY = 25   # gün
DATA_SQL = """
SELECT m.ts, m.price_usd, l.load_forecast_mw, k.total_mw
FROM raw_mcp_hourly m
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly  k ON m.ts = k.ts
WHERE m.ts >= '2021-01-01'
ORDER BY m.ts
"""


def load_data() -> pd.DataFrame:
    df = pd.read_sql(text(DATA_SQL), get_db_engine())
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    # Gün-öncesi tahminlerdeki tekil boşluklar interpolasyonla kapatılıyor;
    # fiyatta boşluk yok (kontrol edildi).
    df[["load_forecast_mw", "total_mw"]] = (
        df[["load_forecast_mw", "total_mw"]].interpolate(limit=6).ffill().bfill())
    return df


def main() -> None:
    df = load_data()
    X, P = lp.build_lear_matrix(df.price_usd, df.load_forecast_mw, df.total_mw)
    print(f"veri  : {len(df)} saat  {df.index.min().date()} → {df.index.max().date()}")
    print(f"matris: {X.shape[0]} gün × {X.shape[1]} özellik  (makale: 247)")

    # Makale §3.1: test dönemi = son 104 hafta (2 yıl)
    test_days = P.index[P.index >= P.index[-1] - pd.Timedelta(weeks=104)]
    print(f"test  : {test_days[0].date()} → {test_days[-1].date()}  "
          f"({len(test_days)} gün)\n")

    # Ara kayıtlı ve kaldığı yerden devam eden döngü. Tek seferde ~50 dakika;
    # önceki sürüm sadece en sonda yazıyordu, yani çökme ya da kapanma
    # bütün hesabı çöpe atıyordu. Mac uykuya dalarsa süreç askıya alınır
    # (ölmez) ama uzun kesintide tekrar başlatmak gerekebilir.
    done = {}
    if PARTIAL.exists():
        prev = pd.read_csv(PARTIAL, index_col=0, parse_dates=True)
        done = {d: prev.loc[d].to_numpy() for d in prev.index}
        print(f"kısmi kayıttan devam: {len(done)} gün hazır\n")

    todo = [d for d in test_days if d not in done]
    import numpy as np
    for i, day in enumerate(todo, 1):
        preds = [lp.lear_predict_day(X, P, day, w) for w in (56, 84, 1095)]
        done[day] = np.nanmean(np.vstack(preds), axis=0)
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            snap = pd.DataFrame(done).T.sort_index()
            snap.columns = [f"h{h:02d}" for h in range(24)]
            snap.to_csv(PARTIAL)
            print(f"  {len(done)}/{len(test_days)} gün · ara kayıt yazıldı", flush=True)

    lear = pd.DataFrame(done).T.sort_index()
    lear.columns = [f"h{h:02d}" for h in range(24)]
    lear.to_csv(CACHE)
    PARTIAL.unlink(missing_ok=True)
    print(f"\n→ {CACHE}  ({lear.shape[0]} gün)")


if __name__ == "__main__":
    main()
