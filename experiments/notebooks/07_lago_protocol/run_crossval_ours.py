#!/usr/bin/env python3
"""
Çapraz doğrulama (1/2): bizim LEAR implementasyonumuzu son 365 gün için,
tek pencere (3 yıl=1092 gün) + AIC seçiciyle koşturur — referans epftoolbox'la
birebir aynı ayarlarla (bkz. run_crossval_reference.py, ayrı venv'de çalışıyor).

    python experiments/notebooks/07_lago_protocol/run_crossval_ours.py
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from db.connection import get_db_engine          # noqa: E402
from src.eval import lago_protocol as lp         # noqa: E402

OUT = Path(__file__).parent / "crossval_ours_1y.csv"
CW = 1092
N_JOBS = 6

DATA_SQL = """
SELECT m.ts, m.price_usd, l.load_forecast_mw, k.total_mw
FROM raw_mcp_hourly m
LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN raw_kgup_hourly  k ON m.ts = k.ts
WHERE m.ts >= '2021-01-01'
ORDER BY m.ts
"""


def main() -> None:
    df = pd.read_sql(text(DATA_SQL), get_db_engine())
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert("Europe/Istanbul")
    df = df.set_index("ts").sort_index()
    df[["load_forecast_mw", "total_mw"]] = (
        df[["load_forecast_mw", "total_mw"]].interpolate(limit=6).ffill().bfill())

    X, P = lp.build_lear_matrix(df.price_usd, df.load_forecast_mw, df.total_mw)
    test_days = P.index[P.index >= P.index[-1] - pd.Timedelta(days=364)]
    print(f"test: {test_days[0].date()} -> {test_days[-1].date()}  ({len(test_days)} gün)")
    print(f"n_jobs={N_JOBS}  pencere={CW} gün  seçici=AIC")

    t0 = time.time()
    preds = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(lp.lear_predict_day)(X, P, day, CW, "aic") for day in test_days)
    out = pd.DataFrame(preds, index=test_days, columns=[f"h{h:02d}" for h in range(24)])
    out.to_csv(OUT)
    print(f"\n{time.time()-t0:.0f}s  -> {OUT}  {out.shape}")


if __name__ == "__main__":
    main()
