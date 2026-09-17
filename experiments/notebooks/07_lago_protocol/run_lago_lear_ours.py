#!/usr/bin/env python3
"""
Faz C (LAGO_BENCHMARK_PLAN.md): BİZİM `lear_predict_day` implementasyonu,
Faz B ile birebir aynı pencere/dönem — kalibrasyon tablosu için.

Faz B (epftoolbox) ile fark: ölçekleme (asinh+StandardScaler vs asinh+medyan/MAD),
λ sonrası düz Lasso refit yok, n≤p'de `noise_variance` yerine CV fallback
(`lear_predict_day` içinde `selector='aic'` + n≤p → LassoLarsCV). Bkz. memory
`lear-impl-vs-epftoolbox`.

    python experiments/notebooks/07_lago_protocol/run_lago_lear_ours.py 56
    ... 84 / 1092 / 1456

Çıktı: ours_lear_cw{W}.csv  (gün × 24 saat, USD/MWh) — uzun pencere
(LONG_TEST_START → TEST_END, 1096g), `04` ile aynı indeks.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from db.connection import get_db_engine          # noqa: E402
from src.eval import lago_protocol as lp          # noqa: E402

N_JOBS = 6
DATA = HERE / "tr_epf_ext.csv"   # Price / Exogenous 1 (yük tahmini) / Exogenous 2 (KGÜP toplam)


def safe_predict(X, P, day, cw):
    try:
        return lp.lear_predict_day(X, P, day, cw, selector="aic")
    except Exception as e:  # noqa: BLE001
        print(f"  {day.date()} çöküş: {e}", flush=True)
        return np.full(24, np.nan)


def main():
    cw = int(sys.argv[1])
    # 2. arg = 'spike' (2022 tam yıl) veya 'YYYY-MM-DD:YYYY-MM-DD'. Yoksa uzun pencere.
    tag = sys.argv[2] if len(sys.argv) > 2 else "long"
    suffix = {"long": "", "spike": "_spike"}.get(tag, "_" + tag.replace(":", "_"))
    out = HERE / f"ours_lear_cw{cw}{suffix}.csv"

    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize("Europe/Istanbul")
    df = df.rename(columns={"Price": "price", "Exogenous 1": "x1", "Exogenous 2": "x2"})

    X, P = lp.build_lear_matrix(df.price, df.x1, df.x2)
    if tag == "long":
        test_days = lp.test_day_index(P, long=True)
    elif tag == "spike":
        test_days = P.index[(P.index >= "2022-01-01") & (P.index <= "2022-12-31")]
    else:
        a, b = tag.split(":")
        test_days = P.index[(P.index >= a) & (P.index <= b)]
    need = test_days[0] - pd.Timedelta(days=cw + 7)
    assert P.index.min() <= need, f"veri {P.index.min()}, {need} gerek"

    print(f"cw={cw}  test {test_days[0].date()} -> {test_days[-1].date()} "
          f"({len(test_days)} gün)  p={X.shape[1]}  seçici=aic(n<=p->CV)  n_jobs={N_JOBS}")
    t0 = time.time()
    preds = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(safe_predict)(X, P, d, cw) for d in test_days)
    tbl = pd.DataFrame(preds, index=test_days, columns=[f"h{h:02d}" for h in range(24)])
    tbl.index = tbl.index.tz_localize(None)
    tbl.to_csv(out)
    nf = int(tbl.isna().all(axis=1).sum())
    print(f"\n{time.time()-t0:.0f}s  ->  {out.name}  {tbl.shape}  (çöküş {nf} gün)")


if __name__ == "__main__":
    main()
