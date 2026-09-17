#!/usr/bin/env python3
"""
Faz B (LAGO_BENCHMARK_PLAN.md): KANONİK Lago LEAR — gerçek epftoolbox kodu.

Tek kalibrasyon penceresi için, UZUN test penceresinde (2023-08-28 -> 2026-08-27,
1096 gün) günlük yeniden kalibrasyonla koşar. 4-pencere ansambl bunun 4 ayrı
çağrısıyla (W = 56, 84, 1092, 1456) üretilir.

    # epftoolbox venv'inde (bkz. memory: epftoolbox-venv):
    EPF=/private/tmp/.../scratchpad
    "$EPF/epf_venv/bin/python" run_lago_lear_reference.py 56
    "$EPF/epf_venv/bin/python" run_lago_lear_reference.py 84
    "$EPF/epf_venv/bin/python" run_lago_lear_reference.py 1092
    "$EPF/epf_venv/bin/python" run_lago_lear_reference.py 1456

Çıktı: lago_ref_lear_cw{W}.csv  (gün × 24 saat, USD/MWh)

n <= p (56, 84 pencereleri; p=247) durumunda epftoolbox'ın stock AIC seçicisi
matematiksel tanımsız → `recalibrate_predict_safe`:
  (1) n<=p ise gürültü varyansı = boş-model (intercept-only) kalıntı varyansı,
      `noise_variance` parametresiyle veriliyor (sklearn >= 1.1),
  (2) `DataScaler('Invariant')`'ın MAD=0 çöküşü (2026 baharında sıfırda
      kenetlenen saatler) için MAD=1 kabul edilip ölçekleme atlanıyor.
n > p (1092, 1456) durumunda bu iki yol devre dışı ve sonuç stock ile birebir.

Kaynak protokol sabitleri: `src/eval/lago_protocol.py`
  LONG_TEST_START = 2023-08-28 · TEST_END = 2026-08-27
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
DATA = HERE / "tr_epf_ext.csv"
TEST_START = "2023-08-28"
TEST_END = "2026-08-27"
N_JOBS = 6

sys.path.insert(0, os.environ.get(
    "EPFTOOLBOX_PATH",
    "/private/tmp/claude-501/-Users-beratkaratasoglu-etkb-intern-project-enerji-fiyat-tahmini/"
    "f99d396b-aa9c-4398-9fec-089a3f3b7e7e/scratchpad/epftoolbox"))
from epftoolbox.data import DataScaler          # noqa: E402
from epftoolbox.models import LEAR              # noqa: E402
from sklearn.linear_model import LassoLarsIC, Lasso  # noqa: E402


def _fit_scaler_safe(data):
    sc = DataScaler("Invariant")
    sc.scaler.fit(data)
    sc.scaler.mad[sc.scaler.mad == 0] = 1.0
    return sc.scaler.transform(data), sc


def recalibrate_predict_safe(Xtrain, Ytrain, Xtest):
    n, p = Xtrain.shape
    Ytr, scalerY = _fit_scaler_safe(Ytrain)
    Xtr_nd, scalerX = _fit_scaler_safe(Xtrain[:, :-7])
    Xtr = Xtrain.copy()
    Xtr[:, :-7] = Xtr_nd

    models = {}
    for h in range(24):
        y = Ytr[:, h]
        nv = np.var(y, ddof=1) if n <= p else None
        alpha = LassoLarsIC(criterion="aic", max_iter=2500,
                            noise_variance=nv).fit(Xtr, y).alpha_
        models[h] = Lasso(max_iter=2500, alpha=alpha).fit(Xtr, y)

    Xte = Xtest.copy()
    Xte[:, :-7] = scalerX.transform(Xte[:, :-7])
    Yp = np.array([models[h].predict(Xte)[0] for h in range(24)])
    return scalerY.inverse_transform(Yp.reshape(1, -1)).flatten()


def predict_one_day(df, day, cw):
    model = LEAR(calibration_window=cw)
    avail = df.loc[:day + pd.Timedelta(hours=23)].copy()
    avail.loc[day:day + pd.Timedelta(hours=23), "Price"] = np.nan
    df_train = avail.loc[:day - pd.Timedelta(hours=1)].iloc[-cw * 24:]
    df_test = avail.loc[day - pd.Timedelta(weeks=2):, :]
    Xtr, Ytr, Xte = model._build_and_split_XYs(
        df_train=df_train, df_test=df_test, date_test=day)
    try:
        return day, recalibrate_predict_safe(Xtr, Ytr, Xte)
    except Exception as e:  # noqa: BLE001
        print(f"  {day.date()} çöküş: {e}", flush=True)
        return day, np.full(24, np.nan)


def main():
    cw = int(sys.argv[1])
    out = HERE / f"lago_ref_lear_cw{cw}.csv"

    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    test_days = pd.date_range(TEST_START, TEST_END, freq="D")
    # 4-yıl penceresi için yeterli geçmiş var mı?
    need = test_days[0] - pd.Timedelta(days=cw + 7)
    assert df.index.min() <= need, f"veri {df.index.min()}, {need} gerekiyor"

    print(f"cw={cw}  test {test_days[0].date()} -> {test_days[-1].date()} "
          f"({len(test_days)} gün)  n_jobs={N_JOBS}")
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(predict_one_day)(df, d, cw) for d in test_days)
    tbl = pd.DataFrame({d: yp for d, yp in res}).T.sort_index()
    tbl.columns = [f"h{h:02d}" for h in range(24)]
    tbl.to_csv(out)
    nf = int(tbl.isna().all(axis=1).sum())
    print(f"\n{time.time() - t0:.0f}s  ->  {out.name}  {tbl.shape}  (çöküş {nf} gün)")


if __name__ == "__main__":
    main()
