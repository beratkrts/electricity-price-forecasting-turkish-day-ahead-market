#!/usr/bin/env python3
"""
Çapraz doğrulama (2/2): referans `epftoolbox` kütüphanesinin LEAR sınıfını,
bizim son 365 günlük test verimizde, tek pencere (3 yıl=1092 gün) + AIC
seçiciyle koşturur. `run_crossval_ours.py` (proje .venv) ile birebir aynı
gün/pencere/seçici kullanıyor — sonuçlar hour-by-hour karşılaştırılabilir.

ÖNEMLİ: Bu script ayrı bir venv'de çalışır (epftoolbox + numpy<2 + tensorflow),
proje .venv'i DEĞİL — epftoolbox'ın kendi kodu NumPy 2.x ile uyumsuz
(bkz. notebook §2, "Yp[h] = self.models[h].predict(X)" skaler atama bug'ı).

    source <epf_venv>/bin/activate
    python run_crossval_reference.py <tr_epf.csv yolu> <çıktı csv yolu> [n_jobs]
"""
import os
# BLAS'ı tek thread'e sabitle — dıştan process paralelliği kullanıyoruz,
# içeride thread paralelliği eklenirse (özellikle Apple'ın MAX_THREADS=3
# sınırlı OpenBLAS'ında) çekirdekler process'ler arasında rekabete girip
# toplamda YAVAŞLATIYOR.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

EPFTOOLBOX_ROOT = Path(__file__).resolve()
# scratchpad/epftoolbox klonu sys.path'e eklenir (repoya eklenmedi, git-dışı)
sys.path.insert(0, os.environ.get(
    "EPFTOOLBOX_PATH",
    "/private/tmp/claude-501/-Users-beratkaratasoglu-etkb-intern-project-enerji-fiyat-tahmini/"
    "64579635-aa8e-44fa-a1e5-bba067d601ee/scratchpad/epftoolbox"))
from epftoolbox.models import LEAR  # noqa: E402

CW = 1092


def predict_one_day(df: pd.DataFrame, day: pd.Timestamp) -> tuple:
    data_available = df.loc[:day + pd.Timedelta(hours=23)].copy()
    data_available.loc[day:day + pd.Timedelta(hours=23), "Price"] = np.nan
    model = LEAR(calibration_window=CW)
    Yp = model.recalibrate_and_forecast_next_day(
        df=data_available, next_day_date=day, calibration_window=CW)
    return day, Yp.flatten()


def main() -> None:
    data_path = sys.argv[1] if len(sys.argv) > 1 else "tr_epf.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "crossval_reference_1y.csv"
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    last_day = df.index.normalize().max()
    test_days = pd.date_range(last_day - pd.Timedelta(days=364), last_day, freq="D")
    print(f"test: {test_days[0].date()} -> {test_days[-1].date()}  ({len(test_days)} gün)")
    print(f"n_jobs={n_jobs}  pencere={CW} gün  seçici=AIC (referans epftoolbox varsayılanı)")

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(predict_one_day)(df, day) for day in test_days)
    out = pd.DataFrame({day: yp for day, yp in results}).T.sort_index()
    out.columns = [f"h{h:02d}" for h in range(24)]
    out.to_csv(out_path)
    print(f"\n{time.time()-t0:.0f}s  -> {out_path}  {out.shape}")


if __name__ == "__main__":
    main()
