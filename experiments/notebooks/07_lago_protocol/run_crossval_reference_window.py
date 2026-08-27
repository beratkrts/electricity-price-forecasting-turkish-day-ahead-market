#!/usr/bin/env python3
"""
Referans epftoolbox LEAR'ını TEK bir kalibrasyon penceresi için, son 365 gün
üzerinde koşturur. Tam 4-pencere ansambl (56, 84, 1092, 1456 gün) bunun 4
ayrı çağrısıyla üretiliyor.

n <= p (56, 84 günlük pencereler; p=247 özellik) durumunda referansın kendi
AIC seçicisi matematiksel olarak tanımsız (bkz. notebook §0 — sklearn PR #21481,
eski sklearn'ün formülü sonradan "bug" ilan edildi, eski sürümü kurmak da bir
çözüm değil). Burada bunun yerine gürültü varyansı için standart bir yedek
kullanılıyor: "boş model" (intercept-only) kalıntı varyansı, yani hedefin
(ölçeklenmiş) kendi varyansı — n<=p altında OLS tahmini imkansız olduğunda
istatistikte bilinen, muhafazakar bir vekil. sklearn'ün `noise_variance`
parametresi (sürüm 1.1'de eklendi) bunu doğrudan besleyip n>p kontrolünü
bypass etmemizi sağlıyor.

İKİNCİ bir kırılganlık, referans kodun KENDİ ölçekleme adımında: 2026 bahar
fiyat çöküşü sırasında bazı saatler (ör. öğlen, güneş bolluğu) 56 günlük
pencerenin yarısından fazlasında $0'da kenetleniyor → medyan mutlak sapma
(MAD) tam sıfır → `(x - medyan) / MAD` sıfıra bölüm, `inf`/`NaN`. Bu bizim
eklediğimiz bir şey değil, referansın `MedianScaler`'ının kendi açığı —
3 yıllık pencerede sorun çıkmıyor çünkü 3 yılda bir saatin 500+ günü aynı
kalması pratikte imkansız. Standart robust-scaling düzeltmesi uygulandı:
MAD=0 olan sütun/saat için MAD=1 kabul edilip ölçekleme atlanıyor (sabit bir
özelliğin ölçeklenmesi zaten anlamsız).

    python run_crossval_reference_window.py <tr_epf.csv> <çıktı csv> <pencere_gün> [n_jobs]
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.environ.get(
    "EPFTOOLBOX_PATH",
    "/private/tmp/claude-501/-Users-beratkaratasoglu-etkb-intern-project-enerji-fiyat-tahmini/"
    "64579635-aa8e-44fa-a1e5-bba067d601ee/scratchpad/epftoolbox"))
from epftoolbox.data import DataScaler
from epftoolbox.models import LEAR
from sklearn.linear_model import LassoLarsIC, Lasso


def _fit_scaler_safe(data):
    """DataScaler('Invariant').fit_transform, MAD=0 koruması ile (bkz. modül
    docstring'i — 2026 çöküşünde sıfırda kenetlenen saatler)."""
    scaler = DataScaler('Invariant')
    scaler.scaler.fit(data)
    scaler.scaler.mad[scaler.scaler.mad == 0] = 1.0
    return scaler.scaler.transform(data), scaler


def recalibrate_predict_safe(Xtrain, Ytrain, Xtest):
    """LEAR.recalibrate()+predict()'in birebir aynısı, iki fark: (1) n<=p ise
    AIC seçicisine 'bos model' kalinti varyansini noise_variance olarak veriyor,
    (2) MAD=0 olan sütun/saatte sıfıra bölüm yerine ölçeklemeyi atlıyor.
    (Referans kod ikisinde de bu durumda çöküyordu.)"""
    n, p = Xtrain.shape
    Ytr, scalerY = _fit_scaler_safe(Ytrain)
    Xtr_nd, scalerX = _fit_scaler_safe(Xtrain[:, :-7])
    Xtr = Xtrain.copy()
    Xtr[:, :-7] = Xtr_nd

    models = {}
    for h in range(24):
        y = Ytr[:, h]
        nv = np.var(y, ddof=1) if n <= p else None
        param_model = LassoLarsIC(criterion='aic', max_iter=2500, noise_variance=nv)
        alpha = param_model.fit(Xtr, y).alpha_
        m = Lasso(max_iter=2500, alpha=alpha)
        m.fit(Xtr, y)
        models[h] = m

    Xte = Xtest.copy()
    Xte[:, :-7] = scalerX.transform(Xte[:, :-7])
    Yp = np.zeros(24)
    for h in range(24):
        Yp[h] = models[h].predict(Xte)[0]
    return scalerY.inverse_transform(Yp.reshape(1, -1)).flatten()


def predict_one_day(df: pd.DataFrame, day: pd.Timestamp, CW: int) -> tuple:
    model = LEAR(calibration_window=CW)
    data_available = df.loc[:day + pd.Timedelta(hours=23)].copy()
    data_available.loc[day:day + pd.Timedelta(hours=23), "Price"] = np.nan
    df_train = data_available.loc[:day - pd.Timedelta(hours=1)].iloc[-CW * 24:]
    df_test = data_available.loc[day - pd.Timedelta(weeks=2):, :]
    Xtr, Ytr, Xte = model._build_and_split_XYs(df_train=df_train, df_test=df_test, date_test=day)
    Yp = recalibrate_predict_safe(Xtr, Ytr, Xte)
    return day, Yp


def main() -> None:
    data_path, out_path, cw = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    last_day = df.index.normalize().max()
    test_days = pd.date_range(last_day - pd.Timedelta(days=364), last_day, freq="D")
    print(f"pencere={cw} gün  test={test_days[0].date()}->{test_days[-1].date()} ({len(test_days)} gün)  n_jobs={n_jobs}")

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(predict_one_day)(df, day, cw) for day in test_days)
    out = pd.DataFrame({day: yp for day, yp in results}).T.sort_index()
    out.columns = [f"h{h:02d}" for h in range(24)]
    out.to_csv(out_path)
    print(f"\n{time.time()-t0:.0f}s  -> {out_path}  {out.shape}")


if __name__ == "__main__":
    main()
