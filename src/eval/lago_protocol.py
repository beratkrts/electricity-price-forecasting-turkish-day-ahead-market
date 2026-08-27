"""
Lago, Marcjasz, De Schutter & Weron (2021) değerlendirme protokolü.

    Forecasting day-ahead electricity prices: A review of state-of-the-art
    algorithms, best practices and an open-access benchmark.
    Applied Energy 293, 116983.  doi:10.1016/j.apenergy.2021.116983

Alanın kabul görmüş ölçütü bu; Türkiye literatürünün hemen hiçbiri uymuyor.

## ⚠️ Bu, referans uygulama DEĞİL

Yazarların `epftoolbox` kütüphanesi PyPI'da yok (yalnız GitHub'da). Git
bağımlılığı eklemek yerine protokolün gerekli parçaları makalenin PDF'inden
okunarak **buradan sıfırdan yazıldı**. Her fonksiyon ilgili denkleme/bölüme
referanslı, ama **referans uygulamayla sayısal olarak karşılaştırılmadı.**

Doğrulanmış olan:
  · DM testi — sentetik veriyle (aynı modele karşı NaN, kötü modele karşı p≈0,
    ters yönde p≈1)
  · naive kaymaları — 24 ve 168 saat, gün bazında
  · LEAR tasarım matrisi — 247 sütun, lag hizası P[d-1]/P[d-7] ile birebir,
    eğitim penceresi test gününü içermiyor, hedef sütunu matriste yok
Doğrulanmamış olan:
  · LEAR'ın makaledeki kıyas veri setlerinde (NP, PJM, EPEX) aynı sayıları
    üretip üretmediği. Kesinlik gerekiyorsa epftoolbox'ı GitHub'dan kurup
    çapraz kontrol et: `pip install git+https://github.com/jeslago/epftoolbox.git`

## Test seti hakkında yaygın bir karışıklık

Makalenin KENDİ veri setleri 2016'da bitiyor (6 yıllık, sabit kıyas kümeleri:
NP, PJM, EPEX-BE/FR/DE). Buradaki 2024-2026 penceresi onların test seti değil —
**protokolü bizim verimize uyguladığımızda** ortaya çıkan pencere. Taşınan şey
veri değil kural: "son 104 hafta test, günlük yeniden kalibrasyon".

DİKKAT — `METRICS.md` ile uyumsuzluk:
Makale rMAE'yi **naive-2** (geçen hafta aynı saat) üzerinden tanımlıyor ve
"bundan sonra rMAE derken bunu kastediyoruz" diyor (§5.4.2). Bizim METRICS.md
ise **naive-3**'ü (Sal-Cum→dün, Cmt/Paz/Pzt→geçen hafta) kullanıyor. İkisi
farklı sayı üretir; kıyas yaparken hangisi olduğu mutlaka söylenmeli.
`rmae()` varsayılan olarak makaleninkini (naive-2) verir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoLarsCV, LassoLarsIC
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Naive tahminler — makale §5.4.2, denklem (8)
# --------------------------------------------------------------------------

def naive_forecast(prices: pd.Series, variant: int = 2) -> pd.Series:
    """Saatlik fiyat serisinden naive tahmin.

    variant 1 : p[d-1, h]              — dün aynı saat
    variant 2 : p[d-7, h]              — geçen hafta aynı saat  (MAKALENİN rMAE'si)
    variant 3 : Sal-Cum → p[d-1, h]    — METRICS.md'nin kullandığı
                Cmt/Paz/Pzt → p[d-7, h]
    """
    if variant == 1:
        return prices.shift(24)
    if variant == 2:
        return prices.shift(168)
    if variant == 3:
        dow = prices.index.dayofweek          # 0=Pzt … 5=Cmt 6=Paz
        return pd.Series(
            np.where(np.isin(dow, [0, 5, 6]), prices.shift(168), prices.shift(24)),
            index=prices.index,
        )
    raise ValueError("variant 1, 2 veya 3 olmalı")


# --------------------------------------------------------------------------
# Nokta tahmin metrikleri — makale §5.4, denklem (4)-(9)
# --------------------------------------------------------------------------

def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def smape(y, yhat) -> float:
    """Denklem (7). Yüzde olarak döner.

    Makale MAPE'yi sıfıra yakın fiyatlarda patladığı için reddediyor; sMAPE'i
    de "tanımsız ortalama, sonsuz varyans" diye destekleyici sayıyor, tek
    başına karar metriği saymıyor.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom == 0, 0.0, 2 * np.abs(y - yhat) / denom)
    return float(100 * np.mean(r))


def rmae(y, yhat, y_naive) -> float:
    """Denklem (9). rMAE = MAE(model) / MAE(naive), ikisi de ÖRNEKLEM DIŞI.

    < 1 ise naive'den iyi. Makale bunu "farklı veri setleri ve kalibrasyon
    pencereleri arasında kıyaslanabilen" metrik olarak öne çıkarıyor.
    """
    return mae(y, yhat) / mae(y, y_naive)


def metric_table(y, preds: dict, y_naive2, y_naive3=None) -> pd.DataFrame:
    """Makalenin raporladığı çekirdek set + METRICS.md süreklilik sütunu."""
    rows = []
    for name, yhat in preds.items():
        row = {
            "model": name,
            "MAE": mae(y, yhat),
            "RMSE": rmse(y, yhat),
            "sMAPE%": smape(y, yhat),
            "rMAE (Lago, naive-2)": rmae(y, yhat, y_naive2),
        }
        if y_naive3 is not None:
            row["rMAE (METRICS.md, naive-3)"] = rmae(y, yhat, y_naive3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(4)


# --------------------------------------------------------------------------
# Diebold-Mariano testi — makale §5.5.1, denklem (10)-(12)
# --------------------------------------------------------------------------

def dm_test(err_a: np.ndarray, err_b: np.ndarray, variant: str = "multivariate",
            p: int = 1) -> float | np.ndarray:
    """DM testi. err_* şekli (n_gun, 24) — günlük satır, saatlik sütun.

    H0: A'nın kaybı B'ninkinden küçük ya da eşit.
    Küçük p-değeri  →  **B, A'dan anlamlı biçimde daha doğru.**

    variant='multivariate' : günlük 24-boyutlu hata vektörünün p-normu
                             üzerinden tek test (denklem 12). Gün öncesi
                             piyasada 24 fiyat aynı anda ve aynı bilgi
                             kümesiyle üretildiği için makale bunu öneriyor.
    variant='univariate'   : saat başına bağımsız 24 test; sonuç 24 p-değeri.
    """
    err_a, err_b = np.asarray(err_a, float), np.asarray(err_b, float)
    if err_a.shape != err_b.shape:
        raise ValueError("hata dizileri aynı şekilde olmalı")

    if variant == "multivariate":
        d = (np.sum(np.abs(err_a) ** p, axis=1) ** (1 / p)
             - np.sum(np.abs(err_b) ** p, axis=1) ** (1 / p))
        return _dm_pvalue(d)
    if variant == "univariate":
        d = np.abs(err_a) ** p - np.abs(err_b) ** p
        return np.array([_dm_pvalue(d[:, h]) for h in range(d.shape[1])])
    raise ValueError("variant 'multivariate' veya 'univariate' olmalı")


def _dm_pvalue(d: np.ndarray) -> float:
    """Tek taraflı asimptotik z-testi. Kayıp farkı kovaryans durağan varsayılır."""
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(1 - stats.norm.cdf(d.mean() / (sd / np.sqrt(n))))


# --------------------------------------------------------------------------
# LEAR — makale §4.2
# --------------------------------------------------------------------------

def build_lear_matrix(prices: pd.Series, x1: pd.Series, x2: pd.Series) -> pd.DataFrame:
    """247 özellikli LEAR tasarım matrisi (makale §4.1).

    Satır = bir gün, sütun = o gün için mevcut girdiler:
      · fiyat lag'leri  p[d-1], p[d-2], p[d-3], p[d-7]        4×24 =  96
      · d günü için gün-öncesi tahminler x1[d], x2[d]         2×24 =  48
      · tahmin lag'leri x1[d-1], x1[d-7], x2[d-1], x2[d-7]    4×24 =  96
      · haftanın günü kuklası                                          7
                                                            toplam  247

    x1, x2 = "ilgilenilen iki değişkenin gün öncesi tahmini". Türkiye için
    doğal karşılık: EPİAŞ yük tahmini ve KGÜP toplamı — ikisi de tahmin
    anında yayımlanmış durumda.
    """
    def daily(s: pd.Series, name: str) -> pd.DataFrame:
        w = s.to_frame(name).copy()
        w["_d"] = w.index.normalize()
        w["_h"] = w.index.hour
        m = w.pivot_table(index="_d", columns="_h", values=name, aggfunc="first")
        m.columns = [f"{name}_h{h:02d}" for h in m.columns]
        return m

    P, X1, X2 = daily(prices, "p"), daily(x1, "x1"), daily(x2, "x2")
    parts = [P.shift(lag).add_suffix(f"_lag{lag}") for lag in (1, 2, 3, 7)]
    parts += [X1, X2]
    parts += [X1.shift(1).add_suffix("_lag1"), X1.shift(7).add_suffix("_lag7"),
              X2.shift(1).add_suffix("_lag1"), X2.shift(7).add_suffix("_lag7")]
    X = pd.concat(parts, axis=1)
    dow = pd.get_dummies(X.index.dayofweek, prefix="dow").set_index(X.index).astype(float)
    return pd.concat([X, dow], axis=1), P


def _asinh(a):
    """Makalenin varyans dengeleyici dönüşümü (§4.2). log1p'in aksine
    negatif ve sıfır fiyatlarda tanımlı — GÖP için gerekli."""
    return np.arcsinh(a)


def _asinh_inv(a):
    return np.sinh(a)


def lear_predict_day(X: pd.DataFrame, P: pd.DataFrame, day, window_days: int,
                     selector: str = "cv") -> np.ndarray:
    """Tek gün için 24 saatlik LEAR tahmini, saat başına ayrı LASSO.

    Kalibrasyon penceresi `day`den geriye `window_days` gün. Hedef ve
    özellikler asinh ile dönüştürülür, özellikler ayrıca standartlaştırılır
    (LASSO ölçeğe duyarlı).

    `selector`:
      "cv"  — LARS + çapraz doğrulama ile λ. Makale §4.2.1'in ikinci
              seçeneği. **Varsayılan bu olmalı:** 8 haftalık pencerede
              örnek sayısı (56) özellik sayısından (247) küçük ve
              `LassoLarsIC` o durumda gürültü varyansını kestiremeyip
              hata veriyor. Makale zaten "parametre-zengin model" diyor,
              yani n < p kasıtlı.
      "aic" — bilgi kriteri; sadece uzun pencerelerde (n > p) geçerli.
    """
    idx = X.index
    pos = idx.get_loc(day)
    lo = max(0, pos - window_days)
    tr = slice(lo, pos)

    Xtr_raw, Xte_raw = X.iloc[tr], X.iloc[pos:pos + 1]
    ok = Xtr_raw.notna().all(axis=1) & P.iloc[tr].notna().all(axis=1)
    Xtr_raw, Ytr_raw = Xtr_raw[ok], P.iloc[tr][ok]
    if len(Xtr_raw) < 30 or Xte_raw.isna().any(axis=None):
        return np.full(24, np.nan)

    sc = StandardScaler().fit(_asinh(Xtr_raw.to_numpy(float)))
    Xtr = sc.transform(_asinh(Xtr_raw.to_numpy(float)))
    Xte = sc.transform(_asinh(Xte_raw.to_numpy(float)))

    out = np.empty(24)
    n_tr = Xtr.shape[0]
    for h in range(24):
        y = _asinh(Ytr_raw.iloc[:, h].to_numpy(float))
        if selector == "aic" and n_tr > Xtr.shape[1]:
            m = LassoLarsIC(criterion="aic")
        else:
            m = LassoLarsCV(cv=min(5, max(2, n_tr // 10)), max_iter=200, n_jobs=1)
        out[h] = _asinh_inv(m.fit(Xtr, y).predict(Xte)[0])
    return out


def lear_ensemble(X: pd.DataFrame, P: pd.DataFrame, test_days,
                  windows=(56, 84, 1095), selector: str = "cv",
                  progress=None) -> pd.DataFrame:
    """Günlük yeniden kalibrasyonla LEAR, pencere ortalaması (makale §4.2/4.4).

    Makale 8 hafta, 12 hafta, 3 yıl ve 4 yıl pencerelerini birlikte kullanıyor
    (kısa + uzun kombinasyonu ampirik olarak daha iyi). Varsayılan burada
    4 yılı DIŞARIDA bırakıyor: bizim ham verimiz 2021'de başlıyor, test
    dönemi 2024-08'de başladığı için 4 yıllık pencere dolmuyor.
    """
    rows = {}
    for i, day in enumerate(test_days):
        preds = [lear_predict_day(X, P, day, w, selector) for w in windows]
        rows[day] = np.nanmean(np.vstack(preds), axis=0)
        if progress and (i + 1) % progress == 0:
            print(f"  LEAR {i + 1}/{len(test_days)} gün", flush=True)
    return pd.DataFrame(rows).T
