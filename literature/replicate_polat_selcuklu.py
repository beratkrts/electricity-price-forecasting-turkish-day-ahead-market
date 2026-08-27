#!/usr/bin/env python3
"""
Polat & Selçuklu (SSRN 4894108) sonucunun tekrar üretimi ve değerlendirme
kurgusunun sökülmesi.

Neden: makale Türkiye GÖP için LightGBM ile MAE 5,981 $/MWh rapor ediyor —
bizim canlı modelimizden iyi görünüyor. Ama bölme yöntemi zaman serisine
rastgele uygulanıyor. Bu script farkın ne kadarının sızıntıdan, ne kadarının
rejim kaymasından geldiğini ayırıyor.

Veri ve kod yazarların kendisinden:
    github.com/TheEmgame/EPF-Turkish-Day-Ahead-Market
    Dataset.rar → Dataset/EXIST_2018.csv   (43.679 saat, 2018-01-01 → 2023-01-01)

Hiperparametreler de onların `tuned_hyper.py`'sinden alındı, değiştirilmedi.

Kullanım:
    python literature/replicate_polat_selcuklu.py --csv /yol/EXIST_2018.csv
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Yazarların tuned_hyper.py'sindeki LightGBM ayarı — aynen
PARAMS = dict(learning_rate=0.2, num_leaves=60, n_estimators=350,
              random_state=35, verbose=-1)

# Açık artırma temizlemesiyle EŞANLI belirlenen sütunlar: piyasa temizlenmeden
# bilinmiyorlar, dolayısıyla gün öncesi tahmininde girdi olamazlar.
POST_CLEARING = ["PISO", "PIBO", "SSOV", "SBOV", "MO", "MB", "TV"]
# Gerçekleşen üretim: gün öncesinde bilinmez (KGÜP ya da ön-tahmin kullanılabilir)
REALISED_GEN = ["TG", "NG", "HYD", "LIG", "RYHD", "ICOAL", "WIND", "SOL",
                "FUEL", "GEO", "ASPH", "BCOAL", "BIO", "IMEX", "WAS"]


def fit_eval(Xtr, ytr, Xte, yte):
    pred = lgb.LGBMRegressor(**PARAMS).fit(Xtr, ytr).predict(Xte)
    return mean_absolute_error(yte, pred), r2_score(yte, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Dataset/EXIST_2018.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ts = pd.to_datetime(df.Date + " " + df.Hour)
    year = ts.dt.year
    d = df.drop(columns=["Date", "Hour"])          # yazarların yaptığı gibi
    X, y = d.drop("MCP", axis=1), d["MCP"]

    print(f"Veri: {len(df)} saat · {ts.min()} → {ts.max()}")
    print(f"MCP ort {y.mean():.3f} · std {y.std():.3f} · min {y.min():.3f} · maks {y.max():.3f}")
    print("  (Makalenin Tablo 1'i maksimumu 66,630 diyor — hatalı; gerçek maksimum yukarıda)\n")

    # --- 0) Test setinin YAPISI ------------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=35)
    tr = set(Xtr.index)
    te = sorted(Xte.index)
    either = sum(1 for i in te if (i - 1) in tr or (i + 1) in tr)
    both = sum(1 for i in te if (i - 1) in tr and (i + 1) in tr)
    same_day = year.copy()  # yer tutucu, aşağıda gerçek hesap
    day = ts.dt.date
    from collections import Counter
    cnt = Counter(day[Xtr.index])
    print("Rastgele bölmede 'test seti' bir DÖNEM değil, dağılmış saatler:")
    print(f"  test aralığı: {ts[Xte.index].min()} → {ts[Xte.index].max()} (eğitimle iç içe)")
    print(f"  test saatlerinin %{100 * either / len(te):.1f}'inin en az bir komşu saati eğitimde")
    print(f"  test saatlerinin %{100 * both / len(te):.1f}'inin her iki komşu saati de eğitimde")
    print(f"  bir test saatinin aynı gününden ortalama "
          f"{sum(cnt[day[i]] for i in te) / len(te):.1f}/24 saat eğitimde\n")

    # --- 1) Makalenin rakamını tekrar üret -------------------------------
    mae, r2 = fit_eval(Xtr, ytr, Xte, yte)
    print(f"1) Makalenin kurgusu (rastgele bölme)   MAE {mae:6.3f}  R² {r2:.3f}"
          f"   ← makale: MAE 5,981 · R² 0,950\n")

    # --- 2) SIZINTIYI İZOLE ET -------------------------------------------
    # Aynı yıl içinde bölersek rejim kayması yok; kalan fark bölmeden geliyor.
    print("2) Sızıntının izolesi — aynı yıl içinde rastgele vs kronolojik")
    print(f"   {'Yıl':<7}{'ort MCP':>9}{'rastgele':>11}{'kronolojik':>13}{'kat':>7}")
    for Y in sorted(year.unique()):
        m = year == Y
        if m.sum() < 2000:
            continue
        Xy, yy = X[m].reset_index(drop=True), y[m].reset_index(drop=True)
        a, b, c, e = train_test_split(Xy, yy, test_size=0.20, random_state=35)
        r_mae, _ = fit_eval(a, c, b, e)
        n = int(len(Xy) * 0.8)
        k_mae, _ = fit_eval(Xy[:n], yy[:n], Xy[n:], yy[n:])
        print(f"   {Y:<7}{yy.mean():>9.1f}{r_mae:>11.2f}{k_mae:>13.2f}{k_mae / r_mae:>6.1f}x")

    # --- 3) DÜRÜST KURGU --------------------------------------------------
    print("\n3) Dürüst kurgu — geçmişle eğit, sonraki yılı tahmin et")
    print(f"   {'Test yılı':<11}{'eğitim':>13}{'ort MCP':>9}{'MAE':>9}{'R²':>8}")
    for Y in sorted(year.unique())[1:]:
        tr_m, te_m = year < Y, year == Y
        if te_m.sum() < 2000:
            continue
        mae, r2 = fit_eval(X[tr_m], y[tr_m], X[te_m], y[te_m])
        print(f"   {Y:<11}{f'2018-{Y - 1}':>13}{y[te_m].mean():>9.1f}{mae:>9.2f}{r2:>8.3f}")

    # --- 4) ADİL KIYAS: rMAE, sık yeniden eğitimle -----------------------
    # İki düzeltme birden:
    #  (a) Dönem zorluğu normalize edilsin diye rMAE (Lago et al. naive:
    #      Pzt/Cmt/Paz → geçen hafta aynı saat, Sal-Cum → dün aynı saat).
    #      MCP_168 ve MCP_24 sütunları zaten veri setinde var.
    #  (b) Bizim model her gün yeniden eğitiliyor; onlara "bir kez eğit, tüm
    #      yılı tahmin et" dayatmak haksızdı. Genişleyen pencere + 30 günde
    #      bir yeniden eğitim veriyoruz. Bu düzeltme ONLARIN lehine.
    dow = ts.dt.dayofweek
    naive = np.where(dow.isin([0, 5, 6]), df.MCP_168, df.MCP_24)

    START, STEP = 365 * 24, 30 * 24
    idxs, preds = [], []
    i = START
    while i < len(X):
        j = min(i + STEP, len(X))
        preds.append(lgb.LGBMRegressor(**PARAMS).fit(X[:i], y[:i]).predict(X[i:j]))
        idxs.append(np.arange(i, j))
        i = j
    idx, pr = np.concatenate(idxs), np.concatenate(preds)

    def rmae_row(label, mask=None):
        k = idx if mask is None else idx[mask]
        pp = pr if mask is None else pr[mask]
        mae = mean_absolute_error(y[k], pp)
        nv = mean_absolute_error(y[k], naive[k])
        print(f"   {label:<44}{mae:>8.2f}{nv:>8.2f}{mae / nv:>8.3f}")

    print("\n5) ADİL KIYAS — genişleyen pencere, 30 günde bir yeniden eğitim")
    print(f"   {'':<44}{'MAE':>8}{'naive':>8}{'rMAE':>8}")
    for Y in sorted(year.unique()):
        m = year.values[idx] == Y
        if m.sum() > 2000:
            rmae_row(f"test {Y} (ort ${y[idx][m].mean():.0f})", m)
    rmae_row("BİRLEŞİK")
    print("   " + "-" * 68)
    print(f"   {'BİZ — canlı model, 2 yıl (METRICS.md §4)':<44}{7.26:>8.2f}{9.60:>8.2f}{0.756:>8.3f}")
    print("   rMAE < 1 = naive'den iyi. Onların modeli hiçbir yılda 1'in altına inmiyor.")

    # --- 6) Eşanlı değişkenler atılınca ----------------------------------
    n = int(len(X) * 0.8)
    print("\n6) Son %20 kronolojik test, girdiler kısıtlanınca")
    for label, cols in [("tüm 33 değişken", []),
                        ("açık artırma çıktıları atıldı", POST_CLEARING),
                        ("+ gerçekleşen üretim atıldı", POST_CLEARING + REALISED_GEN)]:
        Xc = X.drop(columns=cols)
        mae, r2 = fit_eval(Xc[:n], y[:n], Xc[n:], y[n:])
        print(f"   {label:<34} MAE {mae:7.2f}  R² {r2:7.3f}")


if __name__ == "__main__":
    main()
