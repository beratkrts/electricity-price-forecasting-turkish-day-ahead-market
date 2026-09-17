# Deney Raporu - Model & Strateji Karşılaştırması

> **GÜNCELLEME 27 Ağu 2026:** `lgb_lag0_v2` **13 Ağu 2026'da canlıya alındı**
> (commit `d7c67bf`: robust feature set + lag0 oranları + `min_child_samples=10`
> + CQR). Aynı gün `gold.ptf_predictions_daily` toptan yeniden dolduruldu
> (`backfill_gold_predictions.py`, gerçek walk-forward). Yani bugün "canlı model"
> = lgb_lag0_v2 ve tablodaki "Canlı" satırları artık onu gösteriyor. §9 Öncelikli
> Aksiyon 1 KAPANDI; yerine gelen: hidro pre-forecaster + `thermal_requirement_ratio`
> (bkz. `LOW_PRICE_REGIME_ANALYSIS.md`, `LAGO_BENCHMARK_PLAN.md` Faz C.5).
>
> **GÜNCELLEME 1 Eyl 2026:** Faz C.5 bitti. **Çok-pencereli LightGBM ensemble
> (`base+roll90+roll150`) + nedensel konformal band** çöküş körlüğünü kısmen çözdü ve
> canlı tek-modeli DM-anlamlı geçti (çöküş MAE $10.54→$9.62, p<0.0001). Karar kapısı
> geçildi; canlıya kod yazılmadı — devir brief'i `../enerji_fiyat_tahmini/ENSEMBLE_IMPLEMENTASYON.md`.
> Detay §10. **Buraya ara verildi, yeni konuya geçiliyor.**

## Yönetici Özeti

Projede iki ayrı model versiyonu ve çok sayıda deneysel strateji test edilmiştir. Canlıdaki model **LightGBM 3-head quantile regression** (P10/P50/P90) olup **log1p kullanmaz**; **13 Ağu 2026'dan beri `lgb_lag0_v2` konfigü** (lag0 renewable ratios + `min_child_samples=10`). Önceki log1p ve pre-lag0 modeller artık canlıda değildir. `lgb_lag0_v2` lag0 renewable ratios ekleyerek özellikle sıfır-fiyat saatlerinde **kısmi** iyileştirme sağlamıştır — 2026 bahar çöküşünü tam çözmemiştir (Şub-Mar 2026 MAE hâlâ ~$12-13). `lgb_cqr_v2` ise aynı modelin CQR kalibre edilmiş güven aralığı versiyonudur (genel MAE belirgin kötü, canlıda değil).

---

## Model Versiyonları (Durum Tablosu)

| Model | Tablo | Objective | Log1p | Lag0 Ratios | min_child | Güven Aralığı | Durum |
|-------|-------|-----------|-------|-------------|-----------|---------------|-------|
| **lgb_lag0_v2** (Canlı) | `gold.ptf_predictions_daily` | quantile | Hayır | **Evet** | 10 | Native quantile P10/P90 | **CANLI (13 Ağu 2026'dan)** |
| **lgb_cqr_v2** (Deneysel) | `gold.ptf_predictions_experimental` | quantile | Hayır | **Evet** | 10 | **CQR kalibre** P10/P90 | Deneysel — genel MAE $14 (kötü), canlıda değil |
| LightGBM_v1 (pre-lag0) | Yok (artık kullanılmıyor) | quantile | Hayır | Hayır | 20 | Native quantile | **KALDIRILDI (13 Ağu 2026)** |
| Eski Log1p Model | Yok | regression | **Evet** | Hayır | 20 | Yok | **KALDIRILDI** |

Not: 13 Ağu 2026'da `gold.ptf_predictions_daily` **tamamen** yeniden dolduruldu
(2024-08 → 2026-08, `backfill_gold_predictions.py` walk-forward, lag0 feature'larla,
eğitim 2023-01'den). Yani bu tablodaki geçmiş performans metrikleri **artık
lgb_lag0_v2'yi yansıtıyor** — pre-lag0 modelin ayrı bir kaydı kalmadı.

### `lgb_lag0_v2` vs `LightGBM_v1` Farkları
- **Ek feature'lar:** `net_load_lag0`, `renewable_pressure_ratio_lag0`, `solar_peak_pressure_ratio_lag0`, `solar_ramp_rate_lag0`, `renewable_ramp_rate_lag0`, `zero_price_risk_score`
- **min_child_samples:** 10 (canlıda 20)
- **Backtest türü:** Walk-forward (her gün yeniden eğitim)

### `lgb_cqr_v2` vs `lgb_lag0_v2` Farkları
- **P50 tahmini:** Aynı (değişmez)
- **P10/P90 güven aralığı:** CQR (Conformalized Quantile Regression) ile kalibre edilmiş
  - 30 günlük rolling pencere üzerinden non-conformity score hesaplanır
  - Hedef coverage: %80
  - P10 aşağı, P90 yukarı genişletilerek gerçek kapsama oranı iyileştirilir
  - Native quantile'a göre daha güvenilir güven aralığı üretmesi beklenir

---

## 1. Canlı Model (LightGBM Quantile) vs Eski Log1p Benchmark

Notebook'lardaki WAPE rakamları büyük ölçüde **eski log1p modele** aittir. Canlıdaki quantile model ile karıştırılmamalıdır.

### Eski Log1p Model Sonuçları (Referans, artık canlıda değil)

| Model | MAE ($/MWh) | WAPE (%) | Dönem | Kaynak |
|-------|-------------|----------|-------|--------|
| LightGBM Log1p (Eski) | $8.44 | 16.53% | 1 yıl WF | `1year_paper_strategies_summary.csv` |
| LightGBM Log1p + Enhanced | $8.16 | 16.13% | 1 yıl WF | `zero_price_forecasting_experiment.ipynb` |
| LightGBM Log-Transform Quantile | $8.09 | 15.99% | 1 yıl WF | `zero_price_forecasting_experiment.ipynb` |

### Canlı Quantile Model Dashboard Performansı

Canlı modelin performansı `scripts/api_server.py`'daki WAPE/MAE endpoint'inden veya doğrudan DB karşılaştırmasından alınabilir. Notebook'lardaki eski rakamlar canlı modeli yansıtmaz.

### lgb_lag0_v2 Canlı Test (13 Ağustos 2026) — canlıya alma kararının dayanağı

| Model | MAE ($/MWh) | WAPE (%) | 80% Coverage | Kaynak |
|-------|-------------|----------|--------------|--------|
| pre-lag0 canlı model | $16.24 | 28.33% | 25.0% (6/24 saat) | `zero_price_forecasting_experiment.ipynb` |
| **lgb_lag0_v2** | **$9.12** | **15.92%** | **79.2% (19/24 saat)** | `zero_price_forecasting_experiment.ipynb` |

**Tek gün testi** (2026-08-13). Bu sonuç lgb_lag0_v2'nin aynı gün canlıya alınmasını
tetikledi. **UYARI:** Tek güne dayalı; tam Şub-Haz 2026 çöküş dilimine yayıldığında
kazanç çok daha küçük — o dönem lgb_lag0_v2 MAE'si hâlâ ~$12-13
(`LOW_PRICE_REGIME_ANALYSIS.md` §1b: ucuz dilimde pre-lag0 ile fark ~$0). Yani
lag0 oranları **13 Ağustos gibi tek günlerde** yardımcı, **rejim boyunca** yetersiz.

---

## 2. Güven Aralığı Karşılaştırması

| Yöntem | Coverage Hedefi | Avantaj | Dezavantaj |
|--------|----------------|---------|------------|
| **Native Quantile** (Canlı) | ~%80 (nominal) | Basit, model içi | Gerçek coverage nominal'den sapabilir |
| **CQR Kalibre** (lgb_cqr_v2) | %80 (garanti) | Sonlu-örneklem geçerlilik garantisi | Aralık genişleyebilir, ek hesaplama |

CQR'ın temel avantajı: Native quantile'ın gerçek coverage'ı %77 ise, CQR bunu %80'e çıkarır (P10'u aşağı, P90'ı yukarı iterek). lgb_cqr_v2 ile lgb_lag0_v2'nin P50 tahminleri aynıdır — sadece güven aralığı kalibrasyonu farklıdır.

---

## 3. EPNet Derin Öğrenme Deneyleri

### EPNet vs LightGBM Kronolojik Kırılım (359 gün, eski log1p karşılaştırması)

| Dönem | EPNet WAPE | LightGBM (log1p) WAPE | Kazanan |
|-------|------------|----------------------|---------|
| İlk 6 Ay | 10.09% | 10.12% | ~Eşit |
| İlk 9 Ay | 32.61% | 21.57% | LightGBM |
| Tam 12 Ay | 39.67% | 26.88% | LightGBM |

**Kritik bulgu:** EPNet bahar rejim kaymasında çöktü. Hybrid routing (EPNet+LightGBM blend) sadece LightGBM'den kötü sonuç verdi (%32.68 vs %26.88).

### EPNet Grid Search Özeti

| En İyi Strateji | 1M WAPE | 12M WAPE | 12M MAE |
|-----------------|---------|----------|---------|
| ROBUST_02 (Cyclic+Huber) | 24.58% | 39.67% | $9.85 |
| ROBUST_03 (Combined+Huber) | 25.33% | 39.06% | $9.85 |
| STRAT_03 (Pure PTF, Classic) | 33.74% | 32.53% | $10.10 |

---

## 4. Paper-Driven Stratejileri (Eski log1p model üzerinde)

| Strateji | 1 Yıl MAE | 1 Yıl WAPE | Baseline'ı Yendi mi? |
|----------|-----------|------------|---------------------|
| Baseline LightGBM (Log1p) | $8.44 | 16.53% | -- (referans) |
| Multi-Stage Residual Boosting | $8.56 | 16.77% | Hayır |
| Multi-Window Calibration | $8.46 | 16.57% | Hayır |
| Ultimate Paper Hybrid | $8.50 | 16.66% | Hayır |
| ArcSinH MAD Transform | $9.31 | 18.25% | Hayır |

**Sonuç:** Hiçbir paper stratejisi baseline'ı geçemedi.

---

## 5. MWA Baseline Karşılaştırması

| Model | 12M MAE | 12M WAPE |
|-------|---------|----------|
| LightGBM (canlı) | $8.38 | 16.44% |
| MWA Optimized (Scipy) | $9.74 | 19.09% |
| MWA Heuristic | $10.19 | 19.98% |

---

## 6. Pre-Forecaster Sonuçları

| Hedef | Model | WAPE | MAE | Dönem | Kaynak |
|-------|-------|------|-----|-------|--------|
| **Load Forecast** | LightGBM + CDH/HDH | **%2.89** | 1,157 MW | 2 yıl (17,520 saat) | DB hesaplama |
| **Solar KGUP** | Ridge (14 lag) | **%10.53** | 347 MW | 2 yıl (17,544 saat) | DB hesaplama |
| **Wind KGUP** | LightGBM + OpenMeteo | **%18.02** | 859 MW | 2 yıl (17,544 saat) | DB hesaplama |

Hesaplama: `gold.kgup_load_pre_forecasts` vs `raw_load_forecast_hourly` / `raw_kgup_hourly` (2024-08-13 → 2026-08-14)

---

## 7. Kayıp/Eksik Sonuçlar

| Deney Grubu | Toplam | Sonucu Mevcut | Kayıp |
|-------------|--------|---------------|-------|
| PTF Model Karşılaştırma | 8 | 6 | 2 |
| EPNet Grid Search | 26+ | 26 (log dosyaları) | 0 |
| Zero-Price Classifier | 5 | 0 | **5** |
| Pre-Forecaster | 3 | **3** (DB'den hesaplandı) | 0 |
| LSTM Online Learning | 3 | 0 | **3** |

---

## 8. Operasyonel Log Bulguları

| Sorun | Sıklık | Etki |
|-------|--------|------|
| EPİAŞ DNS/SSL timeout | Tekrarlayan | ETL adımı atlanır, retry ile düzelir |
| yfinance timeout | Tekrarlayan | Macro veri ffill ile telafi |
| Pre-forecast `'Index' has no attr 'hour'` | 1 kez | EPİAŞ fallback'e düşer |

---

## 9. Sonuç ve Öncelikli Aksiyonlar

### Kesin Bulgular
1. **Canlı model quantile regression** (P10/P50/P90), log1p değil, **13 Ağu 2026'dan beri lgb_lag0_v2**
2. **lgb_lag0_v2 çöküşü KISMİ iyileştirdi** — tek gün testinde WAPE %28→%16, ama tam
   Şub-Mar 2026 diliminde MAE hâlâ ~$12-13 (sakin aylar ~$8). Çözmedi.
   Sebep kayıp fonksiyonu değil: hidro kaynaklı merit-order rejim kırılması
   (`LOW_PRICE_REGIME_ANALYSIS.md` §3-5) — hidro payı yıllar arası **transfer olmuyor**,
   termal ihtiyaç oranı oluyor.
3. **CQR kalibrasyonu genel MAE'yi bozuyor** ($14 vs $8) — canlıya alınmadı
4. **EPNet ve hybrid routing faydasız** — LightGBM tek başına daha iyi
5. **Paper stratejileri baseline'ı geçememiş**
6. **12/53 deney sonucu kayıp**
7. **Kanonik Lago LEAR (gerçek epftoolbox, 4-pencere ensemble) canlı LightGBM'i
   geçemiyor.** Dondurulmuş protokol, 728g birincil pencere (`04_canonical_lago_benchmark.ipynb`):
   LightGBM (lag0_v2) $7.35 / rMAE(n2) 0.644 vs LEAR ensemble $8.08 / 0.709,
   **DM p=1.9e-9**. Rejim katmanlarının hepsinde LightGBM önde (çöküş: 10.54 vs 12.98).
   Tam beslenmiş 511-özellikli LEAR de geride (`03`: $9.34, p=4e-9).
   Lago'nun 4-yıl (cw1456) penceresi TR'ye transfer olmuyor (çöküşte $16-17).
   Bkz. `07_lago_protocol/`, `LAGO_BENCHMARK_PLAN.md`.
8. **Çok-pencereli LightGBM ensemble çöküş körlüğünü kısmen çözdü — ilk canlıyı geçen
   deneysel değişiklik.** `base(tüm geçmiş) + roll90 + roll150` eşit ağırlık, aynı
   `lgb_lag0_v2` konfigü. 2-yıl WF (proxy'siz): MAE $7.27→$7.04, çöküş $10.54→**$9.62**
   (BIAS +3.76→+1.54), DM base-vs-ensemble p<0.0001 (normal nötr). Konformal band
   (N=60/w=0.5/FLOOR=$0) 3-head quantile'ı değiştiriyor — kapsama her rejimde %80–82
   (canlı %60–76). **Karar kapısı geçildi, canlıya geçiş brief'i:
   `../enerji_fiyat_tahmini/ENSEMBLE_IMPLEMENTASYON.md`.** Detay §10.

### Öncelikli Aksiyonlar
1. ~~`lgb_lag0_v2` canlıya alma~~ **✅ YAPILDI (13 Ağu 2026, commit `d7c67bf`).**
2. **Hidroyu diğer yenilenebilirlerle eşit muamele et** (`LOW_PRICE_REGIME_ANALYSIS.md` §8):
   pre-forecaster'lara hidro ekle (KGÜP hidro planı zaten var), `net_load_lag0`'dan
   hidroyu çıkar, jeotermal + biyokütleyi must-run olarak dahil et.
3. **Açık `thermal_requirement_ratio` feature'ı ekle** — yıllar arası transfer olan tek
   büyüklük (`LOW_PRICE_REGIME_ANALYSIS.md` §5). Mevcut `renewable_pressure_ratio_lag0`
   bunun bulanık yaklaşığı (hidro lag24 yüzünden).
4. **Ölçüm:** 2+3'ü birlikte kur, **Şub-Haz 2026 walk-forward backtest**. Başarı kriteri:
   o dönem MAE'sini $11-13'ten ne kadar indirdiği — **WAPE'e bakma** (payda tuzağı, §2).
   → `LAGO_BENCHMARK_PLAN.md` Faz C.5.
5. **Baraj doluluk geçmişini geriye doldur** (`raw_master_water_energy_provision` /
   active-fullness — DB'de sadece 2026-08 var). Yağış/doluluk hipotezinin ön şartı.
6. Eğitim penceresi pinini (`TRAINING_DATA_START=2023-01-01`) benchmark'ta ölç —
   2022 spike rejimini dışlaması haklı mı? (Faz C.5)
7. Sistematik deney altyapısı (bkz. `EXPERIMENT_WORKFLOW.md`)

---

## 10. Çok-pencereli ensemble + konformal band (Faz C.5, 31 Ağu – 1 Eyl 2026)

Tüm kanıt: `07_lago_protocol/08_collapse_fix_experiments.ipynb`. Metodoloji: `run_wf_lightgbm.py`
proxy'siz WF (kaskad bug 31 Ağu düzeltildi, tüm koşular `_v2`). Framework doğrulaması:
proxy'siz `base` MAE $7.23 ≈ canlı DB $7.27.

**P50 = `(base + roll90 + roll150) / 3`** — 3× LightGBM, aynı `lgb_lag0_v2` konfigü, farklı
eğitim pencereleri (son 90g / son 150g / tüm geçmiş 2023-01+). LEAR ensemble mantığının
LightGBM karşılığı.

| dilim | canlı DB | ensemble | DM (base vs ens) |
|---|---|---|---|
| 2-yıl MAE / rMAE | $7.27 / 0.645 | **$7.04 / 0.628** | p<0.0001 |
| normal MAE | $6.23 | $6.21 (nötr, sıfır-saat 9.8→12.2) | p=0.61 |
| çöküş MAE / BIAS | $10.54 / +3.76 | **$9.62 / +1.54** | p<0.0001 |
| toparlanma MAE / BIAS | $8.63 / +1.51 | $8.20 / −0.81 | p=0.11 (58g) |

- **63 alt-küme tarandı** → plato ~$7.00, `base+r90+r150` en temiz. Uyarlanabilir ağırlık
  (LEAR eq.10) katkı yok.
- **C.5.5 (rejim-tetikli ağırlık) ELENDİ:** çöküş MAE `base` ağırlığından bağımsız (~$9.6 sabit);
  ağırlık sadece bias kolu. Statik `base 0.5×` DM'de hiçbir rejimde üstün değil. Default eşit
  ağırlık, rejim dedektörü kurulmadı.

**P10/P90 = nedensel split-conformal** (3-head quantile'ın yerine): `L = max(0, P50 −
q₀.₉₀(err₆₀ₐ,ₕ) − 0.5·std{üyeler})`, `U = P50 − q₀.₁₀(...) + 0.5·std{...}`. `err = tahmin−gerçek`,
saat bazlı. Kapsama her rejim + son 30/90g'de %80–82 (canlı 3-head %60–76). **Kırılım:**
gerçekleşen $20–60 aralığında ~%70'e düşüyor (modelin kalıntı yukarı biası), $100+ spike'lar
yukarı kaçıyor.

**Durum:** karar kapısı geçildi, deney reposunda doğrulandı, **canlıya kod yazılmadı.**
Devir brief'i `../enerji_fiyat_tahmini/ENSEMBLE_IMPLEMENTASYON.md` (Faz 1 = 2 altyapı bug'ı,
Faz 2 = `src/models/ensemble.py`, Faz 3 = shadow ≥14g).
