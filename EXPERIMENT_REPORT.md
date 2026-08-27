# Deney Raporu - Model & Strateji Karşılaştırması

## Yönetici Özeti

Projede iki ayrı model versiyonu ve çok sayıda deneysel strateji test edilmiştir. Canlıdaki model **LightGBM 3-head quantile regression** (P10/P50/P90) olup **log1p kullanmaz**. Önceki versiyondaki log1p model artık canlıda değildir. Deneysel `lgb_lag0_v2` modeli lag0 renewable ratios ekleyerek özellikle sıfır-fiyat saatlerinde iyileştirme sağlamıştır. `lgb_cqr_v2` ise aynı modelin CQR kalibre edilmiş güven aralığı versiyonudur.

---

## Model Versiyonları (Durum Tablosu)

| Model | Tablo | Objective | Log1p | Lag0 Ratios | min_child | Güven Aralığı | Durum |
|-------|-------|-----------|-------|-------------|-----------|---------------|-------|
| **LightGBM_v1** (Canlı) | `gold.ptf_predictions_daily` | quantile | Hayır | Hayır* | 20 (default) | Native quantile P10/P90 | **CANLI** |
| **lgb_lag0_v2** (Deneysel) | `gold.ptf_predictions_experimental` | quantile | Hayır | **Evet** | 10 | Native quantile P10/P90 | Deneysel (DB'de 730 gün backfill) |
| **lgb_cqr_v2** (Deneysel) | `gold.ptf_predictions_experimental` | quantile | Hayır | **Evet** | 10 | **CQR kalibre** P10/P90 | Deneysel (DB'de 730 gün backfill) |
| Eski Log1p Model | Yok (artık kullanılmıyor) | regression | **Evet** | Hayır | 20 | Yok | **KALDIRILDI** |

*Not: Kod (`feature_engineering.py`) güncellenerek lag0 feature'lar `robust` set'e eklenmiş, ancak canlıdaki geçmiş backfill eski feature set ile üretilmiş durumda. Günlük tahminlerde kod güncel olsa da geçmiş performans metrikleri eski feature set'i yansıtıyor.

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

### lgb_lag0_v2 Canlı Test (13 Ağustos 2026)

| Model | MAE ($/MWh) | WAPE (%) | 80% Coverage | Kaynak |
|-------|-------------|----------|--------------|--------|
| Canlı model (LightGBM_v1) | $16.24 | 28.33% | 25.0% (6/24 saat) | `zero_price_forecasting_experiment.ipynb` |
| **lgb_lag0_v2** | **$9.12** | **15.92%** | **79.2% (19/24 saat)** | `zero_price_forecasting_experiment.ipynb` |

Bu test özellikle düşük fiyat saatlerinde `renewable_pressure_ratio_lag0`'ın etkisini göstermektedir. Coverage farkı (%25 → %79) güven aralığı kalitesindeki iyileşmeyi de yansıtır.

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
1. **Canlı model quantile regression** (P10/P50/P90), log1p değil
2. **lgb_lag0_v2 önemli iyileştirme sağlıyor** (WAPE %28→%16 canlı testte)
3. **CQR kalibrasyonu güven aralığı kalitesini artırabilir** (lgb_cqr_v2)
4. **EPNet ve hybrid routing faydasız** — LightGBM tek başına daha iyi
5. **Paper stratejileri baseline'ı geçememiş**
6. **12/53 deney sonucu kayıp**

### Öncelikli Aksiyonlar
1. `lgb_lag0_v2` canlıya alma — backfill zaten `gold.ptf_predictions_experimental`'da mevcut (730 gün). Sadece:
   - Experimental tablodan production tablosuna veri taşıma (INSERT ... SELECT ... ON CONFLICT)
   - `predict_daily_pipeline.py`'da `min_child_samples: 10` ekleme
   - Model adı güncelleme
2. CQR vs native quantile coverage karşılaştırması (DB'deki verilerle yapılabilir)
3. Kayıp classifier sonuçlarını tekrar üretip kaydetme
4. Sistematik deney altyapısı kurma (bkz. `EXPERIMENT_WORKFLOW.md`)
