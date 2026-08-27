# CLAUDE.md - Enerji Fiyat Tahmini: Deney & Araştırma Reposu

## Bu repo nedir

`enerji_fiyat_tahmini` (canlı dashboard/ETL/model reposu) ile **aynı PostgreSQL
veritabanına bağlanan**, ama ayrı bir git geçmişi olan kardeş repo. 24 Ağustos
2026'da canlı pipeline ile deney/araştırma kodu birbirine karışmaya başladığı
için ayrıldı — canlı repo yalnızca dashboard'u besleyen model/ETL/DB'yi
tutuyor, bu repo tüm deneyleri, literatür taramasını ve kriz/olay istihbarat
araştırmasını tutuyor.

**Canlı repo yolu:** `../enerji_fiyat_tahmini` (sibling dizin). Model
versiyonları, canlı pipeline akışı, veri kaynakları için oradaki `CLAUDE.md`'ye
bak — burada tekrarlanmıyor.

## DB erişimi

`db/connection.py` canlı repodan birebir kopyalandı, `.env` de aynı
credential'larla kopyalandı (**tam erişim, salt-okunur değil** — bazı deney
scriptleri `gold.ptf_predictions_experimental` / `gold.experiment_results` /
`bronze.news_raw` gibi tablolara yazıyor). Canlı `raw_*`/`gold.ptf_predictions_daily`
tablolarına **yazma yapma** — sadece deneysel/gold-experimental şemalara.

## Dizin yapısı

```
electricity_price_forecasting_in_turkish_day_ahead_market/
├── experiments/
│   ├── notebooks/           # Tüm analiz defterleri (çoğu büyük ölçüde git-dışı kalmıştı, artık burada asıl)
│   ├── scripts/              # Backtest runner'lar
│   ├── research_scripts/, generators/
│   └── experiment_master_log.csv
├── literature/                # Literatür taraması (lit_search.py, lit.db, head_to_head.py)
├── docs/
│   └── literature_main/       # Referans PDF'ler (Lago, O'Connor review, vs.)
├── src/
│   ├── models/{epnet,cqr_calibrator}.py   # Deneysel modeller
│   ├── routing/                            # Rejim bazlı yönlendirme (deneysel)
│   ├── eval/lago_protocol.py               # Lago et al. (2021) protokolü
│   └── crisis/                             # Haber etiketleme, kriz analiz şeması
├── scripts/
│   ├── lit_search.py, build_analysis_model.py, crisis_case_report.py
│   └── fetch_news_archive.py               # Haber arşivi toplayıcı (canlı pipeline'ın PARÇASI DEĞİL)
├── db/connection.py            # Canlı repodan kopya
└── requirements.txt             # Ağır ML/araştırma bağımlılıkları (torch, shap, vs.)
```

## Şu anki aktif faz (24 Ağustos 2026'dan devam)

`KATMAN_0_1_2_YOL_HARITASI.md` — kullanıcı model doğrulama + fiyat sürücü
analizini teoriyi kapatarak, hipotez testleriyle kendi eliyle yeniden kuruyor.
AI otonom analiz üretmiyor, ilerlemeyi gözden geçiriyor/tartışıyor. Katman 3
(`EVENT_IMPACT_STUDY.md` — geçmiş olayların fiyata etkisi) bu bitene kadar
bekletiliyor.

Diğer önemli dokümanlar: `LITERATURE_REVIEW.md`, `METRICS.md` (metrik
tanımları + naive baseline), `EXPERIMENT_REPORT.md` (tüm model deney
sonuçları), `LOW_PRICE_REGIME_ANALYSIS.md`, `CRISIS_ANALYSIS_PLAN.md`.

## Önemli notlar

- **`epftoolbox` (gerçek referans LEAR/DNN kütüphanesi) bu repoda değil.**
  `numpy<2` + tensorflow zorunluluğu ana `requirements.txt` ile çakışıyor;
  izole bir venv'de (`scratchpad` veya benzeri geçici bir yerde) kurulup
  kullanılıyor — bkz. `experiments/notebooks/07_lago_protocol/run_crossval_reference*.py`.
- **Pre-forecast lag0 özellikleri** (`renewable_pressure_ratio_lag0` vs.)
  `gold.kgup_load_pre_forecasts`'a bağımlı; bu tablo yalnızca **2024-08-13'ten
  itibaren** dolu. Daha eski tarihli backtest'lerde bu özellikler fallback
  (gerçek/aynı-gün değer) ile dolduruluyor, gerçek ön-tahmin değil.
- **`src/crisis/` + `scripts/fetch_news_archive.py` canlı 04:00 pipeline'ının
  parçası değil** — DB'ye yazıyor (`bronze.news_raw`) ama `daily_update_pipeline.py`
  bunu hiç çağırmıyor, elle/ayrı çalıştırılıyor.
- **Deneysel modeller hiçbiri canlıyı geçemedi** — EPNet, LSTM online-learning,
  hibrit sınıflandırıcı, paper-driven stratejiler — hepsi `EXPERIMENT_REPORT.md`'de.
  Yeni bir mimari denemeden önce oradaki sonuçlara bak.
