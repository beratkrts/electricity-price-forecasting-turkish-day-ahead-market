# Sonnet Aksiyon Planı

Bu plan 4 fazdan oluşur. Her faz bağımsız commit edilebilir.

- [x] **Faz 1: lgb_lag0_v2'yi Canlıya Alma** — **TAMAMLANDI ✅**
- [x] **Faz 2: Tespit Edilen Hataları Düzeltme** — **TAMAMLANDI ✅**
- [x] **Faz 3: Repo Temizliği & Modül İşaretleme** — **TAMAMLANDI ✅**
- [ ] **Faz 4: Sistematik Deney Altyapısı** — **PLANLANDI (Gelecek Aşama ⏳)**

---

## Faz 1: lgb_lag0_v2'yi Canlıya Alma (TAMAMLANDI ✅)

### Ön Bilgi
- `gold.ptf_predictions_experimental` tablosunda `lgb_lag0_v2` adıyla 17,472 satır (2024-08-12 → 2026-08-12) hazır walk-forward backfill var.
- `gold.ptf_predictions_daily` tablosunda `LightGBM_v1` adıyla 17,496 satır var.
- Veriyi taşıyıp pipeline'ı güncellemek yeterli. Yeniden eğitim/backfill gerekmiyor.

### Adım 1.1: Mevcut Canlı Veriyi Deneysel Tabloya Yedekle
Önce mevcut LightGBM_v1 (eski model) verilerini experimental tabloya `lgb_baseline_v1` adıyla kaydet:

```sql
INSERT INTO gold.ptf_predictions_experimental (
    target_ts, model_name,
    predicted_mcp_usd, predicted_mcp_try,
    predicted_mcp_usd_p10, predicted_mcp_try_p10,
    predicted_mcp_usd_p90, predicted_mcp_try_p90
)
SELECT
    target_ts, 'lgb_baseline_v1',
    predicted_mcp_usd, predicted_mcp_try,
    predicted_mcp_usd_p10, predicted_mcp_try_p10,
    predicted_mcp_usd_p90, predicted_mcp_try_p90
FROM gold.ptf_predictions_daily
WHERE model_name = 'LightGBM_v1'
ON CONFLICT (target_ts, model_name) DO NOTHING;
```

Bu sayede eski modelin 17,496 satırlık geçmişi `lgb_baseline_v1` olarak korunur ve ileride karşılaştırma yapılabilir.

### Adım 1.2: lgb_lag0_v2 Verisini Canlıya Taşı
Experimental tablodan production tablosuna veri kopyala:

```sql
INSERT INTO gold.ptf_predictions_daily (
    target_ts, predicted_mcp_usd, predicted_mcp_try,
    predicted_mcp_usd_p10, predicted_mcp_try_p10,
    predicted_mcp_usd_p90, predicted_mcp_try_p90,
    model_name
)
SELECT
    target_ts, predicted_mcp_usd, predicted_mcp_try,
    predicted_mcp_usd_p10, predicted_mcp_try_p10,
    predicted_mcp_usd_p90, predicted_mcp_try_p90,
    'LightGBM_v1'  -- Aynı model adını koru (API ve frontend buna bağlı)
FROM gold.ptf_predictions_experimental
WHERE model_name = 'lgb_lag0_v2'
ON CONFLICT (target_ts, model_name) DO UPDATE SET
    predicted_mcp_usd = EXCLUDED.predicted_mcp_usd,
    predicted_mcp_try = EXCLUDED.predicted_mcp_try,
    predicted_mcp_usd_p10 = EXCLUDED.predicted_mcp_usd_p10,
    predicted_mcp_try_p10 = EXCLUDED.predicted_mcp_try_p10,
    predicted_mcp_usd_p90 = EXCLUDED.predicted_mcp_usd_p90,
    predicted_mcp_try_p90 = EXCLUDED.predicted_mcp_try_p90,
    created_at = CURRENT_TIMESTAMP;
```

**DİKKAT:** model_name'i `LightGBM_v1` olarak bırak. API server ve frontend bu isme bağlı. Değiştirmek frontend'de kırılma yaratır.

### Adım 1.3: predict_daily_pipeline.py Güncelleme
**Dosya:** `scripts/predict_daily_pipeline.py`

3 model tanımında (P50, P10, P90 — satır 190-211) `min_child_samples: 10` ekle:

```python
# Mevcut (satır 190-194):
forecaster = LightGBMForecaster(params={
    'objective': 'quantile', 'alpha': 0.50,
    'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63,
    'verbose': -1, 'random_state': 42
})

# Güncel:
forecaster = LightGBMForecaster(params={
    'objective': 'quantile', 'alpha': 0.50,
    'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63,
    'min_child_samples': 10,
    'verbose': -1, 'random_state': 42
})
```

Aynı değişikliği P10 (satır 198-202) ve P90 (satır 206-210) için de yap.

### Adım 1.4: backfill_gold_predictions.py Güncelleme
**Dosya:** `scripts/backfill_gold_predictions.py`

Aynı şekilde 3 model tanımında (satır 62, 66, 70) `min_child_samples: 10` ekle.
Bu, gelecekte backfill çalıştırılırsa tutarlılık sağlar.

### Adım 1.5: Doğrulama
Pipeline'ın lag0 feature'ları doğru kullandığını doğrulamak için kontrol:
- `predict_daily_pipeline.py:175`: `feature_cols = get_feature_columns('robust', df_feat)`
- `feature_engineering.py:231-232`: `robust_cols` listesinde `renewable_pressure_ratio_lag0`, `net_load_lag0` vb. zaten var
- Ek kod değişikliği gerekmiyor — feature set otomatik olarak doğru

---

## Faz 2: Tespit Edilen Hataları Düzeltme

Tüm hatalar `ISSUES.md`'de detaylı. Aşağıdaki sırayla düzelt:

### Adım 2.1: SMP TRY/USD Feature Adı Düzeltme (ISSUES.md #2)
**Dosya:** `src/features/feature_engineering.py:139`

```python
# Mevcut:
df_feat['smp_usd_lag_48'] = (df_feat['smp_price_try'] / df_feat['usd_try']).shift(48)

# Düzelt — aslında USD'ye çevriliyor (smp_try / usd_try), sadece zaten doğru.
# Ama isim tutarlılığı için ve kodun niyetini açıklamak için yorum ekle:
# SMP TRY -> USD dönüşümü yapılıp 48h lag uygulanıyor
```

Burada dikkat: `smp_price_try / usd_try` zaten USD'ye çeviriyor! Yani feature adı `smp_usd_lag_48` aslında doğru. Sorun yokmuş, sadece `get_feature_columns`'daki referansla eşleştiğini doğrula. **Değişiklik gerekmiyor, ISSUES.md güncelle.**

### Adım 2.2: Veri Boşluğu Kontrolünü Esnet (ISSUES.md #1)
**Dosya:** `scripts/predict_daily_pipeline.py:229-231`

```python
# Mevcut (katı):
if last_available_date != target_today_str:
    raise RuntimeError(...)

# Önerilen (esnek):
days_gap = (pd.Timestamp(target_today_str) - pd.Timestamp(last_available_date)).days
if days_gap > 2:
    raise RuntimeError(f"Data gap too large ({days_gap} days). Latest: {last_available_date}, Expected: {target_today_str}")
elif days_gap > 0:
    logger.warning(f"⚠️ Data gap: {days_gap} day(s). Latest data: {last_available_date}. Proceeding with available data.")
```

### Adım 2.3: run_service.py Duplicate Import (ISSUES.md #5)
**Dosya:** `scripts/run_service.py`

Satır 1-6'yı düzenle. Docstring'i en üste al, duplicate `import sys` kaldır:
```python
"""24/7 Continuous Background Daemon Service.
...
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import time
import logging
# ... (import sys'in ikinci kopyasını kaldır)
```

### Adım 2.4: Log Rotation Ekle (ISSUES.md #6)
**Dosya:** `scripts/daily_update_pipeline.py:38-39`

```python
# Mevcut:
main_file_handler = logging.FileHandler("logs/daily_update.log", encoding="utf-8")
anomaly_file_handler = logging.FileHandler("logs/anomalies.log", encoding="utf-8")

# Güncel:
from logging.handlers import RotatingFileHandler
main_file_handler = RotatingFileHandler("logs/daily_update.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
anomaly_file_handler = RotatingFileHandler("logs/anomalies.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
```

### Adım 2.5: FX prevClose DB Fallback (ISSUES.md #7)
**Dosya:** `scripts/api_server.py`

FX endpoint'inde prevClose boş geldiğinde DB'den önceki günün kurunu çek:
```python
# ExchangeRate-API yanıtı prevClose döndürmüyor.
# Fallback: raw_macro_daily tablosundan dünkü kuru al
if not prev_close_usd:
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT usd_try FROM raw_macro_daily WHERE entry_date < CURRENT_DATE ORDER BY entry_date DESC LIMIT 1"
        )).fetchone()
        if row:
            prev_close_usd = float(row[0])
```

---

## Faz 3: Repo Temizliği

### Adım 3.0: scratch/ Klasörünü Sil
```bash
rm -rf scratch/
```
Tek seferlik debug scriptleri, notebook üreticileri, geçici PNG'ler. Git'e hiç girmemiş (`.gitignore`'da `scratch/` var). Güvenle silinebilir.

### Adım 3.1: scripts/ Klasörünü Ayır

**Canlı pipeline scriptleri** (scripts/ içinde kalsın):
```
scripts/
├── start.sh
├── run_service.py
├── daily_update_pipeline.py
├── predict_daily_pipeline.py
├── fetch_epias_data.py
├── api_server.py
├── backfill_gold_predictions.py
└── backfill_pre_forecasts.py
```

**Deneysel scriptler** → `experiments/scripts/` altına taşı:
```
experiments/scripts/
├── backfill_experimental_predictions.py
├── backfill_cqr_experimental_predictions.py
├── fast_cqr_update.py
├── run_epnet_backtest.py
├── run_epnet_overnight_grid.py
├── run_epnet_zero_price_robust.py
├── run_hybrid_backtest.py
├── run_lightgbm_standalone_backtest.py
├── run_regime_backtest.py
├── test_mwa_lgbm_strategies.py
├── verify_lgbm_and_routing.py
└── db_etl_pipeline.py          # Eski ETL, artık daily_update_pipeline.py ile değiştirildi
```

**Taşıma komutu:**
```bash
mkdir -p experiments/scripts
git mv scripts/backfill_experimental_predictions.py experiments/scripts/
git mv scripts/backfill_cqr_experimental_predictions.py experiments/scripts/
git mv scripts/fast_cqr_update.py experiments/scripts/
git mv scripts/run_epnet_backtest.py experiments/scripts/
git mv scripts/run_epnet_overnight_grid.py experiments/scripts/
git mv scripts/run_epnet_zero_price_robust.py experiments/scripts/
git mv scripts/run_hybrid_backtest.py experiments/scripts/
git mv scripts/run_lightgbm_standalone_backtest.py experiments/scripts/
git mv scripts/run_regime_backtest.py experiments/scripts/
git mv scripts/test_mwa_lgbm_strategies.py experiments/scripts/
git mv scripts/verify_lgbm_and_routing.py experiments/scripts/
git mv scripts/db_etl_pipeline.py experiments/scripts/
```

### Adım 3.2: eda/ Klasörünü Organize Et

**Hedef yapı:**
```
experiments/
├── scripts/                          # Faz 3.1'de taşınan backtest scriptleri
├── notebooks/
│   ├── 01_data_exploration/          # Veri keşfi
│   │   ├── explore_ptf.ipynb
│   │   ├── explore_json.ipynb
│   │   └── comprehensive_eda.ipynb
│   │
│   ├── 02_preforecasters/            # Yük/Güneş/Rüzgar alt-modelleri
│   │   ├── advanced_kgup_load_forecaster.ipynb
│   │   ├── baseline_gen_cons_forecast.ipynb
│   │   └── weather_forecast_impact_analysis.ipynb
│   │
│   ├── 03_ptf_model_comparison/      # PTF model deneyleri
│   │   ├── 02_model_experiments_daily_forecasting.ipynb
│   │   ├── 03_epnet_vs_lightgbm_performance_analysis.ipynb
│   │   ├── mwa_vs_lightgbm_backtest.ipynb
│   │   ├── mwa_baseline_test.ipynb
│   │   └── paper_driven_strategies_experiments.ipynb
│   │
│   ├── 04_zero_price_crisis/         # Sıfır-fiyat analiz ve çözümleri
│   │   ├── regime_shift_analysis.ipynb
│   │   ├── zero_price_forecasting_experiment.ipynb
│   │   ├── hybrid_ptf_classifier.ipynb
│   │   └── lstm_online_learning_experiment.ipynb  ⚠️ AKTİF RUNTIME — taşırken kernel'ı kapatma!
│   │
│   ├── 05_confidence_intervals/      # Güven aralığı deneyleri
│   │   └── (confidence_interval_test.py → notebook'a çevrilebilir)
│   │
│   └── 06_presentation/             # Sunum/rapor notebook'ları
│       └── final_presentation_analysis.ipynb
│
├── research_scripts/                 # EDA Python scriptleri (notebook üretici değil)
│   ├── ptf_ab_test.py
│   ├── ptf_1yr_walk_forward.py
│   ├── hybrid_1yr_walk_forward.py
│   ├── full_two_stage_hybrid_test.py
│   ├── ultimate_hybrid_pipeline.py
│   ├── run_walkforward_backtest.py
│   ├── run_zero_price_experiment.py
│   ├── confidence_interval_test.py
│   ├── zppi_classifier_test.py
│   ├── snowmelt_classifier_test.py
│   ├── zero_price_analysis.py
│   ├── train_mwa_kgup.py
│   ├── train_load_weather.py
│   └── test_wind_openmeteo.py
│
└── generators/                       # Notebook üretici scriptler (çalıştırma gerekmiyor)
    ├── create_notebook_v2.py
    ├── create_final_notebook.py
    ├── create_hybrid_notebook.py
    ├── generate_notebook.py
    └── generate_lstm_notebook.py
```

**Taşıma:**
```bash
# Ana dizinleri oluştur
mkdir -p experiments/notebooks/{01_data_exploration,02_preforecasters,03_ptf_model_comparison,04_zero_price_crisis,05_confidence_intervals,06_presentation}
mkdir -p experiments/{research_scripts,generators}

# lgb_trial.ipynb boş notebook — sil
rm eda/lgb_trial.ipynb

# Notebook'ları taşı (git mv ile)
git mv eda/explore_ptf.ipynb experiments/notebooks/01_data_exploration/
git mv eda/explore_json.ipynb experiments/notebooks/01_data_exploration/
git mv eda/comprehensive_eda.ipynb experiments/notebooks/01_data_exploration/
# ... (diğerleri yukarıdaki yapıya göre)

# Research scripts
git mv eda/ptf_ab_test.py experiments/research_scripts/
# ... (diğerleri)

# Generators
git mv eda/create_notebook_v2.py experiments/generators/
# ... (diğerleri)
```

**Not:** `.gitignore`'da `eda/` ve `*.ipynb` ignore edilmiş. Taşıma sonrası `experiments/` klasörünü ve notebook'ları gitignore'dan çıkarmak gerekebilir ya da ignore bırakılabilir (kullanıcı tercihine göre).

### Adım 3.3: Frontend Temizliği (ISSUES.md #3)
**Dosyalar:** `app/frontend/src/types/energy.ts`, `app/frontend/src/App.tsx`

- `epnetForecast`, `hybridForecast` alanlarını type'lardan kaldır
- `seriesConfigs`'ten EPNet ve Hybrid toggle'larını kaldır
- Sadece LightGBM + P10/P90 bırak
- `smf` alanını tut (referans veri olarak kullanılabilir)

### Adım 3.4: Deneysel Modülleri İşaretle
**Dosyalar:** `src/routing/model_router.py`, `src/routing/regime_detector.py`, `src/models/epnet.py`, `src/models/cqr_calibrator.py`

Her dosyanın docstring'inin başına ekle:
```python
"""
[EXPERIMENTAL - Canlı pipeline'da kullanılmıyor]
...mevcut docstring...
"""
```

---

## Faz 4: Sistematik Deney Altyapısı

### Adım 4.1: Deney Sonuç Tablosu Oluştur
**Dosya:** `db/sql/01_init_schema.sql`'e ekle

```sql
CREATE TABLE IF NOT EXISTS gold.experiment_results (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    feature_set VARCHAR(50),
    hyperparameters JSONB,
    backtest_type VARCHAR(30),
    backtest_days INTEGER,
    train_start DATE,
    test_start DATE,
    test_end DATE,
    mae_usd NUMERIC(10, 4),
    wape_pct NUMERIC(6, 2),
    mape_pct NUMERIC(6, 2),
    rmse_usd NUMERIC(10, 4),
    coverage_80_pct NUMERIC(6, 2),
    low_price_mae_usd NUMERIC(10, 4),
    wape_1m NUMERIC(6, 2),
    wape_3m NUMERIC(6, 2),
    wape_6m NUMERIC(6, 2),
    wape_spring NUMERIC(6, 2),
    runtime_seconds INTEGER,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_name, test_start, test_end)
);
```

### Adım 4.2: Standart Backtest Runner Oluştur
**Dosya:** `experiments/run_experiment.py` (yeni)

Temel yapı:
```python
def run_experiment(
    experiment_name: str,
    model_factory: Callable,
    feature_set: str = 'robust',
    backtest_days: int = 365,
    backtest_type: str = 'walk_forward',
    extra_features: list = None,
) -> dict:
    """
    Standart deney çalıştırıcı.
    1. Veri yükle (load_all_historical_data)
    2. Feature engineering (build_robust_features + opsiyonel ek feature'lar)
    3. Walk-forward veya static backtest
    4. Metrik hesapla (MAE, WAPE, MAPE, RMSE, Coverage, Low-Price MAE)
    5. gold.experiment_results'a yaz
    6. Günlük detayları gold.experiment_daily_results'a yaz (opsiyonel)
    7. Console ve log'a özet yaz
    """
    pass
```

### Adım 4.3: Mevcut Sonuçları Tabloya Kaydet
DB'deki mevcut verilerden ve CSV dosyalarından bilinen deney sonuçlarını `gold.experiment_results`'a kaydet:
- LightGBM_v1 (canlı model) performansı
- lgb_lag0_v2 performansı
- lgb_cqr_v2 performansı
- Pre-forecaster WAPE'leri (Load %2.89, Solar %10.53, Wind %18.02)
- Eski log1p benchmark sonuçları (CSV'lerden)

### Adım 4.4: EXPERIMENT_WORKFLOW.md'yi Güncelle
- İsimlendirme standardını referansla
- `run_experiment.py` kullanım örneği ekle
- Yeni deney ekleme checklist'i yaz

---

## Uygulama Sırası ve Bağımlılıklar

```
Faz 1 (lgb_lag0_v2 canlıya alma)
  ├── 1.1 Mevcut canlıyı yedekle (SQL) → İLK YAPILMALI
  ├── 1.2 lgb_lag0_v2'yi canlıya taşı (SQL) → 1.1 sonrası
  ├── 1.3 predict_daily_pipeline.py    → 1.2'den bağımsız
  ├── 1.4 backfill_gold_predictions.py → 1.3 ile paralel
  └── 1.5 doğrulama                   → 1.1-1.4 sonrası

Faz 2 (bug fix'ler) → Faz 1'den bağımsız, paralel yapılabilir
  ├── 2.1 SMP isim doğrulama          → kontrol, muhtemelen değişiklik yok
  ├── 2.2 veri gap kontrolü           → tek dosya değişikliği
  ├── 2.3 duplicate import            → tek dosya değişikliği
  ├── 2.4 log rotation                → tek dosya değişikliği
  └── 2.5 prevClose fallback          → tek dosya değişikliği

Faz 3 (repo temizliği) → Faz 1-2 commit edildikten sonra
  ├── 3.1 scripts/ ayırma             → git mv komutları
  ├── 3.2 eda/ organize etme          → git mv komutları
  ├── 3.3 frontend temizliği          → component/type düzenleme
  └── 3.4 deneysel modül işaretleme   → docstring ekleme

Faz 4 (deney altyapısı) → Faz 3'ten sonra
  ├── 4.1 DB tablosu                  → SQL
  ├── 4.2 standart runner             → yeni dosya
  ├── 4.3 mevcut sonuçları kaydet     → runner veya SQL
  └── 4.4 doküman güncelle            → markdown
```

## Commit Stratejisi

Her faz ayrı commit:
1. `feat: promote lgb_lag0_v2 to production (min_child_samples=10)`
2. `fix: data gap check, log rotation, duplicate import, prevClose fallback`
3. `refactor: reorganize repo structure (experiments/, scripts/ cleanup)`
4. `feat: add experiment tracking infrastructure (gold.experiment_results)`
