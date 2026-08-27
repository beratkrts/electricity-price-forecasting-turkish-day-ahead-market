# Sistematik Deney Çalışma Rehberi

## Mevcut Sorunlar

Şu ana kadar yapılan deneylerle ilgili tespit edilen yapısal sorunlar:

1. **Sonuçlar dağınık:** Bazıları console çıktısı, bazıları CSV, bazıları notebook output, bazıları log dosyası, bazıları DB tablosu. Tek bir yerden karşılaştırma yapılamıyor.
2. **Notebook output'ları silinmiş:** 5/17 notebook'un sonuçları kayıp (clear output yapılmış veya hiç çalıştırılmamış).
3. **Script'ler sonuçları dosyaya yazmıyor:** `ptf_ab_test.py`, `hybrid_1yr_walk_forward.py`, classifier deneyleri sadece `print()` çıktısı üretiyor.
4. **Backtest'ler çok uzun sürüyor:** 365 gün walk-forward × 3 model = saatlerce çalışma. Sonuç kaybedilince tekrar çalıştırmak maliyetli.
5. **Hyperparameter'ler script'lere hardcode edilmiş:** Farklı deneylerde farklı değerler var, karşılaştırma zor.
6. **Deney isim standardı yok:** STRAT_01, ROBUST_02, Baseline, Enhanced gibi tutarsız isimlendirme.

---

## Önerilen Deney Altyapısı

### 1. Merkezi Sonuç Tablosu: `gold.experiment_results`

Tüm deney sonuçlarının tek bir yere yazılması:

```sql
CREATE TABLE IF NOT EXISTS gold.experiment_results (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,      -- 'lgbm_log1p_baseline'
    model_type VARCHAR(50) NOT NULL,            -- 'lightgbm', 'epnet', 'lstm', 'hybrid'
    feature_set VARCHAR(50),                    -- 'base', 'core', 'full', 'robust', 'robust_v2'
    hyperparameters JSONB,                      -- {"n_estimators": 300, "lr": 0.03, ...}
    backtest_type VARCHAR(30),                  -- 'static', 'walk_forward'
    backtest_days INTEGER,                      -- 365, 180, 90, 30
    train_start DATE,
    test_start DATE,
    test_end DATE,
    -- Metrikler
    mae_usd NUMERIC(10, 4),
    wape_pct NUMERIC(6, 2),
    mape_pct NUMERIC(6, 2),
    rmse_usd NUMERIC(10, 4),
    coverage_80_pct NUMERIC(6, 2),             -- P10-P90 coverage
    low_price_mae_usd NUMERIC(10, 4),          -- MAE for <=15$ hours
    -- Dönemsel kırılım
    wape_1m NUMERIC(6, 2),
    wape_3m NUMERIC(6, 2),
    wape_6m NUMERIC(6, 2),
    wape_spring NUMERIC(6, 2),                 -- Mar-May (kriz dönemi)
    -- Meta
    runtime_seconds INTEGER,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Standart Backtest Runner

Tüm deneylerin aynı framework üzerinden çalışması:

```python
# scripts/run_experiment.py (önerilen yapı)

def run_experiment(
    experiment_name: str,
    model_factory: Callable,           # Model oluşturan fonksiyon
    feature_set: str = 'robust',       # Feature seti
    backtest_days: int = 365,          # Test süresi
    backtest_type: str = 'walk_forward',
    save_daily: bool = True,           # Günlük sonuçları da kaydet
) -> dict:
    """
    Standart backtest runner. Sonuçları hem console'a hem DB'ye yazar.
    """
    # 1. Veri yükle
    # 2. Feature engineering
    # 3. Walk-forward veya static split
    # 4. Günlük metrikler hesapla
    # 5. gold.experiment_results'a yaz
    # 6. Günlük detayları gold.experiment_daily_results'a yaz
    # 7. Özet log'a yaz
    pass
```

### 3. Deney İsimlendirme Standardı

```
{model}_{feature_set}_{varyant}_{versiyon}

Örnekler:
  lgbm_robust_baseline_v1        # Canlı model
  lgbm_robust_lag0ratios_v2      # Enhanced (Lag0 Renewable Ratios)
  lgbm_robust_logtransform_v1    # Log-transform varyantı
  epnet_full_cyclic_huber_v1     # ROBUST_02
  lstm_robust_online_conservative_v1
  hybrid_classifier_zppi_v1
```

### 4. Deney Çalıştırma Sırası (Öncelik)

Şu an yapılması gereken deneyler, öncelik sırasına göre:

| # | Deney | Tahmini Süre | Neden |
|---|-------|-------------|-------|
| 1 | `lgb_robust_lag0ratios_v2` canlıya alma | 0 (sadece kod) | 13 Ağustos testinde WAPE %28→%16, en büyük kazanım |
| 2 | Classifier deneylerini tekrar çalıştır (sonuç kaydet) | ~2 saat | 5 deneyin sonucu kayıp |
| 3 | Load/Wind pre-forecaster WAPE'lerini kaydet | ~4 saat | Solar 11.59% biliniyor, diğerleri kayıp |
| 4 | CQR vs raw quantile A/B testi (DB'de var) | ~30 dk | lgb_cqr_v2 vs lgb_lag0_v2 coverage karşılaştırması |
| 5 | LSTM online learning çalıştır | ~8 saat | Notebook hazır, hiç çalıştırılmamış |

### 5. Repo Temizleme Planı

```
MEVCUT:                          HEDEF:
eda/                             experiments/
  ├── 17 notebook (karışık)        ├── notebooks/
  ├── 19 script (karışık)          │   ├── 01_eda_ptf_exploration.ipynb
  └── CSV sonuçlar                 │   ├── 02_model_comparison.ipynb
                                   │   └── 03_zero_price_analysis.ipynb
scripts/                          ├── backtests/
  ├── run_*_backtest.py              │   ├── run_experiment.py (standart runner)
  ├── test_*                         │   └── configs/ (YAML deney tanımları)
  ├── backfill_*_experimental*       └── results/
  └── fast_cqr_update.py                └── (DB'den çekilir, burada tutulmaz)

scripts/ (canlı, temiz):
  ├── start.sh
  ├── run_service.py
  ├── daily_update_pipeline.py
  ├── predict_daily_pipeline.py
  ├── fetch_epias_data.py
  ├── api_server.py
  ├── backfill_gold_predictions.py
  └── backfill_pre_forecasts.py
```

### 6. Deney Süreci (Her Yeni Deney İçin)

```
1. Deney hipotezini yaz (tek cümle)
2. experiment_name belirle (standart format)
3. run_experiment() ile çalıştır
4. Sonuçlar otomatik olarak gold.experiment_results'a yazılır
5. Baseline ile karşılaştırma tablosu otomatik üretilir
6. Karar: Canlıya al / daha fazla test / reddet
```

---

## Mevcut Deney Envanter Durumu

| Kategori | Toplam | Sonucu Mevcut | Sonucu Kayıp |
|----------|--------|---------------|--------------|
| PTF Model Karşılaştırma | 8 | 6 | 2 |
| EPNet Grid Search | 26+ | 26 (log dosyaları) | 0 |
| Hybrid Routing | 1 | 1 (CSV) | 0 |
| Zero-Price Classifier | 5 | 0 | **5** |
| Pre-Forecaster | 3 | 1 (Solar) | **2** |
| LSTM Online Learning | 3 | 0 | **3** |
| Paper Stratejileri | 5 | 5 (CSV) | 0 |
| MWA Baseline | 2 | 2 | 0 |
| **TOPLAM** | **53+** | **41** | **12** |
