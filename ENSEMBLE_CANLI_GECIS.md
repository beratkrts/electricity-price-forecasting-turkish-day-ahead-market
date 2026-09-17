# Çok-pencereli ensemble + konformal band — canlıya geçiş planı

**Kaynak:** Faz C.5 (`LAGO_BENCHMARK_PLAN.md`), deney defteri
`experiments/notebooks/07_lago_protocol/08_collapse_fix_experiments.ipynb`.
**Hedef repo:** `../enerji_fiyat_tahmini` (canlı pipeline).
**Durum:** deney reposunda doğrulandı, canlıya taşınmadı. **Karar bekliyor.**

---

## 1. Ne değişiyor

### P50 (nokta tahmin) — 3-pencereli ensemble

Şu an: tek LightGBM, tüm geçmişle (2023-01'den) eğitiliyor.

Yeni: **3 LightGBM**, aynı `lgb_lag0_v2` konfigü (`quantile α=0.50, n_est=300, lr=0.03,
depth=8, num_leaves=63, min_child_samples=10`), farklı eğitim pencereleri:

| üye | eğitim penceresi | rolü |
|---|---|---|
| A | son **90 gün** | güncel rejim, hızlı uyum |
| B | son **150 gün** | güncel rejim, biraz daha kararlı |
| C | **tüm geçmiş** (2023-01'den) | yapı + mevsimsellik + sıfır-saat kapsaması |

**P50 = (A + B + C) / 3** (eşit ağırlık; uyarlanabilir ağırlık test edildi, katkı yok).

### P10/P90 — yuvarlanan konformal band

Şu an: ayrı P10 ve P90 quantile LightGBM'leri (α=0.10/0.90), tüm geçmişle. **Kapsama bozuk**
(çöküşte %60, hedef %80).

Yeni: **quantile head'leri kaldır.** Bunun yerine post-process (nedensel split-conformal):

```
her (hedef gün d, saat h) için:
  err[d', h] = ensemble_P50[d', h] − gerçek[d', h]   (d' ∈ son N=60 gün, nedensel)
  L(d,h) = P50(d,h) − quantile_0.90( err[:, h] )  − w · std({A,B,C})(d,h)
  U(d,h) = P50(d,h) − quantile_0.10( err[:, h] )  + w · std({A,B,C})(d,h)
  L(d,h) = max(L(d,h), FLOOR)
```

**Kesinleşen parametreler** (`experiments/scripts/conformal_final.py` taraması, 1 Eyl):

| parametre | değer | gerekçe |
|---|---|---|
| hedef kapsama | %80 (α=0.20) | P10/P90 tanımı |
| çözünürlük | saat-bazlı (24 ayrı hata dağılımı) | gece/gündüz hata profili çok farklı |
| kalibrasyon penceresi `N` | **60 gün** | 45/60/90 taraması: kapsama farkı <%1, 60 = rejim tepkisi/kararlılık dengesi |
| anlaşmazlık ağırlığı `w` | **0.5** | `w=0` saf konformal %74–77 (hedefin altında); `w=0.5` her rejimde %80–82; `w=0.7` fazla geniş (%83+) |
| `FLOOR` | **$0** | TR MCP hiç negatif olmamış (547 saat tam $0, min USD/TRY = 0). `w=0.5`'te 1944 saat (%11) $0'a kırpılıyor — hepsi çöküş/gece saati, kapsama kaybı yok (gerçek de $0) |
| şift yönü | asimetrik (`err = tahmin − gerçek`) | model fazla-tahmin biaslı → band aşağı kayıyor |

- `err` geçmişi: `gold.ptf_predictions_daily` (geçmiş ensemble P50) ⋈ `raw_mcp_hourly` (gerçekleşen).
- `std({A,B,C})` = 3 üye modelin o saatteki P50 tahminlerinin std'si — rejim belirsizliği erken sinyali
  (60g hata geçmişi rejim geçişinde geç kalıyor, anlaşmazlık terimi öne çekiyor).
- İlk 60 gün (soğuk backfill): kalibrasyon geçmişi yok → o dönem için geniş sabit band
  (`P50 ± $25`) fallback.

---

## 2. Doğrulanmış sonuçlar (deney reposu, 2-yıl backtest 2024-08-01→2026-08-27, **proxy'siz**)

> **Not:** ilk sürüm `--proxy-from` kullanıyordu; o bayrağın kaskad bug'ı vardı (her gün
> `pfrom−1`'e çöküyordu). Düzeltildi, tüm koşular `_v2` etiketiyle proxy'siz yeniden üretildi.
> Framework doğrulaması: proxy'siz `base` MAE $7.23 ≈ canlı DB $7.27.

| | canlı DB | base (=canlı politika) | ensemble `base+r90+r150` |
|---|---|---|---|
| 2-yıl MAE / rMAE | $7.27 / 0.645 | $7.23 / 0.645 | **$7.04 / 0.628** (−$0.23, −%3) |
| normal MAE | $6.23 | $6.20 | $6.21 (nötr; sıfır-saat MAE 9.8→12.2) |
| çöküş MAE / BIAS | $10.54 / +$3.76 | $10.49 / +$3.44 | **$9.62 / +$1.54** (−$0.92, BIAS −%59) |
| toparlanma MAE / BIAS | $8.63 / +$1.51 | $8.48 / +$1.27 | **$8.20 / −$0.81** (−$0.43, \|bias\| yarı) |

Karar kapısı: çöküş MAE −$0.92 (eşik −$0.88) ✓, BIAS düzeliyor ✓, toparlanma iyileşiyor,
normal nötr. **GEÇİLDİ.** Hareketli WAPE 1/3/6/12/24 ay: her ufukta iyileşme.

**DM testi (base vs eşit ensemble, `08 §4.1`):** TÜM dönem p<0.0001, çöküş p<0.0001,
çöküş-derin p<0.0001 — kazanç istatistiksel olarak gerçek. Normal p=0.61 (fark yok, beklenen),
toparlanma p=0.11 (58g, yön ensemble lehine). `base 0.5×` çöküşte eşit ensemble'dan
DM-ayırt edilemez (p=0.90), normal/toparlanmada DM-daha kötü → default eşit ağırlık.

**P10-P90 kapsama (hedef %80), ensemble P50 v2 üstünde** (N=60, w=0.5, FLOOR=$0):

| dilim | canlı 3-head | konformal | band (canlı → konformal) |
|---|---|---|---|
| TÜM | 72.1% | **80.8%** | $21.3 → $23.3 |
| normal | 75.7% | 81.0% | $18.8 → $21.3 |
| çöküş | **59.7%** | **80.2%** | $27.6 → $27.0 |
| toparlanma | 75.2% | 81.2% | $24.9 → $29.4 |
| son 90g | 71.7% | 80.2% | $24.5 → $27.7 |
| son 30g | 75.5% | 85.8% | $24.6 → $28.8 |

Bedel: band normalde ~%10 geniş, çöküşte canlıdan **dar** (FLOOR $0 alttan kırpıyor).
Saf konformal (w=0) %74–77'de kalıyor; anlaşmazlık terimi (w=0.5) açığı kapatıyor.

**Kırılım — genel %80 dilimlere eşit dağılmıyor** (`experiments/scripts/band_breakdown.py`,
`08 §7.1`):

| kırılım | kapsama |
|---|---|
| gerçekleşen $0–5 (tam $0: %98) | **91.8%** |
| gerçekleşen **$20–60** | **68–72%** ← zayıf nokta |
| gerçekleşen $60–100 (hakim) | 83.9% |
| gerçekleşen $100+ (spike, n=25) | 48% |
| saat grupları / rejimler | 80–81% (düz) |

$20–60 cebi = modelin yukarı biasının konformal asimetrik şiftle tam yutulamayan kısmı;
C.5.5 (rejim-tetikli ağırlık) bunu da daraltır. Spike'lar yukarı kaçıyor (recency kör).

---

## 3. Kod değişiklikleri (`../enerji_fiyat_tahmini`)

### 3.1 `scripts/predict_daily_pipeline.py` — `run_daily_prediction()`

- **ADIM 4 (eğitim):** P10/P50/P90 3-head yerine → **3× P50 modeli** (A/B/C pencereleri).
  Her biri `df_model.loc[train_end − W : train_end]` ile fit. C = mevcut davranış.
- **ADIM 5d (tahmin):** `future_df` üzerinde 3 modelin P50'sini al → `p50 = mean`,
  `disagree = std` (saat bazlı).
- **Yeni ADIM 5e (band):** son 60 günün `gold.ptf_predictions_daily.predicted_mcp_usd` ⋈
  `raw_mcp_hourly.price_usd` → saat-bazlı `err` quantile'ları → L/U hesapla, `disagree` ile genişlet, FLOOR kırp.
- `results_df`: `predicted_mcp_usd` = p50, `_p10` = L, `_p90` = U. TRY çarpımları aynı.
- **Şema değişikliği YOK** — sütunlar aynı.

### 3.2 `scripts/backfill_gold_predictions.py` — `backfill_historical_predictions()`

- Aynı 3-pencere ensemble + konformal band mantığı, walk-forward.
- Backfill sonrası dashboard'ın geçmiş görünümü yeni yöntemle tutarlı olur.
- **DİKKAT:** backfill'in kendi `err` geçmişi kendi ürettiği P50'lerden gelmeli (nedensel).

### 3.3 Eğitim maliyeti

3× model, ama A/B pencereleri küçük (~2000-3600 saat) → toplam duvar-saati ~**2×**.
Günlük 04:00 koşusu için kabul edilebilir (mevcut ~1-2 dk → ~3-4 dk).

---

## 4. Rollout

> **Detaylı sıralama + önkoşullar:** `../enerji_fiyat_tahmini/ENSEMBLE_AKSIYON_PLANI.md`
> (Opus, 31 Ağu — Faz 0/1/2/3). **Faz 0 (ölçüm çerçevesi düzeltme) TAMAM** — `_v2` proxy'siz
> koşular, kaskad bug'ı kapandı, karar kapısı geçildi. **Faz 1 önkoşulları (ensemble'dan
> bağımsız, yine de gerekli):**
> - `api_server.py`'deki ~12 `gold.ptf_predictions_daily` sorgusuna `model_name` filtresi yok
>   → 2. model yazılınca dashboard JOIN'leri çift satır / `AVG()` iki-model ortalaması verir.
>   Shadow run'dan **önce** eklenmeli (`ACTIVE_MODEL_NAME` sabiti).
> - `backfill_gold_predictions.py`: `engine` döngü içinde (satır 93) kullanılıyor, tanımı
>   satır 119 — FX fallback tetiklenirse `NameError`. Bugün latent (`tr_df['usd_try']` hep dolu).

1. **[ ] Kod:** 3.1 + 3.2'yi canlı repoda uygula (branch: `feat/multiwindow-ensemble`).
2. **[ ] Birim doğrulama:** deney reposundaki `bt2y_*` CSV'leriyle byte-yakın çıktı
   (aynı pencere, aynı gün → aynı P50 ±LightGBM stokastikliği).
3. **[ ] Shadow run (~2 hafta):** yeni tahminleri eskinin YANINDA üret, ikisini de logla
   (`gold.ptf_predictions_daily` `model_name` ayrımıyla: `LightGBM_v1` vs `ensemble_v1`).
   Dashboard eskiyi göstermeye devam etsin.
4. **[ ] Shadow kabul kriterleri:**
   - ensemble P50 MAE ≤ tek-model MAE (haftalık)
   - P10-P90 kapsama ≥ %78 (haftalık, ideal ~%80)
   - normal günlerde MAE regresyonu yok (< +$0.3)
5. **[ ] Geçiş:** shadow onaylarsa `model_name` varsayılanını `ensemble_v1` yap,
   `backfill_gold_predictions.py`'yi tüm geçmişe koş, dashboard'ı yeni seriye çevir.
6. **[ ] İzleme:** haftalık kapsama + BIAS paneli. BIAS < −$3 veya kapsama < %70 → alarm.

---

## 5. Bilinen sınırlar / açık işler

- **Ani tek-gün aşağı-spike körlüğü** (30 Ağu 2026: fiyat $40'a çöktü, ensemble $49 dedi).
  Tüm recency yöntemleri bu hataya açık. C.5.5 kapsamında.
- **Konformal band hızlı rejim geçişinde geç kalıyor** — anlaşmazlık terimi hafifletiyor, bitirmiyor.
- **Kalan çöküş biası +$1.54** (v2) — tam sıfır değil. **Çöküş MAE base ağırlığından bağımsız
  (~$9.6 sabit); ağırlık sadece bias kolu, doğruluk kazandırmıyor** — çöküşte parlayan üye yok
  (`08 §8`). Statik `base 0.5×` (`(0.5·base+r90+r150)/2.5`): çöküş BIAS +$1.54 → +$1.16,
  karşılığında normal/toparlanma ~$0.05–0.16 MAE. Konformal band kapsaması değişmiyor.
  → Default eşit ensemble; `base 0.5×` tek-satır çevrilecek statik kol. **C.5.5 (rejim-tetikli
  dedektör) bu ~$0.4 için kurulmaz.**
- **P10/P90 backtest'i sadece 2024-11'den** (60g kalibrasyon ısınması). Daha uzun tarih için
  ilk dönem fallback band gerekli.
- Bu deneyler `TRAINING_DATA_START = 2023-01-01` sabitini varsayıyor (spike rejimini dışlıyor).
  C member zaten bu; A/B pencereleri de otomatik 2023+ (veri o zaman başlıyor).
