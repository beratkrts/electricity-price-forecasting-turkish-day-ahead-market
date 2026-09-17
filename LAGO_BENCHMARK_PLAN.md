# Lago Benchmark Yarışı — Plan

**Tarih:** 27 Ağustos 2026
**Amaç:** Lago et al. (2021) protokolünü Türkiye GÖP verisine, **tek dondurulmuş
protokolle**, tutarlı biçimde uygulamak. `01`/`02`/`03` defterleri üç ayrı
protokol kullanıyordu (aşağıda); bu plan onları tek çatı altında toplar ve
eksik benchmark'ları ekler.

İlgili: `src/eval/lago_protocol.py`, `experiments/notebooks/07_lago_protocol/`,
`EXPERIMENT_REPORT.md`, `METRICS.md`, `KATMAN_0_1_2_YOL_HARITASI.md` (Faz 1).

---

## ⏸ NEREDE KALDIK (1 Eyl 2026 — buraya ara verildi, yeni konuya geçiliyor)

**Track 1 (model geliştirme) — Faz C.5 BİTTİ.** Kazanan: `base + roll90 + roll150` eşit-ağırlık
çok-pencereli LightGBM ensemble + nedensel konformal band (N=60, w=0.5, FLOOR=$0). Çöküş MAE
$10.54→$9.62, DM p<0.0001. Karar kapısı geçildi. Tüm kanıt `08_collapse_fix_experiments.ipynb`.

**Canlıya geçiş:** tasarım kesin, KOD YAZILMADI. Devir brief'i hazır:
`../enerji_fiyat_tahmini/ENSEMBLE_IMPLEMENTASYON.md` (Opus'a). Sıradaki adım = Faz 1
(iki bağımsız bug: `api_server.py` `model_name` filtresi + `backfill_gold_predictions.py`
`engine` NameError), sonra Faz 2 (`src/models/ensemble.py`), Faz 3 (shadow ≥14g + geçiş).

**Benchmark Track (Faz D–G) — BAŞLANMADI.** D = uyarlanmış LEAR (bizim impl, {56,180,1092,1456}
+ adaptif ağırlık) 1096g'e; E = fARX/CING-LEAR (ops.); F = DNN + Chronos-2 (ayrı); G =
`BENCHMARK_REPORT.md`.

**Açık küçük iş:** ani tek-gün aşağı-spike körlüğü (C.5.5'ten arta kalan, dedektör kurulmadı).

**Yarışa devam edildiğinde:** Faz D'den başla (aşağı bak) ya da canlı geçiş Faz 1'e Opus'la devam.

---

## 0. Mevcut tutarsızlık

| Defter | Test | İmpl | Pencere(ler) | Seçici | Sonuç (rMAE n2, temiz) |
|---|---|---|---|---|---|
| `01_lago_protocol_turkey` | 2 yıl (729g) | bizim `lago_protocol.py` | ensemble(56, 84, 1095) — **1456 yok** | CV | LEAR 0.76 |
| `02_epftoolbox_crossvalidation` | 1 yıl | gerçek `epftoolbox` | ensemble(56, 84, 1092, **1456**) | AIC | LEAR ens. 0.75 |
| `03_lear_feature_ablation` | 1 yıl | bizim | tek(1092) | AIC | taban 0.81 |

Her biri LightGBM'in LEAR'ı DM-anlamlı geçtiği sonucuna varıyor ama mutlak
sayılar kıyaslanamıyor.

---

## 1. Dondurulmuş protokol

Bundan sonra `07_lago_protocol/` altındaki **her** benchmark koşusu şuna uyar:

| Öğe | Karar | Dayanak |
|---|---|---|
| **Birincil test** | Son **728 gün** (104 hafta): 2024-08-30 → 2026-08-27 | Lago §5.3 "last 104 weeks" **+** `gold.ptf_predictions_daily` 2024-08-12'de başlıyor → canlı LightGBM'in kayıtlı tahmin geçmişinin tamamı bu. Daha erken test = LightGBM'i geriye dönük yeniden koşturmak (farklı, zayıf kıyas) |
| **Uzun test (LEAR ailesi + naive)** | **3 yıl**: 2023-08-28 → 2026-08-27 (1096 gün) | LEAR/naive canlı-kayıt kısıtına tabi değil. 4 kalibrasyon penceresi de bu aralıkta geçerli (cw1456 için ~2019-08 verisi gerek, var). Rejim genişliği: 2023 yüksek-fiyat kuyruğu + 2024 normalleşme + 2025 + 2026 çöküş |
| **Rejim katmanları** (birincil pencere) | **normal** 2024-08-30→2026-01-31 · **çöküş** 2026-02-01→2026-06-30 · **toparlanma** 2026-07-01→2026-08-27 | Aylık ort fiyat: normal $65-77, çöküş $47→$13, toparlanma $57-62. Tek toplam sayı %75 normal rejim tarafından domine ediliyor — asıl bulgu katmanlarda |
| **Kalibrasyon** | Günlük yeniden kalibrasyon, 24 saat ileri | Lago §4 |
| **Nokta metrikleri** | MAE, RMSE, sMAPE, rMAE(naive-2), rMAE(naive-3) | Lago §5.4 + `METRICS.md` sürekliliği |
| **naive-2** | `p[d-7]` | Lago rMAE paydası |
| **naive-3** | Sal–Cum → `p[d-1]`, Cmt/Paz/Pzt → `p[d-7]` | `METRICS.md` |
| **Temiz küme** | `|herhangi tahmin| > $500` olan günler dışlanır; **ham + temiz birlikte** raporlanır | LEAR patlama günleri RMSE'yi anlamsızlaştırıyor (`01` §8) |
| **İstatistiksel test** | DM çok değişkenli (p-norm, p=1) + saat-bazında tek değişkenli | Lago §5.5.1 |
| **Fiyat birimi** | USD/MWh | proje standardı |

Sabitler `src/eval/lago_protocol.py`'de modül düzeyinde: `TEST_START/END`
(birincil), `LONG_TEST_START` (uzun), `REGIME_STRATA` (dict), `BLOWN_THRESHOLD_USD`.
Tüm runner'lar oradan okur.

**Neden iki pencere:** Birincil (728g) LightGBM'i içerir ama %75'i tek rejim.
Uzun (3y) rejim çeşitliliği verir ama LightGBM'in gerçek kaydı yok. Rapor iki
katmanlı: (a) "LEAR ailesi + naive, 3 yıl, rejim-katmanlı" — model-sınıfı
hikâyesi; (b) "tam yarış + LightGBM, 728 gün, rejim-katmanlı" — incumbent kıyası.
LightGBM'i uydurmadan hem genişlik hem güncel kıyas.

---

## 2. Üç kavramsal karar

### 2.1 epftoolbox mü, bizim implementasyon mu?

**epftoolbox = kanonik referans.** Başlık sayıları (rapor tablosu, DM testleri)
onunla üretilir. Sebep: hakemli, yayımlı (Applied Energy 293), AGPL, çok atıflı.
"Açık-erişim benchmark'ını bizzat koştuk" iddiası "yeniden yazdık"tan güçlü.

**Bizim `lago_protocol.py` = araştırma aracı.** Pencere duyarlılığı + özellik
ablasyonu için. epftoolbox'ın matris kurucusu katı (`Exogenous 1..N`, sabit
D/D-1/D-7 lag yapısı); TR-özel lag0 özelliklerini temiz enjekte edemiyor.

**Bizim impl ile ölçülen kayıp** (birebir aynı konfig: cw1092, AIC, 365 gün):

| | MAE | rMAE(n2) |
|---|---|---|
| bizim (asinh + StandardScaler, LassoLarsIC.predict) | $10.11 | 0.809 |
| epftoolbox stock (asinh + medyan/MAD, α→düz Lasso refit) | $11.19 | 0.894 |

- Toplam ~$1.1/MWh fark — bu koşuda **bizimki daha iyi** (StandardScaler 2026
  çöküşünde medyan/MAD'den daha az patolojik).
- Gün-gün ort |fark| **$10.5**, medyan $6.4, p95 $31 → aynı model değil.
- n≤p (cw56/84): bizimki CV'ye düşüyor, epftoolbox `noise_variance` proxy
  kullanıyor (patched) — kısa pencerelerde ek ayrışma.

**Kural:** her iki impl **aynı 4 pencerede, aynı 728g testte** koşulur →
kalibrasyon tablosu (Faz C). Araştırma-track sayıları bu tabloyla kanonik
sayılara çevrilebilir kalır. Bizim LEAR'ımız raporda **hiçbir zaman**
"Lago'nun LEAR'ı" diye etiketlenmez.

### 2.2 4 yıllık pencere şart mı?

**İki ayrı şey:**

- **Çok-pencereli ensemble ilkesi → mimari/genel.** Farklı zaman ölçeklerini
  harmanlamak varyans azaltır (tahmin kombinasyon literatürü). Lago bunu
  piyasadan bağımsız "best practice" diye sunuyor. **Korunur.**
- **Spesifik değerler (56/84/1092/1456) → ampirik hiperparametre.** NP/PJM/EPEX
  2011-2016 verisiyle ayarlanmış: olgun, istikrarlı piyasalar, 6 yıl temiz
  geçmiş, bağlayıcı fiyat tavanı yok. **Kutsal değil.**

**Türkiye:** 2021'den beri ≥3 yapısal rejim (COVID sonrası → 2022 gaz+kur krizi
→ 2024-26 yenilenebilir/hidro). 4-yıl penceresi üçünü kapsayıp hiçbirine uymayan
ortalama fitliyor. `03` ablasyonu erken kanıt veriyor:

| Pencere | rMAE(n2) |
|---|---|
| cw56 | **0.768** |
| cw84 | 0.789 |
| cw1092 | 0.894 |
| cw1456 | 0.877 |

Kısa pencereler uzunları eziyor. **Beklenti:** 4-yıl penceresi TR'de nötr/zararlı.

**Karar:** Lago'nun tam konfigünü (4 pencere) **referans olarak koştur**; ayrıca
TR-ayarlı pencere alt-kümesi ara (Faz C). Transfer farkı = raporlanacak bulgu
("Lago'nun istikrarlı-piyasa pencere konfigü rejim-kaymalı gelişen piyasaya
transfer oluyor mu?").

### 2.3 Veri

DB'nin **tüm** tabloları tam 2021-01-01'de başlıyor (mcp, load_fc, kgup, smp,
gas, hava).

- **728g test + 3-yıl pencere: mevcut veriyle tam yeter** (ilk test günü
  2024-08-31 için ~2021-09'a kadar veri gerek, var).
- **728g test + 4-yıl pencere (1456g):** ilk test günü için ~2020-08'e kadar
  veri gerek → **~8 ay eksik**.
- **Çözüm:** `eptr2` ile (kimlik `.env`'de; canlı repo `scripts/fetch_epias_data.py`
  bunu kullanıyor) **2019-01 → 2020-12** çek → bağımsız CSV
  `experiments/notebooks/07_lago_protocol/tr_epf_ext.csv`
  (`Price`, `Exogenous 1` = yük tahmini, `Exogenous 2` = KGÜP toplam).
  **DB'ye yazma yok.** Kanonik LEAR sadece bu 2 dışsalı kullanıyor.
- Pre-2021 Türkiye piyasası yapısal farklı (düşük fiyat, kriz öncesi, farklı
  üretim mixi) — bu zaten 4-yıl penceresi tartışmasının bir parçası, ayrıca
  raporlanır.

---

## 3. Fazlar

### Faz A — Protokol + veri ✅ (27 Ağu 2026)

- [x] `src/eval/lago_protocol.py`'ye dondurulmuş sabitler: `TEST_START =
      2024-08-30`, `TEST_END = 2026-08-27` (728 gün = 104 hafta),
      `BLOWN_THRESHOLD_USD = 500.0`, `test_day_index(P)`, `clean_days(preds)`
- [x] `build_tr_epf_ext.py` → `tr_epf_ext.csv`: **67.104 saat, 2019-01-01 →
      2026-08-27**, boşluksuz, tz-naive. Kolonlar `Price` (USD/MWh),
      `Exogenous 1` (yük tahmini MW), `Exogenous 2` (KGÜP toplam MW).
      2021+ DB'den (03 ile birebir, max |Δ|=0), 2019-2020 `eptr2`'den
      (`mcp.priceUsd`, `load-plan.lep`, `kgup.toplam`), DB'ye yazılmadan.
      Yıllık ort USD: 2019 $46 · 2020 $40 · 2022 $147 · 2026 $41.
- [x] **Veri kalite denetimi:** eptr2 2019-2020 pull %100 temiz (17.544 saat,
      0 NaN, tüm aylar tam). DB 2021+ tek boşluk: **2026-05-12 yük tahmini
      (24h) — EPİAŞ'ta da yok** (eptr `[]` döndürüyor), D-7 (2026-05-05 aynı
      saatler) ile dolduruldu. Başka fallback yok. 2021-22 fiyat serisindeki
      uzun sabit segmentler (₺1745/$117.85 tavan) GERÇEK — kriz dönemi AFL
      kenetlenmesi, eptr + DB birebir aynı. Sıfır fiyatlar gerçek.
- [x] epftoolbox venv doğrulandı: `epf_venv` +
      `.../f99d396b-.../scratchpad/epftoolbox` duruyor. numpy 1.26.4,
      sklearn 1.9.0, pandas 3.0.5. **DNN import'u tf bozukluğu yüzünden
      `models/__init__.py`'de try/except'e alındı** (Faz F'de düzeltilecek).
      cw=1092 VE cw=1456, ilk test günü (2024-08-30) için tahmin üretiyor →
      4-yıl penceresi veri açısından mümkün.
- [x] Test günleri: `lp.test_day_index(P)` → 2024-08-30 → 2026-08-27

### Faz B — Kanonik Lago LEAR (epftoolbox) ✅ (27 Ağu 2026)

Girdi: `tr_epf_ext.csv` (2019-01 → 2026-08). epftoolbox venv: bkz. memory
`epftoolbox-venv`.

- [x] `run_lago_lear_reference.py` — epftoolbox `LEAR`, uzun pencere
      (2023-08-28 → 2026-08-27, 1096g), 4 pencere → `lago_ref_lear_cw{56,84,1092,1456}.csv`.
      Süre: cw56 91s, cw84 148s, cw1092 39dk, cw1456 68dk. **Hepsi 0 çöküş, 0 NaN gün.**
- [x] `recalibrate_predict_safe` tüm pencerelerde (n>p'de stock ile birebir).
- [x] `04_canonical_lago_benchmark.ipynb` — iki tablo (uzun 3y / birincil 728g) +
      rejim katmanları + DM + aylık grafik. LightGBM: `gold.ptf_predictions_daily`.

**Sonuçlar (`04` Sonuç hücresi):**
- **Uzun 3y:** LEAR ensemble MAE **$8.00** / rMAE(n2) **0.690**. Tek pencereler:
  cw56 $8.72 < cw84 $8.79 < cw1456 $8.84 < cw1092 $8.98. Ensemble hepsini DM p≈0 geçiyor.
- **Kısa pencere > uzun pencere** (tek seçilecekse). 4-yıl (cw1456) katkısı zayıf;
  2026'da cw1092/1456 ≈ $13.3-13.7 vs cw56 $10.99 → uzun pencereler çöküşü boğuyor.
- **Birincil 728g:** LightGBM (lag0_v2) **$7.35** / 0.644 vs LEAR ensemble **$8.08** / 0.709.
  **DM: LEAR ensemble ≤ LightGBM → p = 1.9×10⁻⁹.**
- **Rejim katmanları:** LightGBM 3'ünün de en iyisi — normal 6.30, **çöküş 10.54**
  (ensemble 12.98, en iyi LEAR cw56 11.90), toparlanma 8.63. Çöküşte cw1092/1456
  = $16-17. **`03`'teki "LEAR çöküşte rekabetçi" bulgusu kanonik protokolde geçersiz.**
- Faz C'ye devir: `cw1456`'yı ensemble'dan çıkarınca ne olur; TR-ayarlı pencere alt-kümesi.

### Faz C — Bizim impl kalibrasyonu + pencere çalışması ✅ (28 Ağu)

- [x] `run_lago_lear_ours.py` — bizim `lear_predict_day`, uzun 1096g pencere,
      `selector='aic'` (n≤p → LassoLarsCV). ~6-8× hızlı (§Faz C hız notu).
- [x] **Level 1** (Lago'nun 4'ü) + **Level 2** (geniş ızgara) → 11 pencere:
      56/84/120/180/270/365/545/730/1092/1456 (+ cw28 dud, n<30 guard).
      Hepsi 1096g, 0 çöküş (metr `|tahmin|>$500` patlama günü filtreliyor).
- [x] `05_window_sensitivity.ipynb` (17 hücre, koştu) — Sonuç dolu.

**Bulgular:**
1. **Kalibrasyon:** bizim impl pencere başına ~$1.4, ensemble ~$1.0 daha kötü.
   Ama **rejime bağlı:** epftoolbox uzun pencerelerde çöküşte patlıyor (cw1092
   **$17**, cw1456 **$16**), bizim mean/std ölçekleme çöküşte sağlam (**$11**).
   Medyan/MAD dönüşümü 2026 çöküşünde kırılıyor.
2. **Lago'nun {56,84,1092,1456}'sı 15 alt-kümenin EN İYİSİ** (rMAE_epf 0.690).
   cw1456 marjinal katkı sağlıyor (0.694→0.690, kötü-ama-bağımsız üye).
3. **Level 2:** TR-optimal ≈ 56+180+1456 (rMAE 0.772) — Lago-4'ten (0.781) **~%1**,
   gürültü içinde. Kazanç 3-5 pencerede plato. **cw270/cw365 KIRIK** (rMAE 1.04/1.20,
   n≈p'de LassoLarsIC kararsız) — kullanma.
4. **Uyarlanabilir ağırlık (CING-LEAR eq 10): bedava ~%2.** TR-4 uyarlanabilir
   MAE $8.80 / rMAE 0.760. **Çöküşte $10.51 ≈ LightGBM $10.54** — adaptif TR-LEAR
   LightGBM'i tam zayıf olduğu rejimde yakalıyor (normalde hâlâ geride).

**→ Faz D tabanı:** pencere kümesi **{56, 180, 1092, 1456} + uyarlanabilir ağırlık**
(cw84→cw180, cw270/365 dışlanmış). Üzerine `03` en iyi feature seti.

**Hız notu (Faz B ~2sa vs Faz C ~20-40dk, aynı iş):** epftoolbox `_build_and_split_XYs`
247-özellikli matrisi HER GÜN sıfırdan kuruyor (iç içe döngü + pandas `.loc[ts]`);
bizim `build_lear_matrix` bir kez vektörize pivot, her gün `X.iloc` dilim. Ayrıca
epftoolbox saat başına 2 fit (LARS-IC λ + düz Lasso refit, `max_iter=2500`),
bizim 1 fit (`max_iter=500`). Ölçekleme: `statsmodels.mad`×247 vs numpy StandardScaler.
Çarpımsal ~6-8×. Bedeli: epftoolbox sadık referans, bizimki hızlı vekil (~$1 fark,
kalibrasyon tablosu ölçüyor). **15/2047 alt-küme bedava** — 4/11 pencere fit edilir,
alt-kümeler o tahminlerin numpy ortalaması.

### Faz C.5 — Rejim stres testi (1-2 gün)

**Amaç:** Benchmark sadece ölçüm değil, **canlı LightGBM için hangi
değişikliğin işe yarayacağının teşhisi.** Şu an modelin belirgin sorunu
2026 fiyat çöküşlerini kaçırması; 2022 spike'larını yakalayıp yakalamadığı
ise **hiç test edilmedi** (canlı model `TRAINING_DATA_START = 2023-01-01`,
kayıt `gold.ptf_predictions_daily` 2024-08'de başlıyor).

**Canlı model durumu (doğrulandı, 27 Ağu):** Canlı model **2026-08-13'ten
beri `lgb_lag0_v2`** (commit `d7c67bf`: robust feature set + lag0 oranları +
`min_child_samples=10` + CQR). `gold.ptf_predictions_daily` o gün toptan
yeniden dolduruldu (`backfill_gold_predictions.py`, **gerçek walk-forward** —
gün gün yeniden eğitim, yalnız-geçmiş, eğitim 2023-01'den). Yani `03`/`04`'teki
"LightGBM (canlı)" sayıları **güncel lag0_v2 modelin**, adil ve sızıntısız.
`EXPERIMENT_REPORT.md`'nin "öncelikli aksiyon: lgb_lag0_v2 canlıya al" maddesi
**bayat — çoktan yapıldı.** lag0_v2 çöküşü *iyileştirdi* ama *çözmedi*:
`03` aylık kırılımda Şub-Mar 2026 LightGBM ~$12-13 MAE (sakin aylar ~$8),
fed-LEAR ile başabaş.

**İki rejim dilimi:**

| Dilim | Dönem | Neden |
|---|---|---|
| **Spike** | 2022 tam yıl | Küresel enerji krizi, tavana yapışma %25. Canlı model bunu **hiç görmedi** (eğitim 2023'ten) |
| **Çöküş** | 2026-02-01 → 06-30 (150 gün) | Yenilenebilir/hidro bolluğu, sıfır-fiyat saat %3→%32 |

**Koşulacak modeller (dilim başına):**

| Model | Spike | Çöküş | Not |
|---|:-:|:-:|---|
| LEAR ailesi (yuvarlanan kalibrasyon) | ✓ | ✓ | veri hazır (`tr_epf_ext.csv` 2019+), cw≤1092 2022-01'den geçerli. Tek komut |
| Walk-forward LightGBM — **eğitim 2021-01'den** | ✓ | ✓ | `experiments/scripts/run_lightgbm_standalone_backtest.py` diriltilecek. 2024 öncesi lag0 feature'lar fallback (belgeli kısıt) |
| Walk-forward LightGBM — **eğitim 2023-01'den** (canlı politika) | — | ✓ | sabit-pencere politikasının rejimde ne kaybettirdiğini gösterir |
| Canlı LightGBM = lgb_lag0_v2 (`gold.ptf_predictions_daily` kaydı) | — | ✓ | zaten walk-forward, eğitim 2023-01'den. Çöküş dilimini kapsıyor |

- [ ] `06_regime_stress_test.ipynb`: dilim başına metrik tablo (MAE/WAPE/rMAE +
      **spike yakalama:** tavan saatlerinde MAE, **çöküş yakalama:** sıfır-fiyat
      saatlerinde MAE + P10-P90 kapsama) + DM

**Çöküş dilimi — teşhis çoğunlukla YAPILMIŞ.** `LOW_PRICE_REGIME_ANALYSIS.md`
(14 Ağu, tekrar-üretilebilir SQL) + `tavandan_sifira.ipynb` §10-13 çöküş
körlüğünü zaten çözümlemiş, aynı sonuca varıyorlar:
- Model ucuz dilimde **sistematik yukarı sapıyor** (BIAS $0-10 diliminde +$8.56),
  P10-P90 kapsama %80→%43. Shrinkage, kayıp fonksiyonu değil.
- Sebep: **hidro kaynaklı merit-order rejim kırılması.** Hidro payı yıllar arası
  **transfer olmuyor** (aynı %50 hidro: 2024 $69, 2026 $28); **termal ihtiyaç
  oranı** `(total−güneş−rüzgar−hidro−jeo−biyo)/total` transfer oluyor.
- Kod açığı: hidronun T+1 pre-forecaster'ı yok → `kgup_hydro_lag_24` ile 24h
  gecikmeli giriyor; Şub-May rampasında arzı eksik gösteriyor → Şubat +$9.22 bias.
- `net_load_lag0` hidroyu hiç çıkarmıyor; jeo+biyo hiçbir orana girmiyor.

**Çöküş kolu — alt-deneyler (sırayla):**

- [x] **C.5.0 ✅** (29 Ağu) — `04_zero_price_crisis/01_low_price_regime_analysis.ipynb`
      ZATEN VARDI (§7 planına uygun, 12 hücre). `.venv` ile yeniden koşuldu,
      güncel veriyle **teşhis tamamen sağlam**: BIAS merdiveni değişmedi
      ($0-10 dilimi +$8.61, kapsama %43), hidro-transfer-yok / termal-transfer-var
      band tabloları aynı. **+3 yeni hücre: §6 yapısal kırılma testi** (statsmodels):
      `hydro_share→price` eğimi Şub 2026'da p≈0 ile kırılıyor (hidro ≤%20:
      $66→$25); `thermal_ratio→price` çok daha az (termal ≥%55: $72→$68.5, −%3,
      p=0.006). Doc iddiası büyük ölçüde doğru, bir büyüklük mertebesi fark.
- [x] **C.5.1 ön koşul ✅** (29 Ağu) — **pre-forecast backfill.** `gold.kgup_load_pre_forecasts`
      2024-08-13'te başlıyor → öncesi lag0 özellikleri GERÇEKLEŞEN değerle doluyor
      (sızıntı: yük ~%3, güneş ~%14, rüzgar ~%17 forecast hatası "yok" oluyor).
      `run_preforecast_backfill.py` → canlı `pre_forecasters.py`'yi (yük LightGBM /
      güneş Ridge / rüzgar LightGBM + OpenMeteo arşiv) 2021-03 → 2024-08 walk-forward
      koştu → `preforecast_backfill.csv` (29.664 saat, CANLI TABLOYA YAZMADAN).
      Kalite gerçekleşenle: **yük WAPE 2.9%, güneş 13.7%, rüzgar 17.0%** — canlı
      pre-forecaster'larla aynı. `run_wf_lightgbm.py` COALESCE ile okuyor.
- [x] **C.5.1 — stres testi taban tablosu ✅** (30 Ağu) — `06_regime_stress_test.ipynb`
      koşuldu + Sonuç dolduruldu. **Çöküş (2026-02→06, 150g):**
      - LEAR TR-4 uyarlanabilir $10.51 ≈ LightGBM canlı $10.54 genel — **başabaş.** Yön
        farkı: LEAR biassız (+$0.18), LightGBM sistematik yukarı (+$3.76).
      - **Sıfır-fiyat saatlerinde $1.95 (LEAR) vs $3.75 (LightGBM)** — LEAR yarısı.
      - **Eğitim penceresi çöküşte HİÇ fark etmiyor:** WF-LightGBM eğ. 2021 ($10.53) ≈
        eğ. 2023 ($10.49). Canlının `TRAINING_DATA_START=2023-01-01` pin'i çöküşü bozmuyor.
      - epftoolbox LEAR çöküşte kullanılamaz ($13 sıfır-MAE, patlıyor).
      **Spike (2022 tam yıl, 365g):**
      - **WF-LightGBM eğ. 2021 YAKALIYOR:** rMAE(n2) 0.704, MAE $20.59 — naive'i net yeniyor.
        Mimari 2022 krizini kaldırabiliyor; canlı model onu sadece 2023-pin yüzünden görmüyor.
      - WF-LightGBM > LEAR ($20.59 vs $23.83) — LEAR'ın yuvarlanan kalibrasyonu 2022 oynak
        fiyatlarını kovalayıp aşırı-uyum yapıyor. Tavan-saatlerde naive-3 ($21.96) ikisini de yeniyor.
      **Çıkarım:** rejim adaptivitesi (a) pencere politikasından — spike için kritik, çöküş için
      önemsiz; (b) model sınıfından değil (çöküşte başabaş); (c) **çöküşte tek kaldıraç =
      LEAR'ın yuvarlanan kalibrasyonu** (sıfır-saat avantajı). LightGBM'in sabit özellik seti
      merit-order kırılmasını izleyemiyor → **C.5.2 kök-neden deneyi.**
> **⚠ Metodoloji düzeltmesi (31 Ağu, Opus review):** ilk C.5.2/C.5.3 koşuları `run_wf_lightgbm.py
> --proxy-from` kullanıyordu; o bayrağın proxy döngüsünde **kaskad bug'ı** vardı (`prev` df'ten
> okunuyor, önceki iterasyonda ezilmiş → aralıktaki her gün `pfrom−1`'e çöküyor, ~30 exogenous-türevi
> özellik 2 yıl boyunca sabit). Ayrıca `--proxy-from` **zaten gereksizdi** — robust özelliklerin
> hiçbiri gün-D'nin ham `kgup`/`load`/`smp` değerini kullanmıyor (hepsi `shift(24/48/168)` lag'i
> ya `predicted_*_lag0`). Bug düzeltildi (`src = df.copy()` döngü öncesi), tüm koşular `_v2`
> etiketiyle **proxy'siz** yeniden üretildi. Framework doğrulaması: proxy'siz `base` MAE $7.23 ≈
> canlı DB $7.27. Aşağıdaki sayılar `_v2`.

- [x] **C.5.2 — özellik deneyi ~ (v2)** (31 Ağu) — `08 §1`. `thermal_req_ratio_lag0` +
      `hydro_net_load_lag0` (hidro/jeo/biyo = `shift 24`), `run_wf_lightgbm.py --c52`.
      Çöküş MAE **$10.49 → $10.31 (−$0.18)**, rMAE 0.663 → 0.652. **Küçük ama gerçek** — ilk
      bug'lı koşunun "$0.05, ölü" reddi geçersizdi (donmuş girdi). Yine de recency'nin küçük eki,
      ikamesi değil. Çöküş körlüğü ağırlıkla eski rejim eşleşmesinden.
- [x] **C.5.3 — yuvarlanan pencere ✓ (v2)** (31 Ağu) — `08 §2-4`. `run_wf_lightgbm.py --roll-days`.
      2-yıl backtest (2024-08→2026-08-27, 757g, **proxy'siz**):
      - **Tek pencere roll150:** çöküş MAE $10.49 → $9.71, BIAS +$3.44 → +$0.66. AMA normalde
        hafif kötü (MAE $6.20→$6.46, sıfır-saat), toparlanmada BIAS +$1.3 → −$1.6. Rejim-adaptivite
        **takası**.
      - **Sert pencere > recency ağırlığı** (v1: tau=90 $10.35 vs roll120 $9.88; yön açık, v2'de yeniden koşulmadı).
      - **365/730g** base'e yakın, tatlı nokta ~90-150g.
- [x] **C.5.3' — çok-pencereli ensemble ✓ KAZANAN (v2)** (31 Ağu) — `08 §4`. LEAR ensemble
      mantığı: birkaç `--roll-days` + eşit-ağırlık ortalama. 63 alt-küme tarandı → plato ~$7.00
      (3-4 komp arası fark gürültü). **`base + roll90 + roll150`** (en temiz):
      - 2-yıl: canlıya karşı MAE $7.27 → **$7.04** (−%3), rMAE 0.645 → 0.628
      - **Her rejimi iyileştiriyor, rejim cezası yok:** normal $6.23→$6.21 (nötr), çöküş
        $10.54→**$9.62** (BIAS +$3.76→**+$1.54**, −%59), toparlanma $8.63→**$8.20** (BIAS +$1.51→−$0.81)
      - Tek maliyet: normal rejim sıfır-saat MAE 9.8→12.2 (düşük etki)
      - Hareketli WAPE 1/3/6/12/24 ay: her ufukta iyileşme
      - v1'in "normal sıfır-saati bozuyor, toparlanma belirsiz" resmi **kaskad bug'ından**; v2 temiz
      - Uyarlanabilir ağırlık (LEAR eq.10) katkı yapmıyor
      **Benchmark'lara eklendi:** `04 §uzun+birincil+rejim` — ensemble uzun 3y $7.30 / birincil
      $7.15, **LEAR ensemble'ı ve canlı tek-model LightGBM'i DM-anlamlı geçiyor** (p ≤ 4.4e-5).
      `06 §çöküş+spike` — ensemble çöküş $9.62 / spike $19.81, ikisinde de en iyi.
      **DM base vs eşit ensemble (`08 §4.1`):** TÜM/çöküş/çöküş-derin p<0.0001 (kazanç gerçek),
      normal p=0.61 (nötr), toparlanma p=0.11 (58g).
      **Karar kapısı GEÇİLDİ.** Geçiş planı: **`ENSEMBLE_CANLI_GECIS.md`** +
      canlı repoda **`ENSEMBLE_IMPLEMENTASYON.md`** (Opus'a devir brief'i, 1 Eyl).
- [x] **C.5' — P10/P90 konformal band ✓ KESİNLEŞTİ** (1 Eyl) — `08 §7 + §7.1`,
      `conformal_final.py` + `band_breakdown.py`. Canlı 3-head quantile kapsaması bozuk
      (çöküşte %60). Çözüm: quantile head'lerini kaldır → ensemble P50 üstüne nedensel
      split-conformal. **Nihai param: N=60g, w=0.5 (anlaşmazlık ağırlığı), FLOOR=$0
      (TR piyasası hiç negatif değil), saat-bazlı, asimetrik.** Kapsama HER rejim + son
      30/90g'de %80–82 (canlı %60–76), band normalde ~%10 geniş, çöküşte canlıdan dar.
      Saf konformal (w=0) %74–77. **Kırılım (`§7.1`):** genel %80 dilimlere eşit dağılmıyor —
      gerçekleşen $0–5'te %92, **$20–60'ta ~%70** (modelin kalıntı yukarı biası), $60–100'de
      %84, $100+ spike'lar yukarı kaçıyor. Saat/rejim kırılımı düz.
- [~] **C.5.4 — hibrit (naive-2 router) RAFTA** (30 Ağu) — `08 §6`, `route_by_naive.py`.
      naive-2 ≤ eşik → LEAR TR-4 adaptif. Çöküşte MAE ~−$0.5 ama: **(1) LEAR sızıntısı**
      (`tr_epf_ext.csv` Exogenous 2 = o günün gerçekleşen KGÜP toplamı, 04:00'te yok);
      **(2) naive-2 zayıf sinyal** (oracle çöküş MAE $10.5→$7.0, router sadece $0.5);
      **(3) rejim kapısı gerek** (naive-2≤$40 normal rejimde LightGBM'i bozuyor).
      Ensemble'dan karmaşık, daha az kazanç. `src/routing/` + EPNet-blend zaten başarısız.
- [~] **C.5.5 — rejim-tetikli ensemble ağırlığı ~ ELENDİ (dedektör versiyonu)** (1 Eyl) — `08 §8`.
      **Çöküş MAE `base` ağırlığından BAĞIMSIZ (~$9.6 sabit) — çeşitlendirme zaten optimal,
      ağırlık sadece bias kolu.** Çöküşte parlayan üye yok (r90/r150 base'ten ~$0.8 MAE iyi,
      toparlanmada −$2 bias geri ödüyor). Statik `base 0.5×` (`(0.5·base+r90+r150)/2.5`):
      çöküş BIAS +$1.54→+$1.16, bedeli normal/toparlanma ~$0.05–0.16 MAE. **DM (`§4.1`):
      `base 0.5×` çöküşte eşit ensemble'dan ayırt edilemez (p=0.90), normal/toparlanmada
      DM-daha kötü (p<0.001) → hiçbir rejimde DM-üstünlüğü yok.** → default eşit ağırlık;
      `base 0.5×` sadece kodda `BASE_WEIGHT` sabiti. **Rejim dedektörü KURULMAZ.**
      Açık kalan ayrı iş: ani tek-gün aşağı-spike körlüğü (30 Ağu 2026, fiyat $40'a düştü,
      ensemble $49 dedi) — tüm recency yöntemleri buna açık.

**Her sonuç hangi canlı-model kararını besliyor:**

1. **Eğitim penceresini aç mı?** WF-LightGBM 2021'den vs 2023'ten: geniş pencere
   normal rejim doğruluğunu (USD'de) bozmadan spike yeteneği ekliyorsa → `TRAINING_DATA_START`
   pinini kaldır. Bozuyorsa → pin haklı, konu kapanır. **Tahmin değil, karar.**
2. **Spike bu mimariyle yakalanabilir mi?** WF-LightGBM 2022'de eğitilip hâlâ
   kaçırıyorsa → mimari/feature limiti, pini kaldırmakla vakit kaybetme.
   Yakalıyorsa → tek engel eğitim pini.
3. **`thermal_requirement_ratio` + hidro net_load canlıya taşınmalı mı?**
   C.5.2 Şub-Haz MAE'yi $11-13'ten anlamlı indiriyorsa → canlıya PR. Kök-neden
   düzeltmesi, en yüksek öncelik.
4. **LightGBM'e yuvarlanan / recency-ağırlıklı pencere?** C.5.3, LEAR'ın Nisan
   avantajını yuvarlanan pencereyle kapatıyorsa → canlıda `TRAINING_DATA_START`
   sabit yerine yuvarlanan pencere ya da recency `sample_weight`. Rejim-dedektörü
   gerektirmeyen tek-model çözüm.
4b. **Hibrit?** Yalnız C.5.4 statik blend / stacking net kazanç verirse.
   Priorlar zayıf (önceki hibritler başarısız).
5. **OOD dedektörü (spike çözülemezse):** "yenilenebilir baskısı X + gaz payı Y →
   model dağılım dışı, P10-P90 bandını genişlet / düşük güven flag'i." Tahmin
   çözülmese bile gönderilebilir bir güvenlik özelliği.

**Compute:** LEAR dilimleri dakikalar. WF-LightGBM ~365 gün × birkaç sn = dilim
başına ~10-20 dk. Toplam yarım gün + notebook.

### Faz D — Genişletilmiş / uyarlanmış LEAR (bizim impl) (yarım gün)

**Taban (Faz C çıktısı):** pencere kümesi **{56, 180, 1092, 1456}** + **uyarlanabilir
ağırlık** (CING-LEAR eq 10). Bunun üzerine feature ablasyonu.

- [ ] `03_lear_feature_ablation.ipynb`'i 1096g uzun pencereye taşı (şu an 365g),
      yeni taban pencere kümesiyle. `ablation_*_v2.csv`
- [ ] En iyi özellik seti (`+gaz` net, `+hava` marjinal — `03` bulgusu) eklenip
      → **"uyarlanmış LEAR"** tek sayısı (rMAE + rejim-katmanlı)
- [ ] Uyarlanmış LEAR vs kanonik LEAR ensemble ($8.00) vs LightGBM ($7.35) — üçlü DM.
      Çöküş rejiminde özellikle: uyarlanmış LEAR LightGBM'i geçebiliyor mu?
      (C.5 adaptif taban $10.51 zaten ≈ LightGBM $10.54)

### Faz E — Alanı tamamla (opsiyonel, düşük öncelik)

- [ ] fARX (OLS, shrinkage yok) — `lago_protocol.py`'ye `farx_predict_day`.
      "Shrinkage gerçekten gerekli mi" kontrolü
- [ ] fARX-EN (ElasticNet) — LEAR'ın ceza-türü varyantı
- [ ] ARIMA / çift-mevsimsel ETS — `statsmodels`, Lago'da hep sonuncu, tamlık için
- [ ] **CING-LEAR** (Wang et al., `competition_plus.pdf`) — LEAR'ın çok-değişkenli
      uzantısı: 24 saatlik katsayı matrisi B'ye satır-bazlı **group Lasso** → özellik
      ya 24 saatin hepsinde seçili ya hiç. CAISO'da LEAR'ı ~%7, DNN'i geçiyor.
      `sklearn.linear_model.MultiTaskLasso` ile kurulabilir (aynı X, P=N×24 hedef).
      Ön işleme + özellik seti bizim LEAR ile birebir. Ayrı deney: `07_cing_lear.ipynb`.

### Faz F — DNN + TSFM (AYRI, sonraya)

- epftoolbox `DNN` + hyperopt (tree-Parzen, Lago §4.3). Günler sürebilir.
  `06_dnn_benchmark.ipynb`. Lago'nun ikinci referans modeli.
- **Chronos-2** (`competition_plus.pdf` §4.2.7) — önceden eğitilmiş zaman-serisi
  temel modeli (TSFM). Zero-shot ucuz deneme; fine-tune ~30dk/güncelleme.
  CAISO'da LEAR + DNN'i geçti, fine-tuned Chronos ≈ en iyi tekil model.
  **CING-LEAR + Chronos ensemble = makalenin en iyisi** ("bütün > parçalar toplamı",
  tamamlayıcı bilgi). TR'de: zero-shot Chronos'u LightGBM'e karşı dene.
- Faz A-E bitip LEAR tarafı kapanınca planlanır. LEAR tarafı bağımsız raporlanabilir.

### Faz G — Benchmark raporu

- [ ] `BENCHMARK_REPORT.md` + `reports/` altında artifact
- [ ] **İki ana tablo:** (a) LEAR ailesi + naive, 3 yıl, rejim-katmanlı;
      (b) tam yarış + LightGBM, 728g, rejim-katmanlı. Her satır:
      MAE / RMSE / sMAPE / rMAE(n2) / rMAE(n3) / DM-p
- [ ] Rejim-katmanlı alt-tablolar (normal / çöküş / toparlanma)
- [ ] Bulgular:
      1. Hiçbir benchmark canlı LightGBM'i geçemedi (çoklu impl, çoklu konfig) —
         ama rejim kırılımına dikkat (`03`: LEAR Nis-Tem 2026'da LightGBM'i geçti)
      2. Gaz fiyatı LEAR'ın kilit eksiği (kazancın ~%60'ı, `03`)
      3. Lago'nun 4-yıl penceresi TR'ye transfer olmuyor (Faz C)
      4. Bizim impl vs epftoolbox: ~$1/MWh, konfig farkından ($1.8) küçük
      5. ~~LEAR TR'de stresli rejimde rekabetçi~~ — **DÜZELTİLDİ (`04`):** kanonik
         LEAR ensemble çöküş rejiminde de LightGBM'in altında (12.98 vs 10.54);
         en iyi LEAR = kısa pencere cw56 (11.90), yine geride
      6. Canlı LightGBM kaydı 2024-08'de başlıyor → benchmark penceresi bu
         kısıttan doğuyor, keyfi değil
      7. Lago'nun 4-yıl (cw1456) penceresi TR'ye transfer olmuyor — çöküşte
         cw1092/1456 ≈ $16-17, cw56 ≈ $12

---

## 4. Çıktı envanteri (Faz B-D sonunda)

**Kanonik (epftoolbox) ✅ Faz B:** `lago_ref_lear_cw{56,84,1092,1456}.csv`
(1096g) + ensemble + naive-1/2/3. Uzun-3y ensemble $8.00, birincil-728g $8.08.

**Araştırma (bizim impl, 728g):** LEAR cw56/84/1092/1456 + ensemble +
TR-ayarlı ensemble → 6 seri.

**Genişletilmiş (bizim impl, 728g):** ablasyon 5 varyantı (taban, +gaz,
+gaz+hava, +gaz+hava+yenilenebilir, beslenmiş) → 5 seri.

**Referans:** LightGBM (canlı, `gold.ptf_predictions_daily`).

Toplam ~19 model serisi, hepsi tek protokolde, tek DM-karşılaştırma çatısında.
