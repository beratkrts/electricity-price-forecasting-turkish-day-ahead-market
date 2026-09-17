# Düşük Fiyat Rejimi Analizi — 2026 Bahar Çöküşü

**Tarih:** 14 Ağustos 2026
**Soru:** "Model yapısal olarak düşük fiyatları kötü tahmin ediyor" — doğru mu, ve sebebi ne?
**Sonuç:** Doğru, ama sebebi kayıp fonksiyonu değil. Sebep **hidro kaynaklı bir merit-order rejim
kırılması** ve modelin hidroyu güneş/rüzgardan farklı (24h gecikmeli) ele alması.

Bu doküman tüm bulguları **tekrar üretilebilir SQL ile** içerir. Her bölümdeki sorgu
kopyala-yapıştır çalıştırılabilir. Notebook'a dönüştürme planı en sonda (Bölüm 7).

---

## 0. Bağlantı

```python
from sqlalchemy import create_engine, text
import pandas as pd, numpy as np
engine = create_engine("postgresql://postgres:postgres@localhost:5432/enerji_db")
# Alternatif: from db.connection import get_engine; engine = get_engine()
```

CLI için: `docker exec enerji_postgres psql -U postgres -d enerji_db -c "<SQL>"`

**Not:** Canlı tahminler `gold.ptf_predictions_daily`, gerçekleşen fiyat `public.raw_mcp_hourly`
(kolon adı `ts`, `price_usd` — `datetime`/`target_datetime` DEĞİL). Join anahtarı:
`p.target_ts = m.ts`. Tüm analiz **USD bazlı** (TL sabit-kur bug'ından bağımsız olmak için).

---

## 1. Modelin fiyat dilimine göre hatası (2 yıl)

Genel: **N=17.520 saat, WAPE %12.24, MAE $7.26, BIAS +$1.52** (2024-08-12 → 2026-08-14, `LightGBM_v1` dönemi)

> **⚠️ GÜNCELLEME (14 Eyl 2026):** Bu sorgu 14 Ağustos'ta, tabloda tek model
> varken yazıldı. 2 Eylül'de `ensemble_v1` canlıya alındı; tablo artık iki
> modeli aynı anda taşıyor (bkz. `METRICS.md` §5 uyarısı — filtresiz sorgu
> çift satır sayıp BIAS'ı sessizce kaydırıyor, ölçülen fark %39). Aşağıdaki
> sorgu `model_name` filtresiyle güncellendi; **aşağıdaki sonuç tablosu hâlâ
> eski (`LightGBM_v1`) döneme ait** — yeniden koşulmadı.

```sql
SELECT
  CASE WHEN m.price_usd <= 0 THEN 'a) = $0'
       WHEN m.price_usd <= 10 THEN 'b) $0-10'
       WHEN m.price_usd <= 20 THEN 'c) $10-20'
       WHEN m.price_usd <= 30 THEN 'd) $20-30'
       WHEN m.price_usd <= 40 THEN 'e) $30-40'
       WHEN m.price_usd <= 60 THEN 'f) $40-60'
       WHEN m.price_usd <= 80 THEN 'g) $60-80'
       ELSE 'h) $80+' END AS dilim,
  COUNT(*) n,
  ROUND(AVG(m.price_usd), 2) ort_gercek,
  ROUND(AVG(p.predicted_mcp_usd), 2) ort_tahmin,
  ROUND(AVG(ABS(p.predicted_mcp_usd - m.price_usd)), 2) mae,
  ROUND(AVG(p.predicted_mcp_usd - m.price_usd), 2) bias,
  ROUND(100 * SUM(ABS(p.predicted_mcp_usd - m.price_usd)) / NULLIF(SUM(m.price_usd), 0), 1) wape,
  ROUND(100.0 * AVG(((m.price_usd BETWEEN p.predicted_mcp_usd_p10 AND p.predicted_mcp_usd_p90))::int), 1) kapsama
FROM gold.ptf_predictions_daily p
JOIN public.raw_mcp_hourly m ON m.ts = p.target_ts
WHERE p.model_name = 'ensemble_v1'  -- 14 Eyl 2026'da eklendi, bkz. uyarı yukarıda
GROUP BY 1 ORDER BY 1;
```

**Sonuç:**

| Gerçek fiyat | n | Ort. gerçek | Ort. tahmin | MAE | **BIAS** | WAPE | P10-P90 kapsama |
|---|---|---|---|---|---|---|---|
| = $0 | 414 | 0.00 | 3.78 | 3.78 | **+3.78** | ∞ | %42.5 |
| $0–10 | 1361 | 4.73 | 13.29 | 9.48 | **+8.56** | %200.6 | %43.7 |
| $10–20 | 484 | 14.38 | 25.77 | 14.52 | **+11.39** | %100.9 | %46.9 |
| $20–30 | 565 | 24.97 | 33.59 | 14.39 | **+8.62** | %57.6 | %58.8 |
| $30–40 | 1030 | 35.72 | 43.55 | 13.48 | **+7.83** | %37.8 | %62.5 |
| $40–60 | 2596 | 51.19 | 57.25 | 10.76 | **+6.07** | %21.0 | %67.2 |
| $60–80 | 7147 | 71.41 | 70.28 | 5.63 | −1.13 | %7.9 | %83.3 |
| $80–120 | 3923 | 84.59 | 81.35 | 3.96 | −3.24 | %4.7 | %75.1 |

**Okuma:** BIAS sütunu monoton ve tek yönlü — fiyat düştükçe model yukarı kaçıyor. Bu gürültü
değil, sistematik ortalamaya çekme (shrinkage). P10-P90 kapsama nominal %80 olmalı; ucuz dilimde
**%43**'e düşüyor — belirsizlik bandı da orada çalışmıyor.

### 1b. Model versiyonu karşılaştırması

Aynı dilimleri `gold.ptf_predictions_experimental` için `model_name`'e göre koştur.
**Sonuç:** canlı ile `lgb_lag0_v2` ucuz dilimde MAE $8.15 vs $8.16 — **fark yok**.
`lgb_cqr_v2` belirgin kötü (genel MAE $14.05 vs $7.26).
→ **lag0 yenilenebilir feature'ları bu problemi çözmüyor.**

### 1c. Örnek vaka: 15.08.2026

24 saatin **24'ü de** yukarı sapmış. Günlük WAPE %20.08, MAE $10.11, BIAS +$10.11.
- Ucuz saatler (<$45, n=10): WAPE **%44.0**, BIAS +$13.73
- Pahalı saatler (≥$45, n=14): WAPE %11.8, BIAS +$7.52
- En kötü: saat 15 ($29.36 → $51.60, %75.8), saat 10 ($22.33 → $36.26), saat 8 ($33.28 → $52.69)

---

## 2. METRİK UYARISI: WAPE düşük fiyatta yanıltıyor

```sql
SELECT to_char(date_trunc('month', p.target_ts AT TIME ZONE 'Europe/Istanbul'), 'YYYY-MM') ay,
  ROUND(AVG(m.price_usd), 1) gercek,
  ROUND(AVG(ABS(p.predicted_mcp_usd - m.price_usd)), 2) mae,
  ROUND(AVG(p.predicted_mcp_usd - m.price_usd), 2) bias,
  ROUND(100 * SUM(ABS(p.predicted_mcp_usd - m.price_usd)) / NULLIF(SUM(m.price_usd), 0), 1) wape,
  ROUND(STDDEV(m.price_usd), 1) fiyat_std,
  ROUND(100.0 * AVG((m.price_usd < 20)::int), 0) ucuz_pct
FROM gold.ptf_predictions_daily p
JOIN public.raw_mcp_hourly m ON m.ts = p.target_ts
WHERE p.target_ts >= '2025-09-01'
GROUP BY 1 ORDER BY 1;
```

| Ay | Gerçek ort. | **MAE $** | BIAS | WAPE% | fiyat_std | ucuz% |
|---|---|---|---|---|---|---|
| 2025-09 | 66.2 | 5.46 | +0.83 | 8.2 | 20.6 | 6 |
| 2025-10 | 65.7 | 6.21 | +0.70 | 9.5 | 19.4 | 5 |
| 2025-11 | 66.1 | 5.62 | +1.14 | 8.5 | 17.5 | 3 |
| 2025-12 | 69.8 | 4.97 | +1.51 | 7.1 | 11.0 | 0 |
| 2026-01 | 67.2 | 5.99 | +1.13 | 8.9 | 13.3 | 0 |
| **2026-02** | 47.7 | **13.21** | **+9.22** | 27.7 | 23.1 | 13 |
| 2026-03 | 36.8 | 11.62 | +5.23 | 31.6 | 24.4 | 28 |
| 2026-04 | 20.7 | 11.06 | +4.06 | 53.6 | 26.5 | 67 |
| 2026-05 | 12.6 | **7.68** | +2.09 | **61.0** | 23.6 | 86 |
| 2026-06 | 26.9 | 9.07 | −1.56 | 33.8 | 27.3 | 52 |
| 2026-07 | 57.6 | 9.37 | +3.14 | 16.3 | 23.6 | 7 |
| 2026-08 | 60.7 | 7.75 | −1.36 | 12.8 | 20.7 | 5 |

**İki kritik okuma:**

1. **Nisan→Mayıs'ta MAE DÜŞTÜ ($11.06 → $7.68) ama WAPE YÜKSELDİ (%53.6 → %61.0).**
   Payda $20.7'den $12.6'ya indiği için. Mayıs'taki %61 modelin kötüleştiğini göstermiyor.
   → Düşük fiyat rejiminde WAPE ile yön tayin etme. MAE + BIAS kullan.

2. **BIAS günlük yeniden eğitimle kapanıyor:** Şubat +9.22 → Mart +5.23 → Nisan +4.06 →
   Mayıs +2.09 → Haziran −1.56. Kalıcı sapma yok. Şubat 2026'da **tek seferlik rejim kırılması**
   var ve model ~3 ayda uyum sağladı.

---

## 3. Sebep: güneş değil HİDRO

```sql
WITH g AS (
  SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') m,
    AVG(total_mw) tot, AVG(solar_mw) sol, AVG(wind_mw) wnd,
    AVG(dammed_hydro_mw) dam, AVG(river_hydro_mw) riv,
    AVG(natural_gas_mw) gas, AVG(lignite_mw + import_coal_mw + black_coal_mw) coal
  FROM public.raw_actual_generation_hourly WHERE ts >= '2024-01-01' GROUP BY 1),
p AS (SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') m,
    AVG(price_usd) px, 100.0 * AVG((price_usd < 20)::int) ucuz_pct
  FROM public.raw_mcp_hourly WHERE ts >= '2024-01-01' GROUP BY 1)
SELECT to_char(g.m, 'YYYY-MM') ay, ROUND(p.px, 1) fiyat, ROUND(p.ucuz_pct, 1) ucuz_pct,
  ROUND(g.tot) toplam, ROUND(100 * g.sol / g.tot, 1) gunes_pct,
  ROUND(100 * g.wnd / g.tot, 1) ruzgar_pct,
  ROUND(100 * (g.dam + g.riv) / g.tot, 1) hidro_pct,
  ROUND(100 * g.gas / g.tot, 1) gaz_pct, ROUND(100 * g.coal / g.tot, 1) komur_pct
FROM g JOIN p ON p.m = g.m ORDER BY 1;
```

**2025-05 vs 2026-05 (aynı mevsim, farklı yıl):**

| | 2025-05 | 2026-05 | Fark |
|---|---|---|---|
| Güneş | %3.2 | %2.7 | −0.5 |
| Rüzgar | %11.2 | %10.2 | −1.0 |
| **Hidro** | **%29.8** | **%50.4** | **+20.6** |
| Gaz | %16.6 | %11.6 | −5.0 |
| Kömür | %31.5 | %18.4 | −13.1 |
| **Fiyat** | **$63.6** | **$13.0** | **−$50.6** |

Güneş ve rüzgar payı sabit, hatta hafif düşük. Tek başına hidro +20.6 puan.
Bu bir mevsimsellik olayı değil — 2025 baharında aynı çöküş yok (2025-05 fiyat $63.6).

---

## 4. KRİTİK: hidro payı yıllar arası TRANSFER OLMUYOR

```sql
WITH h AS (
  SELECT 100.0 * (g.dammed_hydro_mw + g.river_hydro_mw) / NULLIF(g.total_mw, 0) hp,
         m.price_usd px, EXTRACT(year FROM g.ts AT TIME ZONE 'Europe/Istanbul') yr
  FROM public.raw_actual_generation_hourly g
  JOIN public.raw_mcp_hourly m ON m.ts = g.ts WHERE g.ts >= '2023-01-01')
SELECT width_bucket(hp, 10, 60, 5) b,
  COUNT(*) FILTER (WHERE yr=2023) n23, ROUND(AVG(px) FILTER (WHERE yr=2023),1) px23,
  COUNT(*) FILTER (WHERE yr=2024) n24, ROUND(AVG(px) FILTER (WHERE yr=2024),1) px24,
  COUNT(*) FILTER (WHERE yr=2025) n25, ROUND(AVG(px) FILTER (WHERE yr=2025),1) px25,
  COUNT(*) FILTER (WHERE yr=2026) n26, ROUND(AVG(px) FILTER (WHERE yr=2026),1) px26
FROM h GROUP BY 1 ORDER BY 1;
```

| Hidro payı | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| 30-40% | $83.1 (n=1422) | $65.3 (n=2034) | $71.7 (n=889) | **$40.8** (n=1523) |
| 40-50% | $83.6 (n=171) | $68.8 (n=249) | $79.8 (n=24) | **$28.2** (n=1217) |
| 50-60% | $58.8 (n=7) | $69.6 (n=8) | — | **$29.1** (n=830) |

**Aynı hidro payı, tamamen farklı fiyat.** 2023-2025'te yüksek hidro ucuz fiyat **demek değildi**,
çünkü gaz hâlâ marjinal santraldi. Model bunu **gerçek veriden öğrendi** ve 2026 için yanlış oldu.

→ Bu yüzden hiçbir kayıp fonksiyonu düğmesi (log1p, sample_weight) bunu çözemez.
Sorun kayıpta değil, feature→fiyat eşleşmesinin yıllar arası değişmesinde.

Ek olarak **support problemi:** 40-50% hidro bandında 2023-2025 toplamı 444 saat,
2026 tek başına 1217 saat. Model bu bölgeyi neredeyse hiç görmemişti.

---

## 5. TRANSFER OLAN BÜYÜKLÜK: termal pay

**Tanım:** `termal_pay = (total − güneş − rüzgar − hidro − jeotermal − biyokütle) / total`
Yani dispatch edilebilir termalin (gaz + kömür + fuel-oil) karşılaması gereken kısım.
Merit-order'daki marjinal santrali doğrudan temsil eder.

```sql
WITH h AS (
  SELECT EXTRACT(year FROM g.ts AT TIME ZONE 'Europe/Istanbul') yr, m.price_usd px,
    100.0 * (g.total_mw - (g.solar_mw + g.wind_mw + g.dammed_hydro_mw + g.river_hydro_mw
             + g.geothermal_mw + g.biomass_mw)) / NULLIF(g.total_mw, 0) termal_pay
  FROM public.raw_actual_generation_hourly g
  JOIN public.raw_mcp_hourly m ON m.ts = g.ts WHERE g.ts >= '2023-01-01')
SELECT width_bucket(termal_pay, 20, 70, 5) b,
  COUNT(*) FILTER (WHERE yr=2024) n24, ROUND(AVG(px) FILTER (WHERE yr=2024),1) px24,
  COUNT(*) FILTER (WHERE yr=2025) n25, ROUND(AVG(px) FILTER (WHERE yr=2025),1) px25,
  COUNT(*) FILTER (WHERE yr=2026) n26, ROUND(AVG(px) FILTER (WHERE yr=2026),1) px26
FROM h GROUP BY 1 ORDER BY 1;
```

| Termal pay | 2024 | 2025 | 2026 |
|---|---|---|---|
| 20-30% | $32.7 (n=158) | $26.1 (n=21) | **$12.3 (n=856)** |
| 30-40% | $40.0 | $32.3 | $29.4 |
| 40-50% | $59.6 | $49.7 | $46.0 |
| 50-60% | $70.1 | $66.1 | $65.2 |
| 60-70% | $74.8 | $71.5 | $70.6 |
| 70+% | $73.4 | $71.5 | $70.1 |

**2024/2025/2026 üst dilimlerde neredeyse birebir aynı.** Hidro payının aksine termal pay
yıllar arası taşınıyor. (2023 aykırı — o dönem fiyat seviyesi tamamen farklıydı, 70+%'de $129.)

**2026 krizi tek cümle:** termal pay 20-30% bandına düştü — 2026'da **856 saat**,
2025'te sadece **21 saat**. O bantta fiyat ~$12.

---

## 6. Modelin bu mekanizmadaki açıkları

`src/features/feature_engineering.py:184-191`:

```python
hydro_lag24 = df_feat['kgup_hydro_lag_24'].fillna(0)                      # satır 184
df_feat['net_load_lag0'] = predicted_load_lag0 - (solar_pf + wind_pf)     # satır 186 — hidro YOK
df_feat['renewable_pressure_ratio_lag0'] = \
    (solar_pf + wind_pf + hydro_lag24) / load_lag0_safe                   # satır 187
```

Mekanizma **kısmen** var (`renewable_pressure_ratio_lag0` hidroyu içeriyor), ama üç kusurla:

1. **Hidro T+1 tahmini YOK.** Güneş/rüzgarın pre-forecaster'ı var (`gold.kgup_load_pre_forecasts`:
   `predicted_load_lag0`, `predicted_solar_lag0`, `predicted_wind_lag0`), hidronun yok —
   `kgup_hydro_lag_24` ile **24 saat gecikmeli** giriyor. Şubat-Mayıs rampasında hidro her gün
   artarken lag24 sistematik olarak eksik gösteriyor → arz olduğundan az görünüyor →
   fiyat yukarı tahmin ediliyor. **Şubat 2026'daki +$9.22 bias'ın mekanizması budur.**
2. **`net_load_lag0` hidroyu hiç çıkarmıyor** — 2026'da üretimin yarısını oluşturan kaynak
   bu feature'da yok.
3. **Jeotermal + biyokütle** (must-run, ~%4-5 istikrarlı) hiçbir orana girmiyor.

**Veri boşluğu:** `raw_master_active_fullness` (baraj doluluk) çekiliyor ama DB'de
**sadece 2026-08 var — 89 baraj, tek ay**. Geçmişi olmadığı için feature olarak kullanılamaz.
Yağış/doluluk hipotezini test etmek için önce bu tablonun geçmişini backfill etmek gerekir.

---

## 7. Notebook'a dönüştürme planı

Hedef dosya: `experiments/notebooks/low_price_regime_analysis.ipynb`

| Hücre | İçerik | Kaynak |
|---|---|---|
| 1 | Markdown: başlık + soru + sonuç özeti | Bu dokümanın başı |
| 2 | Import + `engine` kurulumu | Bölüm 0 |
| 3 | Bölüm 1 SQL → df → tablo + **grafik: dilim bazlı BIAS bar chart** (sıfır çizgisi vurgulu) | Bölüm 1 |
| 4 | Markdown: shrinkage yorumu + kapsama %43 uyarısı | Bölüm 1 |
| 5 | Model versiyonu karşılaştırması (`ptf_predictions_experimental`, `model_name` group by) | Bölüm 1b |
| 6 | 15.08.2026 vaka: 24 saatlik gerçek vs tahmin **çizgi grafiği** | Bölüm 1c |
| 7 | Bölüm 2 SQL → **ikili eksen grafiği: MAE (bar) vs WAPE (çizgi)** — makasın açıldığı yer | Bölüm 2 |
| 8 | Markdown: WAPE tuzağı açıklaması — en önemli metodolojik uyarı | Bölüm 2 |
| 9 | Bölüm 3 SQL → **alan grafiği: aylık üretim payları (güneş/rüzgar/hidro/gaz/kömür)** + fiyat overlay | Bölüm 3 |
| 10 | Bölüm 4 SQL → **gruplu bar: hidro payı × yıl → ortalama fiyat** (transfer olmadığını gösterir) | Bölüm 4 |
| 11 | Bölüm 5 SQL → **aynı grafik termal pay için** (transfer olduğunu gösterir) — 10 ve 11 yan yana en güçlü görsel | Bölüm 5 |
| 12 | Markdown: kod açıkları (satır referanslarıyla) + öneriler | Bölüm 6 + 8 |

**Grafik notları:** Tüm grafikler USD bazlı. 10 ve 11 numaralı hücreler aynı y-ekseni ölçeğini
kullanmalı — analizin can alıcı karşılaştırması bu. Renk körlüğü güvenli palet kullan.

---

## 8. Öneriler (öncelik sırasıyla)

1. **Metriği düzelt.** Düşük fiyat rejiminde WAPE ile yön tayin etme; MAE + BIAS raporla.
   Yoksa Mayıs gibi aylarda "kötüleşti" diye modeli yanlış yöne çekersin.
2. **Hidroyu diğer yenilenebilirlerle eşit muamele et.** Pre-forecaster'lara hidro ekle
   (KGÜP hidro planı `raw_kgup_hourly.dammed_hydro_mw + river_hydro_mw` olarak zaten var),
   `net_load_lag0`'dan hidroyu çıkar, must-run'lara jeotermal + biyokütle ekle.
   Doğrudan Şubat tipi rampa hatasını hedefler.
3. **Açık `thermal_requirement_ratio` feature'ı ekle** (Bölüm 5 tanımı) — yıllar arası transfer
   olan tek büyüklük. Mevcut `renewable_pressure_ratio_lag0` bunun yaklaşığı ama hidro lag24
   yüzünden bulanık.
4. **Ölçüm:** 2. ve 3. maddeyi birlikte kurup **Şubat-Haziran 2026 üzerinde walk-forward
   backtest**. Başarı kriteri: o dönemdeki MAE'yi $11-13'ten ne kadar indirdiği (WAPE'e bakma).

### Elenen hipotezler

| Hipotez | Neden elendi |
|---|---|
| Kayıp fonksiyonu ölçeği (log1p / sample_weight) | log1p son 6 ayda WAPE'i %30→%27 yapmış — kaldıraç zayıf. Bölüm 4 asıl sebebi gösteriyor. |
| lag0 yenilenebilir feature'ları eksik | Zaten canlıda (lgb_lag0_v2), ucuz dilimde fark yok (Bölüm 1b). |
| Mevsimsellik | 2025 baharında aynı çöküş yok (Bölüm 3). |
| Güneş kapasitesi artışı | Güneş payı 2025-05 %3.2 → 2026-05 %2.7, düşmüş (Bölüm 3). |
| Baraj doluluk verisi ile test | DB'de sadece 2026-08 var, geçmiş yok (Bölüm 6). |
