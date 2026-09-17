# silver.market_events — küratörleme brief'i (Opus'a)

**Girdi:** `market_events_candidates.json` (54 aday, `build_market_events_candidates.py` üretti).
**Çıktı:** `silver.market_events` tablosu + dolu `market_events_candidates.json`
(her adayın `label / mechanism / direction / description / confidence / chosen_article_ids / drop` alanları).
**Amaç:** dashboard'da fiyat geçmişini "anlatılı" hale getiren olay katmanı — bkz.
`../../../ENSEMBLE_CANLI_GECIS.md` yerine `EVENT_IMPACT_STUDY.md` §2.3 ("rejim bazlı, olay bazlı değil").

Bu, otomatik tespit **değil** (3 testte çürüdü — `04_event_attribution_attempt.ipynb`).
İki mevcut silver serisiyle (`price_cap_official`, `gas_tariff_electricity`) aynı iş bölümü:
**retrieval + tarihleme otomatik yapıldı, seçim + açıklama + dedupe senin.**

---

## Aday dosyasının yapısı

Her kayıt:
- `source_type` — `A_cap` / `A_tariff` / `B_pricemove` / `C_external`
- `start_date`, `end_date` (`end_date=null` → nokta olay)
- `auto_hint` — mekanik özet (ör. "AFL tavanı +43% → 2500 TL")
- `silver_ref` — A adaylarında ilgili silver satırı + kaynak article_id
- `auto_context` — `mcp_usd_before/during/after`, `pct_change` (MCP), `cf_residual_usd`
  (crisis_cf_v5: gerçek − fundamental, **+ = fiyat temellerin üstünde**), `cf_at_cap_share`
- `candidate_news` — pencere içi TF-IDF ile ilk 10 haber (`article_id`, tarih, `similarity`, `section`, `title`, `url`)
- **doldurulacak:** `label`, `mechanism`, `direction`, `description`, `confidence`, `chosen_article_ids`, `drop`

---

## Doldurma kuralları

| alan | değer |
|---|---|
| `label` | kısa Türkçe, ~3-6 kelime (ör. "Rusya-Ukrayna savaşı", "1 Nisan 2022 AUF + AFL paketi") |
| `mechanism` | `maliyet` \| `arz` \| `regülasyon` \| `jeopolitik` \| `talep` \| `kur` — birden fazlaysa asıl olan |
| `direction` | `yukari` \| `asagi` \| `karisik` (fiyat üstündeki net baskı) |
| `description` | 1-2 cümle, **seçilen haberlere dayanarak**. Ne oldu + mekanizma. Sayı verirsen `auto_context`'ten. |
| `confidence` | `kesin` (haber + veri net) \| `muhtemel` (confound ağır / dolaylı) |
| `chosen_article_ids` | `candidate_news`'ten 2-5 tanesi. Daha iyisini `bronze.news_raw`'da elle arayabilirsin (tam metin). |
| `drop` | `true` → aday geçersiz (gürültü, dedupe kurbanı). Gerekçesini `description`'a yaz. |

---

## Bilinen sorunlar — bunları sen çözeceksin

1. **A_cap ↔ A_tariff aynı-tarih çiftleri.** Tavan ve BOTAŞ tarifesi sık sık birlikte değişmiş
   (2022-02-01, 2022-03-01, 2022-09-01, 2023-01-01, 2023-02-01, 2023-03-01, 2023-04-01, 2025-04-05,
   2026-04-04). **Çiftin birini `drop`, diğerini `mechanism=regülasyon` + description'da ikisini de
   belirt.** ya da tek "regülasyon paketi" olayına birleştir.
2. **2021-2023 aylık tarife rampası.** 22 tarife adayının çoğu enflasyon kaynaklı aylık +%15-20 drift —
   tek uzun yörünge, 15 ayrı olay değil. **Sadece büyük mutlak sıçramaları tut:** Tem 2021 (+40%),
   Kas 2021 (+47%), Eyl 2022 (+50%, seri maks 20.625 TL), 2023 geri sarımı (−%13→−%20 zinciri = tek olay).
   Gerisi `drop`.
3. **#20/#21/#22 (2022-04-01)** — cap + tariff + AUF aynı gün. Tek olaya birleştir:
   "1 Nisan 2022 — AUF mekanizması + AFL 1.745→2.500". `EVENT_IMPACT_STUDY.md` §4'te not var
   (AFL ≠ AUF ayrımı).
4. **B_pricemove #51 (2026 çöküşü)** — dört ay birleşik, doğru. Ama haber retrieval zayıf
   (jenerik tohum). `bronze.news_raw`'da "baraj doluluk / kar erimesi / yağış / hidro rekor /
   yenilenebilir pay" ile elle ara. `04_zero_price_crisis/01_low_price_regime_analysis.ipynb` teşhisi.
5. **#14 İran gaz kesintisi** — haberler "İran" demiyor, alt etkisini yazıyor ("sanayiye gaz kısıtı",
   "elektrik molası"). Retrieval doğru, sadece sim düşük. `CRISIS_CASE_IRAN_2022.md` §2'de tam
   zaman çizelgesi var — oradan article_id çek.
6. **#17 / #49 (savaşlar)** — retrieval iyi, TF-IDF sinyal kontrolü de yapıldı (`tfidf_signal.py`).
   Bunlar vaka çalışmasının çekirdeği, en dikkatli bunları yaz.
7. **Hedef:** ~15-25 nihai satır. Şu an 54 aday → ~yarısı drop/merge.

---

## Tablo DDL

```sql
CREATE SCHEMA IF NOT EXISTS silver;
CREATE TABLE silver.market_events (
    event_id           serial PRIMARY KEY,
    start_date         date NOT NULL,
    end_date           date,                    -- NULL = nokta olay
    label              text NOT NULL,
    mechanism          text NOT NULL,           -- maliyet|arz|regülasyon|jeopolitik|talep|kur
    direction          text NOT NULL,           -- yukari|asagi|karisik
    description        text NOT NULL,
    confidence         text NOT NULL,           -- kesin|muhtemel
    source_article_ids bigint[] NOT NULL,
    source_urls        text[] NOT NULL,
    mcp_usd_before     numeric,                 -- auto_context'ten
    mcp_usd_during     numeric,
    cf_residual_usd    numeric,                 -- crisis_cf_v5 pencere ort.
    cf_at_cap_share    numeric,
    curated_by         text DEFAULT 'opus',
    created_at         timestamptz DEFAULT now()
);
```

`source_urls` = seçilen `chosen_article_ids`'lerin `bronze.news_raw.url`'leri.

---

## Dashboard tarafı (sonraki adım, ayrı)

`silver.market_events` hazır olunca: `api_server.py` → `/api/events` endpoint;
frontend `Analysis` sayfası ECharts grafiğine `markLine`/`markArea` + tıkla→yan panel
(pencere gerçek vs kontrafaktüel + seçilen haber başlıkları). ECharts native, düşük risk.

---

## Küratörleme yapıldı (2 Eylül 2026)

`experiments/scripts/curate_market_events.py` → **54 aday → 23 olay**, `silver.market_events` yazıldı.
İz: `market_events_curated.json` (her adayın hangi olaya girdiği / neden düştüğü).

**DDL'de bir değişiklik var:** `description` ve `interpretation` diye **iki** metin alanı var.
Kullanıcı kararı — dashboard olguyu gösterir, mekanizma iddiası ayrı kolonda durur:

| alan | ton | nerede kullanılır |
|---|---|---|
| `description` | nötr — ne oldu, hangi haberde | dashboard yan paneli |
| `interpretation` | mekanizma iddiası ("idari vana" çerçevesi) | sunum + tez |

Ayrıca `candidate_ids`, `pct_change`, `n_hours` kolonları eklendi.

### Kritik: ölçümler final pencereden YENİDEN hesaplandı

Aday dosyasındaki `auto_context` aynı-tarihli çiftlerde birebir aynıydı (pencere aynı), ve
nokta olaylarda "during" penceresi yalnızca ~25 saatti. Bu iki sorun dört yorumu bozdu —
hepsi ilk taslakta yazılmış, dry-run'da yakalanmıştı:

| olay | adayın dediği | final pencere (30 gün) |
|---|---|---|
| Mayıs–Haziran 2022 | tavan arttı, fiyat **−%32** | fiyat **+%20** (−%32 tek günlük ortalamaydı) |
| Ekim 2023 | tarife +%20, fiyat **−%15** | fiyat **+%8**, ama kalıntı −4,6 (kısmi geçiş) |
| Yaz 2022 | kalıntı **−31,9** | 4 aylık ortalamada **+3,7** (−31,9 dar Eylül penceresi) |
| Şubat 2022 | tavan payı **%96** | **%53** (Mart 2022 %64 ile daha yüksek) |

**Ders:** tavan/tarife etkisi ölçüm penceresine aşırı duyarlı. Tek günlük karşılaştırma yanıltıyor.
Yeni olay eklenirse `measure()` üzerinden ölçülmeli, aday dosyasından kopyalanmamalı.

### Kataloğun kendi içindeki kontrolleri

- **Rusya-Ukrayna (2022) ↔ ABD-İsrail-İran (2025)** — aynı tip jeopolitik şok, farklı sonuç.
  Fark idari vananın (BOTAŞ tarifesi) açılıp açılmamasında. Kanıt: `tfidf_signal.py` / `.png`.
- **ABD-İsrail-İran (Haz 2025) ↔ Temmuz 2025 yaz zirvesi** — savaşa atfedilebilecek yükselişin
  aslında mevsimsel olduğunu gösteriyor (kalıntı −0,2).
- **Nisan 2026 ↔ Ekim 2023** — idari araç hareketi ile fiyat hareketi bire bir eşleşmiyor.

### Retrieval'ın kendiliğinden çıkardığı yeni olay

**AUF'a yargı freni (Oca–Mar 2023)** — aday üretiminde hedeflenmemişti, `#34`/`#40`'ın haber
listesinden çıktı. EPDK'nin AUF uygulamasına yargı freni, 1 milyar TL geri ödeme, mekanizmanın
buna rağmen uzatılması.

### Düşenler (6 aday) ve gerekçeleri

`market_events_curated.json` → `dropped`. Özet: 4'ü aylık tarife drifti / tarife değişmemiş
(#1, #2, #5, #31), #11 kur şoku retrieval çuvalladı + hedef zaten USD bazlı, #44 (Nis 2024
−%20) gerçek hareket ama haberlerin tamamı gürültü — belgelenebilir olaya bağlanamadı.

### Sıradaki

`api_server.py` → `/api/events`; frontend ECharts `markArea` + tıkla→yan panel
(`description` + kaynak linkleri + gerçek/kontrafaktüel). `interpretation` dashboard'a **girmez**.
