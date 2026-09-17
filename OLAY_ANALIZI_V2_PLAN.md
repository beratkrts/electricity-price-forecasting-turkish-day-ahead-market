# Olay Bazlı Fiyat Analizi — V2 Çalışma Planı

**Tarih:** 3 Eylül 2026
**Durum:** Yön belirlendi, çalışma başlamadı.
**İlişki:** `EVENT_IMPACT_STUDY.md`'nin §4 (olay listesi) ve §5 (yöntem) bölümlerini
**geçersiz kılar**. §2 (eldeki veriler) ve §7 (otomatik tespitten neden vazgeçildi)
geçerli kalıyor. Metodoloji eleştirisi: `reports/OLAY_HATTI_METODOLOJI.md`.

**Karar (3 Eylül 2026, kullanıcı):**
- Çıktı: **yazılı çalışma + dashboard katmanı, eşit ağırlık.**
- Kapsam: **B + A + C** (aşağıda). Tez derinliği.
- Mevcut `silver.market_events` + dashboard katmanı: **dondur, dokunma.** Yeni çalışma
  ayrı tablolarda/sayfada gelişir; hazır olunca geçiş ayrı karar.

---

## 1. Neden sıfırdan

Mevcut çıktı (23 olay, düz tablo) üç sebeple yürümüyor
(`reports/OLAY_HATTI_METODOLOJI.md` tam gerekçe):

1. **"Olay" tanımlı bir analiz birimi değil** — regülasyon basamağı, arz şoku, savaş
   yörüngesi ve mevsimsel talep aynı 5 kolonla yan yana; kategori hatası.
2. **Ölçümler geçersiz** — `pct_change` penceredeki her şeyi içeriyor; kontrafaktüel
   kalıntı, en çok kullanıldığı mekanizma (maliyet/regülasyon) için "etki" değil
   "geçiş sapması" ölçüyor.
3. **Çıktının kendisi null** — 23 olayın en fazla 2'si kontrafaktüelin gürültü
   bandını aşıyor. Bu bir bulgu ama düz tablo onu "her satır bir ölçüm" gibi
   gösteriyor.

Dürüst çerçeve: **tavanlı, idari yakıt fiyatlı, büyük hidro salınımlı bir piyasada
fiyatı tekil olaylar değil rejim belirliyor; dışsal şoklar fiyata idari kanaldan
(BOTAŞ tarifesi) gecikmeli geçiyor.** V2 bu çerçeveyi ölçer.

---

## 2. Çalışmanın omurgası

### Araştırma sorusu

> Tavanlı ve idari yakıt fiyatlı bir elektrik piyasasında dışsal şoklar toptan
> fiyata nasıl geçer, ve gözlenen fiyat değişkenliğinin ne kadarı tavan tarafından
> bastırılır?

### Üç alt-çalışma, tek iskelet

| | Ne | Rolü | Çıktı |
|---|---|---|---|
| **C** | Yapısal geçiş zinciri: TTF/Brent → BOTAŞ ithal maliyeti → BOTAŞ elektrik tarifesi → marjinal gaz SRMC → MCP; tavan = iki-limitli sansür (Tobit) | **Omurga.** Mekanizmayı ve katsayıları verir | Halka-halka geçiş esneklikleri (düzgün SE), "tavan varyansın ne kadarını bastırıyor" |
| **B** | Kesikli dışsal şoklar doğal deney olarak; iki-kontrafaktüel ayrıştırması | Omurgayı **örnekleyen** vaka çalışmaları + savaş kontrastı | Olay başına: toplam / mekanik kanal / kalıntı kanal + plasebo yüzdeliği |
| **A** | Haberden günlük konu-dikkat endeksi | B'yi besler (aday tarihleme) + **haber pipeline'ının asıl ürünü** | Dikkat → tarife → MCP gecikme yapısı (local projection) |

C katsayıyı verir, B o katsayının belirli şoklarda ne kadar iş gördüğünü gösterir,
A şokların ne zaman geldiğini sürekli bir sinyalden okur.

---

## 3. "Olay" tanımı ve seçim kriterleri

**Ölçümden ÖNCE dondurulacak. Bir kez yazıldıktan sonra değişmez; değişirse
versiyonlanır.**

### Tanım

Bir **olay**:
- **Dışsaldır** — zamanlaması Türkiye elektrik fiyatından/politikasından
  kaynaklanmaz. Savaş, boru hattı kesintisi, deprem, küresel gaz/petrol hareketi,
  kur şoku. → tavan/tarife/AUF değişiklikleri **olay değildir**; içsel politika
  tepkisidir, *kontrol / rejim işareti* olurlar.
- **Tarihlenebilir bir başlangıcı** vardır (≥2 bağımsız kaynak).
- **Hipotezlenen bir etkilenen temeli** vardır: gaz arzı (MW), TTF seviyesi,
  Brent, kur, yük.

### Aday evreni (üç kanaldan, birleşik)

1. **Sabit şok-tipi listesi** — literatür + alan bilgisiyle önceden yazılmış:
   Avrupa gaz arz şokları, Türkiye'ye fiziksel gaz kesintileri, Orta Doğu petrol/
   jeopolitik epizotları, büyük kur şokları, aşırı hava (soğuk/sıcak dalgası),
   büyük talep yıkımları (deprem).
2. **A endeksinde dikkat sıçraması** — ilgili konu endeksinde z > eşik.
3. **Yukarı-akış temelinde büyük hareket** — TTF, gazdan üretim MW, kur serilerinde
   eşik üstü değişim.

### Tutma/düşürme kuralları (yazılı)

Her aday için:
- Dışsallık sağlanıyor mu? (hayır → düşür, kontrole taşı)
- Başlangıç ≥2 kaynakla tarihlenebiliyor mu? (hayır → "muğlak", ayrı işaret)
- Etkilenen temel veride görünüyor mu? (TTF sıçradı / gaz MW düştü / kur koptu)
- Pencere başka bir olayla tamamen örtüşüyor mu? (evet → birleşik "epizot", ayrı ölçülmez)
- **Reddedilenler kayda geçer** — gerekçe + veri. Genişletilmiş "değerlendirildi,
  desteklenmedi" bölümü; mevcut 6 drop yeterli değil.

### Kontroller (olay değil, modelde yer alır)

Tüm tavan (AFL) değişiklikleri, BOTAŞ tarife basamakları, AUF mekanizması
(1 Nis 2022–), YEKDEM, mevsim/takvim, termal pay rejimi.

---

## 4. Veri temeli

### 4.1 Eldeki varlıklar

| Varlık | Durum | Not |
|---|---|---|
| `silver.mcp_with_cap` (`at_cap`) | ✅ güncel | Sansür bayrağı; `MAX(price)` asla |
| `silver.price_cap_official` (AFL, 29 kayıt) | ⚠️ 2026-04'te donmuş | Her satır kaynak haberli |
| `silver.gas_tariff_electricity` (BOTAŞ, 30 kayıt) | ⚠️ 2026-04'te donmuş | Santralin **fiilen ödediği** |
| `gold.crisis_counterfactual` (crisis_cf_v5) | ⚠️ 2026-08-18'de donmuş | Sansürsüzde eğit / her saate tahmin; `analysis_model.py` |
| `bronze.news_raw` (28.102) | ⚠️ 2026-08-16'da donmuş | enerjigunlugu.net; tek kaynak |
| Temeller (mcp/kgup/load/weather/macro/GRF/hidro) | ✅ güncel | Canlı ETL besliyor |
| 182 etiketli altın haber kümesi | ✅ | A'nın süpervize bileşeni için |
| TF-IDF sinyal prototipi (`tfidf_signal.py/.png`) | ✅ | A'nın ilk hali |

### 4.2 Faz 0'da doğrulanmış (2 Eyl 2026, `FAZ0_BULGULAR.md`)

- **TTF** `TTF=F` / yfinance ile geliyor, tam dönem (2021-01 → 2026-09). Olay
  dedektörü **değil** (Türkiye'ye özgü olaylar TTF'de yok), **dışsal maliyet çıpası**.
- **İlk C-halkası ölçüldü:** TTF → BOTAŞ tarifesi eşzamanlı +0,058 (**sıfır**),
  +1 ay **+0,575**, +2 ay +0,432. Seviye korelasyonu +0,71 (kriz +0,76, normalleşme
  +0,53). "İdari vana" artık ölçüm.
- **GDELT DOC 2.0** — toplu iş için uygun değil (5 sn/istek, ağır sorguda 2/9 döner),
  **aday olay başına hedefli retrieval için uygun.** enerjigunlugu'nun kör olduğu
  Türkiye-özgü olayları buluyor (İran kesintisi → 25 makale).

### 4.3 Eksik / kurulacak

| Eksik | Nasıl | Zorluk |
|---|---|---|
| TTF günlük seri (bakımlı) | `TTF=F` yfinance → yeni `silver.upstream_drivers` | düşük |
| BOTAŞ ithal/sınır gaz maliyeti | BOTAŞ PDF'leri yok; **TTF + gecikme + kur ile proxy**, ya da BOTAŞ bülten arşivi taranır | **yüksek** |
| AUF serisi | EPDK duyuruları + haber arşivi; teknoloji bazlı gelir tavanı seviyeleri | orta |
| GDELT jeopolitik risk endeksi | hedefli çekim, aday başına | orta (hız sınırı) |
| Haber dikkat endeksi (günlük, konu bazında) | A çalışması — §5.1 | orta |
| Haber ingestion (artımlı, zamanlı) | `fetch_news_archive.py --since` zaten var; daemon/cron | düşük |
| Kontrafaktüelin bugüne uzatılması | `build_analysis_model.py --write` | düşük |

---

## 5. Yöntem

### 5.1 A — Haber dikkat endeksi

**Amaç:** günlük, konu bazında bir *dikkat* serisi. Etki değil dikkat — gözlenebilir.

**Konular (sabit):** gaz-arz-güvenliği · jeopolitik-enerji · regülasyon-tavan ·
hidro-hava · talep-mevsim.

**Yöntem seçenekleri (Faz 1'de karara bağlanacak, biri seçilip diğerleri ek):**
- **(a) Gömü çıpaları** — her konu bir dizi tohum cümleyle tanımlanır; tüm haberler
  çok-dilli bir gömü modeliyle vektörlenir; günlük dikkat = eşik üstü benzerliklerin
  toplamı, haber öne-çıkması (bölüm, sıra) ile ağırlıklı. Disk hafif, LLM gerektirmez.
- **(b) LLM yapısal çıkarım** — GBNF grameriyle (CRISIS_LLM_SETUP.md hazır) her
  haberden **yalnızca kategorik** alanlar: konu, yön (yukarı/aşağı/nötr), ileriye
  dönük mü (evet/hayır), belirginlik (1-3). **Büyüklük/süre puanı YOK** — onlar
  veriden gelir. Engel: boş disk (2026-08'de 12 GB, kullanıcı durdurdu).
- **(c) Denetimsiz konu modeli** (BERTopic) + kümeleri konulara eşle.

**Doğrulama:** endeks bilinen epizotları yeniden kuruyor mu (savaş haftaları
jeopolitik konuda sıçramalı)? GDELT ile çapraz doğrulama (Faz 0'da doğrulandı).
`tfidf_signal.py` bunu kısmen gösteriyor.

**Bağımsız analiz — local projection:** bir dikkat şokuna {BOTAŞ tarifesi, MCP,
kalıntı} tepkisi, h = 0..180 gün, temeller kontrol. Beklenti: dikkat → tarife (~30g)
→ MCP (~120g); kalıntı tepkisi ~0.

**B'de kullanım:** dikkat sıçramaları = aday olay başlangıçları (§3 kanal 2).

### 5.2 B — Olay-çalışması, iki-kontrafaktüel ayrıştırması

**Olay kümesi:** §3 kriterleriyle dondurulmuş dışsal şoklar. İlk hedef liste
(kesinleşmedi):

| Olay | Etkilenen temel | Rejim | Not |
|---|---|---|---|
| Rusya-Ukrayna işgali (24 Şub 2022) | TTF | tavan bağlıyor | **Merkez vaka.** TTF ±30g +%63 |
| Nord Stream / Gazprom kesintisi (Eyl 2022) | TTF | tavan bağlıyor | TTF zaten zirvede (−%15) — "fiyatlanmış şok" testi |
| İran→Türkiye gaz kesintisi (19 Oca 2022) | gaz MW (−%28,6) | tavan bağlıyor (%73) | Vaka hazır: `CRISIS_CASE_IRAN_2022.md`. Kalıntı alt sınır |
| Kahramanmaraş depremi (6 Şub 2023) | yük | tavan gevşek (%15) | Talep yıkımı; temiz doğal deney |
| ABD-İsrail-İran (13 Haz 2025) | Brent | tavan gevşek (%8) | **2022'nin kontrolü.** TTF tepkisiz, tarife kıpırdamadı |
| Aralık 2021 kur şoku (KKM) | kur | tavan bağlıyor | Hedef USD → kısmen mekanik; ayrı tartış |
| Aşırı hava epizotları | yük | değişken | Aday — literatürden tarihlenecek |

**Tasarım (olay başına):**
- Tahmin penceresi: kontrafaktüel [başlangıç−tampon, başlangıç+pencere+tampon]
  dışındaki sansürsüz saatlerle eğitilir (bloklu OOF — `analysis_model.py` hazır).
- **CF₁ (mekanik sabit):** olayın vurduğu temel (gaz MW / TTF-türevli maliyet / kur)
  olay-öncesi seviyesine sabitlenerek yeniden tahmin.
- **CF₂ (gerçekleşen):** standart kontrafaktüel.
- **Ayrıştırma:**
  - toplam etki = gerçek − CF₁
  - mekanik kanal = CF₂ − CF₁
  - kalıntı kanal = gerçek − CF₂
- **Plasebo:** aynı üçlü, sakin dönemlerde N rastgele pencerede (tavan payı <%5,
  ±30g'de olay yok) → her bileşen için boş dağılım; olayın yüzdelik yeri.
- **Pre-trend:** kalıntı eğimi [başlangıç−30g, başlangıç].
- **Sansür şerhi:** tavan payı yüksekse kalıntı ve toplam **alt sınır**.

**Merkez kontrast — 2022 savaşı vs 2025 İran:** aynı şok sınıfı, iki ayrıştırma yan
yana. BOTAŞ tarifesi + dikkat endeksi bindirilir; tarife kanalı birinde açıldı
(2022), diğerinde açılmadı (2025) — Faz 0'daki +1 ay gecikme buna mekanizma veriyor.
n=2, istatistik değil kontrollü vaka kıyası — raporda böyle sunulur.

### 5.3 C — Yapısal geçiş + sansür

**Zincir:**

1. **TTF/Brent → BOTAŞ elektrik tarifesi.** İdari tepki fonksiyonu:
   tarife_t = f(ithal maliyet geçmişi, gecikme, politik takvim). Faz 0: eşzamanlı
   sıfır, +1 ay +0,575. Asimetri testi (yukarı/aşağı yapışkanlık — memory: iki yönde
   de yapışkan). ARDL kısa/uzun vade.
2. **BOTAŞ tarifesi → marjinal gaz SRMC.** Mekanik: 1000 Sm³ = 10,646 MWh_th
   (9155 kcal/Sm³), verim ~%55 → tarife $/MWh_th × ~1,8 = $/MWh_e.
3. **SRMC → MCP.** Liyakat sırası geçişi, rejim bağımlı (termal pay, tavan
   bağlayıcılığı). Memory: 1 $/MWh gaz → 1,45 $/MWh elektrik (r=0,77, n=30) —
   kontrolsüz düz OLS, C bunu düzeltir: kontroller (termal pay, yük), Newey-West SE,
   rejim etkileşimleri.

**Sansür:** MCP, marjinal teklif AFL'yi aşacağında sağdan sansürlü; 2026'da tabana
yakın (sıfır fiyat) — **iki-limitli Tobit.** Gizli (sansürsüz) fiyat sürecini tahmin
et. "Sansürsüzde eğit / her yere tahmin et" yaklaşımıyla karşılaştır — hakem sorusu
(`EVENT_IMPACT_STUDY.md` §5.3).

**Ana sonuç:** gizli fiyatı simüle et, gözlenenle varyansını kıyasla →
**"tavan varyansın %X'ini bastırıyor."**

---

## 6. Pipeline mimarisi

```
[INGEST]  (zamanlı — repo yerleşimi ayrı karar, bkz. OLAY_YAKALAMA_HATTI_DENETIM.md §4)
  fetch_news_archive.py  --since (artımlı)        → bronze.news_raw
  fetch_upstream_drivers.py (TTF, Brent, kur)     → silver.upstream_drivers
  fetch_gdelt_targeted.py (aday olay başına)      → bronze.gdelt_articles

[INDEX]
  build_attention_index.py (gömü çıpaları / LLM kategorik)
                                                  → silver.news_attention_daily

[CF]
  build_analysis_model.py (genişletildi: CF₁/CF₂) → gold.event_counterfactual

[STUDY]
  event_study.py   (olay başına ayrıştırma + plasebo) → gold.event_decomposition_v2
  passthrough_model.py (C: ARDL + Tobit)              → reports/ tabloları

[SERVE]  (mevcut /api/events ve dashboard DONMUŞ — bunlar yeni)
  /api/event-analysis  → yeni dashboard sayfası
```

Anahtar: **dikkat endeksi haber pipeline'ının ürünü** — bakımlı günlük seri. Bu, işi
"tek seferlik analiz" olmaktan çıkarıp "pipeline" yapan şey.

---

## 7. Fazlar ve karar noktaları

| Faz | İş | Çıktı | Karar noktası |
|---|---|---|---|
| **0 — Temel** | Artımlı ingestion; TTF/kur/AUF serileri; kontrafaktüeli bugüne uzat; **olay evrenini + konu listesini + pencere kurallarını yaz ve dondur** | `silver.upstream_drivers`, güncel `crisis_counterfactual`, `OLAY_EVRENI.md` | Olay listesi süpervizör onayı |
| **1 — A** | Dikkat endeksi: yöntem seç (gömü / LLM / topic), kur, doğrula; local projection | `silver.news_attention_daily` + doğrulama notu | Endeks bilinen epizotları kuruyor mu? Kurmuyorsa B'ye geç, A'yı ertele |
| **2 — B** | CF₁/CF₂ genişletmesi; olay kümesinde ayrıştırma; plasebo dağılımları; savaş kontrastı | `gold.event_decomposition_v2` + vaka bölümleri | Ayrıştırma plasebo bandını aşan olay veriyor mu? |
| **3 — C** | Halka-halka geçiş; ARDL; iki-limitli Tobit; varyans bastırma | Katsayı tabloları + "tavan %X bastırıyor" | Tobit "sansürsüzde eğit" ile tutarlı mı? |
| **4 — Sentez** | Yeni dashboard sayfası (ayrı tablolar); yazılı çalışma; mevcut katmanla değişim kararı | Dashboard sayfası + rapor | Mevcut `silver.market_events` emekliye mi, yan yana mı? |

**Kapsam uyarısı:** B+A+C tez derinliği aylarca iş. Her faz bağımsız değerli bir
çıktı bırakır; faz sonunda dur/devam kararı.

---

## 8. Bilinen riskler / açık sorular

1. **BOTAŞ ithal maliyeti verisi yok.** C'nin 1. halkası için ya BOTAŞ bülten arşivi
   taranmalı ya TTF+gecikme+kur proxy'siyle yetinilmeli. Proxy, C-1'i "TTF geçişi"ne
   indirger — kabul edilebilir ama zayıflık.
2. **Jeopolitik olay n=2.** Rusya-Ukrayna + ABD-İran. İstatistiksel sonuç değil,
   kontrollü vaka kıyası. A endeksi bunu aşan tek yol (her dikkat sıçraması bir
   gözlem).
3. **LLM disk engeli.** A'nın (b) yolu 12 GB boş disk gerektiriyordu, kullanıcı
   2026-08'de durdurdu. (a) gömü yolu bu engeli aşıyor ama gömü modeli de disk
   ister — ölçülmeli. `paraphrase-multilingual-MiniLM` ~450 MB.
4. **Kur şoku kısmen mekanik.** Hedef USD/MWh olduğu için TL değer kaybı zaten
   içeriliyor; Aralık 2021'i olay olarak almak çift sayım riski. Ayrı ele alınacak.
5. **Kontrafaktüel modelin dönem tanıması.** `analysis_model.py` notu: 90 günlük
   yakıt ortalaması "hangi dönemdeyiz" der ve İran sinyalini yutar. CF₁/CF₂
   genişletmesi bu tuzağa dikkat etmeli — sabitlenen feature seti dikkatle seçilmeli.
6. **Reprodüksiyon.** Aday üretim çıktısı hash'lenip dondurulacak; katalog
   versiyonlanacak. Mevcut hattın `DROP TABLE` sorunu yeni tablolarda tekrarlanmayacak
   (slug anahtar + upsert).

---

## 9. Mevcut sistemle ilişki

- `silver.market_events` (23 olay) ve `/api/events` + `MarketEventsTimeline.tsx`:
  **değişmez.** V2 ayrı tablolarda (`_v2` sonek) ve ayrı dashboard sayfasında.
- `gold.crisis_counterfactual` / `crisis_cf_v5`: V2 bunu kullanır ama CF₁/CF₂ için
  **genişletir** (yeni `variant` veya yeni `model_name`), mevcut satırlara dokunmaz.
- Faz 4 sonunda "mevcut katman emekliye mi" kararı — o güne kadar iki sistem yan yana.

---

## 10. İlk somut adım

**Faz 0'ın ilk yarısı, kod gerektirmez:** `OLAY_EVRENI.md` taslağı —
- sabit şok-tipi listesi (literatür + alan bilgisi),
- her aday için hipotezlenen etkilenen temel,
- tutma/düşürme kuralları,
- kontroller listesi.

Bu dondurulmadan ölçüme başlanmaz. Süpervizör onayı bu dosyaya.
