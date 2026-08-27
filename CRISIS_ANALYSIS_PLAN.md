# Kriz & Olay İstihbarat Sistemi — Tasarım ve Fizibilite Raporu

**Tarih:** 14 Ağustos 2026
**Durum:** Fizibilite tamamlandı, inşaya geçilmedi
**Önceki aşama:** Fiyat tahmin modeli (kapandı — 24 saatlik ayrı model deneyi reddedildi, `lgb_lag0_v2` canlı şampiyon)

---

## 1. Amaç

Elektrik piyasasını etkileyen olayları haberlerden toplayıp etiketlemek, fiyatlara kısa/orta/uzun vadeli etkisini **istatistiksel olarak** ölçmek ve dönemsel konjonktür raporu üretmek.

Kritik kısıt: *"Bir haber sitesi öyle dedi"* diye dashboard'da iddiada bulunulamaz. Her çıktının ölçülebilir bir dayanağı ve belirtilmiş belirsizliği olmak zorunda.

---

## 2. Ölçülmüş bulgular (fizibilite)

### 2.1 Haber arşivi erişilebilir

| | Sonuç |
|---|---|
| Kaynak | `enerjigunlugu.net`, sitemap üzerinden |
| Kapsam | 2012'ye kadar tam; **2021+ için 27.419 haber** |
| Günlük hacim | ~13 haber |
| Yayın tarihi | `itemprop="datePublished"`, sitemap `lastmod` ile birebir → `published_at` güvenilir |
| Gövde | `itemprop="articleBody"` mikroverisi |
| Kategori | `itemprop="articleSection"` (Elektrik / Doğalgaz / Kömür / Mevzuat / …) |
| Test sonucu | 40 haber çekildi, **0 hata, hiçbir alan eksik değil** |
| Gövde uzunluğu | medyan 1.175 karakter (~465 token) |
| `robots.txt` | Sitemap bildirimi var, `Disallow` yok |

`enerjiportali.com` şu an HTTP 523 dönüyor; resmî kaynaklar (EPİAŞ duyuruları, BOTAŞ, TEİAŞ) erişilebilir ve ikinci aşamada eklenecek.

### 2.2 Fiyat verisi 2021'e çekildi

Ham veri artık **2021-01-01 → bugün** (24/24 ay, 0 hata, 7.7 dakika). Böylece Ocak 2022 İran gaz kesintisi ve Şubat 2022 savaş şoku kapsanıyor.

**Model eğitim penceresi kasıtlı olarak sabit bırakıldı.** İki ayrı sabit var:

| Sabit | Değer | Kontrol ettiği |
|---|---|---|
| `daily_update_pipeline.HISTORY_START` | `2021-01-01` | Ham verinin geriye kapsamı (kriz analizi için) |
| `predict_daily_pipeline.TRAINING_DATA_START` | `2023-01-01` | Fiyat modelinin eğitim penceresi |

Bu ayrım zorunluydu: `load_all_historical_data()`'nın SQL'inde **hiç tarih filtresi yoktu**, yani 2021-2022 verisi DB'ye girer girmez model onlarla da eğitilmeye başlayacaktı. 2022 azami fiyat limiti rejimiydi; canlı modeli bozardı.

### 2.3 Tavan fiyat sansürü — metodolojiyi belirleyen bulgu

Aylık maksimum fiyata değen saat oranı:

| Ay | Tavan (TL/MWh) | Tavanda geçen saat |
|---|---|---|
| 2021-07 | 617 | %40.7 |
| 2021-08 | 636 | %52.4 |
| **2022-01** (İran gaz kesintisi) | 1.345 | **%39.8** |
| **2022-02** (savaş şoku) | 1.524 | **%54.2** |
| **2022-03** | 1.745 | **%65.3** |
| 2022-09…12 | 4.800 | %8.6–25.4 |
| 2023-01 | 4.200 | %23.5 |

**İncelemek istediğimiz en büyük krizlerin tam ortasında fiyat sansürlü.** Klasik event study ile "anormal fiyat sapması" ölçülürse şok sistematik olarak olduğundan küçük görünür — fiyat fiziksel olarak yükselemiyordu.

> Not: Buradaki "tavan" aylık maksimumdan çıkarım. Yüzlerce saatin tam olarak aynı değerde durması güçlü kanıt, ama EPDK'nın yayımladığı resmî azami fiyat limiti değerleriyle teyit edilmeli.
>
> **GÜNCELLEME (17 Ağustos 2026) — açık soru KAPANDI, çıkarıma gerek kalmadı.**
>
> EPİAŞ/EPDK azami fiyat limitini **her ay duyuruyor** ve haber arşivi bu duyuruların hepsini içeriyor. Resmî seri korpustan derlendi: **`silver.price_cap_official`** — 28 yürürlük kaydı, 2021-02 → 2026-04, 67 ayın 67'si doğrulanmış, her satırın kaynak `article_id`'si kayıtlı. Artık tavan çıkarım değil, **ölçüm**.
>
> Çıkarım yöntemi 24 ayda birebir tuttu ama iki yerde sessizce yanlıştı: **(a)** tavan ay ortasında değişebiliyor (2021-10-15, 2022-05-19, 2025-04-05, 2026-04-04) — aylık maksimum düşük olanı hiç görmüyor, Ekim 2021 %8,1 yerine gerçekte %22,6; **(b)** fiyat tavana hiç değmediği aylarda maksimum tavan değil — Şubat 2021'de resmî tavan 572 TL, verideki maksimum 335 TL.
>
> **Serinin öz-testi ve kör noktası:** kayıt olmayan aylarda önceki değer devam ettiği varsayılıyor; testi "fiyat tavanı aşamaz" (48.000 saatte sıfır aşım). Kör nokta: kaçırılmış bir **düşüş** görünmez. Bu gerçekten oldu — Şubat 2023 indirimi (4.200 → 3.650, haber `52617`) ilk derlemede kaçırıldı. Eklendikten sonra 67 ayın 67'si doğrulandı. Yeni ay geldiğinde fiyatın tavana değmediği aylar için mutlaka duyuru aranmalı.
>
> Bu, planın §5'teki "elle etiket gerektirmeyen kalite testi"nin de ilk somut ürünü: haber hattı, üzerine kurulduğu **ölçümü** düzeltti. (Ham fiyat verisi hiç yanlış değildi; yanlış olan ondan türetilen vekil metrikti.)
>
> İkinci bulgu: aynı duyuru azami fiyatın **GÖP ve DGP'ye birlikte** uygulandığını yazıyor. Bu, §3.5'teki "tamamlayıcı sinyal olarak SMF kullanılabilir" maddesini çürütüyor — SMF de aynı tavanda sansürlü (her ay `smf_max = ptf_tavan`, aşan saat %0,0). Ayrıntı: `CRISIS_CASE_IRAN_2022.md` §3.

Tavan seviyesinin kendisi bir düzenleme olayı (569 TL → 4.800 TL → 2.600 TL) ve veriden doğrudan çıkarılabiliyor. Bu, haber hattını doğrulamak için **elle etiket gerektirmeyen bir kalite testi** sağlıyor: LLM o dönemin haberlerini okuyup tavan değişikliğini yakalayamıyorsa recall'ü kötüdür.

### 2.4 Haberler krizi çıkarılabilir sayılarla anlatıyor

İran gaz kesintisi penceresi (15 Oca – 10 Şub 2022, 440 haber) hedefli tarandı:

| Tarih | Başlık | Faz |
|---|---|---|
| 01-20 | İran gazına teknik arıza engeli | Başlangıç |
| 01-21 | Doğalgaz kesintisi elektrik fiyatını tavana çıkardı | Fiyat etkisi |
| 01-21 | TEİAŞ mesken harici kullanıcılara kısıntı uygulanacak | Talep kısıntısı |
| 01-28 | İran gazı sınırlı miktarda akmaya başladı | Toparlanma |
| 01-31 | Sanayiciye gaz kısıtı oranı %20'ye düşürüldü | Normalleşme |
| 02-07 | Sanayi tesislerine doğal gaz kısıtlaması kaldırılıyor | Bitiş |

Gövdelerden çıkan somut veriler: süre **10 gün**, arz **günlük kontrat miktarının 1/3'üne** düştü, kısıntı **%40 → %20**, yürürlük **31.01.2022 08:00**, aktörler NIGC/BOTAŞ/TEİAŞ.

**Bağımsız doğrulama:** 21 Ocak haberi "22 Ocak günü bütün saat dilimleri 1.345 TL/MWh" diyor. Veriden ölçüldüğünde o gün `ort_try = max_try = 1345` — yani 24 saatin tamamı tavanda. Haber ile veri birebir örtüşüyor. Bu, otomatik tutarlılık kontrolü kurulabileceğini gösteriyor.

**Uyarı:** Pencerede 440 haberin sadece ~17'si krize dair. Ham hacmin büyük kısmı kurumsal/sektörel gürültü (şirket lisansı, yurtdışı kurulum haberi, atama). Alaka filtresinin verimi %10-20 civarında beklenmeli.

---

## 3. Tasarım kararları

### 3.1 Üç katmanlı hiyerarşi — uzun krizler tek satıra sığmaz

"Rusya-Ukrayna savaşı" tek bir kriz kaydı olarak modellenemez: 2.5 yıllık bir pencere, tek tarih, analiz edilemez.

| Katman | Nedir | Analizdeki rolü |
|---|---|---|
| **Olay (atomic)** | Tek haberden çıkan, tek tarihli olgu | Event study'nin birimi; sayı burada yüksek |
| **Faz (episode)** | Ortak mekanizmalı sınırlı pencere (ör. İran kesintisi 20 Oca – 7 Şub) | Kümülatif etkinin ölçüldüğü katman |
| **Tema** | Uzun soluklu şemsiye (ör. Rusya-Ukrayna) | **Sadece gruplama anahtarı**, doğrudan etki analizi yapılmaz |

Temaya tarih aralığı atanmaz. LLM atomik olay çıkarır; fazlar zamansal yoğunlaşma + ortak mekanizmadan türetilir.

### 3.2 Kategori değil, **etki kanalı**

`Geopolitical / Gas_Supply / Regulation` gibi sınıflandırma fiyata *nasıl* geçtiğini söylemiyor. Bunun yerine:

- `fuel_cost` — gaz/kömür/petrol fiyatı, BOTAŞ tarifesi
- `supply_capacity` — santral arızası, bakım, kuraklık, ithalat kesintisi
- `demand` — hava, tatil, sanayi kısıntısı, afet
- `regulation` — tavan fiyat, piyasa kuralı, tarife
- `fx_macro` — kur şoku, enflasyon
- `renewable` — kapasite girişi, aşırı arz

Pratik faydası: modelde **zaten feature'ı olan** kanallar var (`brent_oil_lag_48`, `natural_gas_grf_lag_48`, `usd_try`, hava). Bir olay `fuel_cost` kanalındansa model onu dolaylı olarak görüyor — katkısı düşük. Asıl değer `regulation`, `supply_capacity`, `demand` kanallarında.

### 3.3 **Doz veriden, sebep haberden** — çerçevenin çekirdeği

İlk tasarımda LLM'den her olay için miktar çıkarması isteniyordu ("%40 kısıntı, 10 gün"). Bu yanlıştı, çünkü **olayların çoğu temiz sayı vermeyecek** ("Almanya Rus kömürü almayı sonlandıracak" — doz yok).

Doğrusu: dozu haberden almaya gerek yok, **kendi verimizde zaten var**.

| Ne oldu | Nerede ölçülüyor |
|---|---|
| Gaz arzı kesildi | `kgup_gas_mw` (saatlik) |
| Sanayi kısıntısı | `raw_actual_consumption_hourly` |
| Fiyat baskısı | `raw_mcp_hourly`, tavan oranı |
| Yakıt maliyeti | `natural_gas_grf_try`, `brent_oil_usd` |
| Hava / talep | `temperature_c`, yük tahmini |
| Su durumu | `hydro_water_energy_mwh` |

Haberin gerçekten eklediği üç şey:

1. **Sebep etiketi** — veride 21 Ocak'ta anormallik var, nedenini haber söylüyor.
2. **Zamanlama** — duyurunun ne zaman yapıldığı.
3. **İleriye dönük beklenti** — *en kritik olan.* 20 Ocak'ta yayınlanan "arz **10 gün süreyle** durdurulacak", 21-30 Ocak hakkında verinin 20 Ocak'ta söyleyemediği bir şey söylüyor.

**Haberin veriyi yenebileceği tek yer üçüncüsü.** Geri kalan her şeyde veri daha iyi.

Sonuç olarak LLM'den istenen sadeleşiyor:
- **Sebep etiketi (kanal)** — her haber verebilir
- **İleriye dönük duyuru mu?** — ikili
- **Varsa ileriye dönük miktar/süre** — yoksa `null`, analiz çökmüyor

### 3.4 Kontrafaktüel: 30 günlük ortalama değil, **modelin kalıntısı**

"Krizden önceki 30 günün ortalaması" kötü bir baz — hava, talep, yakıt fiyatı hiçbirini hesaba katmıyor. Ocak'ta soğuk vurduğu için fiyat zaten yükselecekti.

Elimizde fundamentallerden fiyat tahmin eden eğitilmiş bir model var. Tahmini = "bilinen fundamentaller göz önüne alındığında fiyat ne olmalıydı". Kalıntı (gerçek − tahmin) doğal anormal-fiyat ölçüsüdür ve hava/talep/yakıt etkilerinden arındırılmıştır.

**Kısıt:** Canlı model 2023'ten eğitiliyor, 2021-2022 için kontrafaktüel üretemez. Bunun için **ayrı bir analiz modeli** (2021+ eğitilmiş, sadece kontrafaktüel üretmek için, canlıya asla girmeyen) gerekir. Bu ayrım net tutulmalı.

> **GÜNCELLEME (18 Ağustos 2026) — analiz modeli kuruldu (adım 6 kapandı).**
>
> `src/crisis/analysis_model.py` + `scripts/build_analysis_model.py`. Çıktı:
> `gold.crisis_counterfactual`, 2021-01-08 → bugün, 49.064 saat. Canlı modelden
> üç bilinçli ayrım: eğitim 2021'den, **sadece `at_cap = false` saatlerde**, ve
> lag0 gerçekleşen fundamentaller (canlı model tahmin anında bunları göremez).
> Üçüncüsü "açıklanamayan" ölçümü demek, "sürpriz" değil.
>
> **Örneklem-dışı şema: bloklu OOF, ileri yönlü walk-forward DEĞİL.** 28 günlük
> bloklar, iki yanında 8 günlük tampon (168 saatlik lag'ler sızmasın). Gerekçe:
> genişleyen pencere Ocak 2022 için sadece 2021'i görür ve ağaç modeli eğitimde
> gördüğü maksimumun üstünü tahmin edemez — serinin ilk büyük şoku sistematik
> olarak "açıklanamayan" çıkardı, yöntem artefaktı olurdu.
>
> **Doğruluk:** sansürsüz saatlerde MAE $9,96, BIAS +$0,49. Kalıntının boş bandı
> p5=−$21,1 / p95=+$26,6 — bir olayın "anormal" sayılması için bu bandın dışına
> çıkması gerekiyor.
>
> **ÖLÇÜLMÜŞ KARAR 1 — fiyat türevli feature'lar dışarıda.** İki varyant koşuldu.
> `autoregressive` (canlı feature seti) İran epizodunu yutuyor: 5 Şubat'ta
> fundamental +$27,4 derken o +$4,3 diyor, pencere ortalaması +$0,5. Çünkü
> `mcp_usd_lag_24` modele dünkü şok fiyatını veriyor, model şoku "tahmin ediyor",
> kalıntı çöküyor — 10 günlük epizodun 9 günü görünmez oluyor. Bedeli sadece
> $0,79 MAE. **Olay ölçümünde `fundamental` varyant kullanılacak.**
>
> **ÖLÇÜLMÜŞ KARAR 2 — yakıt maliyeti geçmişi EKLENMEYECEK.** Ekim 2022 – Mart
> 2023 arasında kalıntı beş ay boyunca +$12,7 şişikti. Bu bir olay değil: medyan
> = ortalama (dağılımın tamamı kaymış, tepe değil), pozitif saat oranı %70 (boş
> beklenti %50), ve tüm seride kalıntı ile GRF'nin 3 aylık değişimi arasında
> korelasyon −0,43. Sebep: model gaz maliyetinin fiyata ANINDA geçtiğini
> varsayıyor; gerçekte geçiş gecikmeli (gaz %64 düştü, fiyat o hızda düşmedi).
>
> Üç düzeltme denendi, ortak 47.072 saatte:
>
> | Model | Yakıt sütunları | MAE | ort \|aylık sapma\| | Eki22–Mar23 | İran (1-10 Şub) |
> |---|---|---|---|---|---|
> | **v1** | yok | 10,29 | 4,11 | +12,74 | **+16,49** |
> | v2 | mutlak seviye (ma_30d/90d) | 10,34 | 4,29 | **+3,57** | +0,17 ✗ |
> | v3 | sadece 7 günlük | 10,28 | 4,05 | +11,17 | +16,48 |
> | v4 | birimsiz, uzun pencere | 10,57 | 4,53 | +11,15 | +18,57 |
>
> Sapmayı düzelten tek set mutlak seviye sütunları (v2) — ve düzeltmeyi **dönemi
> ezberleyerek** yapıyor: 90 günlük ortalama "anlık maliyet"ten çok "hangi
> dönemdeyiz" demek, dönemi tanıyan model o dönemde ne olduysa onu da açıklıyor.
> İran sinyali +$16,5'ten +$0,2'ye çöküyor — `autoregressive` hatasının aynısı,
> dünkü fiyat yerine son 90 günün gaz maliyeti üzerinden. v3 v1'in aynısı
> (ortalama fark $0,55; gaz günden güne çok yavaş hareket ettiği için 7 günlük
> ortalama lag0'a yeni bilgi katmıyor). v4'ün birimsiz sütunları ne sapmayı
> düzeltti ne de modeli iyileştirdi.
>
> **Sonuç: geçiş gecikmesi ile dönem etkisi bu veriyle kimliklendirilemiyor.**
> Ayırmak için modelin göremediği dışsal bilgi gerekir — BOTAŞ'ın elektrik
> üreticilerine uyguladığı tarifenin gerçek seviyesi. O da haber arşivinden
> derlenebilir (tavan serisi böyle derlendi), ama ayrı bir iş.
>
> **v1 birincil alet.** Ekim 2022 – Mart 2023 **bilinen sapmalı dönem** olarak
> işaretlendi: o dönemdeki her olay ölçümünden ~+$12 düşülmeli, yoksa olaylar
> olduğundan büyük görünür. Dört sürüm de tabloda duruyor (`model_name`).
>
> **Açık kalan:** Ocak–Şubat 2022 kalıntısı −$14 ile −$23 arası ve sebebi
> bilinmiyor. Gaz geçiş gecikmesi DEĞİL — v2'de daha da kötüleşiyor (−$27,4).
> İran vakasının tam ortasında olduğu için öncelikli.

### 3.5 Sansür altında çıktı değişkeni

Doz büyüdükçe fiyat tavana yapışıyor — doz-tepki eğrisi tam da ilgilendiğimiz üst uçta kesik.

İki rejimli çıktı:
- **Sansürsüzken:** model kalıntısı (anormal fiyat, $/MWh)
- **Sansürlüyken:** **tavanda geçen saat oranı** — fiyat yükselemediğinde bilgi seviyede değil süredeymiş

Tamamlayıcı sinyal olarak SMF (`raw_smp_hourly`, 2021'den mevcut) kullanılabilir.

---

## 4. Dashboard ne diyebilir, ne diyemez

**Diyemez:** "Benzer olay yaşandı, %40 kısıntı olacak."

**Diyebilir:** "Bu ölçekte bir arz açığı geçmişte 7 kez görüldü. Ortalama etki +A TL/MWh, %80 aralığı [B, C]. Bu olayların 3'ünde fiyat tavana dayandı. n=7, güven düşük."

Her çıktının yanında **n ve aralık** zorunlu. Nokta iddiası yok.

---

## 5. Düşürülebilir test

Tüm yapının değerli olup olmadığının tek dürüst ölçüsü:

> Olay feature'ları eklendiğinde modelin **örneklem-dışı** hatası düşüyor mu?

Test edilecek feature'lar 3.3'teki üçüncü maddeden gelir: `aktif_duyurulmus_kesinti_gun_sayisi`, `duyurulmus_kisinti_orani` gibi ileriye dönük değişkenler.

Altyapı hazır: `gold.experiment_results` tablosu ve deney runner'ı. Protokol baseline ile aynı walk-forward olmalı.

İyileşme yoksa sistem **anlatı aracıdır, öngörü aracı değildir** — ve raporda öyle yazılmalıdır.

Doz gerektirmeyen ikinci çıktı (betimsel, kendi başına değerli): *"2021'den bu yana en büyük 40 anormal fiyat gününün kaçının tanımlı bir sebebi var, hangi kanaldan?"*

---

## 6. LLM yaklaşımı ve kısıtlar

**Model:** Qwen3-8B, **Q5_K_M** kuantizasyonu (~5.8 GB). 4-bit yerine 5-bit öneriliyor çünkü Türkçe'de kuantizasyon kaybı daha belirgin. Ön filtre kademesi için Qwen3-4B.

**Donanım:** M1 MacBook Air, 16 GB RAM. Metal GPU + unified memory kullanılır (saf CPU çıkarımı değil).

**Hız:** Sınıflandırmada darboğaz prefill (okuma), generation değil.

| | Yaklaşık |
|---|---|
| Prefill | ~100-200 tok/s |
| Haber + prompt | ~1050 token → 5-10 sn |
| Prefix caching ile | ~4-6 sn/haber |
| Ön filtre sonrası ~5.000 haber | **~10 saat, bir gecelik batch** |

**Tasarım uyarlamaları:**
1. Tek kompakt JSON çağrısı (4 ayrı çağrı = 4 kat prefill)
2. Prefix caching — sistem promptu + few-shot sabit tutulur, sadece haber yeniden işlenir
3. MLX de denenmeli (Apple Silicon'da genelde llama.cpp'den hızlı)
4. Kural bazlı ön filtre (site kategorisi + anahtar kelime) hacmi 10-20 kat düşürür
5. Fansız M1 Air saatler süren yükte %30-50 yavaşlar

**Etiket güvenilirliği protokolü (atlanamaz):**
1. **Altın küme:** 200 haber elle etiketlenir (~1 günlük iş)
2. **Uyum:** LLM vs insan, Cohen's kappa. Kanal ve yön için κ > 0.6 hedef; 0.4-0.6 ise sadece filtre olarak kullan
3. **Dışsal geçerlilik:** LLM'in alaka skoru ile gerçek anormal fiyat sapması korelasyonu — etiketlemeyi insan yargısına değil **piyasa verisine** karşı test eder

**API alternatifi:** Dış API erişimi mümkünse Türkçe etiketleme kalitesi belirgin yükselir ve 3-5 binlik korpusun maliyeti küçüktür. Bakanlık ağında erişimin gerçekten kapalı olup olmadığı teyit edilmeli.

---

## 7. Metodolojik uyarılar

- **Sızıntı:** `published_at` (haberin yayın anı) ile `event_date` ayrı tutulmalı. Modele beslenirken **sadece `published_at` < tahmin anı** olan olaylar kullanılır. Geriye dönük atanmış kriz başlangıç tarihi lookahead'dir.
  - Sitenin yayınladığı "Spot elektrik fiyatı … için X TL" haberleri gün öncesi fiyatı içeriyor, ama D günü 14:00 sonrası yayınlandıkları ve tahmin D günü 04:00'te yapıldığı için `published_at` kesimi bunları zaten dışarıda bırakır. Yine de olay değiller — maliyet gerekçesiyle filtrelenmeli.
- **Anomaliden olaya gitme:** "Önce fiyat anomalilerini bul, sonra haberle eşleştir" **döngüseldir**. Doğru sıra: korpusu fiyattan bağımsız topla → olaylar anormal hareketi öngörüyor mu diye test et → anomali dedektörünü sadece **recall kontrolü** için kullan ("bu büyük hareketin karşılığında hiç olay yok, kaçırdık mı?").
- **Nedensellik zayıf:** Olaylar rastgele atanmıyor. Soğuk hava hem gaz kısıntısına hem yüksek talebe yol açar. Confounding ciddi; sonuçlar ilişkisel olarak sunulmalı.

---

## 8. Sıradaki adımlar

| # | İş | Bağımlılık |
|---|---|---|
| 1 | Etiketleme şemasını 3 İran haberi üzerinde Qwen ile test et | — |
| 2 | `bronze.news_raw` şemasını kesinleştir | 1 |
| 3 | Toplayıcıyı tam koşuya çıkar (27.419 haber, ~3-4 saat, LLM yok) | 2 (paralel yürüyebilir) |
| 4 | Kural bazlı ön filtre + alaka kademesi | 3 |
| 5 | 200 haberlik altın küme + kappa ölçümü | 4 |
| 6 | ✅ Analiz modeli (2021+, sadece kontrafaktüel için) — bkz. §3.4 güncellemesi | — |
| 7 | Etki analizi: kalıntı + tavan oranı, faz kümeleme | 5, 6 |
| 8 | Düşürülebilir test: olay feature'ları modeli iyileştiriyor mu | 7 |

**Açık sorular:**
- ~~EPDK'nın resmî azami fiyat limiti değerleri~~ — kapandı, `silver.price_cap_official`
- Ocak–Şubat 2022'nin −$23'lük kalıntısı neden negatif (analiz modeli, §3.4)
- BOTAŞ'ın elektrik üreticisi gaz tarifesi arşivden derlenebilir mi (Eki22–Mar23 sapması için)
- Dış LLM API erişimi var mı
- Resmî duyuru kaynakları (EPİAŞ/BOTAŞ/TEİAŞ) arşiv yapısı — henüz incelenmedi

---

## Ek: Şema taslağı

Medallion mimarisine oturuyor:

| Katman | Tablo | İçerik |
|---|---|---|
| Bronze | `bronze.news_raw` | url, source, `published_at`, section, title, body, body_chars, content_hash |
| Silver | `silver.news_events` | Tekilleştirilmiş atomik olaylar + kanal, yön, `effective_at`, `quantities[]`, `actors[]`, güven |
| Gold | `gold.crisis_episodes` | Olaylardan türetilmiş fazlar |
| Gold | `gold.event_impact_analysis` | Event study çıktıları (kalıntı, tavan oranı, n, aralık) |

Örnek ham kayıt:

```json
{
  "url": "https://www.enerjigunlugu.net/gazprom-...-46439h.htm",
  "source": "enerjigunlugu",
  "published_at": "2022-01-15T12:17:00+03:00",
  "section": "Doğalgaz",
  "title": "Gazprom Letonya'ya uzanan boru hattını bakıma alacak",
  "body": "Enerji Günlüğü - Rus doğal gazının Letonya'ya akışı...",
  "body_chars": 984,
  "content_hash": "547c05af5b082c7f"
}
```

`effective_at` alanı `published_at` ve `event_date`'ten ayrı tutulur: 31 Ocak haberi 09:08'de yayınlandı ama değişiklik 08:00'de yürürlüğe girdi.
