# Geçmiş Olayların Elektrik Fiyatlarına Etkisi — Çalışma Planı

**Tarih:** 21 Ağustos 2026
**Durum:** Yön belirlendi, çalışma başlamadı
**Öncesi:** `CRISIS_ANALYSIS_PLAN.md` (otomatik olay tespiti — denendi, çalışmadı, bkz. §7)

---

## 1. Ne isteniyor

Süpervizör görüşmesinde belirlenen yön:

> Önce enerji piyasasını etkileyen büyük olayları kendimiz araştıralım, fiyatlara
> nasıl etki ettiklerini inceleyelim ve bir rapor çıkaralım. Elektrik fiyatlarının
> geçmişini ve bu olayların etkisini analiz edelim.

Kullanıcının eklediği hedef: **akademik tez derinliğinde** bir çalışma. Gerekçe —
Türkiye elektrik fiyatları üzerine güncel ve kapsamlı bir çalışma görünmüyor.

**Bu, daha önce denenen işten farklı.** Öncekinde sistem olayları haberlerden
kendi buluyordu; bu çalışmada olayları biz seçiyoruz. Otomatik tespit değil,
insan tarafından yürütülen geriye dönük analiz.

---

## 2. Neyin elimizde olduğu

Bu bölüm önemli: çalışmanın değeri, kolay bulunmayan üç seriye dayanıyor.

### 2.1 Nadir olan veriler

| Varlık | Ne | Neden nadir |
|---|---|---|
| `silver.price_cap_official` | Resmî azami fiyat limiti, 29 yürürlük kaydı, 2021-02 → 2026-04 | 67 ayın 67'si doğrulanmış, her satırın kaynak haberi kayıtlı. Çoğu çalışma tavanı yok sayar ya da aylık maksimumdan çıkarır — ve o çıkarımın iki yerde sistematik olarak yanlış olduğunu gösterdik (bkz. §5.1) |
| `silver.gas_tariff_electricity` | BOTAŞ elektrik üretim gaz tarifesi, 30 kayıt, boşluksuz | Marjinal santralin **fiilen ödediği** fiyat. Literatür genelde TTF/spot gaz kullanır. Türkiye'de bu fiyat idari olarak belirleniyor ve piyasa fiyatından ayrışıyor |
| `gold.crisis_counterfactual` | Karşı-olgusal fiyat, saatlik, 2021-2026 | Sadece tavana değmemiş saatlerde eğitilip her saate tahmin üretiyor. Tavanlı piyasada doğru yaklaşım |

### 2.2 Ölçülmüş ilişkiler

**Gaz maliyeti geçişi.** Santrallerin ödediği gaz 1 $/MWh artınca elektrik fiyatı
**1,45 $/MWh** artıyor. %80 aralığı 1,15–1,75, n=30 ay, r=0,77.
Fizikle tutarlı: %55 verimli gaz santrali marjinal olsa katsayı 1,8 olurdu;
1,45 çıkması geçişin tam olmadığını gösteriyor.

*Sınırı:* ölçülebilen 30 ayın 29'unda termal pay %45'in üstünde. Yenilenebilirin
baskın olduğu rejimde (2026 baharı) geçiş çok daha az olmalı ama ayrı ölçüm
yapacak kadar örnek yok.

**Tavan değişikliğinin etkisi ölçülemiyor.** 18 tavan artışının 15'inde fiyat
yükselmiş ama büyüklük çok dağınık (−%43,9 ile +%28,9 arası). Asıl sorun
karıştırıcı: tavan genelde maliyet zaten arttığı için yükseltiliyor. Tavanın
etkisini maliyetin etkisinden ayıramıyoruz. **Rapor bu olaylar için yön söyler,
sayı vermez.**

**Hangi büyüklük yıllar arası taşınıyor.** Hidro payı taşınmıyor, termal pay
taşınıyor:

| Hidro payı | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| %40-50 | $83,6 | $68,8 | $79,8 | **$28,2** |

| Termal pay | 2024 | 2025 | 2026 |
|---|---|---|---|
| %50-60 | $70,1 | $66,1 | $65,3 |
| %60-70 | $74,8 | $71,5 | $70,6 |

Aynı hidro payı yıllara göre tamamen farklı fiyat veriyor; termal pay ise
neredeyse birebir taşınıyor. Sebep: fiyatı belirleyen son santral genelde termal.

**Sansürün büyüklüğü.** 2021-2026 arası saatlerin **%15,4'ü** tavanda geçmiş.
Mart 2022'de bu oran %65,3.

### 2.3 Mevcut rapor

`reports/tavandan_sifira.html` — 2021-2026 fiyat geçmişini **beş rejim** halinde
anlatıyor: üretim kompozisyonu, hangi büyüklüğün taşındığı, günün şeklinin
değişmesi, model performansı. Üreteci `reports/build_report.py`.

**Eksiği:** rejim bazlı kurulmuş, olay bazlı değil. "İran kesintisi fiyata şu
kadar etki etti" cümlesi hiçbir yerde geçmiyor. Bu çalışma o katmanı ekleyecek.

---

## 3. Önerilen çerçeve

"Olayları analiz ettik" bir tez katkısı değil. Malzemenin işaret ettiği çerçeve:

> **Tavanlı ve idari yakıt fiyatlı bir elektrik piyasasında fiyat oluşumu**

Türkiye üç özelliği birlikte taşıyor ve bu kombinasyon nadir:
1. Bağlayıcı bir fiyat tavanı, 29 kez değişmiş
2. Üreticiye idari olarak belirlenen gaz fiyatı, piyasa fiyatından ayrışıyor
3. Çok büyük bir hidro salınımı (%15 → %50 pay)

Üçünün de ölçülmüş serisi elimizde.

**Literatürdeki konumu (21 Ağu 2026'da doğrulandı).** Türkiye'de fiyat tavanının
etkisini ölçen tek çalışma — *Price spikes, temporary price caps…* (Energy Policy,
2022) — **1 May 2016 – 31 Tem 2018** verisiyle çalışıyor ve **etkinin istatistiksel
olarak sıfır** olduğunu buluyor. O dönemde tavan 2.000 TL/MWh'ti ve fiyat ona
neredeyse hiç değmiyordu: **bağlayıcı değildi.** Bizim dönemimizde saatlerin %15,4'ü
tavanda. Çalışmanın çerçevesi bu çelişki üzerine kuruluyor: *aynı soru, tavanın
fiilen bağladığı rejimde.* Ayrıntı ve diğer konumlandırma: `LITERATURE_REVIEW.md`.

### Araştırma soruları

1. Gözlenen fiyat değişkenliğinin ne kadarı tavan tarafından bastırılıyor?
   Bastırılmasa fiyat ne olurdu?
2. İdari gaz tarifesi, piyasa gaz fiyatından farklı mı geçiyor? (Ayrıştıkları
   dönemleri ölçtük — Ara 2021–Mar 2022 tarife piyasanın %45 altında,
   Kas 2022–Mar 2023 %28 üstünde)
3. Rejimler arası istikrarlı ilişki kuran arz değişkeni hangisi?

---

## 4. Olay listesi

Veriden çıkarıldı (en büyük aylık hareketler + rejim kırılmaları), mekanizmaya
göre gruplandı. **Süpervizörle teyit edilmeli, eklenecek olaylar olabilir.**

### Maliyet kaynaklı

| Dönem | Ne oldu | Veriden görünen |
|---|---|---|
| 2021 II. yarı | Küresel gaz fiyatı tırmanışı | Gaz $22 → $50; Tem 2021 fiyat +%28,9 |
| Eylül 2022 | Maliyet zirvesi | Gaz **$105,8** (serinin maksimumu), fiyat **$210,8** |
| 2023 I. yarı | Maliyet normalleşmesi | Gaz $105 → $40; Mar 2023 −%24,7, Haz 2023 −%27,6 |

### Arz kaynaklı

| Dönem | Ne oldu | Veriden görünen |
|---|---|---|
| Ocak 2022 | İran gaz kesintisi | Gazdan üretim 11.837 → 8.453 MW (−%28,6). Vaka çalışması hazır: `CRISIS_CASE_IRAN_2022.md` |
| **2026 baharı** | **Hidro çöküşü** | Şub → May fiyat $47,7 → **$13,0**. Hidro payı %31,6 → %50,1, termal %44,9 → %29,7. **Gaz maliyeti sabit ($32-37)** — yani tamamen arz karması olayı |
| 2026 Haz-Tem | Toparlanma | +%106,1 ve +%114,1; hidro payı geriliyor |

### Düzenleme kaynaklı

| Dönem | Ne oldu |
|---|---|
| 2021-2022 | AFL tırmanışı: 569 → 4.800 TL |
| **1 Nisan 2022** | **AUF (Azami Uzlaştırma Fiyatı) mekanizması devreye girdi** — teknoloji bazlı gelir tavanı: gaz 2.550, ithal kömür 1.800, yenilenebilir 1.700 TL/MWh. **Aynı gün AFL de 1.745 → 2.500'e çıktı**, iki müdahale ayrılamaz |
| 2023 | AFL indirimleri: 4.800 → 2.600 TL |

**AFL ≠ AUF.** AFL teklif tavanıdır ve PTF'yi keser (sansürün kaynağı). AUF gelir
tavanıdır, PTF oluştuktan sonra destekleme bedeli havuzu üzerinden işler — PTF'yi
doğrudan kesmez ama teklif davranışını değiştirir. `silver.price_cap_official`
**AFL'yi** tutuyor (doğru). AUF serisi henüz derlenmedi. Ayrıntı:
`LITERATURE_REVIEW.md` §3.

### Dışsal şoklar

| Dönem | Ne oldu | Not |
|---|---|---|
| Aralık 2021 | Kur şoku (KKM) | USD/TRY 17,78 → 11,19, üç günde |
| Şubat 2022 | Rusya-Ukrayna savaşı | Etkisi tek tarihte değil, yörüngede |
| Şubat 2023 | Deprem | Talep yıkımı; ölçülmedi |

---

## 5. Yöntem

### 5.1 Her olay için verilecekler

1. **Gerçekleşen fiyat** — dönem ortalaması, USD ve TL
2. **Beklenen fiyat** — karşı-olgusal modelin çıktısı (`crisis_cf_v5`)
3. **Fark** — ve boş bandın neresinde olduğu (p5 = −19,9 · p95 = +23,3)
4. **Mekanizmanın veriden ölçümü** — gaz kaç MW düştü, hidro payı ne oldu,
   termal pay nereye gitti, sıcaklık ne, kaç saat tavana dayandı

### 5.2 Kaçınılması gereken üç hata

**Tavanı `MAX(price)` ile çıkarma.** Ay ortası değişiklikleri kaçırır (Ekim 2021
gerçekte %22,6 iken %8,1 görünüyordu) ve fiyatın tavana hiç değmediği aylarda
tamamen yanlış sonuç verir. `silver.mcp_with_cap` kullanılmalı.

**Sansürlü dönemde etkiyi kesin sayı gibi sunma.** Tavana dayanan saatlerde fiyat
daha fazla yükselemediği için ölçülen etki **alt sınırdır**. Bunun kanıtı var:
1 Şubat 2022'de tavan 1.345 → 1.524 TL'ye çıkarıldığında fark aynı gün +$4'ten
+$16,5'e fırladı.

**Karıştırıcıyı görmezden gelme.** Ocak 2022 serinin en soğuk Ocak'ıydı (4,5 °C,
diğerleri 6,4-7,9). İran kesintisinin izole etkisi bu tasarımla ayrılamıyor.

### 5.3 Akademik standart için eksikler

Tez seviyesi hedefleniyorsa bunlar hakem sorusu olur:

| Eksik | Ne gerekiyor |
|---|---|
| Sansür modellenmiyor | **Tobit** veya sansürlü regresyon. Mevcut yaklaşım ("sansürsüzde eğit, her yere tahmin et") savunulabilir ama neden Tobit değil sorusuna cevap gerekir |
| Nedensellik iddiası yok | Ya kimliklendirme stratejisi (fark-içinde-fark, sentetik kontrol) ya da "bu betimleyicidir" şerhi. **Gerekçe artık yazılı:** İberya istisnası sentetik kontrolle ölçüldü ama orada donör havuzu vardı (diğer AB piyasaları); Türkiye enterkonneksiyon bakımından fiilen ada, karşılaştırılabilir kontrol birimi yok. Temel-değişken kontrafaktüeli bu yüzden seçildi |
| Geçiş katsayısı fazla basit | Aylık farklar üzerinde düz OLS, n=30, kontrol yok, otokorelasyon düzeltmesi yok. Gereken: kontroller (termal pay, talep), Newey-West standart hatalar, kısa/uzun vade için ARDL |
| Literatür konumlandırması yok | **Yapılmadı.** "Kapsamlı güncel çalışma yok" iddiası doğrulanmalı |

---

## 6. Sıradaki adımlar

1. ~~**Literatür taraması.**~~ **İlk tur yapıldı (21 Ağu 2026) → `LITERATURE_REVIEW.md`.**
   Sonuç: "kapsamlı güncel çalışma yok" iddiası bu haliyle **yanlış** (tahmin
   literatürü doygun, Polat & Selçuklu LightGBM+SHAP yapmış), ama düzeltilmiş hali
   daha güçlü — bkz. o dosyanın §5. Kapanmadı: YÖK tez taraması, Scopus/WoS ve iki
   paywall'lı temel kaynak eksik (§7).
2. **Olay listesini süpervizörle teyit et.** §4 veriden çıktı; alan bilgisiyle
   eklenecekler olabilir. **AUF (1 Nisan 2022) yeni eklendi.**
3. **Her olay için ölçüm tablosunu üret.** §5.1'deki dört madde. Kod hazır,
   sorgular yazılacak.
4. **Yöntem sağlamlaştırma.** En azından geçiş katsayısına kontroller ve düzgün
   standart hatalar; sansür için Tobit karşılaştırması.
5. **Yazım.**

**Kapsam uyarısı:** akademik derinlik haftalarca iş. Aşamalı gidilmeli.

---

## 7. Neden otomatik olay tespitinden vazgeçildi

Bu bölüm tekrar denenmesin diye duruyor. Ayrıntı:
`experiments/notebooks/05_crisis_analysis/04_event_attribution_attempt.ipynb`

Haberden günlük olay atfetme **üç bağımsız testte** çürüdü:

| Test | Sonuç |
|---|---|
| Kural filtresi: anormal günlerde daha çok aday haber var mı | p = 0,947 — hayır, kontrol günlerinde *daha fazla* (6,11 vs 5,15) |
| 182 haber elle etiketlendi (kör, şemaya uygun): alaka ayrışıyor mu | p = 0,741 — hayır |
| Bilinen 5 kriz penceresinde kanal yoğunluğu sıçrıyor mu | 2/5 |

**Sebebi yapısal.** Model olayın dozunu zaten görüyor (`kgup_gas_lag0`). İran
kesildiğinde gazdan üretim düşüyor, model bunu görüp yüksek fiyat tahmin ediyor.
Yani fark, olayın etkisi *çıkarıldıktan sonra* kalan şey. "Haber farkı açıklıyor
mu" sorusu aslında "olay, kendi etkisinin ötesinde bir şey öngörüyor mu"
demekti.

**Ama bu çalışma için sorun değil** — burada olayları haberden bulmuyoruz, biz
seçiyoruz. Haber arşivi (28.102 haber, tam metin aranabilir) olayları *tarihlemek
ve belgelemek* için hâlâ değerli. Nitekim tavan ve tarife serilerinin ikisi de
o arşivden derlendi.
