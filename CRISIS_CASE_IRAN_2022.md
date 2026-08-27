# Vaka Çalışması — Ocak 2022 İran Gaz Kesintisi

**Tarih:** 17 Ağustos 2026
**Amaç:** Kriz & olay istihbarat sisteminin ilk uçtan uca testi. Haber hattı ile fiyat/fizik
verisi bir olayda gerçekten örtüşüyor mu, ve sansürlü rejimde etki nasıl ölçülür?
**Kapsam:** LLM kullanılmadı. Kural filtresi + SQL + elle okuma.
**Bağlam:** `CRISIS_ANALYSIS_PLAN.md` §2.4, §3.4, §3.5, §7.

---

## Sonuç

Üç katmanlı bir hikâye çıktı ve tek katmanla anlatmak yanlış olurdu:

1. **Yapısal önkoşul.** Ocak 2022'de azami fiyat limiti, gaz santralinin marjinal
   maliyetinin **%68'indeydi**. Santral yakıt maliyetini tavanda karşılayamıyordu.
   Sistem zaten sınırdaydı.
2. **Karıştırıcı.** Ocak 2022 serinin en soğuk Ocak'ı (4,5 °C; diğer yıllar 6,4-7,9 °C).
   Talep baskısı eşzamanlı.
3. **Tetikleyici.** İran kesintisi — gazdan elektrik üretimi %28,6 düştü.

Ve bir metodolojik kazanım: **tavan artık çıkarım değil, ölçüm.** Haber arşivinden resmî
azami fiyat serisi derlendi (`silver.price_cap_official`, 28 yürürlük kaydı) ve 67 ayın
67'sinde doğrulandı. Bu, planın §5'teki "elle etiket gerektirmeyen kalite testi"nin ilk
ürünü — ve beklenenin tersine, haber hattı üzerine kurulduğu **ölçümü** düzeltti.
Detay Bölüm 3'te.

---

## 1. Bağlantı

```python
from sqlalchemy import text
from db.connection import get_db_engine
engine = get_db_engine()
```

CLI: `docker exec enerji_postgres psql -U postgres -d enerji_db -c "<SQL>"`

Pencere: **id 46432 → 46892** (`bronze.news_raw`), 2022-01-15 → 2022-02-10, 440 haber.
Kural filtresi 96'sını seçti (%21,8).

---

## 2. Haber zaman çizelgesi

Kural filtresinden geçen ve elle okunarak olay olduğu doğrulanan haberler:

| Yayın | id | Bölüm | Başlık | Faz |
|---|---|---|---|---|
| 01-20 11:06 | 46514 | Doğalgaz | İran'dan gaz akışı durdu | başlangıç |
| 01-20 17:03 | 46521 | Doğalgaz | İran'daki bir arıza sanayi tesislerini etkileyecek | başlangıç |
| 01-21 14:05 | 46533 | Elektrik | Elektrik fiyatı tavana vurdu | fiyat etkisi |
| 01-21 15:54 | 46538 | Elektrik | TEİAŞ elektrik kesintileri yapılacağını duyurdu | talep kısıntısı |
| 01-24 10:36 | 46565 | Elektrik | Sanayi üretimine üç gün elektrik molası | talep kısıntısı |
| 01-25 16:28 | 46596 | Elektrik | Spot elektrik tavan fiyatı Şubat'ta üçe katladı | **mevzuat** |
| 01-27 11:09 | 46621 | Doğalgaz | Botaş'tan doğal gaz arzına ilişkin açıklama | — |
| 01-28 06:47 | 46635 | Doğalgaz | EPDK asgari doğalgaz stok miktarını belirledi | mevzuat |
| 01-28 19:21 | 46652 | Elektrik | İşte sanayiye elektrik kısıtlamasının sona ereceği tarih | normalleşme |
| 01-28 19:56 | 46653 | Doğalgaz | İran'dan sınırlı gaz akışı başladı | toparlanma |
| 01-31 09:08 | 46682 | Doğalgaz | Sanayiciye doğal gaz kısıtı azaltıldı | normalleşme |
| 02-01 08:21 | 46702 | Doğalgaz | Doğalgazdan elektrik üretme maliyeti arttı | — |
| 02-07 15:31 | 46829 | Doğalgaz | Sanayiye gaz kısıtı sonlandırılıyor | bitiş |

**`46682` planın §7'deki `effective_at` örneğini doğruluyor:** haber 31 Ocak **09:08**'de
yayınlandı, metne göre değişiklik **08:00**'de yürürlüğe girdi. `published_at` ile
`effective_at` gerçekten ayrışıyor.

**`46596` planın öngörmediği ama en değerli haber:** azami fiyat limitinin kendisi bir
düzenleme olayı ve haber resmî değeri veriyor. Bölüm 3'ün tamamı buna dayanıyor.

---

## 3. Sansür problemi — ve tavan çıkarımının doğrulanması

Bu pencerede fiyat seviyesi kullanılamaz: Ocak 2022'de saatlerin %38,6'sı tam olarak
aynı değerde. Ama "aylık maksimum = tavan" varsayımı da her ay geçerli değil.

### 3.0 Sonuç önce: tavan artık çıkarım değil, ölçüm

Bu bölümün ilk sürümü tavanı aylık maksimumdan **istatistikle çıkarıyordu.** Sonra haber
arşivi tarandı ve EPİAŞ/EPDK'nın **her ay yaptığı azami fiyat duyurusu** korpusta bulundu.
Artık resmî seri elde: `silver.price_cap_official`, 28 yürürlük kaydı, 2021-02 → 2026-04,
her satırın kaynak haberi kayıtlı.

**Çıkarım yöntemi 24 ayda birebir tuttu.** Ama iki yerde sessizce yanlıştı:

| Hata | Örnek | Etkisi |
|---|---|---|
| **Tavan ay ortasında değişebiliyor** | 2021-10-15 (718 → 1.078), 2022-05-19, 2025-04-05, 2026-04-04 | Aylık maksimum düşük olanı hiç görmüyor. Ekim 2021: %8,1 yerine gerçekte **%22,6** |
| **Fiyat tavana hiç değmediği aylarda maksimum tavan değil** | Şubat 2021: tavan 572 TL, fiyat 335'i geçmedi | O ay tavan hiç devreye girmedi, oran **%0** |

Bu iki hata düzeltildi; aşağıdaki bütün tavan oranları resmî seriyle hesaplandı
(`silver.mcp_with_cap` görünümü).

**Serinin kendi öz-testi ve kör noktası.** Seri 28 yürürlük kaydı içeriyor ama 67 ayı
kapsıyor — kayıt olmayan aylarda önceki değerin devam ettiği varsayılıyor. Bu varsayımın
testi şu: *fiyat hiçbir saatte tavanı aşamaz.* Aşıyorsa kaçırılmış bir **artış** var.
Test 48.000 saatte sıfır aşım verdi.

Ama testin kör noktası var: kaçırılmış bir **düşüş** görünmez, çünkü fiyat zaten daha
düşük kalır. Ve bu senaryo gerçekten olmuştu — Şubat 2023'te EPDK tavanı 4.200'den
**3.650'ye indirmiş** (haber `52617`, 26 Ocak 2023). Bu kaydı ilk derlemede kaçırmıştım
ve o ayın maksimumu olan 3.650'yi "piyasanın kendi zirvesi" sanmıştım. Kayıt eklendikten
sonra **67 ayın 67'si doğrulanmış** durumda: 28'i doğrudan duyurudan, 39'u fiyatın tavana
değmesiyle.

> **Yeni ay geldiğinde:** aynı kör nokta tekrar açılır. Fiyatın tavana hiç değmediği bir
> ay için mutlaka duyuru aranmalı; sadece "aşım yok" testine güvenilemez.

Bölüm 3.1 ve 3.2 çıkarım yönteminin nasıl doğrulandığını belgeliyor — yöntem artık
kullanılmıyor ama nasıl test edildiği kayıtta kalsın.

### 3.1 Bağlayıcılık testi (çıkarım yönteminin öz-testi)

```sql
WITH m AS (SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') ay, price_try
           FROM public.raw_mcp_hourly),
x AS (SELECT ay, MAX(price_try) mx, COUNT(*) n FROM m GROUP BY 1)
SELECT to_char(x.ay,'YYYY-MM') ay, ROUND(x.mx) maks,
       COUNT(*) FILTER (WHERE m.price_try = x.mx) tam_maksta,
       ROUND(100.0*COUNT(*) FILTER (WHERE m.price_try = x.mx)/x.n, 1) pay_pct
FROM m JOIN x ON x.ay = m.ay GROUP BY 1,2,x.n ORDER BY 1;
```

| Ay | Maksimum | Tam maksimumda saat | Pay |
|---|---|---|---|
| 2021-01 | 356 | 1 | %0,1 |
| 2021-02 | 335 | 1 | %0,1 |
| 2021-03 | 569 | 1 | %0,1 |
| 2021-05 | 578 | 5 | %0,7 |
| **2021-06** | 595 | 38 | **%5,3** |
| 2021-07 | 617 | 297 | %39,9 |
| 2021-08 | 636 | 389 | %52,3 |
| 2021-12 | 1.217 | 204 | %27,4 |
| **2022-01** | **1.345** | 287 | **%38,6** |
| **2022-02** | **1.524** | 361 | **%53,7** |
| 2022-03 | 1.745 | 481 | %64,7 |

Yüzlerce saatin **tam olarak** aynı ondalıkta durması piyasa sonucu olamaz. Kural:

> **≥%5 saat tam maksimumda ⇒ tavan bağlayıcı, maksimum = tavan.**
> Altındaysa maksimum sadece piyasa zirvesidir ve tavan hakkında hiçbir şey söylemez.

Bu test **artık kullanılmıyor** — resmî seri onun yerini aldı. Kayıtta kalmasının sebebi,
çıkarım yönteminin nasıl sınandığını göstermesi. Tavan Haziran 2021'de ilk kez bağlayıcı
oldu ama sürekli bağlayıcı kalmadı; testin "bağlayıcı değil" dediği aylar:

```
2021-01 … 2021-05   (5 ay — piyasa tavana hiç yaklaşmadı)
2022-06, 2022-07
2023-06, 2023-07, 2023-09
2024-01, 2024-02
2025-03, 2025-06
2026-04 … 2026-08   (5 ay — bahar fiyat çöküşü)
```

Örüntü mantıklı: tavan, marjinal maliyet ona yaklaştığında bağlıyor — yani sıkı ve pahalı
dönemlerde. Ucuz dönemlerde (yaz ayları, 2026 bahar çöküşü) fiyat tavandan çok uzakta
oluşuyor ve tavan hiç devreye girmiyor.

### 3.2 Haber ile veri karşılaştırması — iki yönlü test

Haber `46596` resmî değerleri veriyor:

> "EPİAŞ, 1-28 Şubat 2022 tarihleri için Gün Öncesi Piyasası ve Dengeleme Güç
> Piyasası'nda azami fiyatı güncelledi. Azami fiyatın megavatsaat başına **1.524 TL**
> olarak hesaplandığını duyurdu. 2021 yılı Şubat ayı için azami fiyat limiti
> megavatsaat başına **572 lira** olarak belirlenmişti."

| Ay | Haberdeki resmî tavan | Veriden çıkarılan maksimum | Bağlayıcı mı | Sonuç |
|---|---|---|---|---|
| 2022-02 | 1.524 TL | **1.524 TL** | evet (%53,7) | ✅ **birebir** |
| 2021-02 | 572 TL | 335 TL | hayır (%0,1) | ❌ çıkarım geçersiz |

**Her iki sonuç da değerli.** Bağlayıcıyken çıkarım kusursuz — planın §2.3'teki
"EPDK ile teyit edilmeli" uyarısı bu ay için karşılandı. Bağlayıcı değilken çıkarım
tamamen yanlış — Şubat 2021'de piyasa tavana hiç yaklaşmamış (335 vs 572).

### 3.2b Resmî seri — tam doğrulama

Korpustan derlenen resmî tavan serisi ile veriden çıkarılan aylık maksimum karşılaştırıldı:

| Sonuç | Ay sayısı |
|---|---|
| Birebir eşleşen | **24** |
| Fiyat tavana hiç değmediği için eşleşmeyen | 2 (2021-02, 2023-02) |
| Ay ortası değişiklik nedeniyle eksik sayılan | 4 (2021-10, 2022-05, 2023-07 + 2026-04) |

Örnek kayıtlar:

| Yürürlük | Resmî tavan | Kaynak haber |
|---|---|---|
| 2021-07-01 | 617 TL | `43403` |
| 2021-10-01 | 718 TL | `44641` |
| **2021-10-15** | **1.078 TL** | `44932` — ay ortası |
| 2022-01-01 | 1.345 TL | `46117` |
| 2022-02-01 | 1.524 TL | `46596` |
| 2022-04-01 | 2.500 TL | `47685` — kaynak bazlı teklif tavanı |
| 2023-01-01 | 4.200 TL | `52175` |
| 2026-04-04 | 4.500 TL | `67710` |

**Nisan 2022 notu:** EPDK o ay kaynak bazlı teklif tavanı koydu — gaz ve ithal kömür için
2,5 TL/kWh, diğer kaynaklar için 1,2 TL/kWh. Bu **iki ayrı takas fiyatı yaratmıyor**: PTF
saatte tek fiyat, kaynak tavanları sadece teklifleri sınırlıyor. Etkin PTF tavanı en yüksek
olan, yani 2.500. Doğrulama: Nisan 2022 fiyat dağılımında 1.200 civarında hiçbir yığılma yok
(2.500'de 94 saat var).

> **Düzeltildi:** `reports/tavandan_sifira.html` ve üreteci artık `silver.mcp_with_cap`
> üzerinden, resmî seriyle hesaplıyor. `LOW_PRICE_REGIME_ANALYSIS.md` etkilenmiyor —
> o dosyada tavan sütunu yok.

### 3.3 SMF sansürsüz bir alternatif DEĞİL

Plan §3.5 "tamamlayıcı sinyal olarak SMF kullanılabilir" diyordu. Test:

```sql
WITH t AS (SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') ay,
                  MAX(price_try) ptf_cap FROM public.raw_mcp_hourly GROUP BY 1)
SELECT to_char(t.ay,'YYYY-MM') ay, ROUND(t.ptf_cap) ptf_tavan,
       ROUND(MAX(s.system_marginal_price_try)) smf_max,
       ROUND(100.0*AVG((s.system_marginal_price_try > t.ptf_cap*1.001)::int),1) asan_pct
FROM public.raw_smp_hourly s
JOIN t ON t.ay = date_trunc('month', s.ts AT TIME ZONE 'Europe/Istanbul')
WHERE s.ts >= '2021-12-01' AND s.ts < '2022-04-01' GROUP BY 1,2 ORDER BY 1;
```

| Ay | PTF tavanı | SMF maksimumu | SMF'nin tavanı aştığı saat |
|---|---|---|---|
| 2021-12 | 1.217 | 1.217 | %0,0 |
| 2022-01 | 1.345 | 1.345 | %0,0 |
| 2022-02 | 1.524 | 1.524 | %0,0 |
| 2022-03 | 1.745 | 1.745 | %0,0 |

SMF **aynı tavanda** sansürlü. Sebebi haberde yazıyor: azami fiyat "Gün Öncesi Piyasası
**ve Dengeleme Güç Piyasası**'nda" birlikte uygulanıyor. SMF farklı bir seri ama
sansürsüz bir ölçüm sağlamıyor.

**Plan §3.5 bu maddede düzeltilmeli.** Sansürlü rejimde tek geçerli çıktı değişkeni
tavanda geçen saat oranı olarak kalıyor.

---

## 4. Doz veriden — fiziksel etki

Planın §3.3 ilkesi: miktarı haberden değil veriden ölç.

```sql
WITH t AS (SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') ay,
                  MAX(price_try) cap FROM public.raw_mcp_hourly GROUP BY 1),
saatlik AS (
  SELECT m.ts, m.price_try, m.price_usd,
         (m.price_try >= t.cap*0.999)::int tavanda,
         g.natural_gas_mw gaz,
         (g.lignite_mw+g.import_coal_mw+g.black_coal_mw+g.asphaltite_coal_mw) komur,
         c.consumption_mw tuk,
         (m.ts AT TIME ZONE 'Europe/Istanbul')::date gun
  FROM public.raw_mcp_hourly m
  JOIN t ON t.ay = date_trunc('month', m.ts AT TIME ZONE 'Europe/Istanbul')
  JOIN public.raw_actual_generation_hourly g ON g.ts = m.ts
  JOIN public.raw_actual_consumption_hourly c ON c.ts = m.ts)
SELECT CASE
    WHEN gun BETWEEN '2021-12-20' AND '2022-01-19' THEN '1 ÖNCE'
    WHEN gun BETWEEN '2022-01-20' AND '2022-01-27' THEN '2 ŞOK'
    WHEN gun BETWEEN '2022-01-28' AND '2022-02-06' THEN '3 SÜREN'
    WHEN gun BETWEEN '2022-02-07' AND '2022-02-20' THEN '4 SONRA' END faz,
  COUNT(*) saat, ROUND(AVG(price_try)) ptf_try, ROUND(AVG(price_usd),1) ptf_usd,
  ROUND(100.0*AVG(tavanda),1) tavan_pct, ROUND(AVG(gaz)) gaz_mw,
  ROUND(AVG(komur)) komur_mw, ROUND(AVG(tuk)) tuketim_mw
FROM saatlik GROUP BY 1 HAVING ... ORDER BY 1;
```

> **Not — bu sorguda düşülen tuzak:** üretim ve tüketim tabloları saatlik. Bunları
> güne indirmeden `date` üzerinden birleştirirsen 24×24 kartezyen çarpım oluşur
> (744 saat yerine 428.544 satır). Ortalamalar tesadüfen bozulmaz ama sayımlar
> anlamsızlaşır. **Her şey `ts` üzerinden birleşmeli.**

| Faz | Saat | PTF TL | PTF $ | **Tavan %** | Gaz MW | Kömür MW | Tüketim MW |
|---|---|---|---|---|---|---|---|
| 1 ÖNCE (12-20→01-19) | 744 | 1.096 | 82,6 | **27,7** | 11.837 | 14.116 | 38.833 |
| 2 ŞOK (01-20→01-27) | 192 | 1.245 | 92,6 | **69,3** | 8.945 | 14.070 | 37.826 |
| 3 SÜREN (01-28→02-06) | 240 | 1.386 | 102,5 | **84,2** | 8.453 | 14.211 | 38.063 |
| 4 SONRA (02-07→02-20) | 336 | 1.410 | 104,0 | **63,1** | 8.999 | 14.062 | 37.859 |

**Üç okuma:**

1. **Gazdan elektrik üretimi 11.837 → 8.453 MW, %28,6 düşüş.** Doz burada, haberde değil.
2. **Kömür ikame etmedi.** 14.116 → 14.211 MW, fiilen sabit. Açık kapatılmadı.
3. **Fiyat seviyesi yanıltıcı, tavan oranı değil.** TL fiyat 4. fazda en yüksek (1.410)
   ama bunun sebebi Şubat'ta tavanın 1.345'ten 1.524'e çıkması — düzenleme değişikliği,
   piyasa hareketi değil. Tavan oranı %84,2 → %63,1 ile gerçek gevşemeyi gösteriyor.

### Günlük kırılım — 22 Ocak

```
gün   | PTF  | tavanda | gaz   | kömür | tüketim
01-20 | 1131 |   13%   | 13768 | 14672 |  41362
01-21 | 1218 |   67%   |  9615 | 14416 |  41099
01-22 | 1345 |  100%   |  8011 | 14221 |  38924
01-23 | 1336 |   88%   |  8119 | 13624 |  35298
01-27 | 1345 |  100%   |  8582 | 14186 |  37055
01-28 | 1345 |  100%   |  8291 | 14302 |  37723
```

**22 Ocak: 24 saatin 24'ü de tam olarak 1.345,00 TL** (`COUNT(DISTINCT price_try) = 1`).
Haber `46533` 21 Ocak'ta "22 Ocak günü bütün saat dilimleri 1.345 TL/MWh" diyordu.
**Haber ile veri birebir örtüşüyor** — otomatik tutarlılık kontrolü kurulabilir.

---

## 5. Mevsimsellik kontrolü

Aynı takvim penceresi (20 Oca – 6 Şub) diğer yıllarda:

| Yıl | Tavan % | PTF $ | Gaz MW | Kömür MW | Tüketim MW |
|---|---|---|---|---|---|
| 2021 | 0,0 | 40,5 | 11.218 | 12.805 | 36.824 |
| **2022** | **77,5** | **98,1** | **8.672** | 14.148 | 37.958 |
| 2023 | 17,1 | 175,8 | 9.942 | 15.072 | 36.907 |
| **2024** | **2,3** | 66,5 | **5.083** | 14.310 | 39.452 |
| 2025 | 17,1 | 71,2 | 11.570 | 14.851 | 40.674 |
| 2026 | 25,9 | 64,7 | 9.548 | 14.666 | 42.411 |

2022 açık ara en yüksek tavan oranına sahip — mevsimsellik değil.

**Ama 2024 satırı mekanizmayı çürütüyor:** gaz 5.083 MW ile 2022'den %41 daha düşük,
tavan oranı ise sadece %2,3. "Gaz düştü → fiyat tavana" tek başına yanlış.

> **Bağlayıcılık şerhi (Bölüm 3.1):** Ocak ve Şubat 2024 bağlayıcı **olmayan** aylar
> (%4,2 ve %2,3). Yani o satırdaki %2,3 "tavanda geçen saat" değil, "piyasanın kendi
> zirvesinde geçen saat". Bu argümanı zayıflatmıyor, **güçlendiriyor:** 2024'te tavan
> hiç devreye girmedi, dolayısıyla gaz çok daha düşük olmasına rağmen fiyat serbestçe
> oluşabildi. Karşılaştırılabilir tek satır 2022 (bağlayıcı, %77,5).

---

## 6. Neden 2022, neden 2024 değil

Tavanın gaz santralinin marjinal maliyetine oranı (`Tavandan Sıfıra` raporundaki
gaz maliyeti hesabıyla: `GRF ÷ 9,59 MWh ÷ %53 verim`):

| Ay | Tavan TL | Kur | Tavan $ | Gaz $/1000Sm³ | Örtük KÇ maliyeti $ | **Tavan ÷ maliyet** | Sıcaklık |
|---|---|---|---|---|---|---|---|
| 2021-01 | 356 | 7,39 | 48,2 | 176 | 34,7 | **1,39** | 7,6 °C |
| **2022-01** | **1.345** | 13,56 | **99,2** | 738 | 145,1 | **0,68** | **4,5 °C** |
| 2022-02 | 1.524 | 13,64 | 111,7 | 734 | 144,3 | **0,77** | 7,0 °C |
| 2023-01 | 4.200 | 18,79 | 223,6 | 870 | 171,2 | **1,31** | 7,6 °C |
| 2024-01 | 2.700 | 30,08 | 89,8 | 386 | 75,9 | **1,18** | 7,3 °C |
| 2025-01 | 3.000 | 35,53 | 84,4 | 354 | 69,6 | **1,21** | 7,9 °C |
| 2026-01 | 3.400 | 43,22 | 78,7 | 331 | 65,0 | **1,21** | 6,4 °C |

**Ocak 2022 tek istisna: tavan, marjinal maliyetin %68'inde.** Gaz santrali yakıt
maliyetini tavanda karşılayamıyordu. Diğer bütün Ocaklarda tavan maliyetin %18-39 üstünde.

Yani sistem zaten sınırdaydı; İran kesintisi bardağı taşıran damlaydı. 2024'te gaz çok
daha düşük olmasına rağmen tavan maliyetin %18 üstünde olduğu için fiyat serbestçe
oluşabildi.

**Karıştırıcı:** Ocak 2022 aynı zamanda serinin en soğuk Ocak'ı (4,5 °C). Soğuk hem gaz
talebini artırıp kesintiyi ağırlaştırdı, hem elektrik talebini yükseltti. Planın §7'deki
"confounding ciddi" uyarısı burada somut: kesintinin izole etkisi bu tasarımla ayrılamaz.

---

## 7. Ne kanıtlandı, ne kanıtlanmadı

### Kanıtlandı

- **Haber hattı ile veri örtüşüyor.** İki bağımsız doğrulama: 22 Ocak'ın tek fiyatı,
  ve Şubat 2022 tavanının resmî değeri (1.524 TL, birebir).
- **Tavan çıkarımı geçerli** — bağlayıcılık koşuluyla. Bağlayıcılık testi ölçülebilir.
- **Sansürlü rejimde tavan oranı çalışıyor.** %27,7 → %84,2 → %63,1 hareketi olayın
  başlangıcını, sürmesini ve gevşemesini fiyat seviyesinin gösteremediği şekilde gösteriyor.
- **Doz veriden ölçülebiliyor.** Gaz üretimi %28,6 düşüşü, kömürün ikame etmediği,
  hepsi mevcut serilerden.
- **Kural filtresi bu pencerede yeterli.** 440 → 96, kritik olayların 12/12'si içeride.

### Kanıtlanmadı

- **Nedensellik yok.** Soğuk hava karıştırıcısı ayrılamadı; sonuçlar ilişkisel.
- **Kontrafaktüel yok.** Planın §3.4'ü "model kalıntısı" öneriyor ama canlı model
  2023'ten eğitiliyor. 2021+ eğitilmiş **analiz modeli** hâlâ gerekli ve bu vaka
  onun olmadan yapılabilecek tavan noktası.
- **Tek vaka.** n=1. Genellemez.

---

## 8. Planda düzeltilmesi gerekenler

| Plan maddesi | Durum |
|---|---|
| §2.3 "tavan çıkarımı EPDK ile teyit edilmeli" | ✅ Şubat 2022 için teyit edildi (haber `46596`). Ayrıca **bağlayıcılık koşulu** eklendi. |
| §3.5 "tamamlayıcı sinyal olarak SMF kullanılabilir" | ❌ **Yanlış.** SMF aynı tavanda sansürlü; azami fiyat GÖP ve DGP'ye birlikte uygulanıyor. |
| §7 "confounding ciddi" | ✅ Somutlandı: Ocak 2022 serinin en soğuk Ocak'ı. |
| §3.3 "doz veriden, sebep haberden" | ✅ Çalıştı. Gaz %28,6 düşüşü veriden, sebep haberden. |
| §3.4 "kontrafaktüel = model kalıntısı" | ⏸ Analiz modeli olmadan test edilemedi. |

## 9. Sıradaki

1. **`LOW_PRICE_REGIME_ANALYSIS.md` ve `reports/tavandan_sifira.html`'e bağlayıcılık
   şerhi düş** — tavan oranı sütunu %5 altı aylarda yorumlanmamalı.
2. **Analiz modeli** (2021+ eğitilmiş, sadece kontrafaktüel için, canlıya asla girmeyen).
   Bu vakanın eksik kalan tek parçası.
3. **Aynı protokolü ikinci bir olaya uygula** — Şubat 2022 savaş şoku doğal aday,
   korpus ve filtre hazır.
