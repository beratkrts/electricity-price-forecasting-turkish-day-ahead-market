# Literatür Taraması — Türkiye elektrik fiyatları, fiyat tavanı, idari yakıt fiyatı

**Tarih:** 21-22 Ağustos 2026
**Amaç:** `EVENT_IMPACT_STUDY.md` §6.1 — çerçevenin katkı olup olmadığını doğrulamak
**Durum:** İki tur yapıldı (web araması → alıntı grafiği). **Kapanmadı** — bkz. §7.

---

## 1. Bu taramanın ne olduğu ve ne olmadığı

**İki tur yapıldı ve ikincisi birincisini büyük ölçüde geçersiz kıldı.**

*Birinci tur (21 Ağu):* serbest web araması. Tekrar üretilemiyordu, her sorgu
başka sonuç veriyordu ve "hangisi bakmaya değer" sorusunu cevaplamıyordu.

*İkinci tur (22 Ağu):* alıntı grafiği ile snowball — `scripts/lit_search.py`,
kaynak OpenAlex. 12 doğrulanmış tohumdan geri (kaynakçalar) ve ileri (atıf
yapanlar) iki seviye yürüdü. **Havuzun 3.581/3.973'ü graftan geldi, sadece
380'i anahtar kelimeden** — kapsamı asıl veren grafik. Her sorgu, tarihi,
sonuç sayısı ve her eleme kararı `literature/lit.db`'ye yazıldı (PRISMA);
tarama aylar sonra tekrar koşulup **farkı** gösterebilir.

**Yine de Scopus / Web of Science / Google Scholar taraması yapılmadı**, YÖK tez
taraması da yok. OpenAlex kapsamı geniştir ama Türkçe yerel yayınları eksik
görebilir. Bu yüzden aşağıdaki sonuç **güçlü ama kesin değil**; §7'deki eksikler
tez seviyesi için kapatılmalı.

Tam metin erişimi kısmen çözüldü: Polat & Selçuklu (SSRN 4894108) kullanıcı
tarafından sağlandı ve **kodu/verisiyle birlikte** incelendi (§2.1.1); Elsevier
makalelerinin çoğu hâlâ paywall arkasında ve özetleri OpenAlex/Crossref/Semantic
Scholar'ın hiçbirinde yok.

**Sonuç önden:** iddia **bu haliyle yanlış**, ama düzeltilmiş bir hali doğru ve
daha güçlü. Ayrıntı §5'te.

---

## 2. Bulunanlar

§2.0 alanın en sağlam işlerini sıralar; §2.1-2.4 dört tematik
kümeyi ayrı ayrı ele alır.

### 2.0 En sağlam Türkiye çalışmaları — grafik taramasının verdiği liste

> 22 Ağu 2026'da eklendi. `scripts/lit_search.py` ile 3.973 işlik havuzdan
> süzülen 102 kapsam içi çalışmanın (77'sinin başlığında Türkiye geçiyor)
> en güçlü 20'si tek tek incelendi. Tam liste: `literature/search_report.md`.

**Alanın ağırlık merkezi burası — ve tahmin değil.** En sağlam Türkiye
çalışmaları *yenilenebilirin fiyatı düşürmesi* (merit-order) üzerine, ağırlıkla
**Energy Policy**'de, ağırlıkla **kantil regresyonla**:

| Çalışma | Veri dönemi | Yöntem | Ne buluyor | Tavan? |
|---|---|---|---|:-:|
| **Şirin & Yılmaz 2020**, Energy Policy 144 | 2016-2019, saatlik | Kantil regresyon | GÖP'te rüzgar ve nehir hidro için anlamlı negatif merit-order; etki kantile göre değişiyor. YEKDEM ödeme mekanizmasına bağlıyor | ✖ |
| **Şirin & Yılmaz 2021**, Energy Policy | ≈2016-2019 | Kantil + sıralı lojistik | **Dengeleme piyasası (SMF)** — GÖP değil. SMF düşüyor, sistem dengesizliği değişiyor | ✖ |
| **Acar, Selçuk & Daştan 2019**, Energy Policy | **2012-2017** | Saatlik GÖP | İki kaynak aynı teşviki alıyor ama fiyat ve oynaklık etkileri farklı → teşvik ayrıştırılmalı | ✖ |
| **Gökgöz & Yücel 2024**, Utilities Policy 88 | **Oca 2019 – Ara 2022** | 9 kantil, doğrusal + doğrusal olmayan | Değişken yenilenebilir oynaklığı **artırıyor**; dağıtılabilir yenilenebilir gece hariç azaltıyor | ✖ |
| **IJEEP 2024** | 2014-2020, saatlik | Çoklu doğrusal regresyon | Merit-order kazancı **YEKDEM maliyetinden küçük** → perakende maliyet net olarak artıyor | ✖ |
| **Energy Strategy Reviews 2019** — fiyat sıçramaları | **2012-2015** | Sıçrama = ortalamadan 2σ; GÖP vs gerçek zamanlı sapma | Sapma oranlarının %60'ı ±%20 bandında; sıçramaların %56,9'u planlama/arz sorunlarından | ✖ |
| **Renewable Energy 2022** — veri frekansı | 20 Şub 2019 – 26 Mar 2021 | ML vs zaman serisi ekonometrisi, günlük + haftalık | ML ekonometriyi yeniyor; **yüksek frekans performansı artırıyor**; pandemi performansı düşürüyor | ✖ |
| **Energy Policy 2018** — ensemble | raporlanmamış | Ensemble tahmin, piyasa seviyesi açık artırma verisi | Gelişmekte olan piyasada hedging kısıtlı → tahmin risk yönetiminin ana aracı. *Çerçeve olarak değerli* | ✖ |

**Üç sonuç:**

1. **Alanın en sağlam işleri 2022'de bitiyor.** Yukarıdaki sekiz çalışmanın en
   günceli Gökgöz & Yücel 2024, verisi Aralık 2022'de kapanıyor. 2023 tavan
   indirimleri, 2025 kuraklığı ve **2026 hidro çöküşü akademik olarak hiç
   işlenmemiş.** (İncelenen 20 çalışmanın hepsinde durum böyle; kapsam içi
   102'nin tamamı tek tek okunmadı, ama yayın yıllarına bakıldığında 2023
   sonrası veri kullanan başka çalışma görünmüyor.)

2. **Sekizinde sekizi tavanı ele almıyor.** Fiyat sıçramalarını inceleyen tek
   çalışma (ESR 2019) bile **2012-2015** dönemine bakıyor — yani tavanın
   bağlayıcı olmadığı yıllara. Sıçramayı "2σ sapma" diye tanımlıyor; tavanın
   sıçramayı fiziksel olarak kestiği bir rejimde bu tanım çalışmaz.

3. **Yöntem merkezi kantil regresyon.** Alanın kabul görmüş aracı bu. Bizim
   P10/P50/P90 kantil modelimiz literatürle **aynı ailede** — bu, yöntem
   bölümünde konumlandırma avantajı. Farkımız kantili tahmin için değil,
   *sansürlü rejimde karşı-olgusal üretmek* için kullanmamız.

**Elenenler (ilk 20 içinden 6'sı):** PLoS ONE 2017 ve Electronics 2022 fiyat
değil **yük** tahmini; Energies 2022 nükleer senaryo modeli; Energies 2022 EV
şebeke etkileşimi; RSER 2011 Markowitz portföyü. Başlıkta "Türkiye + elektrik"
geçmesi fiyat oluşumu çalışması olduğu anlamına gelmiyor.

**Kapandı (22 Ağu):** Polat & Selçuklu tam metin **ve kodla** incelendi — bkz.
§2.1.1. SSRN 5472209 (2025 sürümü) hâlâ okunmadı ama 4894108 ile aynı veri seti
ve aynı kurguyu kullanıyor.

### 2.1 Türkiye günlük/saatlik fiyat tahmini — literatür KALABALIK ama SIĞ

> **Düzeltme (21 Ağu 2026, aynı gün).** Bu bölümün ilk hali "alan doygun" diyordu.
> Üç makalenin tam metni okununca bu **yanlış** çıktı. Çalışma *sayısı* çok; ama
> test setleri, değerlendirme kurgusu ve sürücü kapsamı çoğunda zayıf. Aşağıdaki
> tablo tam metinden doğrulanmış sayılarla yeniden kuruldu.

| Çalışma | Veri dönemi | **Test seti** | Sürücüler | Metrikler | Tavan? |
|---|---|---|---|---|---|
| **Akpınar (2026)**, IAREJ 10(1):021-027 — *Probabilistic forecasting…* ✅tam metin | Kas 2022 – Kas 2025, saatlik | **60 gün** (≈5 Eyl – 7 Kas 2025) | saat/gün/ay + **gerçek zamanlı üretim** (gaz, kömür, hidro, rüzgar, güneş) + fiyat lag'leri | RMSE **434,82** TL/MWh · CRPS **194,98** TL/MWh · **PICP(%80) = 0,737** | ❌ |
| **Arifoğlu & Kandemir (2022)**, MAKÜ İİBF 9(2):1433-1458 ✅tam metin | Eğitim 13.06.2016–04.10.2020 · Doğrulama 05.10.2020–28.02.2021 (3.528 s) | **28 gün** (01–28.03.2021, 672 saat) | 7 dışsal: gün/tatil, RTC(-24s), HPP(-24s), SMP(-24s), LFP, sıcaklık, rüzgar hızı | MAPE: LSTM **8,15** · MLP 8,44 · GRU 8,72 · CNN 9,27 (dönem ort. PTF 228,08 TL) | ❌ |
| **Özdemir & Yılmaz (2026)**, JISE 10(1):189-200 ✅tam metin | 15.04.2023 07:00 – 29.09.2023 05:00, 4.001 nokta | **25 nokta** (bağımsız test) + 10-kat CV | 11 değişken: RTC, kaynak bazlı üretim, teklif hacmi, eşleşme miktarı, **USD kuru** | EGPR R² 0,908 (CV) / **0,913** (test); test MAE **61,56**, RMSE 87,94 TL | ❌ |
| Yılan & Beykent (2026), CMC 86(1) — XGBoost ⚠️özet | **sadece 2023**, 8.760 saat | %20 → ≈1.752 saat (bölme biçimi doğrulanamadı) | belirtilmemiş; SHAP → gaz üretimi baskın | MAE **144,8** · RMSE 201,8 TL · R² 0,923 | ❌ |
| **Polat & Selçuklu (2024)**, SSRN 4894108 ✅tam metin **+ kod** | Oca 2015–Ara 2022 toplanmış, eksik veri yüzünden **2018-2022**'ye daraltılmış | **%20 RASTGELE** — kronolojik değil (aşağı bak) | 33 değişken: fiyat lag'leri, teklif eğrisi hacimleri, **eşleşen miktar**, kur, kaynak bazlı üretim, **BOTAŞ gaz fiyatı**, sıcaklık | LightGBM **test** (Tablo 4): R² 0,950 · MAE **5,981 $/MWh** · RMSE 11,248 · **MAPE %49,6** — *eğitim* (Tablo 3): R² 0,996 · MAE 2,265 | ❌ |
| ESWA 224 (2023) — TEDSE transformer ⚠️özet | **2017–2021** | belirtilmemiş | rejim alt-grupları (Covid) | RMSE 3,14 · R² 0,94 | ❌ (veri tavan öncesi biter) |
| Şimşek (2024) — *içeriden atıf* | 17.04.2023–16.04.2024, 8.772 s | belirtilmemiş | kaynak bazlı üretim + talep | XGBoost en iyi | ❌ |
| Demirezen & Çetin (2021) — *içeriden atıf* | 01.01.2019–10.03.2020 | %16 | işlem hacmi kritik çıkıyor | RF en iyi | ❌ |

✅ = tam metin okundu · ⚠️ = özet/ikincil kaynak, doğrulanmadı

**Ortaya çıkan tablo — dört sistematik zayıflık:**

1. **Test setleri çok küçük.** Özdemir & Yılmaz **25 nokta** ile "bağımsız test"
   diyor. Arifoğlu & Kandemir 28 gün. Akpınar 60 gün — en iyisi. Karşılaştırma:
   bizde `gold.ptf_predictions_experimental` içinde **730 günlük walk-forward**
   backfill var.
2. **Zaman serisine rastgele bölme / k-kat CV.** Özdemir & Yılmaz 10-kat CV
   kullanıyor; Yılan & Beykent 80/20 (bölme biçimi netleşmedi). Zaman serisinde
   bu **sızıntı** demek — gelecekten öğrenip geçmişi tahmin ediyor.
3. **Eşzamanlı, tahmin anında bilinmeyen özellikler.** Akpınar *gerçek zamanlı
   üretim*i (gaz/kömür/hidro/rüzgar/güneş MW) doğrudan girdi yapıyor. Gün öncesi
   tahminde bu değerler **bilinmiyor**; ancak KGÜP ya da bir ön-tahmin kullanılabilir.
   Yani o kurgu işletilebilir bir gün öncesi tahmini değil, açıklayıcı bir uyum.
   Arifoğlu & Kandemir bu konuda **doğru** davranıyor (RTC/HPP/SMP'yi -24 saat
   gecikmeli alıyor, LFP ve hava tahminini eşzamanlı — ikisi de gerçekten mevcut).
4. **Hiçbiri tavanı ele almıyor.** Sekiz çalışmanın sekizinde de azami fiyat limiti
   yok. Akpınar'ın Şekil 4'ü bunu görsel olarak ele veriyor: test penceresindeki
   fiyatlar 3.000-3.400 TL bandında **düz bir tavana** yaslanıyor ve model bandın
   üstünü tahmin etmeye çalışıyor. PICP'in %80 hedefe karşı **0,737** çıkması
   (yetersiz kapsama) bununla tutarlı.

**Sürücü kapsamı — kimse tam sete sahip değil:**

| Sürücü | Akpınar | Arifoğlu | Özdemir | Polat & S. | **Bu proje** |
|---|:-:|:-:|:-:|:-:|:-:|
| Takvim / tatil | ✔ | ✔ | ✖ | ✔ | ✔ |
| Fiyat lag/rolling | ✔ | ✖ | ✖ | ✔ | ✔ |
| Talep (gerçekleşen / tahmin) | ✖ | ✔ | ✔ | ✔ | ✔ |
| Kaynak bazlı üretim | ✔ (eşzamanlı) | kısmi (hidro) | ✔ (eşzamanlı) | ✔ | ✔ (KGÜP + ön-tahmin) |
| Hava (sıcaklık/rüzgar) | ✖ | ✔ | ✖ | ? | ✔ (26 bölge ağırlıklı) |
| Kur (USD/TRY) | ✖ | ✖ | ✔ | ✔ | ✔ |
| **Gaz maliyeti** | ✖ | ✖ | ✖ | ✔ (piyasa gazı) | ✔ (**BOTAŞ idari tarifesi**) |
| **Fiyat tavanı** | ✖ | ✖ | ✖ | ✖ | ✔ (`silver.mcp_with_cap`) |

**Bunun anlamı — ilk halinden farklı:**

- "Türkiye PTF'sini tahmin ettik / ML karşılaştırdık" hâlâ bir tez katkısı **değil**.
  Bu kadar çok çalışma varken n+1'inci karşılaştırma kimseyi ilgilendirmiyor.
- **Ama "alan doygun" demek de yanlıştı.** Alan kalabalık ve sığ. Kimse tavanı
  modellemiyor, kimse idari gaz tarifesini kullanmıyor, test setleri 25 noktadan
  60 güne kadar değişiyor.
- Dolayısıyla projenin **operasyonel kurgusu** (730 günlük walk-forward, tahmin
  anında gerçekten mevcut olan girdiler, T+1 için ön-tahmin alt modelleri, sansür
  farkındalığı) literatürdekilerin çoğundan **daha sağlam** — ve bu, tezde
  *ikincil bir yöntemsel katkı* olarak söylenebilir. Birincil katkı değil;
  birincil katkı §5'teki dört madde.
- Yön değişikliği yine de **doğru**: değerli olan tahmin performansı değil,
  tavan + idari gaz tarifesi ekseni.

### 2.1.1 En yakın komşu: tam metin, kod, tekrar üretim ve kafa kafaya kıyas

> 22 Ağu 2026. Kullanıcı makaleyi `docs/ssrn-4894108.pdf` olarak sağladı; kod ve
> veri de açık: `github.com/TheEmgame/EPF-Turkish-Day-Ahead-Market`.

**Polat & Selçuklu (2024)** bu projenin literatürdeki en yakın komşusu: Türkiye
GÖP, LightGBM, SHAP, kur ve **BOTAŞ gaz fiyatı** dahil 33 değişken, hedef
USD/MWh. Rapor edilen sonuç LightGBM ile **MAE 5,981 $/MWh, R² 0,950**.

**Bu rakam bizim rakamlarımızla karşılaştırılamaz.** Sebebi üç tanesi de kodla
doğrulanmış:

**1. Zaman serisine rastgele bölme — sızıntı.** `tuned_hyper.py`:

```python
df.drop(columns=["Date", "Hour"], inplace=True)      # zaman bilgisi atılıyor
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=35)           # shuffle=False YOK
```

`sklearn`'ün varsayılanı `shuffle=True`. Yani test saatleri geçmişin içinden
rastgele seçiliyor ve **komşu saatler eğitim setinde kalıyor**. Özellikler
arasında `MCP-24`, `MCP-168`, `MCP-672` var; SHAP sıralamasında da ilk sırada
`MCP_24` çıkıyor. Bir saatin fiyatını, hem dünkü hem bir sonraki saatteki
fiyatı görerek tahmin etmek gün öncesi tahmini değildir.

*Dayanak kodun kendisidir, eğitim/test farkı değil.* Aksine o fark büyük:
LightGBM eğitimde MAE 2,265 · MSE 10,944, testte MAE 5,981 · MSE 126,528 —
MAE 2,6 kat, MSE on kattan fazla açılıyor. Yani ayrıca ciddi bir aşırı-uyum
var; ama sızıntıyı gösteren şey bölme yönteminin ta kendisi.

**2. Aykırı değerler silinmiş.** Makale "outliers are removed based on
statistical analysis" diyor. Tavana dayanan saatler ve sıfır fiyatlar tam da
incelenmesi gereken olgu; silinince model kolay rejimde ölçülmüş oluyor.

**3. Yedi değişken açık artırmanın ÇIKTISI.** `SSOV`, `SBOV`, `PISO`, `PIBO`
teklif eğrisi hacimleri; `MO`/`MB` **eşleşen** teklif miktarları; `TV` işlem
değeri. Bunlar MCP ile **eşanlı** belirleniyor — piyasa temizlenmeden önce
bilinmiyorlar. `SSOV` SHAP sıralamasında 8. sırada. Ayrıca kaynak bazlı üretim
sütunları gerçekleşen üretim, KGÜP değil.

**Tekrar üretildi ve söküldü (22 Ağu 2026).** Yazarların kendi verisi ve kendi
hiperparametreleriyle koştuk — `literature/replicate_polat_selcuklu.py`.
Makalenin rakamı **birebir çıktı** (MAE 5,981 · R² 0,950), yani kurgu doğru
anlaşılmış.

*Önce: "test seti" nedir?* Rastgele bölmede bir dönem değil, 2018-2022 boyunca
dağılmış saatler:

- Test saatlerinin **%96,1'inin en az bir komşu saati eğitim setinde**
- **%63,8'inin her iki komşu saati de** eğitimde
- Bir test saatinin **aynı gününden ortalama 18,4/24 saat** eğitimde

Yani model, tahmin ettiği saatin hem öncesini hem sonrasını görmüş oluyor.
Sorulan soru "yarını ne kadar iyi tahmin eder" değil, "ezberlediği serideki
deliği ne kadar iyi doldurur".

*Sızıntı ne kadar şişiriyor?* Aynı yıl içinde bölersek rejim kayması devre dışı
kalıyor; kalan fark tamamen bölme yönteminden geliyor:

| Yıl | ort. MCP | rastgele MAE | kronolojik MAE | kat |
|---|--:|--:|--:|--:|
| 2018 | 47,4 | 3,45 | 4,79 | 1,4× |
| 2019 | 46,0 | 3,78 | 4,41 | 1,2× |
| 2020 | 40,1 | 2,74 | 4,38 | 1,6× |
| 2021 | 55,6 | 3,71 | **13,64** | **3,7×** |
| 2022 | 147,5 | 15,44 | 37,18 | 2,4× |

*Dürüst kurgu ne veriyor?* Geçmişle eğit, sonraki yılı tahmin et:

| Test yılı | Eğitim | ort. MCP | MAE | R² |
|---|---|--:|--:|--:|
| 2019 | 2018 | 46,0 | **7,03** | 0,529 |
| 2020 | 2018-19 | 40,1 | **8,06** | 0,160 |
| 2021 | 2018-20 | 55,6 | **10,83** | 0,260 |
| 2022 | 2018-21 | 147,5 | **60,18** | **−0,720** |

**İki sonuç.** (a) Makalenin karşılaştırılabilir rakamı 5,98 değil, durağan
yıllarda **7-11 $/MWh**. (b) **2022'de R² −0,720**: ortalamayı tahmin etmekten
kötü. Tavanlı kriz rejimi 2018-2021'den öğrenilemiyor. Sızıntı düzeltilince
modelin asıl kırılganlığı görünüyor — ve o kırılganlık tam da bu çalışmanın konusu.

**Kendi modelimizle kıyas — dikkatli yapılmalı.** `gold.ptf_predictions_daily`,
17.736 saat (12 Ağu 2024 – 23 Ağu 2026): **MAE 7,27 $/MWh · WAPE %12,25**
(ort. gerçekleşen 59,36 $). Yıl bazında:

| Yıl | Saat | Ort. gerçek | MAE | BIAS | WAPE | P10-P90 kapsama |
|---|--:|--:|--:|--:|--:|--:|
| 2024 (kısmi) | 3.408 | 71,07 | 5,57 | +1,05 | %7,8 | %74,5 |
| 2025 | 8.760 | 66,42 | 6,50 | +0,81 | %9,8 | %75,0 |
| 2026 (Ağu'a kadar) | 5.568 | 41,11 | **9,51** | +2,98 | **%23,1** | %65,9 |

Bu iki rakam **aynı bantta ama aynı deney değil**: dönemler (2019-21 vs 2024-26),
fiyat seviyeleri (MAE seviyeyle ölçeklenir) ve girdi kümeleri farklı. Seviyeden
arındırmak için WAPE'e bakılırsa bizim %12,25'e karşı onların %15,3 / %20,1 /
%19,5 çıkıyor — ama onların 2022'si de bizim 2026'mız da uç rejim.

**Kafa kafaya kıyas (22 Ağu 2026) — `literature/head_to_head.py`.**
Dönem örtüşmezliği ortak *ham* veriyle aşıldı: onların CSV'si 2018-2023, bizim
DB'miz 2021-2026, kesişim 2021-2022. Hedef serilerin **birebir aynı** olduğu
doğrulandı (17.500 ortak saat, fark 0,0000). Aynı test seti (2022, 8.760 saat),
aynı yeniden eğitim kadansı (30 gün, genişleyen pencere), aynı naive.

| Model | MAE | naive | **rMAE** |
|---|--:|--:|--:|
| Onlar — 33 değişken, **2018'den** (3 yıl fazla veri) | 27,10 | 25,90 | **1,046** |
| Onlar — 33 değişken, 2021'den (eşit geçmiş) | 27,06 | 25,90 | **1,045** |
| **Biz** — 71 özellik, 2021'den, **ön-tahminler YOK** | **24,69** | 25,90 | **0,953** |

**İki handikap kasıtlı olarak bizim aleyhimize bırakıldı:** (1)
`gold.kgup_load_pre_forecasts` 2021-2022 için boş, yani ölçülmüş en etkili
özelliklerimiz (`renewable_pressure_ratio_lag0` ve türevleri) bu koşuda NaN;
(2) onlara 3 yıl fazla eğitim verisi verildi — ki neredeyse hiç işe yaramıyor
(27,10 → 27,06).

**Sonuç:** bizim özellik kümemiz **%8,9 daha düşük MAE** veriyor ve **rMAE=1
eşiğini geçen tek taraf biz**: onlar naive baseline'ı yenemiyor (1,046), biz
ancak yeniyoruz (0,953).

**Ama bu bir ezici üstünlük değil, ve tek yıllık.** Üç şerh:

1. **2022 ikimiz için de kötü bir test yılı.** Tavan rejiminin en sert dönemi;
   `CLAUDE.md` `TRAINING_DATA_START`'ı tam da bu yüzden 2023'te tutuyor. Her iki
   model de kendi konfor alanının dışında, ikisi de naive'e yakın.
2. **Başka ortak yıl yok.** Bizim ham verimiz 2021'de başladığı için 2021'i test
   edecek eğitim penceresi kalmıyor. Tek karşılaşma bu.
3. Bizim canlı dönemimizdeki rMAE 0,756 — buradaki 0,953'ten belirgin iyi.
   Aradaki fark rejim zorluğu ve eksik ön-tahmin özellikleri.

**Savunulabilir ifade: "aynı test setinde, aynı kadansla, bizim özellik
kümemiz onlarınkinden iyi çıkıyor (rMAE 0,953 vs 1,046) — ve onların modeli
naive baseline'ı yenemiyor."** Bu, "onlardan iyiyiz"in ölçülmüş hali.
Rakam yarışından daha önemli olan §5'teki argüman ise değişmedi: yayımlanmış
Türkiye MAE/MAPE'leri kurgu farkı yüzünden ölçüt olarak kullanılamaz.

Not: son %20'lik kronolojik testte (test dönemi = 2022) MAE 63,48 çıkıyor;
eşanlı açık artırma sütunları ve gerçekleşen üretim atılınca 62,2 ve 62,9.
Yani o dönemde asıl belirleyici sızıntı değil rejim kayması — ikisini
karıştırmamak için yukarıdaki yıl-içi tablo kullanılmalı.

**Yine de değerli iki şey var:**

- **`GASP` (BOTAŞ gaz fiyatı) SHAP'ta 3. sırada** — fiyat lag'lerinden hemen
  sonra. Bizim gaz tarifesi geçişi hattımızı bağımsız olarak destekliyor;
  literatürde bu değişkeni kullanan tek çalışma bu.
- **MAPE %49,6 ile R² 0,950 aynı modelde.** Sıfıra yakın fiyatlarda payda
  çöküyor. Bu, bizim WAPE/MAPE tuzağı bulgumuzun (`METRICS.md` §2) yayımlanmış
  bir örneği — ve makale bunu sorun olarak tartışmıyor.

**Bir de tablo hatası:** Tablo 1'de MCP ortalaması 67,449 ama maksimumu 66,630
görünüyor; dört fiyat sütununda da maksimum ortalamadan küçük. Şekil 7'deki
saçılım ~265 $/MWh'a kadar gidiyor. Maksimum sütunu hatalı.

**Sonuç — iki ayrı şey, karıştırılmamalı:**

1. **Yayımlanmış rakamla kıyas yanlış olur.** Onların 5,981'i ile bizim 7,27'mizi
   yan yana koymak anlamsız: kurgular karşılaştırılabilir değil. Bu, §5'teki
   "ikincil yöntemsel katkı" argümanının somut dayanağı.
2. **Ama özellik kümeleriyle kıyas yapılabilir** — yeter ki ikisi de aynı test
   setinde, aynı kurguyla koşsun. Yukarıdaki kafa kafaya deney tam olarak bunu
   yapıyor ve sonucu bizim lehimize (rMAE 0,953 vs 1,046).

Yani "karşılaştırılamaz" olan onların *yayımladığı sayı*, modelin kendisi değil.

### 2.2 Türkiye piyasa yapısı, oynaklık, merit-order

| Çalışma | Kapsam |
|---|---|
| *Measuring the long-term impact of wind, run-of-river, solar on MCP* (Renewable Energy, 2024) | Uzun dönem merit-order |
| *Türkiye Enerji Piyasasında Yapısal Kırılmalar ve Oynaklık Modellemesi* (ULİSBUD) | ARMA(7,7)-EGARCH(1,1), **1 Tem 2015 – 31 Ara 2022** günlük ağırlıklı PTF |
| *Makroekonomik değişkenler ile enerji piyasaları* (DergiPark) | ARDL + Hacker–Hatemi-J bootstrap nedensellik, PTF bağımlı |
| *An Assessment of Electricity Markets in Turkey: Price Mechanisms, Regulations, and Methods* (Springer, 2022) | Betimleyici kitap bölümü |
| Enerji Uzmanları Derneği, *Organize Toptan Elektrik Piyasalarında Fiyat Limitleri* | **Kurumsal referans**: AFL nasıl belirleniyor — tepe santral sabit maliyet geri kazanımı + VoLL'un ≥%12,5'i, DUY dayanağı |

(Gökgöz & Yücel 2024 ve Şirin & Yılmaz'ın iki makalesi **§2.0'da** ele alındı,
burada tekrarlanmıyor. Not: Gökgöz & Yücel'in dergisi **Utilities Policy 88**'dir;
bu belgenin ilk sürümünde yanlışlıkla "Energy Strategy Reviews" yazılmıştı.)

**Ortak sınır:** hemen hepsi **2022 sonunda bitiyor**. 2023 tavan indirimleri, 2024-2025
kuraklık, **2026 hidro çöküşü** akademik literatürde yok. Sadece gri literatürde var
(Ember Türkiye Electricity Review 2025/2026, basın).

### 2.3 Fiyat tavanı ekonometrisi — Türkiye'de TEK çalışma, ve yanlış dönem

**En önemli bulgu.**

> *Price spikes, temporary price caps, and welfare effects of regulatory
> interventions on wholesale electricity markets*, **Energy Policy**, Şubat 2022
> (S0301421522000416)

Türkiye piyasası. Veri: **1 May 2016 – 31 Tem 2018**. EXIST + TCMB + MGM.
Bulgu: **geçici tavan müdahalesinin PTF üzerindeki etkisi istatistiksel olarak sıfır**;
sabahın iki saati dışında anlamlı etki yok.

Bu çalışma bizim sorumuzu soruyor ama **tamamen farklı bir rejimde**. 2016-2018'de
tavan 2.000 TL/MWh'ti ve fiyat ona neredeyse hiç değmiyordu — yani **bağlayıcı değildi**.
Bizim dönemimizde (2021-2026) saatlerin **%15,4'ü tavanda**, Mart 2022'de **%65,3**.

**Bu, çerçeveyi kuran çelişki.** Literatürdeki tek Türkiye tavan çalışması "etki yok"
diyor; bizim veri dönemimiz tavanın fiilen fiyatı kestiği dönem. Aynı soru, bağlayıcı
rejimde, yeniden sorulmalı — ve cevabın farklı çıkması bekleniyor.

Uluslararası karşılaştırmalar:
- *Navigating the crisis: Fuel price caps in the Australian NEM* (Energy Economics)
- *Revisiting the crisis: An empirical analysis of the NEM suspension* (Energy Economics, 2024)
- *The price cap regulation paradox in the electricity sector* (Energy Policy)
- *A data-analytics approach for price cap formulation* (Energy, 2025)

### 2.4 Maliyet geçişi ve kontrafaktüel yöntem

**İberya istisnası, metodolojik şablon:**

> *The effects of the Iberian exception mechanism on wholesale electricity prices
> and consumer inflation: a synthetic-controls approach*, **Applied Economics
> Letters**, 2024 (10.1080/13504851.2024.2425834)

Sentetik kontrol ile ölçüm: Tem 2022 – Haz 2023 arası spot fiyatta **~%40 düşüş**.
Bu, "idari müdahalenin toptan fiyata etkisi" sorusunun kabul görmüş cevap biçimi.

**İberya bizim vakamıza neden birebir uymuyor:** İspanya/Portekiz'de müdahale gaz
maliyetine *tavan* koyup farkı sübvanse etti ve **karşılaştırılabilir kontrol birimleri
vardı** (diğer AB piyasaları). Türkiye'de:
- Müdahale iki katmanlı (teklif tavanı + idari gaz tarifesi), tek değil
- **Kontrol birimi yok.** Türkiye piyasası enterkonneksiyon bakımından fiilen ada;
  sentetik kontrol için donör havuzu kurulamaz

Bu, `crisis_cf_v5`'in *temel değişken tabanlı kontrafaktüel* yaklaşımının neden
seçildiğinin **savunması**. Şimdiye kadar bu gerekçe yazılı değildi — hakem sorusuydu,
artık cevabı var: donör havuzu yokluğu.

Diğer geçiş literatürü:
- *Pass-through from fossil fuel market prices to electricity* (Holladay)
- *Energy price pass-through with long-term contracts* (Economics Letters, 2025) —
  geçişin gecikmeli ve eksik olduğu, bizim 1,45 katsayımızla tutarlı
- *District heating under high CO2 prices: pass-through from emission cost to
  electricity prices* (arXiv) — geçiş katsayısı tahmininin standart kurgusu
- BOTAŞ'ın rolü: **Oxford Institute for Energy Studies, Insight 113 (Nisan 2022)** —
  BOTAŞ'ın gaz ithalatındaki tekel konumu ve elektrik üretimi gaz tarifesini **aylık**
  belirlemesi. Bizim `silver.gas_tariff_electricity` serisinin kurumsal dayanağı

**Sansür / Tobit:** Elektrik fiyat tavanı sansürünü Tobit ile ele alan bir çalışma
**bulunamadı**. Bu ya gerçek bir boşluk ya da taramanın sınırı (§7). Genel censored
regression referansları (censReg, penalized Tobit) yöntemsel dayanak olarak yeterli.

---

## 3. Kritik bulgu: AFL ile AUF karıştırılmamalı

Tarama sırasında çıktı, **veri tarafını doğrudan ilgilendiriyor.**

Türkiye'de 2022'den beri **iki ayrı** tavan var ve bunlar farklı şeyler:

| | **AFL** — Azami Fiyat Limiti | **AUF** — Azami Uzlaştırma Fiyatı |
|---|---|---|
| Ne | GÖP/DGP **teklif** tavanı | Santral **gelir** tavanı |
| Dayanak | DUY + fiyat limitleri piyasa kuralları | AUM mekanizması, 1 Nisan 2022 |
| PTF'yi keser mi | **EVET** — sansürü yaratan bu | **HAYIR** — PTF oluştuktan sonra işler |
| Ayrım | Tek değer (2022-04'ten sonra kaynak bazlı teklif tavanı) | **Teknoloji bazlı**: gaz 2.550, ithal kömür 1.800, yenilenebilir 1.700 TL/MWh |
| Nasıl işler | Teklif bu değerin üstüne çıkamaz | Fark "destekleme bedeli" havuzuna girer; düşük AUF'lu santral öder, yüksek AUF'lu alır |

**Bizim `silver.price_cap_official` doğru olanı tutuyor — AFL.** 2022-04-01 satırının
notu bunu zaten belgeliyor ("kaynak bazlı teklif tavanı: gaz/ithal kömür 2,5 TL/kWh,
diğer 1,2 TL/kWh… etkin tavan 2500; veride 1200 kümesi YOK (doğrulandı)"). Yani
sansür modellemesi sağlam.

**Ama AUF çalışmada hiç yok, ve olmalı.** Üç sebeple:

1. **1 Nisan 2022 bir düzenleme olayı** ve `EVENT_IMPACT_STUDY.md` §4'ün "Düzenleme
   kaynaklı" tablosunda yok. Aynı gün AFL de 1.745 → 2.500'e çıktı; iki müdahale
   **aynı tarihte**, yani birbirinden ayrılamaz — bu, o olay için ölçüm verilirken
   söylenmesi gereken bir karıştırıcı.
2. **Teklif davranışını değiştirir.** Geliri zaten AUF ile sınırlanan santralin
   PTF'ye teklif verme güdüsü değişir. Yani AUF, PTF'yi *dolaylı* etkiler.
3. **Hakem sorusu.** "Türkiye'de tavan" diyen bir teze mutlaka "hangi tavan"
   sorulur. Ayrımı açıkça yapmayan çalışma savunulamaz.

**Yapılacak:** AUF yürürlük dönemleri ve teknoloji bazlı değerleri ayrı bir seri
olarak derlenmeli (haber arşivinde var — AA, EPDK duyuruları, uzatma kararları
bulundu). `silver.price_cap_official` **değiştirilmemeli**; yanına `silver.auf_official`
gelmeli.

---

## 4. Literatürün planı düzelttiği diğer noktalar

| Bulgu | `EVENT_IMPACT_STUDY.md`'ye etkisi |
|---|---|
| Türkiye tavan çalışması var (2016-18, "etki sıfır") | §3'e **doğrudan konumlandırma** girdi: aynı soru, bağlayıcı rejimde |
| Sentetik kontrol İberya'da standart, Türkiye'de donör havuzu yok | §5.3'teki "nedensellik iddiası yok" satırı artık **gerekçeli**: sentetik kontrol uygulanamaz, temel-değişken kontrafaktüeli tercih edildi |
| Merit-order Türkiye çalışmaları 2022'de bitiyor | 2023-2026 kapsamı **gerçek bir uzatma**; 2026 hidro çöküşü akademik olarak bakir |
| Polat & Selçuklu LightGBM+SHAP yapmış | Tahmin performansı **katkı olarak sunulmamalı**; model sadece araç |
| Geçiş literatürü gecikmeli/eksik geçiş buluyor | 1,45 katsayımız literatürle **tutarlı**, aykırı değil — bu iyi haber, ama ARDL ile kısa/uzun vade ayrımı standart beklenti |
| AFL'nin VoLL ve tepe santral maliyetine bağlı belirlenmesi | Tavan değişikliklerinin **neden** yapıldığına kurumsal cevap — §5.2'deki karıştırıcı argümanını güçlendiriyor |
| Ember 2025/2026 raporları | 2025 kuraklığı (hidro payı %16), 2026 toparlanma (Mart %36, Nisan %42) — bizim veriyle **bağımsız doğrulama** kaynağı |

---

## 5. "Kapsamlı güncel çalışma yok" iddiasının değerlendirmesi

**Bu haliyle savunulamaz.** Türkiye PTF'si üzerine 2022-2026 arasında çok sayıda
çalışma var; merit-order ve oynaklık tarafı işlenmiş, tahmin tarafı kalabalık.

**Ama "çalışma çok" ile "soru cevaplanmış" aynı şey değil.** §2.1'in tam metin
incelemesi gösteriyor ki tahmin literatürünün test setleri 25 nokta ile 60 gün
arasında, bir kısmı zaman serisine rastgele bölme/k-kat CV uyguluyor, bir kısmı
tahmin anında bilinmeyen eşzamanlı üretim verisini girdi yapıyor — ve **hiçbiri
fiyat tavanını modellemiyor.**

**Savunulabilir hali:**

> Türkiye elektrik piyasasında **bağlayıcı fiyat tavanının yarattığı sansür** ile
> **üreticiye idari olarak belirlenen gaz tarifesinin** fiyat oluşumuna etkisi,
> 2021-2026 dönemini kapsayacak biçimde ve saatlik çözünürlükte birlikte
> incelenmemiştir.

Bunu dört gözlem taşıyor:

1. Türkiye tavan ekonometrisindeki tek çalışma tavanın bağlayıcı **olmadığı** dönemi
   (2016-2018) ölçüyor ve sıfır etki buluyor.
2. Mevcut Türkiye literatürü ağırlıkla **2022'de bitiyor** — 2023 tavan indirimleri,
   2025 kuraklığı ve 2026 hidro çöküşü akademik olarak işlenmemiş.
3. **BOTAŞ elektrik üretim gaz tarifesinin** PTF'ye geçişini ölçen bir çalışma
   bulunamadı. Literatür TTF/spot gaz kullanıyor; Türkiye'de marjinal santralin
   ödediği fiyat idari ve piyasadan ayrışıyor (Ara 2021–Mar 2022 %45 altı,
   Kas 2022–Mar 2023 %28 üstü — bizim ölçümümüz).
4. Sansürlü elektrik fiyatı için Tobit/censored kurgusu uygulayan çalışma
   bulunamadı.

**Katkı iddiası bu dört maddeye dayandırılmalı, "çalışma yok"a değil.**

**İkincil (yöntemsel) katkı — ayrı tutulmalı.** Projenin değerlendirme kurgusu
(730 günlük walk-forward, tahmin anında gerçekten mevcut girdiler, T+1 için
ön-tahmin alt modelleri) Türkiye tahmin literatüründeki örneklerin çoğundan daha
sağlam. Bu bir *tez katkısı* değil ama **yöntem bölümünde savunma** olarak
kullanılır: "neden yayımlanmış MAPE rakamlarıyla karşılaştırma yapmıyoruz"
sorusunun cevabı burada. Karşılaştırılabilir değiller — §2.1 ve §2.1.1.

---

## 6. Çerçevenin durumu

`EVENT_IMPACT_STUDY.md` §3'teki çerçeve — *"Tavanlı ve idari yakıt fiyatlı bir
elektrik piyasasında fiyat oluşumu"* — **ayakta**, ama iki düzeltmeyle:

- Üç özellikten biri (büyük hidro salınımı) tek başına katkı değil; merit-order
  literatürü bunu işliyor. **Katkı, üçünün etkileşiminde**: hidro payı %50'ye
  çıktığında idari gaz tarifesinin geçişi ne oluyor? (§2.2'deki "30 ayın 29'unda
  termal pay %45 üstü" sınırı tam da bu soruyu açık bırakıyor.)
- "Tavan" ikiye ayrılmalı: **AFL** (sansür kaynağı) ve **AUF** (gelir tavanı,
  dolaylı). Ayrım yapılmadan çerçeve kurulmamalı.

---

## 7. Bu taramanın kapatmadığı eksikler

Tez seviyesi için aşağıdakiler yapılmalı — **bunlar yapılmadan §5'in sonucu kesin
değil.**

| Eksik | Ne gerekiyor |
|---|---|
| Scopus / WoS taraması yok | Atıf zinciri **yapıldı** (OpenAlex, 3.973 iş — `scripts/lit_search.py`), ama Scopus/WoS ile çapraz kontrol edilmedi. OpenAlex Türkçe yerel yayınları eksik görebilir |
| İki temel kaynak okunmadı | Energy Policy S0301421522000416 (Türkiye tavan çalışması) ve SSRN 5472209 tam metinleri. Üniversite erişimi gerekli. *SSRN 4894108 okundu — §2.1.1* |
| Elsevier özetleri erişilemiyor | Kapsam içi işlerin bir kısmının özeti OpenAlex/Crossref/Semantic Scholar'ın üçünde de yok; o işlerin kapsam kararı yalnızca grafik yakınlığına dayanıyor (`search_report.md`'de "Belirsiz" listesi) |
| YÖK tez taraması yok | Türkiye'de bu konuda **yazılmış tezler** taranmadı. En muhtemel örtüşme kaynağı burası |
| EPDK / EPİAŞ resmî raporları taranmadı | Kurumların kendi piyasa gelişim raporları — hem veri hem konumlandırma |
| Tobit + elektrik tavanı için ikinci tur arama | "Bulunamadı" sonucu tek turlu aramaya dayanıyor, zayıf |
| AUF serisi derlenmedi | §3 — haber arşivinden çıkarılabilir |

---

## Kaynaklar

**Türkiye — fiyat tahmini ve piyasa analizi** (✅ = tam metin okundu)
- ✅ Akpınar, K.N. (2026), *Probabilistic forecasting of short-term electricity prices in the Turkish day-ahead market*, **International Advanced Researches and Engineering Journal 10(1): 021-027**. DOI [10.35860/iarej.1820591](https://doi.org/10.35860/iarej.1820591) · [PDF](https://dergipark.org.tr/en/download/article-file/5406461)
- ✅ Arifoğlu, A. & Kandemir, T. (2022), *Electricity Price Forecasting in Turkish Day-Ahead Market via Deep Learning Techniques*, **MAKÜ İİBF Dergisi 9(2): 1433-1458**. DOI [10.30798/makuiibf.1097686](https://doi.org/10.30798/makuiibf.1097686) · [PDF](https://dergipark.org.tr/en/download/article-file/2349722)
- ✅ Özdemir, V. & Yılmaz, M. (2026), *Modelling Turkey's Hourly Electricity Market Clearing Prices Using Exponential Gaussian Process Regression*, **Journal of Innovative Science and Engineering 10(1): 189-200**. DOI [10.38088/jise.1738364](https://doi.org/10.38088/jise.1738364)
- ✅ Polat, Ö. & Selçuklu, S. (2024), *Explainable Forecast of Electricity Market Price Using Machine Learning Algorithms: A Case Study of the Turkish Day-Ahead Market*, **SSRN 4894108** — tam metin `docs/ssrn-4894108.pdf`, kod+veri [github.com/TheEmgame/EPF-Turkish-Day-Ahead-Market](https://github.com/TheEmgame/EPF-Turkish-Day-Ahead-Market). [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4894108)
- Polat, Ö. & Selçuklu, S. (2025), *Impact of Market Factors on Day-Ahead Electricity Prices…*, **SSRN 5472209** — 4894108'in güncel sürümü, okunmadı. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5472209)
- Gökgöz, F. & Yücel, Ö. (2024), *Merit-order of dispatchable and variable renewable energy sources in Turkey's day-ahead electricity market*, **Utilities Policy 88**, 101758 — Oca 2019–Ara 2022, 9 kantil, tavan yok. [DOI](https://doi.org/10.1016/j.jup.2024.101758)
- Şirin, S.M. & Yılmaz, B.N. (2020), *Variable renewable energy technologies in the Turkish electricity market: Quantile regression analysis of the merit-order effect*, **Energy Policy 144** — 2016-2019. [DOI](https://doi.org/10.1016/j.enpol.2020.111660)
- Şirin, S.M. & Yılmaz, B.N. (2021), *The impact of variable renewable energy technologies on electricity markets: An analysis of the Turkish balancing market*, **Energy Policy** — dengeleme piyasası (SMF). [DOI](https://doi.org/10.1016/j.enpol.2020.112093)
- Acar, B., Selçuk, O. & Daştan, S.A. (2019), *The merit order effect of wind and river type hydroelectricity generation on Turkish electricity prices*, **Energy Policy** — 2012-2017. [DOI](https://doi.org/10.1016/j.enpol.2019.07.006)
- [*Measuring the long-term impact of wind, run-of-river, solar renewable energy alternatives on market clearing prices* (Renewable Energy, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0960148124023607)
- [*Electricity price estimation using deep learning approaches: Turkish markets in normal and Covid-19 periods* (ESWA, 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0957417423005286)
- [*Electricity Price Forecasting in Turkish Day-Ahead Market via Deep Learning Techniques* (MAKÜ İİBF)](https://dergipark.org.tr/en/pub/makuiibf/issue/69071/1097686)
- [*Day-Ahead Electricity Price Forecasting Using XGBoost: Turkish Electricity Market* (CMC 86(1))](https://www.techscience.com/cmc/v86n1/64421)
- [*Market Clearing Price Prediction in the Electricity Market in Turkey Using ML* (Springer, 2025)](https://link.springer.com/chapter/10.1007/978-3-031-98304-7_33)
- [*Türkiye Enerji Piyasasında Yapısal Kırılmalar ve Oynaklık Modellemesi* (ULİSBUD)](https://dergipark.org.tr/tr/pub/ulisbud/article/1563448)
- [*Türkiye Enerji Piyasasında PTF Tahmini: Makine Öğrenimi Yöntemlerinin Karşılaştırılması* (TİSEJ)](https://tisej.com/index.php/pub/article/view/1509)
- [*Probabilistic forecasting of short-term electricity prices* (DergiPark)](https://dergipark.org.tr/en/download/article-file/5406461)

**Fiyat tavanı ve düzenleyici müdahale**
- [*Price spikes, temporary price caps, and welfare effects of regulatory interventions on wholesale electricity markets* (Energy Policy, 2022) — **Türkiye, 2016-2018**](https://www.sciencedirect.com/science/article/abs/pii/S0301421522000416)
- [*Navigating the crisis: Fuel price caps in the Australian national wholesale electricity market* (Energy Economics)](https://www.sciencedirect.com/science/article/pii/S0140988323007351)
- [*Revisiting the crisis: An empirical analysis of the NEM suspension* (Energy Economics, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0140988324006911)
- [*The price cap regulation paradox in the electricity sector* (Energy Policy)](https://www.sciencedirect.com/science/article/abs/pii/S104061901630015X)
- [*A data-analytics approach for price cap formulation of electricity markets* (Energy, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0360544225041532)

**Kontrafaktüel ve maliyet geçişi**
- [*The effects of the Iberian exception mechanism on wholesale electricity prices and consumer inflation: a synthetic-controls approach* (Applied Economics Letters, 2024)](https://www.tandfonline.com/doi/full/10.1080/13504851.2024.2425834)
- [*The Iberian Exception and Its Impact* (Columbia CGEP, 2023)](https://www.energypolicy.columbia.edu/publications/the-iberian-exception-and-its-impact/)
- [*Energy price pass-through with long-term contracts* (Economics Letters, 2025)](https://www.sciencedirect.com/science/article/pii/S0165176525006378)
- [Holladay, *Pass-Through from Fossil Fuel Market Prices to Electricity*](http://web.utk.edu/~jhollad3/PassThrough.pdf)
- [*District heating systems under high CO2 emission prices: pass-through from emission cost to electricity prices* (arXiv 1810.02109)](https://arxiv.org/pdf/1810.02109)
- [*Assessment of the impact of supply and demand shocks on sustained price increases: wholesale electricity market in Portugal* (2026)](https://www.tandfonline.com/doi/full/10.1080/15567249.2026.2652403)

**Kurumsal ve düzenleyici**
- [Enerji Uzmanları Derneği, *Organize Toptan Elektrik Piyasalarında Fiyat Limitleri*](https://www.enerjiuzmanlari.org.tr/dergi/organize-toptan-elektrik-piyasalarinda-fiyat-limitleri/)
- [EPİAŞ, *Procedures and Principles for the Determination of the Min. and Max. Price Limits in the DAM and the BPM*](https://www.epias.com.tr/wp-content/uploads/2023/07/PROCEDURES-AND-PRINCIPLES-FOR-THE-DETERMINATION-OF-THE-MIN.-AND-MAX.-PRICE-LIMITS-IN-THE-DAM-AND-THE-BPM.pdf)
- [EPİAŞ Şeffaflık, *Azami Uzlaştırma Fiyatı (AUF)*](https://seffaflik.epias.com.tr/electricity/electricity-markets/maximum-settlement-price-msp)
- [EPDK, *Azami Uzlaştırma Fiyat Mekanizması için uzatma kararı*](https://www.epdk.gov.tr/Detay/Icerik/2-12999/azami-uzlastirma-fiyat-mekanizmasi-icin-uzatma-ka)
- [Anadolu Ajansı, *EPDK azami uzlaştırma fiyat mekanizmasının uygulama süresini 6 ay uzattı*](https://www.aa.com.tr/tr/ekonomi/epdk-azami-uzlastirma-fiyat-mekanizmasinin-uygulama-suresini-6-ay-uzatti/2859391)
- [Oxford Institute for Energy Studies, *Turkey's supply-demand balance and renewal of its LTCs*, Insight 113 (2022)](https://www.oxfordenergy.org/wpcms/wp-content/uploads/2022/04/Insight-113-Turkeys-supply-demand-balance-and-renewal-of-its-LTCs.pdf)
- [PwC Türkiye, *Overview of the Turkish Electricity Market 2023*](https://www.pwc.com.tr/tr/sektorler/enerji/2024/overview-of-the-turkish-electricity-market-2023.pdf)
- [*An Assessment of Electricity Markets in Turkey: Price Mechanisms, Regulations, and Methods* (Springer, 2022)](https://link.springer.com/chapter/10.1007/978-3-031-16620-4_11)

**Gri literatür — 2025-2026 dönemi**
- [Ember, *Türkiye Electricity Review 2026*](https://ember-energy.org/app/uploads/2026/04/Turkiye-Electricity-Review-2026.pdf)
- [Ember, *Türkiye Electricity Review 2026 — Hydropower*](https://ember-energy.org/latest-insights/turkiye-electricity-review-2026/hydropower/)
- [Ember, *Türkiye Electricity Review 2025 — Renewables*](https://ember-energy.org/latest-insights/turkiye-electricity-review-2025/renewables/)
- [Daily Sabah, *Record renewable output drives Turkish power prices to historic lows*](https://www.dailysabah.com/business/energy/record-renewable-output-drives-turkish-power-prices-to-historic-lows)
- [bne IntelliNews, *Turkey hikes wholesale electricity price ceiling by 4%*](https://www.intellinews.com/turkey-hikes-wholesale-electricity-price-ceiling-by-4-283972/)
