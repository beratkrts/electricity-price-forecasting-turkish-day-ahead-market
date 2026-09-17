# Olay Evreni — dondurulmuş tasarım (v1)

**Tarih:** 3 Eylül 2026
**Statü:** DONDURULDU. Ölçüm bu listeyle yapılır. Değişiklik = v2, gerekçeli.
**Bağlam:** `OLAY_ANALIZI_V2_PLAN.md` §3. Süpervizör zamanı yok — liste ve kurallar
tek elden (AI) yazıldı; kriterler tekrarlanabilir olsun diye açık bırakıldı.

---

## 1. "Olay" tanımı

Bir **olay** şu üç şartı birden sağlar:

1. **Dışsal** — başlangıç zamanı Türkiye elektrik fiyatından veya EPİAŞ/EPDK/BOTAŞ
   politikasından kaynaklanmaz. (Savaş, boru hattı kesintisi, LNG tesisi patlaması,
   deprem, küresel gaz/petrol hareketi, aşırı hava, kur şoku.)
2. **Tarihlenebilir** — başlangıcı ≥2 bağımsız kaynakla bir güne / dar pencereye
   oturuyor. (Diffüz epizotlar ayrı işaretlenir.)
3. **Bir temele bağlı** — hipotezlenen etkilenen değişken veride izlenebilir:
   TTF, Brent, gazdan üretim (MW), yük, sıcaklık, USD/TRY.

**Olay DEĞİLDİR:** tavan (AFL) değişiklikleri, BOTAŞ tarife basamakları, AUF, YEKDEM,
piyasa kuralı değişiklikleri. Bunlar **içsel politika tepkisi** — modelde *kontrol*
olarak yer alır (§4).

---

## 2. Seçim kanalları

Aday evreni üç kanaldan toplandı, sonra §3 kurallarıyla elendi:

| Kanal | Yöntem | Bu turda kullanılan |
|---|---|---|
| **K1 — sabit şok-tipi listesi** | Literatür + alan bilgisiyle önceden yazılmış tipoloji | Avrupa gaz arz şokları, Türkiye'ye fiziksel gaz kesintisi, Orta Doğu petrol/jeopolitik, büyük kur şoku, aşırı hava, büyük talep yıkımı |
| **K2 — yukarı-akış temelinde büyük hareket** | TTF aylık \|Δ%\|>25; gazdan üretim 7g-ort haftalık Δ<−18%; USD/TRY 5g Δ>8%; sıcaklık anomalisi \|>6°C\| 5+ gün | TTF: 11 ay işaretlendi (2021-09…2026-03). Gaz MW: 2023 sonrası gürültülü (yenilenebilir salınım), tek başına dedektör değil. Kur: 3 epizot. Sıcaklık: ~17 spell |
| **K3 — haber dikkat sıçraması** | A endeksi (henüz yok) — Faz 1'de bu kanal eklenecek, liste v2'ye güncellenir | — (bu turda yok) |

`bronze.news_raw` (enerjigunlugu, 28.102) her aday için tarihleme kontrolünde
kullanıldı. **Kör noktaları ölçüldü:** kur şoku ("KKM"/"kur korumalı" → ~0 sonuç,
enerji sitesi finans haberi tutmuyor) ve Türkiye-özgü gaz kesintileri (İran olayı
"İran" adıyla değil sonuçlarıyla geçiyor). Bu ikisi için GDELT hedefli retrieval
(Faz 0'da doğrulandı) veya ikinci kaynak gerekir.

---

## 3. Tutma / düşürme kuralları

Her aday için sırayla:

1. Dışsal mı? → hayır ise **kontrole taşı**, listeden çıkar.
2. Başlangıç ≥2 kaynakla tarihlenebiliyor mu? → hayır ise **"diffüz"** işaretle,
   nokta-olay ölçümü yapılmaz, sadece epizot penceresi.
3. Etkilenen temel veride hareket etti mi? → hayır ise **reddet** (§6'ya gerekçeyle).
4. Pencere başka bir olayla tamamen örtüşüyor mu? → evet ise **birleşik epizot**,
   ayrı ölçülmez, ama iç bileşenler not edilir.
5. Sansür durumu: olay penceresinde tavan payı >%40 ise ölçüm **alt sınır** olarak
   işaretlenir (kalıntı ve toplam etki eksik tahmin).

**Katman ataması:**
- **T1** — temiz doğal deney: dışsal, tarihlenebilir, temel hareketi net, örtüşme az.
  Tam ayrıştırma (CF₁/CF₂ + plasebo) uygulanır.
- **T2** — kısmen mekanik / ağır confound: ölçülür ama sonuç şerhli, "üst/alt sınır".
- **T3** — anlatı / kontrol: kendi başına zayıf sinyal; bir T1 olayının kontrolü
  olarak değerli (ör. transmisyonun olmadığı benzer şok).

---

## 4. Kontroller (modelde yer alır, olay listesinde değil)

- **AFL (tavan)** — `silver.price_cap_official`, 29 kayıt. Rejim değişkeni + sansır limiti.
- **BOTAŞ elektrik gaz tarifesi** — `silver.gas_tariff_electricity`, 30 kayıt.
  C zincirinin çıktısı; B'de kontrol.
- **AUF** — 1 Nis 2022'den itibaren teknoloji bazlı gelir tavanı. Seri derlenecek
  (Faz 0). Post-2022-04 dönemde teklif davranışı kontrolü.
- **Mevsim / takvim / sıcaklık** — canlı feature seti.
- **Termal pay rejimi** — geçiş katsayısının rejim etkileşimi.
- **YEKDEM maliyet öngörüsü** — varsa; yenilenebilir destekleme yükü.

---

## 5. Olay evreni (dondurulmuş liste)

Başlangıç tarihleri haber arşivi + kamuya açık kronolojiden. "Temel" = hipotezlenen
etkilenen değişken. "Tavan payı" = o pencerede `at_cap` ortalaması (kabaca, mevcut
kataloğun ölçümünden; final ölçüm Faz 2'de).

### 5.1 Küresel gaz arz şokları (kanal: TTF → BOTAŞ tarife → SRMC → MCP)

| # | Olay | Başlangıç | Temel | Tavan | Katman | Not |
|---|---|---|---|---|---|---|
| G1 | 2021 Avrupa gaz tırmanışı | ~Eyl 2021 (diffüz) | TTF (+49/+37/+40% Eyl-Ara) | bağlıyor | T2 | Diffüz onset: düşük depo + düşük Rus akışı + Asya LNG rekabeti. Epizot penceresi, nokta değil |
| G2 | **Rusya-Ukrayna işgali** | 24 Şub 2022 | TTF (+59% Mar) | bağlıyor (~%19-64) | **T1** | **Merkez vaka.** Faz 0: TTF ±30g +%63/zirve +%182 |
| G3 | Freeport LNG patlaması | 8 Haz 2022 | TTF (+60% Tem) | bağlıyor (~%9) | T1 | ABD LNG ihracat kapasitesi −%20; tesis Kas 2022'ye dek kapalı. Temiz dışsal, tekil tarih |
| G4 | Nord Stream akış durması + sabotaj | ~2 Eyl 2022 (sabotaj 26 Eyl) | TTF | bağlıyor (~%9) | T1 | Faz 0: TTF −%15 (zirvede, fiyatlanmış). **"Fiyatlanmış şok" testi** — beklenti: mekanik kanal ≈ 0 |
| G5 | 2022-23 Avrupa gaz normalleşmesi | ~Eki 2022 (diffüz) | TTF (−33/−45% Eki-Oca) | bağlıyor→gevşek | T2 | Ilıman kış + dolu depo + talep yıkımı. Aşağı yönlü geçiş testi (yapışkanlık) |
| G6 | **2026 Hürmüz / Rus-Türk hat tehdidi** | ~25 Şub 2026 (Hürmüz kapatma 4 Mar; "hatlar saldırı altında" 19 Mar) | TTF (+63% Mar), Brent (80$) | **gevşek (~%3)** | **T1** | **Üçüncü jeopolitik epizot, düşük-fiyat rejiminde.** G2'nin farklı rejimdeki kontrolü. Hidro toparlanmasıyla örtüşüyor (§5.6 H2) — dikkat |

### 5.2 Türkiye'ye fiziksel gaz kesintisi

| # | Olay | Başlangıç | Temel | Tavan | Katman | Not |
|---|---|---|---|---|---|---|
| S1 | **İran → Türkiye gaz kesintisi** | 19 Oca 2022 (~10 gün) | Gazdan üretim (11.837→8.453 MW, −%28,6) | **bağlıyor (%73)** | **T1** (alt sınır) | Vaka hazır: `CRISIS_CASE_IRAN_2022.md`. En soğuk Ocak (4,5°C) confound — sıcaklık kontrolü şart. Haberde "İran" değil "sanayiye gaz kısıtı / elektrik molası" |

### 5.3 Orta Doğu petrol / jeopolitik (kanal: Brent; gaza zayıf geçiş)

| # | Olay | Başlangıç | Temel | Tavan | Katman | Not |
|---|---|---|---|---|---|---|
| O1 | **ABD-İsrail-İran çatışması** | 13 Haz 2025 (~2 hafta) | Brent (>78$) | **gevşek (%8)** | **T3** | **G2'nin kontrolü.** Faz 0: TTF tepkisiz (−%0,3), BOTAŞ tarifesi kıpırdamadı. Beklenti: hem mekanik hem kalıntı kanal ≈ 0. "Anlatı var, transmisyon yok" |
| O2 | 2026 ABD-İran petrol savaşı | ~2 Mar 2026 | Brent | gevşek | T3 | G6 ile aynı pencere/kaynak; ayrı ölçülmez, G6'nın petrol bileşeni |

### 5.4 Kur şokları (kanal: USD/TRY; hedef USD olduğu için kısmen mekanik)

| # | Olay | Başlangıç | Temel | Tavan | Katman | Not |
|---|---|---|---|---|---|---|
| K1 | Kasım-Aralık 2021 lira çöküşü + KKM | ~18 Kas 2021 (KKM 20 Ara) | USD/TRY (11→18→11) | bağlıyor | **T2** (çift sayım riski) | Hedef USD/MWh → TL değer kaybı zaten mekanik olarak içeriliyor. **Ayrı bölüm:** kur şoku TL bazlı fiyatta ne yaptı, USD bazlıda ne kaldı. Haber arşivi kör → GDELT/ikincil kaynak |

### 5.5 Talep yıkımı

| # | Olay | Başlangıç | Temel | Tavan | Katman | Not |
|---|---|---|---|---|---|---|
| D1 | **Kahramanmaraş depremi** | 6 Şub 2023 | Yük (bölgesel tüketim düşüşü) | **gevşek (%15)** | **T1** | **En temiz doğal deney.** 3 il + 9 ilçe gaz akışı durdu, bölge kesintileri. Şubat 2023 soğuk spell (−7,6°C anomali) confound. Mevcut katalog: kalıntı ≈ 0 (fiyat düşüşü tamamen yükle açıklanıyor) — V2 bunu ayrıştırmayla doğrulayacak |

### 5.6 Hidrolojik rejim (kanal: hidro üretim payı)

| # | Olay | Başlangıç | Temel | Tavan | Katman | Not |
|---|---|---|---|---|---|---|
| H1 | 2026 hidro bolluğu / fiyat çöküşü | ~1 Şub 2026 (yavaş onset) | Hidro payı (%31,6→%50,1), termal pay (%44,9→%29,7) | **gevşek→çok gevşek (%3)** | **T1** (yavaş onset) | Sürücü yağış/kar erimesi = dışsal. **G2'nin karşı-kutbu:** tavanın bağlamadığı, fiyatın tamamen arzdan oluştuğu rejim. `04_zero_price_crisis/01_low_price_regime_analysis.ipynb` |
| H2 | 2026 hidro normalleşmesi / toparlanma | ~1 Haz 2026 | Hidro payı geriliyor | çok gevşek | T2 | H1'in simetrik karşılığı. G6 ile örtüşüyor — jeopolitik ve hidro bileşenleri ayrılmalı |

### 5.7 Aşırı hava (çoğu KONTROL — sıcaklık zaten feature)

Sıcaklık model girdisi olduğu için aşırı hava spelli'leri normalde kontroldür. Ancak
bir T1 olayıyla örtüşenler **confound olarak** açıkça işaretlenir:
- Oca 2022 soğuk (−6,2°C anomali) → **S1 İran ile örtüşüyor**
- Mar 2022 soğuk (−8,2°C) → G2 penceresinde
- Şub 2023 soğuk (−7,6°C) → **D1 deprem ile örtüşüyor**
- Haz 2024 sıcak (+6,5°C) → mevcut katalogda "yaz zirvesi", olay değil

Bağımsız bir "aşırı hava olayı" ancak sıcaklık feature'ının yakalayamadığı bir etki
(ör. şebeke kısıtı, ani rampa) hipotezi varsa listeye eklenir. **Bu turda yok.**

---

## 6. Reddedilenler ve gerekçe

| Aday | Kanal | Gerekçe |
|---|---|---|
| TANAP gaz teslimat azalması (Mar 2021) | K1/haber | Küçük, planlı, gazdan üretimde iz yok |
| Mavi Akım bakımları (May 2021, May 2022) | haber | Planlı bakım — dışsal değil, öngörülü; MW etkisi mevsim içinde |
| CBRT başkan değişimi (Mar 2021) | K2 kur | USD/TRY +%9 ama dönem kapsam öncesi etkisi zayıf, elektrik fiyatı verisi seyrek; K1'e göre küçük |
| Haziran 2023 seçim sonrası kur | K2 kur | Politika kaynaklı (içsel-ish), kademeli; ayrıca 2023 geri sarımıyla tam örtüşük |
| Norveç gaz grevi (Tem 2022) | haber | Hükümet 1 günde müdahale etti, TTF'de kalıcı iz yok |
| 2021 Süveyş tıkanması (Mar 2021) | K1 | Konteyner krizi; LNG/gaz akışına ölçülebilir etki yok, dönem başı |
| Groningen kapanışı (2023) | haber | Kademeli, yıllardır bilinen; şok değil |

**Kapsam sınırı:** K3 (haber dikkat sıçraması) kanalı Faz 1'de eklenince bu liste
gözden geçirilir. Şu an K1+K2 ağırlıklı — yani "bilinen büyük şoklar + veriden büyük
hareketler". Dikkat endeksinin çıkaracağı sürpriz adaylar v2'ye girer.

---

## 7. Ölçüm önceliği

1. **G2 + O1 kontrastı** (2022 savaşı vs 2025 İran) — merkez sonuç, minimum şart.
2. **G6** (2026 Hürmüz) — üçüncü nokta, düşük-fiyat rejiminde jeopolitik.
3. **S1 + D1** — temiz doğal deneyler, ayrıştırma yönteminin doğrulaması.
4. **H1** — tavanın bağlamadığı rejim, C'nin "varyans bastırma" sonucunun karşı-örneği.
5. **G3, G4** — LNG/Nord Stream, "fiyatlanmış şok" ve tekil-tarih testleri.
6. G1, G5, K1, H2 — epizot / şerhli, rapor gövdesinde ikincil.

---

## 8. Açık kalemler

- **G6'nın haber tarihlemesi** taze arşivde eksik (bronze.news_raw 16 Ağu'da bitiyor;
  G6 Şubat-Mart 2026 → aslında kapsamda, kontrol edilecek). ✅ düzeltme: G6 kapsamda,
  başlığlar mevcut ("İran Hürmüz'ü kapattı" 2026-03-04 arşivde var).
- **K1 kur şoku** için ikinci kaynak / GDELT şart.
- **H1/H2 ↔ G6 örtüşmesi** — Şubat-Temmuz 2026'da hidro rejimi + jeopolitik + petrol
  aynı anda. Ayrıştırma bu pencerede en zor; CF₁'de hangi temelin sabitleneceği
  dikkatli seçilmeli.
- **AUF serisi** derlenmedi — D1 ve sonrası için kontrol eksik.
