# Faz 0 — Kaynak doğrulama bulguları (2 Eylül 2026)

Amaç: olay boru hattına (K0-K4) geçmeden önce iki kaynağın gerçekten iş görüp
görmediğini ölçmek. Sondalar: `experiments/scripts/probe_ttf_tariff.py`,
`experiments/scripts/probe_gdelt.py`.

**Karar: GEÇ.** Üç kaynağın üçü de kullanılabilir, biri beklenmedik bir bulgu verdi.

---

## 1. TTF (Avrupa gaz referansı) — ✅ GEÇ

`TTF=F` / yfinance ile geliyor: 1.424 gün, 2021-01-04 → 2026-09-01. Tam dönem.

### Olay tarihleme yeteneği SEÇİCİ

| olay | TTF ±30 gün |
|---|---|
| Rusya-Ukrayna başlangıcı | ort **+63%**, zirve **+182%** |
| Gazprom akışı tamamen kesti (Eyl 2022) | −15% (zirvedeydi, fiyatlanmıştı) |
| İsrail-İran (Haz 2025) | −0,3% ort, +14% zirve |
| **İran→Türkiye gaz kesintisi (Oca 2022)** | **−22% — görünmez** |

TTF yalnızca **Avrupa gaz arz şoklarını** tarihliyor. Türkiye'ye özgü olaylar TTF'de
yok. Dolayısıyla TTF bir olay dedektörü **değil**, dışsal maliyet çıpası.

### Beklenmedik bulgu: idari vana ölçüldü

BOTAŞ elektrik gaz tarifesi USD'ye çevrilip TTF ile karşılaştırıldı (63 ay, aylık %Δ):

```
TTF_t → tarife_t+0 ay   +0.058     ← eşzamanlı: SIFIR
TTF_t → tarife_t+1 ay   +0.575     ← en güçlü
TTF_t → tarife_t+2 ay   +0.432
TTF_t → tarife_t+3 ay   +0.222

seviye korelasyonu:  +0.707 genel · +0.764 (2021-22 kriz) · +0.528 (2023-26)
```

**Tarife Avrupa gaz fiyatını izliyor ama aynı ay içinde değil — bir ay gecikmeyle.**
Eşzamanlı değişim korelasyonunun sıfır olması vananın tanımı: fiyat sinyali anında
geçmiyor, bir karar bekliyor. Kriz döneminde bağ daha sıkı (+0,76), normalleşmede
takdir payı artıyor (+0,53).

Bu, `silver.market_events`'teki "Rusya-Ukrayna" yorumuna mekanizma veriyor:
TTF Şubat 2022'de sıçradı → tarife bir ay sonra kıpırdadı → tarifenin MCP'ye geçişi
kendi gecikmesini ekledi → fiyat Temmuz'da geldi.

**Bu artık yorum değil ölçüm.** `EVENT_IMPACT_STUDY.md` §3 araştırma sorusu #2'nin
ilk sayısal cevabı.

---

## 2. GDELT DOC 2.0 — ⚠️ KOŞULLU GEÇ

### Ölçülmüş kısıtlar (iki başarısız koşudan sonra)

| kısıt | değer |
|---|---|
| Hız sınırı | **5 saniyede 1 istek.** İhlal → 429, ardından bağlantı düşer (SSL UNEXPECTED_EOF) |
| Doğru aralıkla gecikme | istek başına 3-5 sn |
| Sorgu sözdizimi | Parantez **yalnızca** OR için. `(a b)` hata, `(a OR b)` ve `a b` geçerli |
| Güvenilirlik | Ağır sorgularda (~yıl boyu timeline + filtre) **düşük** — 9 istekten 2'si döndü |
| Türkçe olmayan ASCII sorgu | `doğalgaz`, `sourcelang:turkish` düştü; `turkey electricity` döndü |

Toplu iş için DOC API uygun değil; BigQuery veya ham dosya indirme gerekir.
**Hedefli retrieval için (aday olay başına artlist) uygun.**

### Değeri: tam olarak enerjigunlugu'nun kör olduğu yerde

"İran→Türkiye gaz kesintisi, Oca-Şub 2022" sorgusu 25 makale döndürdü:

```
20220119 [Turkey ] TR  İran, teknik arıza gerekçesiyle Türkiye'ye gaz arzını 10 gün...
20220121 [Turkey ] TR  10 gün sürecek deniyordu! İran, Türkiye'ye doğal gaz akışını...
20220121 [Turkey ] TR  Türkiye'ye doğal gaz akışı yeniden başladı mı?
20220122 [Turkey ] TR  İran Türkiye'ye gaz akışını bilerek mi durdurdu?
20220127 [Turkey ] TR  İran: Doğalgazdaki sorun Türkiye kaynaklı
20220124 [Iran   ] FA  قطع گاز وارداتی از ایران باعث توقف تولید...
20220120 [Armenia] HY  Իրանից Թուրքիա բնական գազի մատակարարումը դադարեցվել է 10 օրով
```

Karşılaştır: **bizim arşivimizde bu olayın haberlerinin hiçbiri "İran" demiyor** —
"sanayiye gaz kısıtı", "elektrik molası" diyor (bkz. `MARKET_EVENTS_BRIEF.md` §5).

GDELT üç şey veriyor ki bizde yok:
1. **Olayın kendisi** — sonucu değil, sebebi ve failli
2. **Kesin başlangıç tarihi** — 19 Ocak, "teknik arıza gerekçesiyle"
3. **İLERİYE DÖNÜK SÜRE — "10 gün".** Bu tam olarak
   `labeling_schema.FIELD_ORDER`'daki `sure_gun` alanı, ve fiyat serisinde
   **olması imkânsız** olan bilgi türü. Boru hattının tahmin tarafındaki tek
   gerçek kozu bu.
4. Çok dillilik — Farsça/Ermenice/Azerice karşı taraf çerçevesi

Türkiye enerji kapsamı da var: `turkey electricity` 2025'te 29.516 makale,
348/348 gün dolu (~85/gün — enerjigunlugu'nun 13/gün'üne karşı).

### İş bölümü

| kaynak | rolü |
|---|---|
| **enerjigunlugu** | yurt içi sektör ayrıntısı, tarife/tavan duyuruları, EPDK/EPİAŞ kararları |
| **GDELT** | dışsal şokun kendisi, yayı, kesin tarihi, ileriye dönük süresi, çok dilli çerçeve |
| **TTF / Brent** | dışsal maliyet çıpası; Avrupa gaz şoklarının objektif tarihlemesi |

---

## 3. Korpus ve kümeleme ölçeği — ✅ GEÇ

```
26.054 haber · 2021-01-01 → 2026-08-16 · 2.008 dolu gün
günlük: ort 13,0 · medyan 14 · maks 33 · yıllar arası dengeli (11,8-15,1)
gövde: medyan 1.089 karakter · %90'ı 2.373 · boş gövde YOK
```

14 günlük kayan pencerede ~182 haber → ~16 bin çift. **Tam pairwise benzerlik
pencere başına bedava**; global O(n²) gereksiz.

`sklearn 1.9` içinde `HDBSCAN` ve `AgglomerativeClustering` var — K2 için ek
bağımlılık yok. Yalnız embedding modeli lazım (~500 MB, çok dilli).

Yan not: şemadaki `BODY_LIMIT = 2400` tam 90. persentile (2.373) denk geliyor.
İyi kalibre edilmiş, dokunma.

---

## 4. `CRISIS_LLM_SETUP.md`'nin disk kısıtı BAYAT

Doküman "12 GB boş disk → tek backend, tek model" diyor (17 Ağu 2026 ölçümü).
**Şu an 81 GB boş (%13 dolu).** Yerel LLM'in disk engeli yok.

Kalan kısıtlar: 16 GB RAM (8B 6-bit + KV ≈ 7,5 GB — sığar), fansız M1 termal
kısılma, ve dokümanın kendi yazdığı Türkçe kalite riski (κ < 0,4 ihtimali).

Seçim artık "API mi yerel mi" değil, **"para mı zaman mı"**:

| | API (Batch, Opus 5) | Yerel Qwen3-8B |
|---|---|---|
| maliyet | ~$133 tek seferlik (tahmin; `count_tokens` ile ölçülmeli) | $0 |
| süre | saatler | geceler |
| Türkçe kalite | risk yok | κ ölçülene kadar bilinmiyor |
| kurulum riski | yok | `outlines`/`mlx-lm` sürüm çakışması |

26k haber bir kez etiketlenir; sonrası günde ~13 haber. Tek seferlik maliyet.

---

## Faz 1'e girerken değişen tasarım

1. **GDELT toplu değil hedefli kullanılacak.** Aday olay başına artlist sorgusu,
   5 sn aralık, yeniden deneme sarmalı zorunlu.
2. **TTF K0'a giriyor** ama dedektör olarak değil, çıpa olarak.
3. **`sure_gun` / `etki_baslangici` alanları önceliklendirilmeli.** GDELT'in
   "10 gün" örneği, boru hattının tahmin değeri taşıyabilecek tek yerinin
   ileriye dönük duyurular olduğu tezini doğruluyor.
4. **Altın küme GDELT'ten de örnek almalı** — Türkçe olmayan kaynakların
   etiketlenmesi ayrı bir kalite sorusu.
