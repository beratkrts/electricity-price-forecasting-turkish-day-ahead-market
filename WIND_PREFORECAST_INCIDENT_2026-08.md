# Rüzgar Ön-Tahmin Arızası — 28-30 Ağustos 2026

**Tespit:** Kullanıcı, canlı fiyat tahmininin "3 gündür patladığını" bildirdi (29 Ağustos 2026).
**Kapsam:** Teşhis + bu repoda düzeltme testi. Canlıya (`../enerji_fiyat_tahmini`) henüz
taşınmadı.

> **GÜNCELLEME (29 Ağu 2026) — kök sebep §4'te yazılandan farklı.** Aşağıdaki §6, hipotezi
> walk-forward testle sınadı. Sonuç: sorun **lag özelliklerinin fazla önemi değil**, ileri
> rüzgar hızının modele **hiç ulaşmaması** (OpenMeteo forecast API üretimde çağrılmıyor,
> `daily_update.log`'da 0 çağrı). Doğru meteo verilince mevcut lag 1-14 modeli zaten iyi;
> lag azaltmak BIAS'ı KÖTÜLEŞTİRİYOR. Düzeltme: Ağu 28-30 fiyat fazla-tahmini +$11.9 → +$6.8 (sızıntısız, T→T+1 proxy'li).
> Detay + defter: `experiments/notebooks/07_lago_protocol/07_wind_preforecast_incident.ipynb`.

---

## 1. Semptom

`gold.ptf_predictions_daily` içindeki günlük ortalama mutlak fiyat hatası ($/MWh):

| Gün | Ort. \|hata\| | Not |
|---|---|---|
| 25 Ağu | 7,11 | |
| 26 Ağu | 9,10 | |
| 27 Ağu | 5,52 | haftanın en iyi günü |
| **28 Ağu** | **15,81** | |
| **29 Ağu** | **11,83** | |
| 30 Ağu (kısmi) | 15,07 | |

En büyük tekil hatalar, gerçek fiyatın çöktüğü ama modelin yüksek tahmin ettiği saatlerde:
örn. 28 Ağu 12:00 (gerçek $31,23, tahmin $70,30, hata +$39,07), 30 Ağu 07:00 (gerçek $5,41,
tahmin $28,07, hata +$22,66). Sistematik yön: **model neredeyse hep fazla tahmin ediyor**
(fiyatın gerçekte düştüğü saatlerde).

---

## 2. İlk soru: canlı pipeline'da mı hata var?

**Yöntem:** `gold.kgup_load_pre_forecasts`'taki (immutable) pre-forecast değerleriyle,
canlının kullandığı BİREBİR AYNI hiperparametrelerle (`quantile, alpha=0.50, n_estimators=300,
learning_rate=0.03, max_depth=8, num_leaves=63, min_child_samples=10, random_state=42`),
son 7 günü walk-forward olarak DB'ye yazmadan yeniden ürettim (`replay_recent_days.py`,
bu commit'e ekli — bkz. `experiments/scripts/`).

**Sonuç:** Replay ile canlı tahmin arasındaki günlük ortalama fark **$0,08–0,78/MWh** —
LightGBM'in kendi küçük stokastikliği kadar. **Canlı pipeline'ın çalışma mantığında
(feature inşası, eğitim, inference) hata yok.** Sorun girdi verisinin kalitesinde.

---

## 3. Girdi tarafı: rüzgar ön-tahmini büyük ve büyüyen bir açık veriyor

`predicted_wind_lag0` (ön-tahminci çıktısı) vs `raw_kgup_hourly.wind_mw` (gerçekleşen):

| Gün | Ort. rüzgar hatası (tahmin − gerçek) |
|---|---|
| 25 Ağu | -2.975 MW |
| 26 Ağu | -1.552 MW |
| 27 Ağu | -3.812 MW |
| 28 Ağu | -5.854 MW |
| 29 Ağu | -6.263 MW (bazı saatler -6.064 MW) |

`predicted_wind_lag0` bu süre boyunca neredeyse sabit (~2.300–2.900 MW) kalırken gerçek
rüzgar üretimi 7.000–9.700 MW'a tırmandı. Fiyat hatası ile rüzgar hatası korelasyonu
**-0,46** (rüzgar ne kadar az tahmin edilirse fiyat o kadar fazla abartılıyor — mantıklı:
düşük rüzgar tahmini → düşük `renewable_pressure_ratio_lag0` → model arz baskısını az
görüyor → fiyatı yüksek tahmin ediyor, gerçekte fazladan rüzgar fiyatı çöktürüyor).

**İkincil katkı — yük ön-tahmini:** `predicted_load_lag0` hatası da 7 gün boyunca
sistematik (günlük ortalama -2.400 ile +2.056 MW arası, işareti gün gün değişiyor,
korelasyon 0,23). Özellikle 24 Ağustos öğle saatleri ve 26 Ağustos sabahında rüzgar
hatası küçükken bile fiyat hatası büyüktü — o saatlerde yük hatası -4.000/-5.267 MW gibi
uç değerlere ulaşıyordu. **Güneş ön-tahmini temiz** (korelasyon ~0).

**Sonuç: rüzgar başlıca ve en büyük sebep, ama tek sebep değil.**

---

## 4. Rüzgar ön-tahmincisinin kök sebebi (23 Ağustos'takinden FARKLI)

23 Ağustos'taki arıza "veri hiç gelmiyor" idi (arşiv API geleceği veremiyor, NaN →
sessizce 14 günlük gecikmeye düşme). O zaten düzeltildi (`fetch_openmeteo_wind_forecast`,
commit `86774c94`). Bu sefer:

- **Rüzgar hızı verisi doğru geliyor mu?** Evet — hem arşiv (`raw_wind_history_hourly`,
  son günlerde 20-33 aralığında, gerçekten yüksek) hem de canlı forecast fonksiyonu
  (test edildi, 30-31 Ağustos için de yüksek hız dönüyor) çalışıyor.
- **Model fiziksel olarak yüksek MW çıkaramıyor mu (extrapolation tavanı)?** Hayır —
  2023'ten beri veri 12.291 MW'a kadar rüzgar görmüş (en son 13-14 Ağustos'ta
  11.500+ MW), model bu aralığı eğitimde zaten görmüş.

**Asıl sebep — özellik önemi ölçümü** (`train_and_predict_pre_forecasters`'daki
`lgb_wind`, 25 Ağustos'a kadarki veriyle yeniden eğitildi):

| Özellik grubu | Önem payı |
|---|---|
| `kgup_wind_lag_1d...14d` (geçmiş 14 günün gerçek rüzgarı) | **%57,5** |
| `wind_power_{şehir}` + `wind_{şehir}` (ileriye bakan rüzgar hızı) | %27,8 |
| `hour`, `month` | %14,7 |

`kgup_wind_lag_1d` tek başına en güçlü özellik. Model, **geleceğe bakan meteoroloji
sinyalinden çok yakın geçmişteki rüzgar seviyesine yaslanıyor** — normal/durağan rejimde
işe yarayan bu tasarım, **sürekli çok günlü bir tırmanışta** (25 Ağustos'tan beri
devam eden) modeli geçmiş-ortalamaya çapalıyor; düzelme, tırmanışın hızından yavaş
kaldığı için açık kapanmıyor, büyüyor.

---

## 5. Düzeltme yönleri (henüz uygulanmadı — tartışılacak)

- 14 günlük gecikme sayısını azaltmak (ör. sadece 1-3 gün) — otokorelasyon avantajının
  çoğu zaten `lag_1d`'de, geri kalanı gürültü + aşırı ağırlık katıyor olabilir.
- Gecikme özelliklerinin ağırlığını sınırlamak (ör. `monotone_constraints`,
  `feature_fraction`, ya da ayrı bir "sapma düzeltmesi" modeli: önce fizik-tabanlı
  `wind_power` ile tahmin et, sonra lag'lerle sadece REZİDÜEL düzelt).
  düzeltme).
- Rejim değişimi flag'i eklemek (ör. son 24-48 saatteki wind_speed trendi) — modelin
  "bu bir ramp, geçmişe güvenme" sinyalini görmesi için.
- Alternatif: lag özelliklerini tamamen çıkarıp sadece meteorolojik girdiyle test etmek,
  hata büyür mü küçülür mü ölçmek (bu, lag'lerin normal rejimde gerçekten katkısı olup
  olmadığını da netleştirir).

**Karar/deney bu repoda yapılacak; canlıya taşınmadan önce en az bu 3 günlük pencerede
(ve tercihen daha geniş bir rejim-değişimi örnekleminde) geriye dönük test edilecek.**

---

## 6. Düzeltme testi (29 Ağu 2026) — §4 hipotezi çürüdü

Defter: `experiments/notebooks/07_lago_protocol/07_wind_preforecast_incident.ipynb`
Scriptler: `experiments/scripts/wind_preforecast_lag_test.py`, `fix_preforecast_recent.py`,
`run_wf_lightgbm.py --preforecast-override`

### 6.1 Kök sebep: ileri rüzgar hızı üretimde modele hiç ulaşmıyor

- Canlı `predicted_wind_lag0` **25 Ağustos'a kadar** gerçeği makul takip ediyor; o günden
  sonra **düz ~2300-2500 MW**'a çöküyor (gerçek rüzgar 6000-8700 MW'a çıkarken).
- `daily_update.log`: `api.open-meteo.com/v1/forecast` çağrısı = **0**, `archive-api` = 63.
  Arşiv API yarını veremez → hedef saatlerin rüzgar hızı NaN → `train_and_predict_pre_forecasters`
  `kgup_wind_lag_1d..14d`'ye düşüyor → rampa-öncesi seviyede (~2400 MW) takılıyor.
- Bu, 23 Ağustos arızasıyla **aynı sınıf** ("rüzgar hızı modele gelmiyor"), farklı yüzey.

### 6.2 Lag azaltmak yanlış yön

Ağu 25-29 walk-forward, meteo = OpenMeteo arşiv reanaliz ("iyi meteo elde olsaydı"):

| varyant | BIAS (MW) | MAE (MW) |
|---|---|---|
| **base** (mevcut: hour+month + wind_power×3 + wind×3 + lag 1-14) | **≈ −400 … −600** | ≈ 900 |
| lag3 / lag1 / nolag (lag azalt) | −950 … −1000 | 1150-1210 |
| **meteoNaN** (lag'ler baş başa, canlı senaryo) | **≈ −2900 … −4000** | 3000+ |
| ff05 (feature_fraction 0.5 + bagging) | ≈ −390 | ≈ 880 |

`meteoNaN` canlı arızayı yeniden üretiyor. `base` doğru meteo ile zaten iyi. Lag azaltmak
BIAS'ı **kötüleştiriyor** — otokorelasyon sinyali gerçek katkı sağlıyor, sorun onun tek
başına kalması.

### 6.3 Düzeltme sonucu — arıza penceresi 28-29-30 Ağustos

`train_and_predict_pre_forecasters` **değiştirilmeden**, sadece meteo girdisi arşiv+forecast
birleşimiyle dolu → walk-forward (`preforecast_recent_fixed.csv`). 30 Ağustos için gerçek KGÜP
kırılımı henüz yayınlanmadı → o gün canlı pipeline'ın **T→T+1 proxy'si** (`run_wf_lightgbm.py
--proxy-missing`: dünün gerçekleşenini yarına saat-eşli kopyala) ile koşuldu, tıpkı canlının
yaptığı gibi.

Rüzgar ön-tahmin kalitesi (Ağu 28-29; 30 Ağu gerçek rüzgar da henüz yok):

| | WAPE | BIAS |
|---|---|---|
| bozuk (canlı) | **71.9%** | −5901 MW |
| düzeltilmiş | **15.0%** | −1136 MW |

Fiyat modeline etki, **Ağu 28-30** (WF-LightGBM, canlı `lgb_lag0_v2` konfigü **+ T→T+1 proxy**
`--proxy-from 2026-08-25`, tek değişen = ön-tahmin):

| model | modelin dediği ort $ | MAE | BIAS | WAPE |
|---|---|---|---|---|
| canlı `ptf_predictions_daily` | $63.3 | $13.8 | +$11.9 | 26.8% |
| WF-LGBM, bozuk ön-tahmin (canlıyı üretir) | $64.6 | $14.6 | +$13.2 | 28.4% |
| WF-LGBM, **düzeltilmiş rüzgar** | **$58.2** | **$12.2** | **+$6.8** | **23.7%** |

_(gerçekleşen ort. fiyat Ağu 28-30 = $51.5; rMAE(naive-24h) canlı 1.04 → düzeltilmiş 0.92)_

Gün gün — modelin tahmin ettiği ort. fiyat (gerçek):

| gün | gerçek $ | canlı der | rüzgar düzeltilince der |
|---|---|---|---|
| 28 Ağu | 60.9 | $75 (+$14) | **$69 (+$8)** |
| 29 Ağu | 53.8 | $64 (+$10) | **$57 (+$3)** |
| 30 Ağu | 39.6 | $51 (+$12) | **$48 (+$9)** |

**Rüzgar hızı girdisinin kaynağı:** 29-30 Ağu = OpenMeteo **forecast API** (gerçek D-1 tahmini),
25-28 Ağu = arşiv reanaliz (hafif iyimser). İlk koşumda fiyat modelinin hedef-gün yük/KGÜP'ü
gerçekleşendi (~$1 daha düşük bias); yukarıdaki tablo T→T+1 proxy'li, sızıntısız sürüm.

- Sistematik **+$12-14 fazla-tahmin** → +$6.8'e (29 Ağu tek başına +$10 → +$3) neredeyse yarıya
  iniyor. Canlı model 28-30 Ağustos'ta naive-24h'den bile kötüydü (rMAE ≈ 1.04).
- Kalan fazla-tahmin (30 Ağu +$9: fiyat $40'a çökmüş, model $48 diyor) **rüzgar değil** —
  model düşük-fiyat rejimini göremiyor: çöküş körlüğü / feature→fiyat eşleşmesi
  (`LOW_PRICE_REGIME_ANALYSIS.md`, Faz C.5.2).

### 6.4 Güncellenmiş düzeltme listesi (canlıya taşınacak, öncelik sırası)

1. **`load_wind_features`: forecast API yarını kapsamıyorsa `raise`** — merge sonrası `gap`'in
   hedef 24 saatin hepsini kapsamasını şart koş; `fc` None/boş ise `return hist` yerine hata.
2. **`assert_wind_available`'ı sıkılaştır** — `sub.isna().all(axis=None)` tek non-NaN ile
   geçiyor; "herhangi bir hedef saatte 3 şehir birden NaN" veya kapsama < 24/24 ise raise.
3. **forecast API çağrısına 3× retry + backoff** (şu an tek deneme, sessiz `continue`).
4. **Neden forecast API üretimde hiç çağrılmıyor** araştır — kod ağacında
   `fetch_openmeteo_wind_forecast` var, çalışan serviste log'u yok (deploy / kod-yolu).
5. **Düz-eğri monitörü** — koşu sonrası son N gün `predicted_wind_lag0` aynı 24s şeklin
   ±X MW'ı içindeyse alarm.
6. **D-1 çapraz kontrol** — akşam fiyat-çekim koşusu: `predicted_wind_lag0` vs EPİAŞ yayınlanan
   yarınki KGÜP rüzgarı; fark > 2000 MW ise logla.
7. (opsiyonel) `lgb_wind`'e `feature_fraction=0.5` + bagging — küçük ama bedava (~%5 MAE).

**§5'teki "lag sayısını azalt" / "lag ağırlığını kıs" önerileri düşürüldü — test ters yönü gösterdi.**

---

## Ek: reprodüksiyon

- `experiments/scripts/replay_recent_days.py` — canlı ile aynı girdilerle son N günü
  DB'ye yazmadan yeniden üretir, karşılaştırma tablosu basar.
- Ham karşılaştırma verisi: `experiments/scratch_replay_comparison.csv`.
- `experiments/notebooks/07_lago_protocol/07_wind_preforecast_incident.ipynb` — §6'nın tamamı.
- Çıktı CSV'leri (`experiments/notebooks/07_lago_protocol/`): `preforecast_recent_fixed.csv`,
  `wf_lgbm_recent_{dbpf,fixedpf}.csv`, `wind_fix_price_impact.csv`, `wind_lag_test_results.csv`.
