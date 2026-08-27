# Omurga makaleler

Bu klasör tezin yöntem omurgasını taşıyan makaleleri tutuyor. Dördü birlikte
tutarlı bir zincir: **protokol → güncel harita → yarışma kanıtı → yöntem halefi.**

| Dosya | Künye |
|---|---|
| `lago_review.pdf` | Lago, J., Marcjasz, G., De Schutter, B. & Weron, R. (2021). *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark.* **Applied Energy 293, 116983.** (elimizdeki arXiv 2008.08004v2) |
| `review_2025.pdf` | O'Connor, C., Bahloul, M., Prestwich, S. & Visentin, A. (2025). *A Review of Electricity Price Forecasting Models in the Day-Ahead, Intra-Day, and Balancing Markets.* **Energies 18(12), 3097.** |
| `iise_pge_competition_review.pdf` | Ezzat, A.A., Mansouri, M., Yildirim, M. & Fang, X. (2026). *IISE PG&E Energy Analytics Challenge 2024: Forecasting day-ahead electricity prices.* **IISE Transactions 58(1), 117-129.** |
| `competition_plus.pdf` | Wang, K., Ji, J., Mansouri, M. & Ezzat, A.A. (2026). *Day-Ahead Electricity Price Forecasting Using a Multivariate Group Lasso Method.* **arXiv 2605.27781v2** |

---

## 1. Lago et al. 2021 — protokol

Alanın değerlendirme ölçütünü koyan makale. Şart koştukları: 6 yıllık veri,
**son 104 hafta test**, **günlük yeniden kalibrasyon**, LEAR ve DNN'e karşı kıyas,
MAE/RMSE/sMAPE/**rMAE**, ve **Diebold-Mariano testi**. MAPE'yi sıfıra yakın
fiyatlarda patladığı için reddediyorlar.

**rMAE'yi naive-2 (geçen hafta aynı saat) üzerinden tanımlıyorlar** — bkz.
`METRICS.md`, bizim naive-3 kullandığımız ve bunun kıyası bozduğu orada yazılı.

Uygulaması: `src/eval/lago_protocol.py`, defter
`experiments/notebooks/07_lago_protocol/`.

## 2. O'Connor et al. 2025 — güncel harita

Weron 2014'ten sonra alanın en güncel geniş derlemesi. Yeniliği: sadece gün
öncesi değil **gün içi ve dengeleme** piyasalarını da kapsıyor.

Bulguları:
- Gün öncesinde **hibrit ve topluluk modelleri** öne çıkıyor
- Gün içinde **RNN'ler** baskın
- Dengelemede **LEAR gibi basit modeller** — oynaklık yüksek, veri seyrek

**§4.6 bizim naive sorunumuzu bağımsız olarak doğruluyor:** rMAE'nin
*"usefulness hinges on the appropriateness of the benchmark, which varies
substantially across studies."* Yani hangi naive olduğunu yazmayan çalışmanın
rMAE'si kıyaslanamaz — `METRICS.md`'ye eklediğimiz uyarının literatürdeki hali.

Türkiye dört yerde geçiyor (Türkiye GÖP ve gün içi piyasası çalışmaları).

## 3. Ezzat et al. 2026 — yarışma kanıtı

**Gizli test setli gerçek bir yarışma.** CAISO NP-15, eğitim 2020-2022 (3 yıl),
test 2023'ün ilk iki haftası, katılımcılardan saklanmış.

| Model | MAE | RMSE | MAPE% | Skill* |
|---|--:|--:|--:|--:|
| **Team 1 — LSTM + 46 dışsal özellik** | **18,92** | 24,39 | 12,04 | **%53,9** |
| Team 3 — VAR + XGBoost (1 özellik) | 26,82 | 35,17 | 17,08 | %34,6 |
| Team 2 — ağaç topluluğu (4 özellik) | 29,08 | 36,03 | 18,16 | %29,1 |
| ARIMA-X (dışsal değişkenli) | 32,3 | 44,0 | 21,5 | %21,2 |
| Mevsimsel naive (24 saat) | 38,0 | 46,3 | 23,3 | %7,3 |
| ARIMA (dışsal değişkensiz) | 38,4 | 48,1 | 23,1 | %6,4 |
| Naive | 41,01 | 50,36 | 24,70 | 0 |

\* `Skill = 100 − 100 × MAE(model)/MAE(naive)`, yani `100 × (1 − rMAE)`.

**Ana ders: dışsal bilgi zenginliği kazandırıyor.** Team 1'in 46 eklenen özelliği
(hava, üretim karması, ithalat, depolama, takvim) skill'i 34,6'dan 53,9'a
çıkarıyor. Üç takımın ortalaması naive'i %39,2, ARIMA-X'i %22,8 geçiyor.

**İkinci ders — dikkat, incelikli:** Team 2 kendi raporunda hazır LSTM ve
transformer'ları denemiş, **aşırı uyum yapıp geride kalmışlar**. Ama yarışmayı
kazanan da bir LSTM. Yani ayrım "derin öğrenme çalışmaz" değil:
**hazır kullanılan derin öğrenme kaybediyor, bağlama uyarlanmış ve zengin
dışsal girdi verilen derin öğrenme kazanıyor.**

**Uyarı:** test seti **iki hafta**. Titiz bir yarışma bile bu kadar kısa bir
pencere kullanmış; Lago'nun iki yıllık şartı daha zorlayıcı. Skill skorları da
kendi naive'lerine göre — o naive (41,01) kendi mevsimsel naive'lerinden (38,0)
bile zayıf, dolayısıyla bu yüzdeler naive-2 tabanlı rMAE ile kıyaslanamaz.

## 4. Wang et al. 2026 — CING-LEAR, yöntem halefi

LEAR'ın **Group Lasso** uzantısı. Fikir: açıklayıcı değişkenlerin fiyata etkisi
ardışık saat blokları boyunca sürüyor ("temporal group effects"), ceza yapısı
bunu yakalayacak şekilde kuruluyor.

Değerlendirme: iki tam yıl CAISO, kıyas olarak **Lago'nun LEAR ve DNN'i**,
nokta *ve olasılıksal* metrikler (MAE, RMSE, **CRPS**).

İki güçlü doğrulama:
- Yarışmaya geriye dönük sokulduğunda **ikinci sırada** — üstelik ek özellik
  kullanmadan. **LEAR üçüncü**, yani üç takımdan ikisini geçiyor.
- ABD'de büyük bir şirketin **sahada çalışan iki modeline** karşı test edilmiş;
  7 aylık pencerede birinci.

---

## Bu proje için ne anlama geliyor

**Lehimize olan.** Yarışmanın ana dersi (dışsal bilgi zenginliği kazandırır)
bizim mimarimizi destekliyor: 65+ özellik, 26 bölge ağırlıklı hava, KGÜP,
üretim karması, kur, BOTAŞ gaz tarifesi. Team 1'in kazanma sebebi tam olarak bu.

**Yeniden açılması gereken.** `EXPERIMENT_REPORT.md` EPNet'in çöktüğünü yazıyor
ve bundan "derin öğrenme çalışmıyor" sonucu çıkarılmıştı. Yarışmayı bir LSTM
kazandı. Doğru soru şu: EPNet mimari yüzünden mi çöktü, yoksa yeterince zengin
dışsal girdi verilmediği için mi? Bu kapalı bir dosya değil.

**Kıyas ciddiye alınmalı.** LEAR, yarışmada üç takımdan ikisini geçen bir model.
07_lago_protocol defterindeki LEAR bir hasım, kolay bir referans değil.

**Metrik disiplini.** Üç makale de aynı şeyi söylüyor: hangi naive/kıyas
kullanıldığı yazılmadan rMAE ya da skill skoru kıyaslanamaz. `METRICS.md`
buna göre düzeltildi (22-23 Ağu 2026).

---

# Eklenmesi gereken — ortak-atıf analiziyle

23 Ağu 2026. Yöntem: Lago 2021'e atıf yapan **486 çalışma** çekildi,
kaynakçaları toplandı (15.181 ayrı iş), Lago ile **birlikte** en çok anılanlar
sıralandı. Kanonu tahminle değil ölçümle bulmanın yolu bu.
Üreteç sorgusu `scripts/lit_search.py` altyapısıyla aynı (OpenAlex).

## A. Mutlaka — Lago'nun üstüne kurulduğu iki temel

Bunlar **Lago 2021'den önce** ama onsuz protokol anlaşılmıyor; ortak-atıfta
en üst sıradalar.

| Birlikte anılma | Künye | Neden |
|--:|---|---|
| **108** | **Lago, De Ridder & De Schutter (2018).** *Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms.* **Applied Energy 221, 386-405.** 626 atıf · OA · doi:10.1016/j.apenergy.2018.02.069 | 2021 kıyasındaki **DNN buradan geliyor**. Ortak-atıfta 1. sıra |
| 45 | **Uniejewski, Nowotarski & Weron (2016).** *Automated Variable Selection and Shrinkage for Day-Ahead Electricity Price Forecasting.* **Energies 9(8), 621.** OA · doi:10.3390/en9080621 | **LEAR'ın kökeni** (özgün adı LassoX). Kıyas modelinin kaynağı |
| 73 | **Ziel & Weron (2018).** *Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks.* **Energy Economics 70.** 242 atıf · OA | Tek değişkenli mi çok değişkenli mi — 24 saati ayrı ayrı mı birlikte mi modellemeli. Bizim saat-başına-model kararımızın literatürdeki tartışması |

## B. Son 5 yılda eklenenler — asıl sorunun cevabı

Alan Lago 2021'den sonra **üç yönde** ilerledi.

### B1. Nokta tahminden dağılımsal tahmine — en büyük kayma

| Künye | Neden bizim için kritik |
|---|---|
| **Marcjasz, Narajewski, Weron & Ziel (2023).** *Distributional neural networks for electricity price forecasting.* **Energy Economics 126, 106843.** doi:10.1016/j.eneco.2023.106843 | Kırılmanın kendisi. Bizde P10/P50/P90 zaten var ama nokta tahmin mantığıyla değerlendiriliyor |
| **Lipiecki, Uniejewski & Weron (2024).** *Postprocessing of point predictions for probabilistic forecasting of day-ahead electricity prices.* **Energy Economics 139, 107934.** OA · doi:10.1016/j.eneco.2024.107934 | **Doğrudan bizim CQR hattımız.** `src/models/cqr_calibrator.py` "deneysel" rafında duruyor; alanın gittiği yer tam burası |
| **Uniejewski & Weron (2021).** *Regularized quantile regression averaging for probabilistic electricity price forecasting.* **Energy Economics 95, 105121.** doi:10.1016/j.eneco.2021.105121 | QRA — kantil tahminlerini birleştirme |
| Ziel & Steinert (2018). *Probabilistic mid- and long-term electricity price forecasting.* **RSER 94.** OA | Olasılıksal EPF'in derlemesi, bağlam |

### B2. Mimari halefi

| Künye | Neden |
|---|---|
| **Olivares, Challú, Marcjasz, Weron & Dubrawski (2022).** *Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx.* **IJF 39(2).** 246 atıf · OA · doi:10.1016/j.ijforecast.2022.03.001 | Lago'nun DNN kıyasının halefi. Aynı ekip, aynı protokol |

### B3. Bağımsız doğrulama ve kıyas genişlemesi

| Künye | Neden |
|---|---|
| **Tschora, Pierre, Plantevit & Robardet (2022).** *Electricity price forecasting on the day-ahead market using machine learning.* **Applied Energy 313, 118752.** 213 atıf · doi:10.1016/j.apenergy.2022.118752 | **Ortak-atıfta 2. sıra (77).** Kaçırılmıştı; kanona en yakın işlerden |
| Billé, Gianfreda, Grosso & Ravazzolo (2022). *Forecasting electricity prices with expert, linear, and nonlinear models.* **IJF.** OA | |
| Lehna, Scheller & Herwartz (2021). *Forecasting day-ahead electricity prices: A comparison of time series and neural network models.* **Energy Economics 106, 105742.** OA | |

## C. Güncel derleme ve bağlam

| Künye | Neden |
|---|---|
| **Maciejowska, Uniejewski & Weron (2023).** *Forecasting Electricity Prices.* **Oxford Research Encyclopedia of Economics and Finance.** | **Weron 2014'ün halefi**, aynı okuldan. Güncel harita |
| Hong, Pinson, Wang, Weron, Yang & Zareipour (2020). *Energy Forecasting: A Review and Outlook.* **IEEE OAJPE.** 608 atıf · OA | Enerji tahmininin tamamı; ortak-atıfta üst sıralarda |
| Diebold & Mariano (1995). *Comparing Predictive Accuracy.* **J. Business & Economic Statistics 13(3).** | Testin kendisi. Uyguluyoruz, atıflamalıyız |

## Okuma sırası önerisi

1. `lago_review.pdf` — protokol *(okuyorsun)*
2. **Lago 2018** — 2021'in DNN'i, ortak-atıfta 1. sıra
3. `review_2025.pdf` (O'Connor) veya Maciejowska 2023 — güncel harita
4. **Marcjasz 2023 + Lipiecki 2024** — dağılımsal dönüş, bizim P10/P90 ve CQR'a doğrudan bakan iki iş
5. `iise_pge_competition_review.pdf` — yarışma kanıtı
6. Gerisi ihtiyaca göre

**Bizim projemiz açısından en yüksek getirili ikili: Marcjasz 2023 ve
Lipiecki 2024.** Sebebi: elimizde zaten kantil tahminler var (P10/P50/P90,
kapsama %65-75 aralığında dolaşıyor, nominal %80) ve raflanmış bir CQR
kalibratörü duruyor. Alanın 2023-2024'te tam bu problemi çözdüğü yer burası.
