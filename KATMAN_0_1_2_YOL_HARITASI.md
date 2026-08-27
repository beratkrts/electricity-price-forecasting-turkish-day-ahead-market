# Katman 0-1-2 Yeniden İnşa Yol Haritası

**Tarih:** 24 Ağustos 2026
**Amaç:** Repo şu an içi dolu ama AI'ın düşük gözetimli oynama alanı durumunda.
Bu belge, mevcut analizleri (model doğrulama + fiyat sürücü) **kendi başına,
teoriyi kapatarak, hipotez testleriyle sistematikleştirerek** yeniden inşa etmek
için bir sıra veriyor. Katman 3 (olay/kriz etki çalışması) bilinçli olarak
bekletiliyor — bkz. `EVENT_IMPACT_STUDY.md`.

**Kural:** Her faz için önce teori, sonra o teoriyle o fazın analizini yeniden
yap. Teoriyi hepsini baştan okuyup sonra tüm analizlere geçme — kaynağı
unutursun, analiz teoriyle bağlantısız kalır.

---

## Faz 0 — Zemin (yarım gün)

**Oku:**
- `docs/literature_main/lago_review.pdf` (zaten okunuyor) — protokolün kendisi:
  6 yıl veri, son 104 hafta test, günlük yeniden kalibrasyon, LEAR+DNN kıyası,
  MAE/RMSE/sMAPE/rMAE, Diebold-Mariano.
- Diebold & Mariano (1995), *Comparing Predictive Accuracy*, J. Business &
  Economic Statistics 13(3). **`literature_main`'de yok, eklenmeli** — testin
  kendisini kullanıyoruz, orijinal makaleyi okumadan atıflayamayız.

**Yeniden yap:**
- `METRICS.md`'yi kapat, sil, **kendi kelimelerinle** yeniden yaz: WAPE neden
  düşük fiyatta çöküyor (payda), naive-2 vs naive-3 farkı ne, rMAE'nin
  yorumlanma eşikleri (0,5-0,8) nereden geliyor. Kaynak SQL'ler zaten dosyada,
  onları çalıştır, sayıları kendi gözünle doğrula.

**Bitti sayılır ne zaman:** WAPE'nin neden yanılttığını, formülü yazmadan,
sözlü olarak birine anlatabiliyorsan.

---

## Faz 1 — LEAR / benzeşik kıyas kökeni (1-2 gün)

**Oku (ikisi de `literature_main`'de yok, eklenmeli — OpenAlex ortak-atıf
listesinde A bölümünde):**
- Uniejewski, Nowotarski & Weron (2016), *Automated Variable Selection and
  Shrinkage for Day-Ahead Electricity Price Forecasting*, Energies 9(8), 621.
  **LEAR'ın kökeni** (orijinal adı LassoX) — OA, doi:10.3390/en9080621.
- Ziel & Weron (2018), *Day-ahead electricity price forecasting with
  high-dimensional structures*, Energy Economics 70. Tek değişkenli/saat-başı
  model kararının tartışması — OA.

**Yeniden yap:**
- `src/eval/lago_protocol.py`'deki `build_lear_matrix` ve `lear_predict_day`'i
  kapat, **247 özelliğin her birinin neden orada olduğunu** (fiyat lag'leri,
  dışsal tahminler, dummy'ler) kendi diyagramınla çıkar, sonra kodu tekrar yaz.
- `experiments/notebooks/07_lago_protocol/`'daki her iki defteri (LEAR
  ensemble kıyası, epftoolbox çapraz doğrulama) sıfırdan, kendi hücrelerinle
  yeniden üret. Bugünkü oturumda bulduğumuz 3 kırılganlığı (n≤p AIC tanımsızlığı,
  MAD=0 çöküşü, LEAR lehine sızıntı) **kendi kodunla tekrar keşfet** — cevapları
  biliyorsun ama neden orada olduklarını kod okuyarak değil, veriye bakarak
  bulmaya çalış.

**Bitti sayılır ne zaman:** `noise_variance` yamasının neden gerektiğini
sklearn kaynağına bakmadan anlatabiliyorsan.

**Ek not — bugünkü LEAR sızıntı bulgusuyla bağlantılı:** `experiments/notebooks/
02_preforecasters/` (3 notebook) tam bu sorunun çözümü — yük ve KGÜP'ün T+1'de
gerçekten bilinmediğini, bu yüzden ayrı alt-modellerle (LightGBM yük/rüzgar,
Ridge güneş) **tahmin edilmesi** gerektiğini gösteriyor. `advanced_kgup_load_
forecaster.ipynb`'i bu fazda oku — LEAR'ın besleme SQL'inin neden gerçek
değerleri değil bu pre-forecaster çıktılarını kullanması gerektiğini (bugün
paket bırakılan düzeltme) burada somut olarak gör.

---

## Faz 2 — Model mimarisi: neden LightGBM, neden quantile — GERÇEK İTERASYON TARİHİ (3-4 gün)

Bu faz ilk yazımda zayıftı — sadece `EXPERIMENT_REPORT.md`'ye bakıp geçmiştim.
`experiments/notebooks/` taraması sonrası gerçek resim çok daha zengin: **11
notebook** (03_ptf_model_comparison'da 7, 04_zero_price_crisis'te 2, artı
06_presentation'daki final sunum). Bunları **kronolojik/mantıksal sırayla**
oku — bu, canlı modelin nasıl bugünkü haline geldiğinin gerçek hikâyesi.

**Oku (teori):**
- `docs/literature_main/review_2025.pdf` (O'Connor et al. 2025) — zaten elde,
  güncel harita.
- Lago, De Ridder & De Schutter (2018), *Forecasting spot electricity prices:
  Deep learning approaches...*, Applied Energy 221. **`literature_main`'de yok,
  eklenmeli** — 2021 kıyasındaki DNN buradan geliyor, ortak-atıfta 1. sıra.

**Yeniden yap — sıralı notebook listesi:**

1. **`03_ptf_model_comparison/02_model_experiments_daily_forecasting.ipynb`**
   — köken noktası. DENEME 1 (LightGBM baseline) vs DENEME 2 (log1p hedef +
   hidro su enerjisi + güneş/rüzgar zirve oranları, "nihai başarılı"). Buradan
   sonra log1p'in neden **kaldırıldığını** (canlı modelde yok, `EXPERIMENT_REPORT.md`
   diyor) bu notebook'la şimdiki kod arasındaki farktan çıkar.
2. **`03_ptf_model_comparison/03_epnet_vs_lightgbm_performance_analysis.ipynb`**
   — EPNet (CNN-LSTM) 359 günlük walk-forward'da LightGBM'e karşı. Fine-tuning
   doyma/unutma analizi, rejim bazlı hata kırılımı. **EPNet'in 9. aydan sonra
   çöktüğünü** (WAPE %21→%79) burada göreceksin — bu "derin öğrenme çalışmıyor"
   sonucunun kaynağı.
3. **`04_zero_price_crisis/lstm_online_learning_experiment.ipynb`** —
   **2'nin doğal devamı, hemen ardından oku.** IISE PG&E 2024 şampiyonu
   (Team 1) metodolojisini temel alan PyTorch LSTM + online öğrenme, 3 senaryo.
   Bu notebook zaten `literature_main`'in flagladığı soruyu ("EPNet mimari
   yüzünden mi çöktü yoksa yetersiz dışsal girdi yüzünden mi?") sınamaya
   çalışıyor — sonucunu oku, EPNet'in çöküş sebebiyle karşılaştır: aynı mı,
   farklı mı?
4. **`03_ptf_model_comparison/mwa_baseline_test.ipynb`** +
   **`mwa_vs_lightgbm_backtest.ipynb`** — basit ağırlıklı hareketli ortalama
   kıyası, ek bir sağduyu kontrolü (naive-2/3 ve LEAR'a ek üçüncü bir taban).
5. **`03_ptf_model_comparison/paper_driven_strategies_experiments.ipynb`** —
   **Faz 1/3 teorisiyle doğrudan bağlantılı, en yüksek öncelikli.** Wang et al.
   (Group Lasso, arcsinh+MAD dönüşüm, çoklu pencere ansambl) ve Ezzat et al.
   (residual boosting, zengin dışsal özellik) stratejilerini **ampirik olarak
   zaten test etmiş.** Faz 1'de Uniejewski/Ziel&Weron'u, Faz 3'te
   Marcjasz/Lipiecki'yi okuduktan sonra bu notebook'a dönüp sonuçları teoriyle
   birlikte oku — teori-pratik köprüsü burada kurulu, sıfırdan kurmana gerek yok.
6. **`04_zero_price_crisis/zero_price_forecasting_experiment.ipynb`** —
   `renewable_pressure_ratio_lag0` özelliğinin **keşif anı** (13 Ağustos 2026
   canlı vaka: WAPE %28,3→%15,9, kapsama %25→%79,2). Şu ana kadarki tek en
   büyük belgelenmiş iyileştirme — ama tek günlük test, örneklem büyüklüğü
   sorusunu burada sor.
7. **`04_zero_price_crisis/hybrid_ptf_classifier.ipynb`** — sıfır-fiyat için
   sınıflandırıcı+regresör hibrit Ar-Ge, muhtemelen yarım kalmış/terk edilmiş
   bir kol — neden terk edildiğini (kod var, sonuç/karar notu yoksa) kendi
   yargınla değerlendir.
8. **`03_ptf_model_comparison/04_live_results_rmae_analysis.ipynb`** —
   `METRICS.md`'nin çekirdek metrik setinin canlı veriden çıkarılışı (Faz 0
   ile örtüşüyor, çapraz kontrol için iyi).
9. **`03_ptf_model_comparison/05_gas_tariff_feature_backtest.ipynb`** —
   Katman 2/3 sınırında: kriz analizinde bulunan BOTAŞ tarife bulgusunun
   (`05_crisis_analysis/03_gas_tariff_discovery.ipynb`, Faz 4'te okunacak)
   **canlı modelde de işe yarayıp yaramadığının** testi. Faz 4'ü bitirince
   buraya geri dön.
10. **`06_presentation/final_presentation_analysis.ipynb`** — kapanış/sentez:
    öznitelik önemi, "Günlük Ortalama WAPE vs Hacim Ağırlıklı WAPE" matematiksel
    paradoksu (bunu ayrıca anla, ince ama önemli bir tuzak), EPNet vs LightGBM
    log dosyalarından gerçek kıyas.

**Bitti sayılır ne zaman:** (a) EXPERIMENT_REPORT.md'deki her tablo satırı için
"bu iddia neye dayanıyor, örneklem büyüklüğü ne" sorusuna cevap verebiliyorsan;
(b) EPNet'in çöküş sebebini (mimari mi, veri zenginliği mi) LSTM online-learning
denemesinin sonucuyla birlikte savunabiliyorsan; (c) "Günlük Ortalama WAPE"
ile "Hacim Ağırlıklı WAPE" arasındaki farkı bir örnekle gösterebiliyorsan.

---

## Faz 3 — Olasılıksal tahmin: P10/P90 ve CQR (1-2 gün)

**Oku (ikisi de `literature_main`'in kendi README'sinde "en yüksek getirili
ikili" diye işaretli, eklenmesi gerekiyor):**
- Marcjasz, Narajewski, Weron & Ziel (2023), *Distributional neural networks
  for electricity price forecasting*, Energy Economics 126. OA.
- Lipiecki, Uniejewski & Weron (2024), *Postprocessing of point predictions
  for probabilistic forecasting...*, Energy Economics 139. OA — **doğrudan
  bizim CQR hattımız.**

**Yeniden yap:**
- `src/models/cqr_calibrator.py` şu an "deneysel" rafında. P10-P90 kapsaması
  şu an %65-75 (nominal %80) — bu bilinen, çözülmemiş bir açık
  (`model-shrinkage-low-prices` hafızasında da var). Lipiecki 2024'ü okuduktan
  sonra: CQR'ı canlıya almanın maliyeti/faydası nedir, karar ver. Kod
  yazmasan bile bir karar notu (`neden rafta kalıyor` ya da `neden alınmalı`)
  üret.

**Bitti sayılır ne zaman:** Kapsama açığının (%65-75 vs %80) sebebini
(quantile crossover koruması mı, model bias'ı mı, kalibrasyon mu) ayırt
edebiliyorsan.

---

## Faz 4 — Piyasa mekaniği: fiyat sürücüleri ve rejimler (3-4 gün)

Bu faz da ilk yazımda zayıftı — sadece `LOW_PRICE_REGIME_ANALYSIS.md`'ye
bakmıştım. Asıl omurga başka yerde: **`06_presentation/tavandan_sifira.ipynb`**,
13 bölümde 2021-2026 Türkiye elektrik piyasasının neredeyse tüm yapısal
hikâyesini veritabanından yeniden üretiyor — hiçbir sayı elle yazılmamış, her
bölüm SQL→tablo/grafik→yorum sırası izliyor. Bunu görmemiştim, şimdi omurga bu.

**Oku (teori — `literature_main` burada zayıf, EPF tahmin literatürü piyasa
**mekaniğini** değil tahmin yöntemini kapsıyor; aşağıdakiler 24 Ağustos 2026'da
ayrıca araştırılıp doğrulandı, sırayla oku):**

*Kavramsal temel (global):*
1. Kirschen, D.S. & Strbac, G., *Fundamentals of Power System Economics*, Wiley.
   Alanın standart ders kitabı — merit-order'ın kendisi, yenilenebilir
   entegrasyonunun piyasaya etkisi.
2. Stoft, S. (2002), *Power System Economics: Designing Markets for
   Electricity*, IEEE Press/Wiley. Fiyat tavanı/Lerner endeksi bölümleri
   AFL/AUF'u anlamak için özellikle ilgili. **Ücretsiz PDF:**
   `stoft.com/books-and-papers`.

*Türkiye'ye özel ampirik literatür:*
3. Sirin, S.M. & Yilmaz, B.N. (2020), *Variable renewable energy technologies
   in the Turkish electricity market: Quantile regression analysis of the
   merit-order effect*, Energy Policy 144, 111660. doi:10.1016/j.enpol.2020.111660
   — VRE'nin Türkiye'de merit-order etkisi, kantil regresyonla.
4. Gökgöz, F. & Yücel, Ö. (2024), *Merit-order of dispatchable and variable
   renewable energy sources in Turkey's day-ahead electricity market*,
   Utilities Policy 88, 101758. doi:10.1016/j.jup.2024.101758 — daha güncel
   (2019-2022), hidro/termal ayrımı `low-price-regime-hydro-2026` bulgusuyla
   doğrudan konuşuyor.
5. Durmaz, T., Acar, S. & Kızılkaya, S. (2024), *Generation failures,
   strategic withholding, and capacity payments in the Turkish electricity
   market*, Energy Policy 184, 113897. doi:10.1016/j.enpol.2023.113897 —
   arz tarafı stratejik davranış.
6. **Emre, T. (2025), *Hidden subsidies and behavioral targeting: Insights
   from Türkiye's three-pillar energy poverty strategy (2019–2023)*, Utilities
   Policy, 102043. doi:10.1016/j.jup.2025.102043 — en kritik, doğrudan AUF
   mekanizmasının kendisini analiz ediyor. `LITERATURE_REVIEW.md` §3'teki
   AFL≠AUF ayrımına buradan kaynak ekle.**

*Resmî birincil kaynaklar (akademik değil, otorite — mekanizmanın yasal/teknik
tanımı buradan gelir):*
7. EPİAŞ, *Gün Öncesi Elektrik Piyasası PTF Belirleme Yöntemi* (2025 güncel
   sürüm), `epias.com.tr/wp-content/uploads/2025/03/Gun-Oncesi-Elektrik-Piyasasi-PTF-Belirleme-Yontemi.pdf`
   — PTF'nin arz-talep eğrisinden oluşumunun resmi metodolojisi.
8. EPDK Yönetmelikler sayfası (`epdk.gov.tr/Detay/Icerik/3-0-0-49/yonetmelikler`)
   ve ilgili Resmî Gazete tebliğleri — AFL/AUF'un yasal dayanağı ve değişiklik
   tarihleri, `LITERATURE_REVIEW.md`'deki olay tablosunu doğrulamak için.
9. Enerji Uzmanları Derneği, *Organize Toptan Elektrik Piyasalarında Fiyat
   Limitleri*, `enerjiuzmanlari.org.tr` — akademik değil ama erişilebilir bir
   Türkçe giriş, AFL/AUF ayrımına hızlı başlangıç için.

**Yeniden yap — sıralı notebook listesi:**

1. **`06_presentation/tavandan_sifira.ipynb` §1-9** — omurga. Sırayla: veri
   kapsamı, 2021-2026 fiyat seyri/rejimleri, tavan/taban yapışma zamanları,
   üretim kompozisyonu değişimi, **yıllar arası neyin taşındığı** (hidro payı
   mı termal pay mı — bu soru `low-price-regime-hydro-2026` hafızasının
   kaynağı), $70 plato seviyesinin kökeni, sürücü ağırlıklarının yıldan yıla
   değişimi, ördek eğrisi, lisanssız güneşin ölçüm boşluğu. Her bölümün SQL'ini
   çalıştır, sayıyı kendi gözünle doğrula.
2. **`04_zero_price_crisis/regime_shift_analysis.ipynb`** — tavandan_sifira'yı
   tamamlıyor: 2024-2026 yenilenebilir baskısının büyümesi, sıfır-fiyat
   saatlerinin patlaması, ördek eğrisinin derinleşmesi — yıl-yıl kıyaslamalı,
   sunum/makale için tasarlanmış üç kanıt.
3. **`04_zero_price_crisis/01_low_price_regime_analysis.ipynb`** — hidro rejim
   kırılması teşhisi (ilk yazımda buradaydı, sırası şimdi 3.). Buradaki
   "hidro kaynaklı merit-order kırılması" iddiasını **hipotez testine çevir:**
   yapısal kırılma testi (Chow test ya da benzeri) ile "hidro payı ile termal
   payın Şubat-Mayıs 2026 arasında rejim değiştirdiği" iddiasını sayısal
   olarak test et, sadece grafikle gösterme.
4. **`05_crisis_analysis/01_counterfactual_model.ipynb`** — bir olayın etkisini
   arka plan sürücülerinden (hava, hidro, yakıt) nasıl ayrıştırırsın sorusunun
   yöntemi (`crisis_cf_v5`, örneklem-dışı şema, boş bant). **v1'i değil v5'i
   temel al** — notebook'un kendi uyarısı bunu söylüyor.
5. **`05_crisis_analysis/02_model_bias_investigation.ipynb`** +
   **`03_gas_tariff_discovery.ipynb`** — kontrafaktüel aletin kendi
   sapmasının avı (BOTAŞ tarife vs EPİAŞ GRF ayrımı, korelasyon 0,852). Bu
   bulgu Faz 2 adım 9'da (`05_gas_tariff_feature_backtest.ipynb`) canlı
   modelde test edildi — oraya geri dönüp iki sonucu karşılaştır.
6. **`tavandan_sifira.ipynb` §10-13`** — model performansı, WAPE tuzağı,
   fiyat dilimine göre sistematik sapma, 16 Ağustos 2026 vakası. Bunlar
   Katman 1 ile kesişiyor, Faz 0/2 ile çapraz kontrol için iyi.
- `docs/VERI_SIZINTISI_PROTOKOLU.md`'yi de bu fazda tekrar oku — fiyat
  sürücü analizlerinde hangi verinin ne zaman bilindiği kritik, aynı sızıntı
  hatası (bugün LEAR'da bulduğumuz) buraya da sızmış olabilir, kontrol et.

**Bitti sayılır ne zaman:** "Hidro rejim kırılması" iddiasını bir p-değeriyle
savunabiliyorsan (sadece grafikle değil), ve BOTAŞ tarife bulgusunun kriz
modelinde mi canlı modelde mi (ya da ikisinde de) işe yaradığını net
ayırt edebiliyorsan.

---

## Faz 5 — Geri kalanlar (fırsat buldukça, zorunlu değil)

- Tschora et al. (2022) — ortak-atıfta 2. sıra, kaçırılmış önemli bir iş.
- Maciejowska, Uniejewski & Weron (2023) — Weron 2014'ün halefi, güncel harita.
- Olivares et al. (2022), NBEATSx — Lago'nun DNN kıyasının mimari halefi.
- `docs/literature_main/iise_pge_competition_review.pdf` ve
  `competition_plus.pdf` — zaten elde, yarışma kanıtı ve CING-LEAR. İlk 4
  fazda dolaylı kullanıldı, tam okuma bu fazda.

---

## Genel ilerleme kuralı

Her faz sonunda **eski notebook'u SİLME, yanına `_v2` ile yenisini yaz**, ikisini
kıyasla, sonra eskisini kaldır. Bu hem "ne değişti" sorusuna cevap bırakır hem
de AI'ın yazdığı orijinal analizle senin yeniden ürettiğin arasındaki farkı
görünür kılar — paketleme aşamasında bu fark kendi başına bir metodoloji notu
olabilir ("otonom üretilen ilk analiz ile hipotez-güdümlü yeniden yapılan analiz
arasındaki fark").

Katman 3'e (`EVENT_IMPACT_STUDY.md`) dönüş koşulu: Faz 0-4 bitti VE literatür
taramasındaki açık kalemler (YÖK tez taraması, Scopus/WoS, 2 paywall'lı kaynak
— bkz. `LITERATURE_REVIEW.md` §7) kapandı.
