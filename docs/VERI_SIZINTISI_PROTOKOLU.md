# 🛡️ EPİAŞ PTF Tahmin Motoru — Veri Kullanım ve Sıfır-Sızıntı Protokolü

**Amaç:** Tahmin modellerinin canlı üretim ortamında hatasız çalışması ve veri sızıntısının (Data Leakage) önlenmesi için bağlayıcı kural belgesidir.

---

## 📌 ALTIN KURAL

Yarınki PTF fiyatları tahmin edilirken, kullanılan verilerin **tahmin edilecek hedef günden kaç gün öncesine kadar alınabileceği** aşağıdaki 2 kurala tabidir.

---

## 🟢 1. Tahminlenen Günden En Fazla "1 Gün Öncesinden" Alınabilecek Veriler

_(Gün Öncesi Piyasası'nda ilan edilen planlar ve geçmiş fiyatlar)_

1. **Yük Tahmini:** EPİAŞ resmi talep tahmini.
2. **KGÜP Kaynak Üretim Planı:** Santrallerin hidro, güneş, rüzgar, gaz ve kömür üretim programları.
3. **Geçmiş PTF Fiyatları:** Kesinleşmiş geçmiş Dolar PTF fiyatları.

---

## 🟡 2. Tahminlenen Günden En Fazla "2 Gün Öncesinden" Alınabilecek Veriler

_(Piyasa onayları, sayaç okumaları ve veri yansıma gecikmesi olanlar)_

1. **Makro Göstergeler:** Dolar/TL Kuru, Brent Petrol, Doğal Gaz GRF Fiyatı.
2. **Gerçekleşen Santral Üretimi:** Sayaçlardan okunan gerçek üretim verisi.
3. **Gerçekleşen Elektrik Tüketimi:** Sayaçlardan okunan gerçek tüketim verisi.
4. **Gerçekleşen Sıcaklık Ölçümleri:** İstasyon geçmiş sıcaklık verileri.
5. **SMF (Sistem Marjinal Fiyatı):** Dengeleme piyasası gerçekleşen fiyatı.

---

## 🚫 3. YASAKLAR (VERİ SIZINTISI)

- ❌ **Tahmin edilecek günün kendisine ait KGÜP veya Yük Tahminini girdi yapmak YASAKTIR.** _(Çünkü o günün planı henüz ilan edilmemiştir)._
- ❌ **Tahmin edilecek günün veya 1 gün öncesinin Gerçekleşen Üretim / Tüketim verisini girmek YASAKTIR.** _(Çünkü sayaçlar onaylanmamıştır)._
- ❌ **Tahmin edilecek günün veya 1 gün öncesinin Dolar Kurunu / SMF Fiyatını girmek YASAKTIR.** _(Çünkü piyasalar henüz kapanmamıştır)._
