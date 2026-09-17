"""Feature engineering — canlı repodan vendor edilmiş temel.

`feature_engineering.py` ve `holidays.py`, `../enerji_fiyat_tahmini`'nin
`src/features/`'inden BİREBİR kopya olarak başladı:

    kaynak commit : d7c67bf (dosyaların son değiştiği), canlı HEAD fde25b84
    senkron tarihi: 2026-08-27

Amaç: bu deney reposunu canlı repodan bağımsız kılmak (bkz. CLAUDE.md). Canlı
modelle yapılan kıyaslar ("LightGBM canlı") ancak bu kopya canlıyla AYNI feature
mantığını taşıdığı sürece geçerli.

## İş akışı (kullanıcının kuralı)

Feature değişikliği ÖNCE burada yapılır, sonuç alınırsa canlıya taşınır — canlı
repo takip eder, bu repo öncüdür. Yani bu dosyalar zamanla canlıdan **kasıtlı
olarak** ayrışabilir; her ayrışma bir deneydir, drift değil.

Canlıdan gelen bir düzeltmeyi geri almak için:
    cp ../enerji_fiyat_tahmini/src/features/{feature_engineering,holidays}.py src/features/
"""
