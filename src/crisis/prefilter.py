"""Kural bazlı ön filtre — LLM'e gidecek haberleri seçer.

Bu modül bir optimizasyon değil, **taşıyıcı altyapı**. Sıfır bütçeyle yerel bir
8B model kullanıyoruz; haber başına ~8 saniye. 28.100 haberin tamamını göndermek
5 gece, %20'sini göndermek 1-2 gece. Bu dosya o farkı belirliyor.

Tasarım ilkesi: **recall pahalı, precision ucuz.**
  - Kaçırılan olay görünmez — analizde hiç var olmamış gibi davranır.
  - Geçen gürültü 8 saniyeye mal olur ve LLM'in `alaka_skoru`'su onu ikinci
    kademede eler (bkz. labeling_schema).
  Bu yüzden dışlama kuralları sadece **olay OLMASI imkânsız** olan şeyleri
  hedefler. Şüphe varsa geçir.

Katmanlar:
  A. Kesin dışlama — boilerplate seri, ilgisiz bölüm, olay olamayacak konular
  C. Kanal bazlı dahil etme — altı etki kanalına bağlı terimler
     (B katmanı yok: coğrafi dışlama denendi ve terk edildi — Rusya/Ukrayna
      gaz haberleri kritik, "yurtdışı" ile ayırmak recall'ü kırıyor)

Ölçülmüş sonuçlar (8.340 haberlik kısmi korpus, 17 Ağustos 2026):
  geçen oran            %18,6  (5,4× azalma)
  İran penceresi recall  14/14  (planın §2.4'te saydığı olayların tamamı)
  İran penceresi geçen  %23,2  (440 -> 102)
  tahmini precision      ~%45  (elle bakılan 24 örnek üzerinden)

Korpus tamamlandığında `--calibrate` ile yeniden ölçülmeli.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------- A: dışlama

#: Tekrarlayan otomatik seri. Korpustan ölçüldü: başlık iskeletlerinden en sık
#: olanı "Spot elektrik fiyatı # için # TL" ve tek başına korpusun ~%7'si.
#: Bunlar olay değil, **verinin kendisi** — gün öncesi fiyatı zaten
#: raw_mcp_hourly'de var (plan §7).
EXCLUDE_TITLE_PREFIX = ("Spot elektrik fiyatı",)

#: Bölüm bazlı dışlama. Not: plan `Mevzuat` bölümünü sayıyordu ama korpusta
#: **sadece 1 haber** var — o bölüme güvenilemez, mevzuat sinyali terimle
#: yakalanıyor.
EXCLUDE_SECTIONS = ("Kariyer/Atama", "Elektrikli Araçlar", "Etkinlikler")

#: Olay olması imkânsız konular. Her satırın gerekçesi var; keyfi eklenmez.
EXCLUDE_TERMS: dict[str, tuple[str, ...]] = {
    # Elektrikli araç / otomotiv: bölüm filtresi %0,9 yakalıyor ama aynı içerik
    # 'Elektrik' bölümünde de çıkıyor (ölçüldü: Tesla, MG SUV, Norveç EV payı).
    "arac": ("elektrikli araç", "otomobil", "şarj istasyonu", "tesla",
             "suv", "hibrit araç", "motosiklet", "otomotiv"),

    # Pompa yakıtı: benzin/motorin zammının toptan elektrik fiyatıyla ilgisi yok.
    # Brent zaten modelde feature (brent_oil_lag_48), plan §3.2.
    "pompa": ("benzine", "motorine", "akaryakıt", "pompa fiyat", "lpg",
              "benzin fiyat", "motorin fiyat"),

    # Arama/keşif: ruhsat ve sondaj haberleri yıllar sonrasına dair, fiyat olayı değil.
    "arama": ("petrol ara", "gaz aranacak", "gaz arama ruhsat", "sondaj",
              "kuyu aç", "saha keşf", "keşfetti", "rezerv keşf", "önlisans"),

    # Kurumsal/CSR gürültüsü. 'tiyatro' gerçek bir yanlış pozitiften geldi:
    # "Enerjisa çocuklara enerji tasarrufunu tiyatroyla anlatıyor" -> 'tasarruf'
    # terimine takılıyordu.
    "kurumsal": ("abone", "müşteri sayısı", "istihdam", "ödül", "sponsor",
                 "imza töreni", "iş birliği protokol", "tiyatro",
                 "iş ortaklığı program", "eğitim programı"),

    # Akaryakıt depolama tesisi tarifeleri — 'tarife' terimine takılıyor
    # ama elektrik piyasasıyla ilgisi yok (POAŞ/Petrol Ofisi depo tarifeleri).
    "depo_tarife": ("depolama tesisi tarifesi", "depolama tesisi tarife"),
}

# ---------------------------------------------------------------------------- C: dahil etme

#: Kanal bazlı dahil etme terimleri. Anahtarlar labeling_schema.Kanal ile
#: eşleşiyor — hangi kanalın hangi terimlerle yakalandığı ölçülebilir olsun.
#:
#: Not: `fuel_cost` ve `fx_macro` kanalları kasten dar. Plan §3.2: bu iki kanalı
#: model zaten dolaylı görüyor (brent_oil_lag_48, natural_gas_grf_lag_48,
#: usd_try). Asıl değer regulation / supply_capacity / demand kanallarında,
#: o yüzden onlar geniş.
INCLUDE_TERMS: dict[str, tuple[str, ...]] = {
    "supply_capacity": (
        "kesinti", "kısınt", "kısıt", "arıza", "devre dışı", "arz güvenliğ",
        "akış durdu", "akışı durdu", "planlı bakım", "baraj doluluk",
        "kuraklık", "kurak", "yağış", "su seviyes", "stok miktar", "depola",
        "santral durdu", "üretim durdu", "ithalat durdu",
    ),
    "demand": (
        "tüketim rekor", "puant", "talep rekor", "tasarruf", "soğuk hava",
        "sıcak hava", "kısıtlama", "elektrik molası",
    ),
    "regulation": (
        "epdk", "epiaş", "teiaş", "azami fiyat", "tavan fiyat", "tarife",
        "yönetmelik", "tebliğ", "piyasa kural", "yekdem", "kgüp", "dengeleme",
        "mevzuat", "düzenleme", "kademeli tarife",
    ),
    "fuel_cost": (
        "botaş", "referans fiyat", "gaz fiyat", "doğal gaz tarife",
        "kömür fiyat", "ttf", "zam yapıl", "fiyat artır", "fiyat indir",
    ),
    "fx_macro": ("kur şok", "döviz kur", "enflasyon"),
    "renewable": ("kapasite artı", "devreye al", "kurulu güç", "santral devreye"),
}


# ---------------------------------------------------------------------------- SQL üretimi

#: Aranan alan: başlık + spot + editör etiketleri. Gövde KASTEN dışarıda —
#: ölçümde gövdeyi eklemek geçen oranı %18,6'dan %40'ın üstüne çıkarıyor
#: (uzun metinlerde her terim eninde sonunda geçiyor) ve precision çöküyor.
#: `keywords` alanı burada belirleyici: recall testinde iki kritik haber
#: ("Elektrik fiyatı tavana vurdu", "İran'dan sınırlı gaz akışı başladı")
#: sadece etiketler sayesinde yakalandı.
HAYSTACK_SQL = (
    "lower(title || ' ' || coalesce(description,'') || ' ' || "
    "coalesce(array_to_string(keywords,' '),''))"
)


def _regex(terms: tuple[str, ...]) -> str:
    """Terim listesini tek POSIX regex'e çevirir. Terimler zaten küçük harf."""
    return "(" + "|".join(t.replace("(", r"\(").replace(")", r"\)") for t in terms) + ")"


def exclude_sql() -> str:
    """A katmanı: TRUE ise haber LLM'e gitmez."""
    parts = [
        " OR ".join(f"title LIKE '{p}%'" for p in EXCLUDE_TITLE_PREFIX),
        "section IN (" + ", ".join(f"'{s}'" for s in EXCLUDE_SECTIONS) + ")",
    ]
    parts += [f"{HAYSTACK_SQL} ~ '{_regex(t)}'" for t in EXCLUDE_TERMS.values()]
    return "(" + " OR ".join(parts) + ")"


def include_sql() -> str:
    """C katmanı: TRUE ise haberde en az bir kanal sinyali var."""
    return "(" + " OR ".join(
        f"{HAYSTACK_SQL} ~ '{_regex(t)}'" for t in INCLUDE_TERMS.values()
    ) + ")"


def channel_hits_sql() -> str:
    """Hangi kanalların isabet ettiğini dizi olarak döndürür — hata ayıklama ve
    kalibrasyon için. Bir haber birden çok kanala isabet edebilir."""
    parts = [
        f"CASE WHEN {HAYSTACK_SQL} ~ '{_regex(terms)}' THEN '{kanal}' END"
        for kanal, terms in INCLUDE_TERMS.items()
    ]
    return "array_remove(ARRAY[" + ", ".join(parts) + "], NULL)"


def decision_sql(where: Optional[str] = None) -> str:
    """Karar CTE'si: her haber için geçti/düştü/dışlandı + isabet eden kanallar."""
    filtre = f"WHERE {where}" if where else ""
    return f"""
WITH karar AS (
  SELECT article_id, published_at, section, title,
         {exclude_sql()} AS dislandi,
         {include_sql()} AS dahil,
         {channel_hits_sql()} AS kanallar
  FROM bronze.news_raw
  {filtre}
)
SELECT *, (NOT dislandi AND dahil) AS gecti FROM karar"""


# ---------------------------------------------------------------------------- kalibrasyon

CALIBRATE_SQL = f"""
WITH k AS ({decision_sql()})
SELECT COUNT(*) AS toplam,
       COUNT(*) FILTER (WHERE dislandi)              AS dislandi,
       COUNT(*) FILTER (WHERE NOT dislandi AND NOT dahil) AS sinyal_yok,
       COUNT(*) FILTER (WHERE gecti)                 AS gecen,
       ROUND(100.0 * COUNT(*) FILTER (WHERE gecti) / NULLIF(COUNT(*),0), 1) AS gecen_pct,
       ROUND(COUNT(*)::numeric / NULLIF(COUNT(*) FILTER (WHERE gecti),0), 1) AS azalma_katsayisi
FROM k
"""

CHANNEL_BREAKDOWN_SQL = f"""
WITH k AS ({decision_sql()})
SELECT kanal, COUNT(*) AS gecen
FROM k, unnest(kanallar) AS kanal
WHERE gecti GROUP BY 1 ORDER BY 2 DESC
"""

#: İran gaz kesintisi penceresi — recall çıpası. Bu id aralığı sitemap'ten
#: ölçüldü (2022-01-15 -> 2022-02-10, 440 haber; planın §2.4'teki sayıyla birebir).
IRAN_WINDOW = (46432, 46892)

#: Planın §2.4'te olay olarak saydığı + fizibilitede tespit edilen haberler.
#: Bunların HEPSİ geçmek zorunda; biri düşerse filtre fazla agresif.
IRAN_MUST_PASS: dict[int, str] = {
    46514: "İran'dan gaz akışı durdu",
    46521: "İran'daki bir arıza sanayi tesislerini etkileyecek",
    46533: "Elektrik fiyatı tavana vurdu",
    46596: "Spot elektrik tavan fiyatı Şubat'ta üçe katladı",   # azami fiyat limiti değişikliği
    46621: "Botaş'tan doğal gaz arzına ilişkin açıklama",
    46635: "EPDK asgari doğalgaz stok miktarını belirledi",
    46652: "İşte sanayiye elektrik kısıtlamasının sona ereceği tarih",
    46653: "İran'dan sınırlı gaz akışı başladı",
    46682: "Sanayiciye doğal gaz kısıtı azaltıldı",
    46689: "Spot elektrik Şubat'a tavandan girdi",
    46702: "Doğalgazdan elektrik üretme maliyeti arttı",
    46829: "Sanayiye gaz kısıtı sonlandırılıyor",
}


# ---------------------------------------------------------------------------- CLI

def _main() -> int:
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sqlalchemy import text
    from db.connection import get_db_engine

    ap = argparse.ArgumentParser(description="Kural bazlı ön filtre kalibrasyonu")
    ap.add_argument("--calibrate", action="store_true", help="hacim raporu")
    ap.add_argument("--recall", action="store_true", help="İran penceresi recall testi")
    ap.add_argument("--sample", type=int, metavar="N", help="geçen N haberi listele")
    ap.add_argument("--rejected", type=int, metavar="N", help="düşen N haberi listele")
    a = ap.parse_args()
    if not any((a.calibrate, a.recall, a.sample, a.rejected)):
        a.calibrate = a.recall = True

    eng = get_db_engine()

    if a.calibrate:
        with eng.connect() as c:
            r = c.execute(text(CALIBRATE_SQL)).one()
            print("=== HACİM ===")
            print(f"  toplam           {r.toplam:>7,}".replace(",", "."))
            print(f"  A: dışlandı      {r.dislandi:>7,}".replace(",", "."))
            print(f"  C: sinyal yok    {r.sinyal_yok:>7,}".replace(",", "."))
            print(f"  GEÇEN            {r.gecen:>7,}  (%{r.gecen_pct})".replace(",", "."))
            print(f"  azalma           {r.azalma_katsayisi}×")
            if r.gecen:
                sn = 8
                print(f"  ~LLM süresi      {r.gecen*sn/3600:.1f} saat "
                      f"(haber başına {sn} sn varsayımı)")
            print("\n=== KANAL DAĞILIMI (geçenler) ===")
            for row in c.execute(text(CHANNEL_BREAKDOWN_SQL)):
                print(f"  {row.kanal:<18} {row.gecen:>6,}".replace(",", "."))

    if a.recall:
        with eng.connect() as c:
            k = c.execute(text(decision_sql(
                f"article_id BETWEEN {IRAN_WINDOW[0]} AND {IRAN_WINDOW[1]}"))).all()
            if not k:
                print("\n=== RECALL === İran penceresi henüz DB'de değil.")
                return 0
            durum = {r.article_id: r for r in k}
            gecen = sum(1 for r in k if r.gecti)
            print(f"\n=== RECALL — İran penceresi (id {IRAN_WINDOW[0]}-{IRAN_WINDOW[1]}) ===")
            print(f"  pencere {len(k)} haber, geçen {gecen} (%{100*gecen/len(k):.1f})")
            kayip = []
            for aid, ad in IRAN_MUST_PASS.items():
                r = durum.get(aid)
                if r is None:
                    print(f"  ?  {aid}  (DB'de yok)     {ad}")
                elif r.gecti:
                    print(f"  ✓  {aid}  {','.join(r.kanallar):<28} {ad}")
                else:
                    kayip.append(aid)
                    neden = "A'da dışlandı" if r.dislandi else "kanal sinyali yok"
                    print(f"  ✗  {aid}  {neden:<28} {ad}")
            n = len([1 for aid in IRAN_MUST_PASS if aid in durum])
            print(f"\n  recall: {n-len(kayip)}/{n}")
            if kayip:
                print(f"  !! FİLTRE FAZLA AGRESİF — kaybedilen: {kayip}")
                return 1

    if a.sample:
        with eng.connect() as c:
            print(f"\n=== GEÇEN ÖRNEKLER ({a.sample}) ===")
            for r in c.execute(text(
                    f"WITH k AS ({decision_sql()}) SELECT * FROM k WHERE gecti "
                    f"ORDER BY random() LIMIT {a.sample}")):
                print(f"  {r.article_id}  {','.join(r.kanallar):<24} {r.title[:64]}")

    if a.rejected:
        with eng.connect() as c:
            print(f"\n=== DÜŞEN ÖRNEKLER ({a.rejected}) — recall riski burada ===")
            for r in c.execute(text(
                    f"WITH k AS ({decision_sql()}) SELECT * FROM k "
                    f"WHERE NOT gecti AND NOT dislandi ORDER BY random() LIMIT {a.rejected}")):
                print(f"  {r.article_id}  {r.section or '-':<18} {r.title[:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
