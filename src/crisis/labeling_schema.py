"""Haber etiketleme şeması — Kriz & Olay İstihbarat Sistemi.

Tek doğruluk kaynağı. Buradaki enum'lardan hem JSON Schema (MLX / outlines için)
hem GBNF grameri (llama.cpp için) türetilir, böylece iki artefakt birbirinden
kayamaz.

Tasarım gerekçeleri (bkz. CRISIS_ANALYSIS_PLAN.md):

  §3.2 — Kategori değil **etki kanalı**. `Geopolitical/Gas_Supply` gibi bir
         sınıflandırma fiyata *nasıl* geçtiğini söylemiyor.

  §3.3 — **Doz veriden, sebep haberden.** Modelden miktar çıkarmasını istemiyoruz;
         doz zaten `kgup_gas_mw`, tüketim ve fiyat serilerinde var. Modelden
         istenen üç şey: sebep etiketi (kanal), ileriye dönük duyuru mu,
         ve varsa **ileriye dönük** miktar/süre. Hepsi null olabilir —
         null gelmesi analizi çökertmez.

  §7   — `published_at` ile `effective_at` ayrı tutulur. 31 Ocak 2022 haberi
         09:08'de yayınlandı ama değişiklik 08:00'de yürürlüğe girdi.

Şemadan bilinçli olarak ÇIKARILANLAR:

  `aktorler` — planın silver taslağında var ama `bronze.news_raw.keywords`
      zaten editörün elle girdiği kurum/kişi etiketlerini bedava veriyor
      ("Enerji ve Tabii Kaynaklar Bakanlığı", "TÜREB", "BOTAŞ"). Modelden
      tekrar istemek üretim token'ı harcar ve yeni bir hata yolu açar.

  `guven` — 8B bir modelin öz-güven skoru zayıf kalibre. Yerine tek sürekli
      `alaka_skoru` var: planın §6 doğrulama protokolü "LLM'in alaka skoru ile
      gerçek anormal fiyat sapması korelasyonu"nu ölçmeyi şart koşuyor ve bu
      sürekli bir değişken gerektiriyor. Eşik kodda tunable kalır, modelin
      çıktısına gömülmez.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------- enum'lar


class Kanal(str, Enum):
    """Olayın fiyata geçtiği mekanizma. Sıra prompt'ta da bu sırayla anlatılır."""

    FUEL_COST = "fuel_cost"              # gaz/kömür/petrol fiyatı, BOTAŞ tarifesi
    SUPPLY_CAPACITY = "supply_capacity"  # santral arızası, bakım, kuraklık, ithalat kesintisi
    DEMAND = "demand"                    # hava, tatil, sanayi kısıntısı, afet
    REGULATION = "regulation"            # tavan fiyat, piyasa kuralı, tarife
    FX_MACRO = "fx_macro"                # kur şoku, enflasyon
    RENEWABLE = "renewable"              # kapasite girişi, aşırı arz


class Yon(str, Enum):
    """Olayın **kendi mekanizmasının** fiyatı ittiği yön.

    Piyasanın o gün ne yaptığı değil — olayın izole etkisi. Sanayi kısıntısı
    talebi düşürür, yani mekanik olarak `ASAGI`; kıtlığın *belirtisi* olması
    bunu değiştirmez. Bu ayrım olmadan model bağlamla mekanizmayı karıştırıyor.
    """

    YUKARI = "yukari"
    ASAGI = "asagi"
    BELIRSIZ = "belirsiz"


class Birim(str, Enum):
    """İleriye dönük miktarın birimi. Kapalı liste — serbest metin birim analizi bozar."""

    YUZDE = "yuzde"        # kısıntı oranı, indirim
    GUN = "gun"            # duyurulmuş süre
    MW = "mw"              # kapasite
    MCM_GUN = "mcm_gun"    # milyon m³/gün — gaz arzı
    TL_MWH = "tl_mwh"
    USD_MWH = "usd_mwh"
    ORAN = "oran"          # birimsiz çarpan (ör. kontratın 1/3'ü -> 0.33)


# ---------------------------------------------------------------------------- şema

#: Alan sırası **sabittir** ve gramerde de bu sırayla zorlanır. İki faydası var:
#: arama uzayı küçülür (8B model için kayda değer), ve akış hâlinde ayrıştırma
#: önemsizleşir.
FIELD_ORDER = (
    "alaka_skoru",
    "kanal",
    "yon",
    "ileriye_donuk",
    "etki_baslangici",
    "sure_gun",
    "miktar",
    "ozet",
)

OZET_MAX = 140

#: Hedef haberin gövde sınırı. Medyan 1.175 karakter; kuyruk uzun ve uzun
#: gövdeler prefill'i şişirmekten başka bir şey yapmıyor.
BODY_LIMIT = 2400

#: Few-shot örneklerinin gövde sınırı — kasten daha dar. Örnekler etiket
#: eşlemesini öğretiyor, okuma pratiği yaptırmıyor. Bu blok her haberde
#: birebir aynı kaldığı için prefix cache'e girer; küçük tutmak her koşuda
#: kazanç değil, ama cache ısınmasını ve bellek baskısını azaltır.
EXAMPLE_BODY_LIMIT = 900

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(FIELD_ORDER),
    "properties": {
        "alaka_skoru": {
            "type": "number", "minimum": 0, "maximum": 1,
            "description": "Bu haber Türkiye elektrik piyasasını etkileyen bir OLAY mı? "
                           "0 = ilgisiz/kurumsal gürültü, 1 = doğrudan piyasa olayı.",
        },
        "kanal": {
            "type": ["string", "null"], "enum": [k.value for k in Kanal] + [None],
            "description": "Etki kanalı. Alakasızsa null.",
        },
        "yon": {
            "type": "string", "enum": [y.value for y in Yon],
            "description": "Olayın kendi mekanizmasının fiyatı ittiği yön.",
        },
        "ileriye_donuk": {
            "type": "boolean",
            "description": "Haber, GELECEĞE dair bir duyuru mu içeriyor? "
                           "Olmuş bitmiş bir olayın raporu ise false.",
        },
        "etki_baslangici": {
            "type": ["string", "null"],
            "pattern": r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$",
            "description": "Değişikliğin yürürlüğe girdiği an, metinde AÇIKÇA yazıyorsa. "
                           "Yayın anından farklı olabilir. Yazmıyorsa null.",
        },
        "sure_gun": {
            "type": ["integer", "null"], "minimum": 1, "maximum": 9999,
            "description": "Duyurulmuş süre, gün. Metinde yazmıyorsa null.",
        },
        "miktar": {
            "type": ["object", "null"], "additionalProperties": False,
            "required": ["deger", "birim"],
            "properties": {
                "deger": {"type": "number"},
                "birim": {"type": "string", "enum": [b.value for b in Birim]},
            },
            "description": "İleriye dönük miktar. Geçmişi anlatan sayı BURAYA YAZILMAZ.",
        },
        "ozet": {
            "type": "string", "maxLength": OZET_MAX,
            "description": "Tek cümle, ne olduğu. İnsan gözden geçirmesi için.",
        },
    },
}


# ---------------------------------------------------------------------------- GBNF

def build_gbnf() -> str:
    """llama.cpp için grameri enum'lardan üretir.

    Yerel bir gecelik koşuda gramer kısıtı pazarlığa kapalı: onsuz çıktının
    %10-15'i ayrıştırılamaz hâlde gelir ve sabahı hangi satırın bozuk olduğunu
    ayıklamakla geçirirsin.
    """
    def alts(values, quoted=True) -> str:
        return " | ".join(f'"\\"{v}\\""' if quoted else f'"{v}"' for v in values)

    return f'''# ÜRETİLMİŞ DOSYA — elle düzenleme. Kaynak: src/crisis/labeling_schema.py
# Yeniden üret: python -m src.crisis.labeling_schema --write-grammar

root ::= "{{" nl
  "  \\"alaka_skoru\\": " prob "," nl
  "  \\"kanal\\": " kanal "," nl
  "  \\"yon\\": " yon "," nl
  "  \\"ileriye_donuk\\": " bool "," nl
  "  \\"etki_baslangici\\": " (tarih | null) "," nl
  "  \\"sure_gun\\": " (tamsayi | null) "," nl
  "  \\"miktar\\": " (miktar | null) "," nl
  "  \\"ozet\\": " ozet nl
  "}}"

kanal   ::= {alts([k.value for k in Kanal])} | null
yon     ::= {alts([y.value for y in Yon])}
birim   ::= {alts([b.value for b in Birim])}

miktar  ::= "{{\\"deger\\": " sayi ", \\"birim\\": " birim "}}"

# 0, 1, ya da 0.NN — model ara değer uydurmasın diye dar tutuldu
prob    ::= "0" | "1" | "0." [0-9] [0-9]?
sayi    ::= "-"? [0-9]+ ("." [0-9]+)?
tamsayi ::= [1-9] [0-9]? [0-9]? [0-9]?
bool    ::= "true" | "false"
null    ::= "null"

# ISO tarih, opsiyonel saat: "2022-01-31" veya "2022-01-31T08:00"
tarih   ::= "\\"" d d d d "-" d d "-" d d ("T" d d ":" d d)? "\\""
d       ::= [0-9]

# Özet: kaçış gerektiren karakterler dışlandı, uzunluk Python tarafında kırpılır
ozet    ::= "\\"" [^"\\\\\\n]+ "\\""

nl      ::= "\\n"
'''


GRAMMAR_PATH = Path(__file__).resolve().parent / "grammar" / "news_event.gbnf"


def write_grammar(path: Path = GRAMMAR_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_gbnf(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- doğrulama

class LabelError(ValueError):
    """Model çıktısı şemaya uymuyor."""


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$")


def validate(obj: Any) -> dict[str, Any]:
    """Model çıktısını doğrular ve normalize eder.

    Gramer zaten sözdizimini garanti ediyor; buradaki kontroller anlamsal
    tutarlılık için — gramer "ileriye_donuk=false ama sure_gun=10" gibi bir
    çelişkiyi yakalayamaz.
    """
    if not isinstance(obj, dict):
        raise LabelError(f"nesne beklendi, {type(obj).__name__} geldi")

    eksik = [f for f in FIELD_ORDER if f not in obj]
    if eksik:
        raise LabelError(f"eksik alan: {', '.join(eksik)}")
    fazla = set(obj) - set(FIELD_ORDER)
    if fazla:
        raise LabelError(f"bilinmeyen alan: {', '.join(sorted(fazla))}")

    out: dict[str, Any] = {}

    skor = obj["alaka_skoru"]
    if not isinstance(skor, (int, float)) or not 0 <= skor <= 1:
        raise LabelError(f"alaka_skoru 0-1 aralığında sayı olmalı: {skor!r}")
    out["alaka_skoru"] = round(float(skor), 2)

    kanal = obj["kanal"]
    if kanal is not None and kanal not in {k.value for k in Kanal}:
        raise LabelError(f"geçersiz kanal: {kanal!r}")
    out["kanal"] = kanal

    if obj["yon"] not in {y.value for y in Yon}:
        raise LabelError(f"geçersiz yon: {obj['yon']!r}")
    out["yon"] = obj["yon"]

    if not isinstance(obj["ileriye_donuk"], bool):
        raise LabelError(f"ileriye_donuk bool olmalı: {obj['ileriye_donuk']!r}")
    out["ileriye_donuk"] = obj["ileriye_donuk"]

    eb = obj["etki_baslangici"]
    if eb is not None and not _DATE_RE.match(str(eb)):
        raise LabelError(f"etki_baslangici ISO tarih olmalı: {eb!r}")
    out["etki_baslangici"] = eb

    sg = obj["sure_gun"]
    if sg is not None and not (isinstance(sg, int) and 1 <= sg <= 9999):
        raise LabelError(f"sure_gun 1-9999 tamsayı olmalı: {sg!r}")
    out["sure_gun"] = sg

    mik = obj["miktar"]
    if mik is not None:
        if not isinstance(mik, dict) or set(mik) != {"deger", "birim"}:
            raise LabelError(f"miktar {{deger, birim}} olmalı: {mik!r}")
        if not isinstance(mik["deger"], (int, float)):
            raise LabelError(f"miktar.deger sayı olmalı: {mik['deger']!r}")
        if mik["birim"] not in {b.value for b in Birim}:
            raise LabelError(f"geçersiz miktar.birim: {mik['birim']!r}")
        mik = {"deger": float(mik["deger"]), "birim": mik["birim"]}
    out["miktar"] = mik

    ozet = obj["ozet"]
    if not isinstance(ozet, str) or not ozet.strip():
        raise LabelError("ozet boş olamaz")
    out["ozet"] = ozet.strip()[:OZET_MAX]

    # --- anlamsal tutarlılık: gramerin göremediği çelişkiler ---
    if out["kanal"] is None and out["alaka_skoru"] >= 0.5:
        raise LabelError("alaka_skoru >= 0.5 ise kanal zorunlu")
    if not out["ileriye_donuk"] and (out["sure_gun"] is not None or out["miktar"] is not None):
        raise LabelError("ileriye_donuk=false iken sure_gun/miktar dolu olamaz "
                         "(geçmişi anlatan sayı miktar alanına yazılmaz)")
    return out


def parse_output(raw: str) -> dict[str, Any]:
    """Modelin ham metnini JSON'a çevirip doğrular. Gramer varsa bu hiç patlamamalı."""
    raw = raw.strip()
    if not raw.startswith("{"):  # gramer kapalıysa model önüne laf edebilir
        i, j = raw.find("{"), raw.rfind("}")
        if i == -1 or j <= i:
            raise LabelError(f"çıktıda JSON yok: {raw[:120]!r}")
        raw = raw[i:j + 1]
    try:
        return validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise LabelError(f"JSON ayrıştırılamadı: {exc}") from exc


# ---------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = """Türkiye elektrik piyasası haberlerini etiketleyen bir analistsin.

Her haber için SADECE JSON döndür. Açıklama yazma, JSON dışına çıkma.

## Görev

Haberin Türkiye elektrik toptan fiyatını (PTF) etkileyen bir OLAY olup olmadığını
belirle; olaysa mekanizmasını etiketle.

## alaka_skoru

0.0-0.2  Piyasayla ilgisiz: şirket haberi, atama, yurtdışı kurulum, elektrikli
         araç haberi, ihale sonucu, lisans işlemi.
0.0      Günlük fiyat raporu ("Spot elektrik fiyatı DD.MM.YYYY için X TL").
         Bu bir olay DEĞİL, verinin kendisi — bizde zaten var.
0.3-0.6  Dolaylı: sektörel eğilim, kapasite planı, uzun vadeli yatırım.
0.7-1.0  Doğrudan piyasa olayı: arz kesintisi, tarife değişikliği, kısıntı
         kararı, santral arızası, mevzuat değişikliği, yakıt fiyatı kararı.

## kanal (alaka_skoru >= 0.5 ise zorunlu)

fuel_cost        Gaz/kömür/petrol fiyatı, BOTAŞ tarifesi, YEKDEM maliyeti
supply_capacity  Arz tarafı: santral arızası/bakımı, ithalat kesintisi, kuraklık
demand           Talep tarafı: hava, tatil, sanayi kısıntısı, afet
regulation       Piyasa kuralı: azami fiyat limiti, tarife, piyasa yapısı
fx_macro         Kur şoku, enflasyon, faiz
renewable        Yenilenebilir kapasite girişi, aşırı arz, YEKDEM kapsamı

## yon

Olayın KENDİ MEKANİZMASININ fiyatı ittiği yön. Piyasanın o gün ne yaptığı değil.
Örnek: sanayiye kısıntı talebi düşürür -> "asagi". Kıtlığın belirtisi olması
bunu değiştirmez, mekanizmaya bak.

## ileriye_donuk — en kritik alan

true  ise haber GELECEĞE dair bir taahhüt/duyuru içeriyor: "10 gün süreyle
      durdurulacak", "1 Şubat'tan itibaren uygulanacak", "kısıntı %20'ye
      düşürülecek". Verinin o an bilemediği bir şey söylüyor.
false ise olmuş bitmiş bir şeyin raporu: "fiyat tavana çıktı", "üretim düştü".
      Bunlar alakalı olabilir ama veride zaten görünürler.

sure_gun ve miktar SADECE ileriye_donuk=true iken doldurulur. Geçmişi anlatan
sayıyı miktar alanına YAZMA — dozu veriden ölçüyoruz, haberden değil.

## etki_baslangici

Metinde yürürlük anı AÇIKÇA yazıyorsa yaz ("31.01.2022 saat 08:00 itibarıyla"
-> "2022-01-31T08:00"). Yayın tarihinden farklı olabilir. Yazmıyorsa null."""


@dataclass(frozen=True)
class GoldExample:
    """Elle yazılmış altın cevap. Metin DB'den çekilir, cevap burada sabit."""

    etiket: str
    article_id: Optional[int]          # bilinen id (negatif örnekler)
    title_like: Optional[str]          # id bilinmiyorsa başlık kalıbı (İran haberleri)
    beklenen: dict[str, Any]
    gerekce: str                       # neden bu cevap — insan gözden geçirmesi için


#: Few-shot bloğu. Beş örnek bilinçli: negatif örnekler pozitiflerden çok,
#: çünkü korpusun ~%90'ı gürültü ve model varsayılan olarak her şeye
#: "alakalı" demeye meyilli.
GOLD_EXAMPLES: tuple[GoldExample, ...] = (
    GoldExample(
        etiket="günlük fiyat raporu — olay değil",
        article_id=40687, title_like=None,
        beklenen={
            "alaka_skoru": 0.0, "kanal": None, "yon": "belirsiz",
            "ileriye_donuk": False, "etki_baslangici": None,
            "sure_gun": None, "miktar": None,
            "ozet": "Ertesi gün için gün öncesi piyasa takas fiyatı duyurusu.",
        },
        gerekce="Verinin kendisi. Her gün bir tane var, korpusun ~%7'si. "
                "Kural filtresi bunu başlık kalıbıyla zaten elemeli.",
    ),
    GoldExample(
        etiket="ilgisiz — elektrikli araç",
        article_id=40695, title_like=None,
        beklenen={
            "alaka_skoru": 0.0, "kanal": None, "yon": "belirsiz",
            "ileriye_donuk": False, "etki_baslangici": None,
            "sure_gun": None, "miktar": None,
            "ozet": "Otomotiv üreticisinin teslimat hedefi haberi.",
        },
        gerekce="Bölüm 'Elektrik' ama piyasayla ilgisi yok. Bölüm filtresinin "
                "tek başına yetmediğinin kanıtı.",
    ),
    GoldExample(
        etiket="ileriye dönük arz kesintisi — süre + miktar",
        article_id=None, title_like="İran gazına teknik arıza engeli",
        beklenen={
            "alaka_skoru": 1.0, "kanal": "supply_capacity", "yon": "yukari",
            "ileriye_donuk": True, "etki_baslangici": None,
            "sure_gun": 10, "miktar": {"deger": 0.33, "birim": "oran"},
            "ozet": "İran gaz arzı teknik arıza nedeniyle 10 gün süreyle "
                    "kontrat miktarının üçte birine düşürüldü.",
        },
        gerekce="Kanonik ileriye dönük olay: 20 Ocak'ta yayınlanan 'arz 10 gün "
                "durdurulacak' bilgisi, verinin 20 Ocak'ta bilemediği şey. "
                "Haberin veriyi yendiği tek yer bu (plan §3.3).",
    ),
    GoldExample(
        etiket="alakalı ama geriye dönük rapor",
        article_id=None, title_like="Doğalgaz kesintisi elektrik fiyatını tavana çıkardı",
        beklenen={
            "alaka_skoru": 0.8, "kanal": "supply_capacity", "yon": "yukari",
            "ileriye_donuk": False, "etki_baslangici": None,
            "sure_gun": None, "miktar": None,
            "ozet": "Gaz kesintisinin elektrik fiyatını azami limite taşıdığı "
                    "raporlandı.",
        },
        gerekce="Yüksek alaka ama ileriye_donuk=false: fiyatın tavana çıktığı "
                "zaten raw_mcp_hourly'de görünüyor. Modelin en sık karıştırdığı "
                "ayrım bu, o yüzden few-shot'ta.",
    ),
    GoldExample(
        etiket="yürürlük anı yayın anından farklı",
        article_id=None, title_like="Sanayiciye gaz kısıtı oranı",
        beklenen={
            "alaka_skoru": 0.9, "kanal": "demand", "yon": "yukari",
            "ileriye_donuk": True, "etki_baslangici": "2022-01-31T08:00",
            "sure_gun": None, "miktar": {"deger": 20, "birim": "yuzde"},
            "ozet": "Sanayiye uygulanan gaz kısıntısı oranı %40'tan %20'ye "
                    "indirildi, 31 Ocak 08:00 itibarıyla.",
        },
        gerekce="etki_baslangici != published_at (haber 09:08'de, değişiklik "
                "08:00'de). yon=yukari çünkü kısıntının GEVŞEMESİ talebi "
                "yükseltir — mekanizmaya bakılıyor, bağlama değil.",
    ),
)


def to_grammar_json(obj: dict[str, Any]) -> str:
    """Nesneyi GBNF'in ürettiği metnin BİREBİR aynısına çevirir.

    Neden gerekli: `json.dumps` few-shot'ı tek satır ve `1.0` / `0.0` biçiminde
    yazar; gramer ise çok satırlı ve `prob ::= "0" | "1" | "0." [0-9] [0-9]?`
    yüzünden `1.0`'ı reddeder. İkisi ayrışırsa few-shot modeli gramerin
    yasakladığı bir biçime doğru çeker — kısıt kazanır ama model boşuna zorlanır.
    Öz-kontrol bu fonksiyonun çıktısının gramerden geçtiğini doğrular.
    """
    def prob(v: float) -> str:
        if v in (0, 1):
            return str(int(v))
        return f"{v:.2f}".rstrip("0")           # 0.90 -> "0.9", 0.33 -> "0.33"

    def sayi(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else repr(round(float(v), 4))

    def q(t: str) -> str:
        return json.dumps(t, ensure_ascii=False)

    satirlar = [
        f'  "alaka_skoru": {prob(obj["alaka_skoru"])},',
        f'  "kanal": {q(obj["kanal"]) if obj["kanal"] else "null"},',
        f'  "yon": {q(obj["yon"])},',
        f'  "ileriye_donuk": {"true" if obj["ileriye_donuk"] else "false"},',
        f'  "etki_baslangici": {q(obj["etki_baslangici"]) if obj["etki_baslangici"] else "null"},',
        f'  "sure_gun": {obj["sure_gun"] if obj["sure_gun"] is not None else "null"},',
    ]
    m = obj["miktar"]
    satirlar.append(
        f'  "miktar": ' + ("null" if m is None
                           else '{"deger": %s, "birim": %s}' % (sayi(m["deger"]), q(m["birim"]))) + ","
    )
    satirlar.append(f'  "ozet": {q(obj["ozet"])}')
    return "{\n" + "\n".join(satirlar) + "\n}"


def format_article(title: str, section: Optional[str], published_at: str,
                   body: str, body_limit: int = BODY_LIMIT) -> str:
    """Haberi prompt bloğuna çevirir. Gövde kırpılır — medyan 1.175 karakter,
    kuyruk uzun ve uzun gövdeler prefill'i şişirmekten başka bir şey yapmıyor."""
    govde = body if len(body) <= body_limit else body[:body_limit].rsplit(" ", 1)[0] + " […]"
    return (f"Tarih: {published_at}\n"
            f"Bölüm: {section or '-'}\n"
            f"Başlık: {title}\n"
            f"Metin: {govde}")


def build_prompt(article_block: str, examples: list[tuple[str, dict[str, Any]]]) -> str:
    """Sistem promptu + few-shot + hedef haber. Sabit önek prefix cache'e girer."""
    parts = [SYSTEM_PROMPT, ""]
    for blok, cevap in examples:
        parts += ["## Örnek", blok, "JSON:", to_grammar_json(cevap), ""]
    parts += ["## Etiketle", article_block, "JSON:"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------- CLI

_GRAMMAR_LINE_RE = re.compile(
    r'^\{\n'
    r'  "alaka_skoru": (?:0|1|0\.[0-9][0-9]?),\n'
    r'  "kanal": (?:"(?:fuel_cost|supply_capacity|demand|regulation|fx_macro|renewable)"|null),\n'
    r'  "yon": "(?:yukari|asagi|belirsiz)",\n'
    r'  "ileriye_donuk": (?:true|false),\n'
    r'  "etki_baslangici": (?:"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?"|null),\n'
    r'  "sure_gun": (?:[1-9][0-9]{0,3}|null),\n'
    r'  "miktar": (?:\{"deger": -?[0-9]+(?:\.[0-9]+)?, "birim": '
    r'"(?:yuzde|gun|mw|mcm_gun|tl_mwh|usd_mwh|oran)"\}|null),\n'
    r'  "ozet": "[^"\\\n]+"\n'
    r'\}$'
)


def _assert_grammar_conform(metin: str, etiket: str) -> None:
    """GBNF'in kabul edeceği biçimi bağımsız bir regex ile doğrular.

    Gerçek doğrulama llama.cpp'nin gramer ayrıştırıcısıyla yapılmalı; bu kontrol
    serializer ile gramerin ayrışmasını yakalamak için — ikisi de aynı enum
    listesinden üretildiği için sürüklenme buradan görülür.
    """
    if not _GRAMMAR_LINE_RE.match(metin):
        raise AssertionError(f"gramer biçimine uymuyor [{etiket}]:\n{metin}")


def _self_check() -> None:
    """Altın cevapların hepsi kendi doğrulayıcısından geçmeli."""
    for ex in GOLD_EXAMPLES:
        try:
            validate(dict(ex.beklenen))
        except LabelError as exc:
            raise AssertionError(f"altın cevap şemadan geçmedi [{ex.etiket}]: {exc}") from exc

    # Doğrulayıcı gerçekten reddediyor mu?
    kotu = [
        ({**GOLD_EXAMPLES[2].beklenen, "ileriye_donuk": False}, "geçmiş + miktar çelişkisi"),
        ({**GOLD_EXAMPLES[0].beklenen, "alaka_skoru": 0.9}, "yüksek skor + kanal null"),
        ({**GOLD_EXAMPLES[2].beklenen, "kanal": "gas_supply"}, "geçersiz kanal"),
        ({**GOLD_EXAMPLES[4].beklenen, "etki_baslangici": "31.01.2022"}, "ISO olmayan tarih"),
    ]
    for obj, ad in kotu:
        try:
            validate(obj)
        except LabelError:
            continue
        raise AssertionError(f"doğrulayıcı yakalamalıydı: {ad}")

    # parse_output uçtan uca
    ham = json.dumps(GOLD_EXAMPLES[2].beklenen, ensure_ascii=False)
    assert parse_output(f"Elbette, işte JSON:\n{ham}\n")["sure_gun"] == 10

    # Few-shot biçimi gramerden geçiyor mu, ve tur-gidiş-dönüş kayıpsız mı?
    for ex in GOLD_EXAMPLES:
        metin = to_grammar_json(ex.beklenen)
        _assert_grammar_conform(metin, ex.etiket)
        geri = parse_output(metin)
        beklenen = validate(dict(ex.beklenen))
        if geri != beklenen:
            fark = {k: (beklenen[k], geri[k]) for k in beklenen if beklenen[k] != geri[k]}
            raise AssertionError(f"tur kaybı [{ex.etiket}]: {fark}")

    print(f"öz-kontrol OK — {len(GOLD_EXAMPLES)} altın cevap gramer-uyumlu ve "
          f"kayıpsız, {len(kotu)} negatif test geçti")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Etiketleme şeması araçları")
    ap.add_argument("--write-grammar", action="store_true", help="GBNF dosyasını üret")
    ap.add_argument("--show-schema", action="store_true", help="JSON Schema'yı yazdır")
    ap.add_argument("--show-prompt", action="store_true", help="sistem promptunu yazdır")
    ap.add_argument("--check", action="store_true", help="öz-kontrol koştur")
    a = ap.parse_args()

    if a.write_grammar:
        print(f"yazıldı: {write_grammar()}")
    if a.show_schema:
        print(json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2))
    if a.show_prompt:
        print(SYSTEM_PROMPT)
    if a.check or not any(vars(a).values()):
        _self_check()
