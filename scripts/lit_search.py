#!/usr/bin/env python3
"""
Sistematik literatür taraması — anahtar kelime yerine ALINTI GRAFİĞİ.

Sorun: anahtar kelime araması her seferinde başka bir şey döndürüyor, tekrar
üretilemiyor ve "hangisi bakmaya değer" sorusunu cevaplamıyor.

Bu araç iki soruyu AYIRIYOR:

  KAPSAM  — bu iş Türkiye elektrik fiyatları hakkında mı?   (evet/hayır)
  SAĞLAMLIK — kapsam içindekiler arasında hangisi daha sağlam?  (sıralama)

Konu etiketi (tahmin / merit-order / oynaklık / politika / geçiş) sadece
ETİKET; sıralamaya girmiyor. Böylece "hangi konuyu merak ediyorsam o üste
çıkar" sapması olmuyor — sıralama konudan bağımsız.

Sağlamlık, tam metin okumadan görülebilen dört işaretin yüzdelik bileşimi:
  · atıf hızı (atıf/yıl)      — alanın işi kullanıp kullanmadığı
  · derginin h-endeksi        — hakemlik eşiği için kaba vekil
  · tohumlarla bağ sayısı     — konuşmanın merkezinde mi, kenarında mı
  · güncellik                 — 2021+ rejimini görüyor mu

Bunlar VEKİL. Gerçek sağlamlık (test seti uzunluğu, bölme biçimi, girdilerin
tahmin anında mevcut olup olmadığı) ancak tam metinde görülüyor — `extract`
komutu o formu sabitliyor, çünkü karşılaştırılabilir tablo ancak aynı alanlar
her seferinde doldurulunca çıkıyor.

Her sorgu, tarihi, sonuç sayısı ve her eleme kararı gerekçesiyle DB'ye
yazılıyor (PRISMA). Tarama 3 ay sonra tekrar koşulup FARK görülebilir.

Kaynak: OpenAlex (anahtar gerekmez). Sadece stdlib.

Kullanım:
    python scripts/lit_search.py seed
    python scripts/lit_search.py sweep
    python scripts/lit_search.py expand --depth 2
    python scripts/lit_search.py venues        # dergi h-endekslerini çek
    python scripts/lit_search.py rank
    python scripts/lit_search.py report --top 30
    python scripts/lit_search.py screen W123 include "gerekçe"
    python scripts/lit_search.py stats
"""

import argparse
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "literature" / "lit.db"
MAILTO = "beratc237@gmail.com"
API = "https://api.openalex.org/works"
SOURCES_API = "https://api.openalex.org/sources"
THIS_YEAR = datetime.now().year

# --------------------------------------------------------------------------
# Tohumlar: ilgili olduğu DOĞRULANMIŞ işler. Ağırlık Türkiye + elektrik fiyatı.
# Son ikisi yöntem tohumu — kapsam dışı kalacaklar ama grafiği açıyorlar.
# --------------------------------------------------------------------------
SEEDS = [
    ("doi:10.30798/makuiibf.1097686", "Arifoğlu & Kandemir 2022 — DL, test 28 gün"),
    ("doi:10.35860/iarej.1820591", "Akpınar 2026 — quantile GBR, PICP 0.737"),
    ("doi:10.38088/jise.1738364", "Özdemir & Yılmaz 2026 — EGPR, test 25 nokta"),
    ("doi:10.1016/j.eswa.2023.120026", "ESWA 2023 — TEDSE transformer, 2017-2021"),
    ("title:Merit-order of dispatchable and variable renewable energy sources in Turkey's day-ahead electricity market", "Gökgöz & Yücel 2024"),
    ("title:Variable renewable energy technologies in the Turkish electricity market quantile regression merit-order", "Türkiye merit-order kantil"),
    ("title:Price spikes temporary price caps and welfare effects of regulatory interventions on wholesale electricity markets", "Energy Policy 2022 — Türkiye tavan"),
    ("title:On the determination of European day ahead electricity prices The Turkish case", "Türkiye GÖP fiyat oluşumu"),
    ("title:Measuring the long-term impact of wind run-of-river solar renewable energy alternatives on market clearing prices", "Türkiye yenilenebilir → PTF"),
    ("title:Forecasting the day-ahead price in electricity balancing and settlement market of Turkey", "Türkiye DGP fiyat tahmini"),
    # Yöntem tohumları (kapsam dışı; grafiği genişletmek için)
    ("doi:10.1080/13504851.2024.2425834", "İberya istisnası — sentetik kontrol"),
    ("title:Navigating the crisis Fuel price caps in the Australian national wholesale electricity market", "Avustralya yakıt tavanı"),
]

# --------------------------------------------------------------------------
# KAPSAM: iki koşul birlikte sağlanmalı.
# --------------------------------------------------------------------------
SCOPE_GEO = [
    "turkey", "türkiye", "turkiye", "turkish", "epias", "epiaş", "exist",
    "epdk", "teias", "teiaş", "botas", "botaş",
]
SCOPE_TOPIC = [
    "electricity price", "power price", "market clearing price", "day-ahead",
    "day ahead", "spot price", "electricity market", "wholesale electricity",
    "energy price", "price forecast", "ptf", "mcp", "balancing market",
    "system marginal price", "electricity tariff",
]

# --------------------------------------------------------------------------
# KONU ETİKETLERİ — sadece kapsama alınmış işleri sınıflar. Skora GİRMEZ.
# Sıra önemli: ilk eşleşen kazanır.
# --------------------------------------------------------------------------
TOPICS = [
    ("politika/tavan", ["price cap", "price ceiling", "price limit", "maximum price",
                        "regulatory intervention", "price control", "subsid",
                        "market design", "regulation", "liberali"]),
    ("maliyet-geçişi", ["pass-through", "passthrough", "pass through", "fuel cost",
                        "gas price", "marginal cost", "cost of generation",
                        "fuel switching", "merit order", "merit-order"]),
    ("nedensellik", ["counterfactual", "synthetic control", "difference-in-differences",
                     "event study", "causal", "identification strategy",
                     "regression discontinuity"]),
    ("oynaklık", ["volatility", "garch", "structural break", "jump", "regime switch",
                  "risk premium", "value at risk"]),
    ("yenilenebilir-etkisi", ["renewable", "wind power", "solar", "hydropower",
                              "hydro", "penetration", "intermittent"]),
    ("tahmin", ["forecast", "prediction", "predicting", "lstm", "gru", "xgboost",
                "random forest", "neural network", "deep learning",
                "machine learning", "arima", "hybrid model", "estimation of"]),
]

REVIEW_HINTS = ["review", "survey", "systematic literature", "state of the art",
                "overview", "taxonomy"]

# --------------------------------------------------------------------------
# Anahtar kelime taraması — grafikten kopuk kümeleri yakalamak için.
# Türkiye + elektrik fiyatı ekseninde GENİŞ tutuldu.
# --------------------------------------------------------------------------
SWEEPS = [
    "Turkish electricity market price",
    "Turkey day-ahead electricity market clearing price",
    "Turkey electricity price forecasting",
    "Turkey electricity market liberalization reform price",
    "Turkey electricity price volatility structural break",
    "Turkey renewable energy merit order effect electricity price",
    "Turkey electricity market regulation price cap EPDK",
    "Turkey natural gas electricity generation cost price",
    "Turkey electricity market balancing system marginal price",
    "Turkey hydropower drought electricity price",
    "Turkey electricity demand price elasticity",
    "Turkish electricity market imbalance price bidding",
]


# ==========================================================================
# HTTP
# ==========================================================================

def api_get(url_base: str, params: dict, retries: int = 3) -> dict:
    params = {**params, "mailto": MAILTO}
    url = f"{url_base}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  ! API hatası: {exc}", file=sys.stderr)
                return {"results": [], "meta": {"count": 0}}
            time.sleep(2 * (attempt + 1))
    return {"results": [], "meta": {"count": 0}}


def unwrap_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        pos.extend((i, word) for i in idxs)
    pos.sort()
    return " ".join(w for _, w in pos)


def short_id(oa_id: str) -> str:
    return oa_id.rsplit("/", 1)[-1] if oa_id else ""


# ==========================================================================
# DB
# ==========================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY, doi TEXT, title TEXT, year INTEGER,
    venue TEXT, venue_id TEXT, work_type TEXT,
    cited_by INTEGER, n_refs INTEGER, is_oa INTEGER, oa_url TEXT,
    abstract TEXT, discovered_via TEXT, first_seen TEXT
);
CREATE TABLE IF NOT EXISTS edges (src TEXT, dst TEXT, PRIMARY KEY (src, dst));
CREATE TABLE IF NOT EXISTS seeds (id TEXT PRIMARY KEY, note TEXT);
CREATE TABLE IF NOT EXISTS venues (
    id TEXT PRIMARY KEY, name TEXT, h_index INTEGER,
    mean_citedness REAL, is_doaj INTEGER
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    q TEXT, ran_at TEXT, n_hits INTEGER, n_new INTEGER
);
CREATE TABLE IF NOT EXISTS screening (
    work_id TEXT PRIMARY KEY, decision TEXT, reason TEXT, decided_at TEXT
);
CREATE TABLE IF NOT EXISTS extraction (
    work_id TEXT PRIMARY KEY, data_period TEXT, test_set TEXT, split_type TEXT,
    inputs TEXT, cap_handled TEXT, metrics TEXT, notes TEXT, extracted_at TEXT
);
CREATE TABLE IF NOT EXISTS scores (
    work_id TEXT PRIMARY KEY, in_scope INTEGER, topic TEXT, is_review INTEGER,
    seed_links INTEGER, velocity REAL, venue_h INTEGER,
    p_velocity REAL, p_venue REAL, p_links REAL, p_recency REAL,
    strength REAL
);
CREATE TABLE IF NOT EXISTS expansion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    depth INTEGER, frontier_size INTEGER, added INTEGER, ran_at TEXT
);
"""


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


WORK_FIELDS = (
    "id,doi,title,display_name,publication_year,primary_location,type,"
    "cited_by_count,referenced_works,open_access,abstract_inverted_index"
)


def upsert_work(conn, w: dict, via: str) -> bool:
    wid = short_id(w.get("id", ""))
    if not wid:
        return False
    if conn.execute("SELECT 1 FROM works WHERE id=?", (wid,)).fetchone():
        return False
    src = (w.get("primary_location") or {}).get("source") or {}
    conn.execute(
        "INSERT INTO works (id,doi,title,year,venue,venue_id,work_type,cited_by,"
        "n_refs,is_oa,oa_url,abstract,discovered_via,first_seen) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            wid,
            (w.get("doi") or "").replace("https://doi.org/", ""),
            w.get("title") or w.get("display_name") or "",
            w.get("publication_year"),
            src.get("display_name") or "",
            short_id(src.get("id") or ""),
            w.get("type") or "",
            w.get("cited_by_count") or 0,
            len(w.get("referenced_works") or []),
            1 if (w.get("open_access") or {}).get("is_oa") else 0,
            (w.get("open_access") or {}).get("oa_url") or "",
            unwrap_abstract(w.get("abstract_inverted_index")),
            via, now(),
        ),
    )
    for ref in w.get("referenced_works") or []:
        conn.execute("INSERT OR IGNORE INTO edges (src,dst) VALUES (?,?)",
                     (wid, short_id(ref)))
    return True


# ==========================================================================
# Komutlar
# ==========================================================================

def cmd_seed(args) -> None:
    conn = get_db()
    for spec, note in SEEDS:
        kind, _, value = spec.partition(":")
        if kind == "doi":
            data = api_get(API, {"filter": f"doi:{value}", "select": WORK_FIELDS})
        else:
            data = api_get(API, {"search": value, "per-page": 1, "select": WORK_FIELDS})
        results = data.get("results") or []
        if not results:
            print(f"  ✖ ÇÖZÜLEMEDİ: {note}")
            continue
        w = results[0]
        wid = short_id(w["id"])
        upsert_work(conn, w, "seed")
        conn.execute("INSERT OR REPLACE INTO seeds (id,note) VALUES (?,?)", (wid, note))
        print(f"  ✓ {wid}  {w.get('publication_year')}  atıf={w.get('cited_by_count')}"
              f"\n      {(w.get('title') or '')[:88]}")
        time.sleep(0.15)
    conn.commit()
    print(f"\n{conn.execute('SELECT COUNT(*) FROM seeds').fetchone()[0]} tohum. Sıradaki: sweep")


def cmd_sweep(args) -> None:
    conn = get_db()
    queries = [args.query] if args.query else SWEEPS
    for q in queries:
        data = api_get(API, {
            "search": q, "per-page": args.per_page,
            "filter": f"from_publication_date:{args.since}-01-01",
            "select": WORK_FIELDS,
        })
        hits = data.get("meta", {}).get("count", 0)
        new = sum(upsert_work(conn, w, f"sweep:{q[:40]}") for w in data.get("results") or [])
        conn.execute("INSERT INTO queries (q,ran_at,n_hits,n_new) VALUES (?,?,?,?)",
                     (q, now(), hits, new))
        conn.commit()
        print(f"  {hits:>7} sonuç · {new:>3} yeni · {q}")
        time.sleep(0.2)
    print("\nSıradaki: expand")


def fetch_batch(base: str, ids: list[str], fields: str) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(ids), 50):
        data = api_get(base, {
            "filter": "openalex_id:" + "|".join(ids[i:i + 50]),
            "per-page": 50, "select": fields,
        })
        out.extend(data.get("results") or [])
        time.sleep(0.15)
    return out


def cmd_expand(args) -> None:
    conn = get_db()
    frontier = [r[0] for r in conn.execute("SELECT id FROM seeds")]
    if not frontier:
        sys.exit("Tohum yok. Önce: seed")

    for depth in range(1, args.depth + 1):
        print(f"\n─── Derinlik {depth} · sınır {len(frontier)} iş ───")
        added = 0

        refs: set[str] = set()
        for wid in frontier:
            refs.update(r[0] for r in conn.execute("SELECT dst FROM edges WHERE src=?", (wid,)))
        known = {r[0] for r in conn.execute("SELECT id FROM works")}
        missing = sorted(refs - known)
        if missing:
            print(f"  geri: {len(missing)} kaynak çekiliyor…")
            for w in fetch_batch(API, missing, WORK_FIELDS):
                added += upsert_work(conn, w, f"bwd:d{depth}")
            conn.commit()

        for wid in frontier:
            page, seen = 1, 0
            while page <= args.max_pages:
                data = api_get(API, {"filter": f"cites:{wid}", "per-page": 100,
                                     "page": page, "select": WORK_FIELDS})
                results = data.get("results") or []
                if not results:
                    break
                for w in results:
                    added += upsert_work(conn, w, f"fwd:{wid}")
                    seen += 1
                if len(results) < 100:
                    break
                page += 1
                time.sleep(0.2)
            if seen:
                print(f"  ileri: {wid} → {seen} atıf")
            conn.commit()
            time.sleep(0.15)

        conn.execute("INSERT INTO expansion_log (depth,frontier_size,added,ran_at) "
                     "VALUES (?,?,?,?)", (depth, len(frontier), added, now()))
        conn.commit()
        print(f"  → {added} yeni (toplam {conn.execute('SELECT COUNT(*) FROM works').fetchone()[0]})")

        if depth < args.depth:
            # Sonraki sınır: KAPSAM İÇİ ve en sağlam işler. Hepsini yürümek
            # kombinatoryal patlama; konu ayrımı burada YAPILMIYOR.
            cmd_venues(argparse.Namespace(quiet=True))
            cmd_rank(argparse.Namespace(quiet=True))
            frontier = [r[0] for r in conn.execute(
                "SELECT work_id FROM scores WHERE in_scope=1 "
                "ORDER BY strength DESC LIMIT ?", (args.frontier_limit,))]
            if not frontier:
                print("  (kapsam içi yeni iş yok, duruldu)")
                break
    print("\nSıradaki: venues → rank → report")


def cmd_venues(args) -> None:
    """Dergi h-endeksi — hakemlik eşiği için kaba vekil."""
    conn = get_db()
    known = {r[0] for r in conn.execute("SELECT id FROM venues")}
    need = sorted({r[0] for r in conn.execute(
        "SELECT venue_id FROM works WHERE venue_id!='' AND venue_id IS NOT NULL")} - known)
    if not need:
        if not getattr(args, "quiet", False):
            print("Dergi bilgisi güncel.")
        return
    if not getattr(args, "quiet", False):
        print(f"{len(need)} dergi çekiliyor…")
    for s in fetch_batch(SOURCES_API, need, "id,display_name,summary_stats,is_in_doaj"):
        stats = s.get("summary_stats") or {}
        conn.execute("INSERT OR REPLACE INTO venues VALUES (?,?,?,?,?)",
                     (short_id(s["id"]), s.get("display_name") or "",
                      stats.get("h_index") or 0,
                      stats.get("2yr_mean_citedness") or 0.0,
                      1 if s.get("is_in_doaj") else 0))
    conn.commit()
    if not getattr(args, "quiet", False):
        print(f"→ {conn.execute('SELECT COUNT(*) FROM venues').fetchone()[0]} dergi kayıtlı")


JATS_TAG = __import__("re").compile(r"<[^>]+>")


def cmd_abstracts(args) -> None:
    """Eksik özetleri Crossref'ten tamamla.

    OpenAlex, Elsevier başta olmak üzere birçok yayıncının özetini TUTMUYOR.
    Bu, kapsam kararını sessizce bozuyordu: başlığında ülke adı geçmeyen
    gerçek Türkiye çalışmaları eleniyordu — Energy Policy 2022 tavan makalesi
    tam olarak böyle kayboldu. Heuristikle telafi etmek yerine veriyi
    tamamlamak doğru çözüm.

    Sadece aday işler çekiliyor: başlığı konu terimi içerenler ya da
    tohumlarla bağı olanlar. Havuzun tamamı gereksiz.
    """
    conn = get_db()
    cand = [
        (wid, doi) for wid, doi, title in conn.execute(
            "SELECT w.id,w.doi,w.title FROM works w LEFT JOIN scores s ON s.work_id=w.id "
            "WHERE (w.abstract IS NULL OR w.abstract='') AND w.doi!='' "
            "AND (COALESCE(s.seed_links,0) >= 1 OR w.title LIKE '%lectricit%' "
            "     OR w.title LIKE '%ower market%' OR w.title LIKE '%nergy market%')"
        )
        if title
    ]
    print(f"{len(cand)} işin özeti eksik, Crossref'ten çekiliyor…")
    filled = 0
    for i, (wid, doi) in enumerate(cand, 1):
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                msg = json.loads(r.read().decode()).get("message", {})
        except Exception:  # noqa: BLE001
            continue
        raw = msg.get("abstract")
        if raw:
            text = JATS_TAG.sub(" ", raw)
            text = " ".join(text.split())
            conn.execute("UPDATE works SET abstract=? WHERE id=?", (text, wid))
            filled += 1
        if i % 50 == 0:
            conn.commit()
            print(f"  {i}/{len(cand)} · {filled} özet dolduruldu")
        time.sleep(0.05)
    conn.commit()
    print(f"→ {filled} özet eklendi ({len(cand)} denendi). Sıradaki: rank")


def has_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def count_hits(text: str, terms: list[str]) -> int:
    t = text.lower()
    return sum(t.count(term) for term in terms)


def in_scope_of(title: str, abstract: str) -> bool:
    """Türkiye ∧ elektrik fiyatı.

    Tek bir 'Turkey' geçişi yetmiyor — ülke listelerinde geçen çalışmalar
    ('100% renewable electricity in Australia', çok ülkeli paneller) aksi
    halde havuzu dolduruyor. Şart: coğrafya ya BAŞLIKTA, ya da özette
    en az iki kez.
    """
    title = title or ""
    abstract = abstract or ""
    geo_in_title = has_any(title, SCOPE_GEO)
    geo_in_abs = count_hits(abstract, SCOPE_GEO)
    if not (geo_in_title or geo_in_abs >= 2):
        return False
    return has_any(f"{title} {abstract}", SCOPE_TOPIC)


def best_topic(text: str) -> str:
    """En çok eşleşen konu kazanır — ilk eşleşen değil.

    İlk-eşleşen kuralında 'regulation' kelimesi geçen her tahmin makalesi
    'politika/tavan' oluyordu.
    """
    scored = [(count_hits(text, terms), i, name) for i, (name, terms) in enumerate(TOPICS)]
    best = max(scored)
    return best[2] if best[0] > 0 else "diğer"


def percentile_ranks(values: list[float]) -> dict[int, float]:
    """Sıra → 0..1 yüzdelik. Ölçek birimlerini karşılaştırılabilir yapıyor."""
    if not values:
        return {}
    order = sorted(range(len(values)), key=lambda i: values[i])
    n = len(values)
    out: dict[int, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 / max(1, n - 1)   # beraberliklerde ortalama
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def cmd_rank(args) -> None:
    conn = get_db()
    seeds = {r[0] for r in conn.execute("SELECT id FROM seeds")}
    vh = dict(conn.execute("SELECT id,h_index FROM venues"))
    conn.execute("DELETE FROM scores")

    rows = conn.execute(
        "SELECT id,title,abstract,year,cited_by,venue_id,work_type FROM works"
    ).fetchall()

    staged = []
    rescued = 0
    for wid, title, abstract, year, cited_by, venue_id, wtype in rows:
        text = f"{title or ''} {abstract or ''}"
        topic = best_topic(text)

        links = conn.execute(
            "SELECT COUNT(DISTINCT peer) FROM ("
            f"  SELECT src AS peer FROM edges WHERE dst=? AND src IN ({','.join('?' * len(seeds))})"
            "  UNION "
            f"  SELECT dst AS peer FROM edges WHERE src=? AND dst IN ({','.join('?' * len(seeds))})"
            ")", (wid, *seeds, wid, *seeds)).fetchone()[0] if seeds else 0

        # Kapsam kararı — ÜÇ durum, iki değil.
        #   1 = metinle doğrulandı (Türkiye ∧ elektrik fiyatı)
        #   2 = BELİRSİZ: özet hiçbir kaynakta yok, metin testi uygulanamıyor
        #   0 = kapsam dışı
        # Belirsizleri 1'e katmak yanlıştı: grafik yakınlığı 'Türkiye' demek
        # değil, Türkiye makaleleri genel literatürü zaten alıntılıyor. İtalyan
        # merit-order ve MISO makaleleri bu yüzden Türkiye listesine giriyordu.
        if wid in seeds:
            in_scope = 1                      # elle doğrulanmış
        elif in_scope_of(title, abstract):
            in_scope = 1
        elif not (abstract or "").strip() and links >= 2 and has_any(title or "", SCOPE_TOPIC):
            in_scope = 2                      # ayrı listede, elle kontrol için
            rescued += 1
        else:
            in_scope = 0

        age = max(1, THIS_YEAR - (year or THIS_YEAR) + 1)
        velocity = (cited_by or 0) / age
        staged.append({
            "id": wid, "in_scope": in_scope, "topic": topic,
            "is_review": 1 if has_any(title or "", REVIEW_HINTS) or wtype == "review" else 0,
            "links": links, "velocity": velocity, "venue_h": vh.get(venue_id, 0),
            "year": year or 0,
        })

    # Yüzdelikler SADECE kapsam içi havuzda hesaplanıyor — kenar işler
    # dağılımı bozmasın diye.
    scoped = [s for s in staged if s["in_scope"] == 1]
    p_vel = percentile_ranks([s["velocity"] for s in scoped])
    p_ven = percentile_ranks([float(s["venue_h"]) for s in scoped])
    p_lnk = percentile_ranks([float(s["links"]) for s in scoped])
    idx = {s["id"]: i for i, s in enumerate(scoped)}

    for s in staged:
        if s["in_scope"] == 1:
            i = idx[s["id"]]
            pv, pn, pl = p_vel[i], p_ven[i], p_lnk[i]
            pr = 1.0 if s["year"] >= 2021 else (0.5 if s["year"] >= 2017 else 0.15)
            strength = 0.35 * pv + 0.25 * pn + 0.25 * pl + 0.15 * pr
        else:
            pv = pn = pl = pr = 0.0
            strength = 0.0
        conn.execute(
            "INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["id"], s["in_scope"], s["topic"], s["is_review"],
             s["links"], round(s["velocity"], 2), s["venue_h"],
             round(pv, 3), round(pn, 3), round(pl, 3), round(pr, 3),
             round(strength, 4)))
    conn.commit()

    if not getattr(args, "quiet", False):
        n_scope = conn.execute("SELECT COUNT(*) FROM scores WHERE in_scope=1").fetchone()[0]
        print(f"Kapsam içi (metinle doğrulanmış): {n_scope} / {len(staged)}")
        print(f"Belirsiz (özet yok, elle kontrol): {rescued}")
        print("\nKonu dağılımı (kapsam içi):")
        for topic, n in conn.execute(
            "SELECT topic,COUNT(*) FROM scores WHERE in_scope=1 "
            "GROUP BY topic ORDER BY COUNT(*) DESC"
        ):
            print(f"  {n:>4}  {topic}")
        print("\nSıradaki: report")


def cmd_report(args) -> None:
    conn = get_db()
    tot = conn.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    n_scope = conn.execute("SELECT COUNT(*) FROM scores WHERE in_scope=1").fetchone()[0]

    out = [
        "# Türkiye elektrik fiyatları — literatür tarama raporu",
        "",
        f"`python scripts/lit_search.py report` · {now()[:10]}",
        "",
        f"**Havuz:** {tot} iş · **kapsam içi:** {n_scope} "
        f"(Türkiye ∧ elektrik fiyatı) · "
        f"{conn.execute('SELECT COUNT(*) FROM seeds').fetchone()[0]} tohum · "
        f"{conn.execute('SELECT COUNT(*) FROM queries').fetchone()[0]} kayıtlı sorgu",
        "",
        "**Sıralama = sağlamlık vekili**, konu değil: "
        "`0,35·atıf hızı + 0,25·dergi h-endeksi + 0,25·tohum bağı + 0,15·güncellik` "
        "(hepsi kapsam içi havuzda yüzdelik).",
        "",
        "> Bunlar tam metin okumadan görülebilen VEKİLLER. Test seti uzunluğu, "
        "bölme biçimi ve girdilerin tahmin anında mevcut olup olmadığı ancak "
        "tam metinde görülüyor — `extract` formu onun için var.",
        "",
        "## En sağlam işler (konu farketmeksizin)",
        "",
        "| # | Güç | Yıl | Atıf/yıl | Dergi h | Bağ | OA | Özet | Konu | Başlık |",
        "|--:|--:|--:|--:|--:|--:|:-:|:-:|---|---|",
    ]
    rows = conn.execute(
        "SELECT w.id,w.title,w.year,w.venue,w.doi,w.is_oa,w.abstract,"
        "       s.strength,s.velocity,s.venue_h,s.seed_links,s.topic,s.is_review "
        "FROM scores s JOIN works w ON w.id=s.work_id "
        "WHERE s.in_scope=1 ORDER BY s.strength DESC LIMIT ?", (args.top,)
    ).fetchall()
    for i, (wid, title, year, venue, doi, is_oa, abstract, strength, vel, vh,
            links, topic, is_rev) in enumerate(rows, 1):
        link = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{wid}"
        t = (title or "")[:60].replace("|", "·")
        badge = " 📖" if is_rev else ""
        has_abs = "✔" if (abstract or "").strip() else "✖"
        out.append(f"| {i} | {strength:.3f} | {year or '?'} | {vel:.1f} | {vh} | {links} "
                   f"| {'✔' if is_oa else '·'} | {has_abs} | {topic} | [{t}]({link}){badge} |")
    out += [
        "",
        "📖 = derleme/survey — alana giriş için önce bunlar.",
        "",
    ]

    # -- Belirsizler: özet hiçbir kaynakta yok, metin testi uygulanamadı ------
    unk = conn.execute(
        "SELECT w.id,w.title,w.year,w.venue,w.doi,s.seed_links,s.topic "
        "FROM scores s JOIN works w ON w.id=s.work_id "
        "WHERE s.in_scope=2 ORDER BY s.seed_links DESC, w.year DESC"
    ).fetchall()
    if unk:
        out += [
            f"## Belirsiz — elle kontrol gerekli ({len(unk)} iş)", "",
            "Bu işlerin özeti **OpenAlex, Crossref ve Semantic Scholar'ın üçünde "
            "de yok** (Elsevier özet paylaşmıyor), başlıklarında da ülke adı "
            "geçmiyor. Türkiye çalışması olup olmadıkları metinden anlaşılamıyor; "
            "listeye sadece **alıntı grafiği yakınlığı** (≥2 tohumla bağ) ile "
            "girdiler. Grafik yakınlığı 'Türkiye' demek DEĞİL — Türkiye "
            "çalışmaları genel literatürü zaten alıntılıyor. Bu yüzden ana "
            "sıralamaya katılmadılar.", "",
            "| Bağ | Yıl | Dergi | Başlık |", "|--:|--:|---|---|",
        ]
        for wid, title, year, venue, doi, links, topic in unk:
            link = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{wid}"
            out.append(f"| {links} | {year or '?'} | {(venue or '?')[:26]} "
                       f"| [{(title or '')[:70].replace('|', '·')}]({link}) |")
        out.append("")

    # -- Kesin kapsam: başlığında ülke adı geçenler ---------------------------
    certain = [
        r for r in conn.execute(
            "SELECT w.title,w.year,w.venue,w.doi,w.id,s.strength,s.topic "
            "FROM scores s JOIN works w ON w.id=s.work_id "
            "WHERE s.in_scope=1 ORDER BY s.strength DESC")
        if has_any(r[0] or "", SCOPE_GEO)
    ]
    out += [
        f"## Kesin kapsam — başlığında Türkiye geçenler ({len(certain)} iş)", "",
        "Ana sıralamada, özeti bir ülke listesinde 'Turkey' geçtiği için giren "
        "birkaç yabancı çalışma olabiliyor. Bu liste o gürültüden arınmış: "
        "başlığın kendisi Türkiye diyor. **Tez kaynakçasının çekirdeği burası.**",
        "",
        "| Güç | Yıl | Konu | Dergi | Başlık |", "|--:|--:|---|---|---|",
    ]
    for title, year, venue, doi, wid, strength, topic in certain:
        link = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{wid}"
        out.append(f"| {strength:.3f} | {year or '?'} | {topic} | {(venue or '?')[:26]} "
                   f"| [{(title or '')[:72].replace('|', '·')}]({link}) |")
    out.append("")

    out += ["## Konu dağılımı — neyin işlenmiş, neyin işlenmemiş olduğu", "",
            "| Konu | n | En sağlam örnek |", "|---|--:|---|"]
    for topic, n in conn.execute(
        "SELECT topic,COUNT(*) FROM scores WHERE in_scope=1 GROUP BY topic "
        "ORDER BY COUNT(*) DESC"
    ):
        top = conn.execute(
            "SELECT w.title,w.doi,w.year FROM scores s JOIN works w ON w.id=s.work_id "
            "WHERE s.in_scope=1 AND s.topic=? ORDER BY s.strength DESC LIMIT 1", (topic,)
        ).fetchone()
        title = (top[0] or "")[:52].replace("|", "·") if top else ""
        link = f"https://doi.org/{top[1]}" if top and top[1] else ""
        cell = f"[{title}]({link}) ({top[2]})" if link else title
        out.append(f"| {topic} | {n} | {cell} |")
    out.append("")

    out += ["## Her konunun en sağlam 5'i", ""]
    for topic, in conn.execute(
        "SELECT topic FROM scores WHERE in_scope=1 GROUP BY topic ORDER BY COUNT(*) DESC"
    ):
        sub = conn.execute(
            "SELECT w.id,w.title,w.year,w.venue,w.doi,s.strength,s.velocity "
            "FROM scores s JOIN works w ON w.id=s.work_id "
            "WHERE s.in_scope=1 AND s.topic=? ORDER BY s.strength DESC LIMIT 5", (topic,)
        ).fetchall()
        if not sub:
            continue
        out += [f"### {topic}", ""]
        for wid, title, year, venue, doi, strength, vel in sub:
            link = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{wid}"
            out.append(f"- **{strength:.3f}** · {year or '?'} · *{(venue or '?')[:44]}* — "
                       f"[{(title or '')[:74]}]({link})")
        out.append("")

    path = Path(args.out) if args.out else PROJECT_ROOT / "literature" / "search_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"→ {path}\n")
    print("\n".join(out[:14 + min(args.top, 30)]))


def cmd_screen(args) -> None:
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO screening VALUES (?,?,?,?)",
                 (args.work_id, args.decision, args.reason, now()))
    conn.commit()
    print(f"✓ {args.work_id} → {args.decision}: {args.reason}")


def cmd_extract(args) -> None:
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO extraction VALUES (?,?,?,?,?,?,?,?,?)",
                 (args.work_id, args.data_period, args.test_set, args.split_type,
                  args.inputs, args.cap_handled, args.metrics, args.notes, now()))
    conn.commit()
    print(f"✓ çıkarım kaydedildi: {args.work_id}")


def cmd_stats(args) -> None:
    conn = get_db()
    print("PRISMA sayıları\n" + "─" * 46)
    for label, sql in [
        ("Tohum", "SELECT COUNT(*) FROM seeds"),
        ("Toplam havuz", "SELECT COUNT(*) FROM works"),
        ("  ↳ anahtar kelimeden", "SELECT COUNT(*) FROM works WHERE discovered_via LIKE 'sweep:%'"),
        ("  ↳ geri alıntıdan", "SELECT COUNT(*) FROM works WHERE discovered_via LIKE 'bwd:%'"),
        ("  ↳ ileri alıntıdan", "SELECT COUNT(*) FROM works WHERE discovered_via LIKE 'fwd:%'"),
        ("Kapsam içi", "SELECT COUNT(*) FROM scores WHERE in_scope=1"),
        ("Elenmiş", "SELECT COUNT(*) FROM screening WHERE decision='exclude'"),
        ("Dahil", "SELECT COUNT(*) FROM screening WHERE decision='include'"),
        ("Tam metin çıkarımı", "SELECT COUNT(*) FROM extraction"),
    ]:
        print(f"{label:<26} {conn.execute(sql).fetchone()[0]:>6}")
    print("\nKayıtlı sorgular:")
    for q, ran, hits, new in conn.execute(
        "SELECT q,ran_at,n_hits,n_new FROM queries ORDER BY id"
    ):
        print(f"  {ran[:10]}  {hits:>7} sonuç  {new:>3} yeni  {q}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed").set_defaults(func=cmd_seed)

    sp = sub.add_parser("sweep")
    sp.add_argument("--query"); sp.add_argument("--per-page", type=int, default=50)
    sp.add_argument("--since", type=int, default=2010)
    sp.set_defaults(func=cmd_sweep)

    ep = sub.add_parser("expand")
    ep.add_argument("--depth", type=int, default=1)
    ep.add_argument("--max-pages", type=int, default=5)
    ep.add_argument("--frontier-limit", type=int, default=25)
    ep.set_defaults(func=cmd_expand)

    sub.add_parser("venues").set_defaults(func=cmd_venues, quiet=False)
    sub.add_parser("abstracts").set_defaults(func=cmd_abstracts)
    sub.add_parser("rank").set_defaults(func=cmd_rank, quiet=False)

    rp = sub.add_parser("report")
    rp.add_argument("--top", type=int, default=30)
    rp.add_argument("--out")
    rp.set_defaults(func=cmd_report)

    scp = sub.add_parser("screen")
    scp.add_argument("work_id")
    scp.add_argument("decision", choices=["include", "exclude", "maybe"])
    scp.add_argument("reason")
    scp.set_defaults(func=cmd_screen)

    xp = sub.add_parser("extract")
    xp.add_argument("work_id")
    for f in ["data-period", "test-set", "split-type", "inputs", "cap-handled",
              "metrics", "notes"]:
        xp.add_argument(f"--{f}", default="")
    xp.set_defaults(func=cmd_extract)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
