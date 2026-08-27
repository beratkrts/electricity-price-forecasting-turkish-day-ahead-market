"""Haber Arşivi Toplayıcı — Kriz & Olay İstihbarat Sistemi (Bronze Katmanı).

enerjigunlugu.net sitemap'inden haber URL'lerini çıkarır, her haberi çeker,
schema.org mikroverisini ayrıştırır ve `bronze.news_raw` tablosuna yazar.

Tasarım kararları:
  - **Kesintiye dayanıklı.** Her parti hemen commit edilir; yeniden başlatıldığında
    DB'de olmayan URL'lerden devam eder. Ctrl-C en fazla bir partiyi kaybettirir.
  - **published_at yetkilidir.** Sitemap `lastmod` alanı düzenlenmiş haberlerde
    yayın tarihinden sapar, bu yüzden sadece kaba ön seçim için kullanılır;
    tarih filtresi sayfadaki `itemprop=datePublished` üzerinden uygulanır.
  - **article_id sıralıdır.** URL'deki `...-41334h.htm` numarası yayın sırasıyla
    monotondur; birincil anahtar ve doğal sıralama anahtarı olarak kullanılır.
  - **Katman ham kalır.** Alaka filtresi ve LLM etiketleri buraya yazılmaz.

Kullanım:
    python scripts/fetch_news_archive.py --limit 40        # duman testi
    python scripts/fetch_news_archive.py                   # tam koşu (~28 bin haber)
    python scripts/fetch_news_archive.py --retry-errors    # başarısızları yeniden dene
    python scripts/fetch_news_archive.py --stats           # sadece durum raporu
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db.connection import get_db_engine  # noqa: E402

# ---------------------------------------------------------------------------- ayarlar

SOURCE = "enerjigunlugu"
BASE = "https://www.enerjigunlugu.net"
SITEMAP_INDEX = f"{BASE}/sitemap.xsd"

# Ham verinin geriye kapsamı. daily_update_pipeline.HISTORY_START ile hizalı:
# Ocak 2022 İran gaz kesintisi ve Şubat 2022 savaş şoku kapsansın.
HISTORY_START = "2021-01-01"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) enerji-arastirma/1.0 (+beratc237@gmail.com)"
)
REQUEST_TIMEOUT = 30
DEFAULT_DELAY = 0.4          # saniye/istek — nazik olmak ücretsiz
MAX_ATTEMPTS = 3
BATCH_SIZE = 50              # bu kadar haberde bir commit
ID_RE = re.compile(r"-(\d+)h\.htm$")

# ---------------------------------------------------------------------------- log

Path("logs").mkdir(exist_ok=True)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
_fh = RotatingFileHandler("logs/news_archive.log", maxBytes=10 * 1024 * 1024,
                          backupCount=3, encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger = logging.getLogger("NewsArchive")
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.addHandler(_sh)
logger.propagate = False

# ---------------------------------------------------------------------------- kesinti

_stop = False


def _handle_sigint(_signum, _frame):
    global _stop
    if _stop:
        logger.warning("İkinci kesinti — hemen çıkılıyor.")
        sys.exit(130)
    _stop = True
    logger.warning("Kesinti alındı. Açık parti yazılıp çıkılacak; tekrar Ctrl-C zorla çıkar.")


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# ---------------------------------------------------------------------------- HTTP

class Fetcher:
    """requests.Session sarmalayıcı: yeniden deneme, geri çekilme, nezaket gecikmesi."""

    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        })
        self._last = 0.0

    def get(self, url: str, attempts: int = MAX_ATTEMPTS) -> requests.Response:
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            gap = self.delay - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                # 4xx kalıcıdır (429 hariç); yeniden denemek anlamsız.
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if i < attempts - 1:
                    backoff = 2.0 * (2 ** i)
                    logger.debug(f"  yeniden deneme {i+1}/{attempts-1} ({backoff:.0f}s): {url} — {exc}")
                    time.sleep(backoff)
        raise last_exc  # type: ignore[misc]


def _decompress(raw: bytes) -> str:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------- sitemap

@dataclass(frozen=True)
class SitemapEntry:
    article_id: int
    url: str
    lastmod: str


def discover_sitemaps(fetcher: Fetcher) -> list[str]:
    """Sitemap index'ten haber sitemap'lerini bulur."""
    text_ = _decompress(fetcher.get(SITEMAP_INDEX).content)
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", text_, re.S)
    news = [u for u in locs if "sitemap-news" in u]
    if not news:
        raise RuntimeError(f"Sitemap index'te haber sitemap'i yok: {locs}")
    logger.info(f"Haber sitemap'i: {len(news)} adet — {', '.join(u.rsplit('/', 1)[-1] for u in news)}")
    return news


def parse_sitemap(fetcher: Fetcher, url: str) -> Iterator[SitemapEntry]:
    """Tek bir sitemap'i ayrıştırır. ID'si okunamayan URL'ler atlanır (galeri/video)."""
    body = _decompress(fetcher.get(url).content)
    for loc, lastmod in re.findall(r"<loc>\s*(.*?)\s*</loc>\s*<lastmod>\s*(.*?)\s*</lastmod>", body, re.S):
        m = ID_RE.search(loc)
        if m:
            yield SitemapEntry(int(m.group(1)), loc, lastmod)


def collect_candidates(fetcher: Fetcher, since: str) -> list[SitemapEntry]:
    """Tüm haber sitemap'lerini tarayıp `since` sonrası adayları döndürür.

    lastmod'a göre kaba eleme yapılır — düzenlenmiş eski haberler bu ağa takılabilir,
    kesin tarih filtresi sayfadaki datePublished ile uygulanır (bkz. ingest_article).
    """
    seen: dict[int, SitemapEntry] = {}
    for sm in discover_sitemaps(fetcher):
        n_before = len(seen)
        for e in parse_sitemap(fetcher, sm):
            if e.lastmod[:10] >= since:
                seen[e.article_id] = e
        logger.info(f"  {sm.rsplit('/', 1)[-1]}: +{len(seen) - n_before} aday")
    out = sorted(seen.values(), key=lambda e: e.article_id)
    logger.info(f"Toplam aday: {len(out):,} (id {out[0].article_id} → {out[-1].article_id})".replace(",", "."))
    return out


# ---------------------------------------------------------------------------- ayrıştırma

@dataclass
class Article:
    article_id: int
    url: str
    published_at: datetime
    modified_at: Optional[datetime]
    section: Optional[str]
    title: str
    description: Optional[str]
    body: str
    keywords: list[str]
    author: Optional[str]
    http_status: int

    @property
    def body_chars(self) -> int:
        return len(self.body)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.title}\n{self.body}".encode("utf-8")).hexdigest()


class ParseError(ValueError):
    """Zorunlu bir mikroveri alanı bulunamadı."""


def _prop(soup: BeautifulSoup, name: str):
    return soup.find(attrs={"itemprop": name})


def _prop_value(soup: BeautifulSoup, name: str) -> Optional[str]:
    """itemprop değerini okur: önce `content` özniteliği, yoksa görünen metin."""
    el = _prop(soup, name)
    if el is None:
        return None
    val = el.get("content") or el.get_text(" ", strip=True)
    val = re.sub(r"\s+", " ", (val or "")).strip()
    return val or None


def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def parse_article(html: str, entry: SitemapEntry, http_status: int) -> Article:
    soup = BeautifulSoup(html, "lxml")

    published = _parse_dt(_prop_value(soup, "datePublished"))
    if published is None:
        raise ParseError("datePublished yok veya ayrıştırılamadı")

    title = _prop_value(soup, "headline")
    if not title:
        og = soup.find("meta", property="og:title")
        title = (og.get("content").strip() if og and og.get("content") else None)
    if not title:
        raise ParseError("headline yok")

    body_el = _prop(soup, "articleBody")
    if body_el is None:
        raise ParseError("articleBody yok")
    # Gövde içindeki reklam/ilgili-haber gürültüsünü at, sonra düz metne indir.
    for junk in body_el.find_all(["script", "style", "iframe", "figure", "figcaption",
                                  "noscript", "ins", "form"]):
        junk.decompose()
    body = re.sub(r"[ \t ]+", " ", body_el.get_text("\n", strip=True))
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        raise ParseError("articleBody boş")

    kw_raw = _prop_value(soup, "keywords") or ""
    keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]

    return Article(
        article_id=entry.article_id,
        url=entry.url,
        published_at=published,
        modified_at=_parse_dt(_prop_value(soup, "dateModified")),
        section=(_prop_value(soup, "articleSection") or None),
        title=title,
        description=_prop_value(soup, "description"),
        body=body,
        keywords=keywords,
        author=(_prop_value(soup, "author") or None),
        http_status=http_status,
    )


# ---------------------------------------------------------------------------- DB

UPSERT = text("""
    INSERT INTO bronze.news_raw (
        article_id, url, source, published_at, modified_at, section,
        title, description, body, body_chars, keywords, author,
        content_hash, http_status
    ) VALUES (
        :article_id, :url, :source, :published_at, :modified_at, :section,
        :title, :description, :body, :body_chars, :keywords, :author,
        :content_hash, :http_status
    )
    ON CONFLICT (article_id) DO NOTHING
""")

LOG_ERROR = text("""
    INSERT INTO bronze.news_fetch_errors (url, article_id, http_status, error, attempts, last_try_at)
    VALUES (:url, :article_id, :http_status, :error, 1, NOW())
    ON CONFLICT (url) DO UPDATE SET
        attempts    = bronze.news_fetch_errors.attempts + 1,
        http_status = EXCLUDED.http_status,
        error       = EXCLUDED.error,
        last_try_at = NOW()
""")

CLEAR_ERROR = text("DELETE FROM bronze.news_fetch_errors WHERE url = :url")


def existing_ids(engine) -> set[int]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text("SELECT article_id FROM bronze.news_raw"))}


def persist(engine, rows: list[Article], errors: list[tuple[SitemapEntry, int, str]]) -> int:
    """Bir partiyi tek işlemde yazar. Kısmi parti kaybı en fazla BATCH_SIZE kadardır."""
    if not rows and not errors:
        return 0
    with engine.begin() as conn:
        for a in rows:
            conn.execute(UPSERT, {
                "article_id": a.article_id, "url": a.url, "source": SOURCE,
                "published_at": a.published_at, "modified_at": a.modified_at,
                "section": a.section, "title": a.title, "description": a.description,
                "body": a.body, "body_chars": a.body_chars, "keywords": a.keywords,
                "author": a.author, "content_hash": a.content_hash,
                "http_status": a.http_status,
            })
            conn.execute(CLEAR_ERROR, {"url": a.url})
        for entry, status, msg in errors:
            conn.execute(LOG_ERROR, {
                "url": entry.url, "article_id": entry.article_id,
                "http_status": status, "error": msg[:2000],
            })
    return len(rows)


# ---------------------------------------------------------------------------- koşu

def _fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}sa {m:02d}dk" if h else f"{m}dk {s:02d}sn"


def run(engine, fetcher: Fetcher, targets: list[SitemapEntry], since: str) -> dict:
    stats = {"yazildi": 0, "atlandi_tarih": 0, "ayristirilamadi": 0, "http_hata": 0}
    batch: list[Article] = []
    errors: list[tuple[SitemapEntry, int, str]] = []
    started = time.monotonic()
    total = len(targets)

    for i, entry in enumerate(targets, 1):
        if _stop:
            logger.info("Kesinti — açık parti yazılıyor.")
            break
        try:
            resp = fetcher.get(entry.url)
            if resp.status_code != 200:
                stats["http_hata"] += 1
                errors.append((entry, resp.status_code, f"HTTP {resp.status_code}"))
            else:
                art = parse_article(resp.text, entry, resp.status_code)
                # Kesin tarih filtresi: sitemap lastmod düzenlemede kayar, datePublished kaymaz.
                if art.published_at.date().isoformat() < since:
                    stats["atlandi_tarih"] += 1
                else:
                    batch.append(art)
        except ParseError as exc:
            stats["ayristirilamadi"] += 1
            errors.append((entry, 200, f"ParseError: {exc}"))
            logger.warning(f"  ayrıştırılamadı: {entry.url} — {exc}")
        except Exception as exc:  # noqa: BLE001
            stats["http_hata"] += 1
            errors.append((entry, 0, f"{type(exc).__name__}: {exc}"))
            logger.warning(f"  çekilemedi: {entry.url} — {exc}")

        if len(batch) + len(errors) >= BATCH_SIZE:
            stats["yazildi"] += persist(engine, batch, errors)
            batch, errors = [], []
            hiz = i / max(time.monotonic() - started, 1e-9)
            logger.info(
                f"[{i:>6,}/{total:,}] {100*i/total:5.1f}% | {hiz:4.1f} haber/sn | "
                f"kalan ~{_fmt_eta((total - i) / max(hiz, 1e-9))} | "
                f"yazıldı={stats['yazildi']:,} hata={stats['http_hata'] + stats['ayristirilamadi']:,}"
                .replace(",", ".")
            )

    stats["yazildi"] += persist(engine, batch, errors)
    stats["sure_sn"] = round(time.monotonic() - started, 1)
    return stats


def show_stats(engine) -> None:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) n, MIN(published_at)::date ilk, MAX(published_at)::date son,
                   ROUND(AVG(body_chars)) ort_karakter,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY body_chars)::int medyan_karakter,
                   COUNT(DISTINCT content_hash) tekil_icerik
            FROM bronze.news_raw
        """)).one()
        print(f"\nbronze.news_raw: {row.n:,} haber".replace(",", "."))
        if row.n:
            print(f"  kapsam        : {row.ilk} → {row.son}")
            print(f"  gövde         : ort {int(row.ort_karakter):,} / medyan {row.medyan_karakter:,} karakter".replace(",", "."))
            print(f"  tekil içerik  : {row.tekil_icerik:,} ({row.n - row.tekil_icerik} mükerrer)".replace(",", "."))
            print("\n  Yıla göre:")
            for r in conn.execute(text("""
                SELECT EXTRACT(year FROM published_at)::int yil, COUNT(*) n
                FROM bronze.news_raw GROUP BY 1 ORDER BY 1
            """)):
                print(f"    {r.yil}  {r.n:>6,}".replace(",", "."))
            print("\n  En sık bölüm (ilk 12):")
            for r in conn.execute(text("""
                SELECT COALESCE(section, '(yok)') bolum, COUNT(*) n
                FROM bronze.news_raw GROUP BY 1 ORDER BY 2 DESC LIMIT 12
            """)):
                print(f"    {r.bolum:<22} {r.n:>6,}".replace(",", "."))
        err = conn.execute(text("SELECT COUNT(*) FROM bronze.news_fetch_errors")).scalar_one()
        print(f"\n  bekleyen hata : {err:,}".replace(",", "."))


# ---------------------------------------------------------------------------- CLI

def main() -> int:
    p = argparse.ArgumentParser(description="enerjigunlugu.net haber arşivi toplayıcı (bronze katmanı)")
    p.add_argument("--since", default=HISTORY_START, help=f"YYYY-MM-DD, varsayılan {HISTORY_START}")
    p.add_argument("--limit", type=int, help="ilk N adayı çek (duman testi)")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"istekler arası saniye (varsayılan {DEFAULT_DELAY})")
    p.add_argument("--retry-errors", action="store_true", help="sadece bronze.news_fetch_errors'daki URL'leri yeniden dene")
    p.add_argument("--stats", action="store_true", help="sadece durum raporu yazdır, çekme yapma")
    args = p.parse_args()

    engine = get_db_engine()

    if args.stats:
        show_stats(engine)
        return 0

    fetcher = Fetcher(delay=args.delay)

    if args.retry_errors:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT url, article_id FROM bronze.news_fetch_errors
                WHERE attempts < 5 ORDER BY article_id
            """)).all()
        targets = [SitemapEntry(r.article_id or 0, r.url, "") for r in rows]
        logger.info(f"Yeniden deneme: {len(targets):,} URL".replace(",", "."))
    else:
        logger.info(f"Sitemap taranıyor (>= {args.since})...")
        candidates = collect_candidates(fetcher, args.since)
        have = existing_ids(engine)
        targets = [e for e in candidates if e.article_id not in have]
        logger.info(
            f"DB'de mevcut: {len(have):,} | çekilecek: {len(targets):,}".replace(",", ".")
        )

    if args.limit:
        targets = targets[: args.limit]
        logger.info(f"--limit {args.limit} uygulandı.")

    if not targets:
        logger.info("Çekilecek yeni haber yok.")
        show_stats(engine)
        return 0

    tahmini = len(targets) * (args.delay + 0.35)
    logger.info(f"Başlıyor: {len(targets):,} haber, tahmini süre ~{_fmt_eta(tahmini)}".replace(",", "."))

    stats = run(engine, fetcher, targets, args.since)

    logger.info("--- ÖZET ---")
    for k, v in stats.items():
        logger.info(f"  {k:18s}: {v}")
    show_stats(engine)
    return 130 if _stop else 0


if __name__ == "__main__":
    sys.exit(main())
