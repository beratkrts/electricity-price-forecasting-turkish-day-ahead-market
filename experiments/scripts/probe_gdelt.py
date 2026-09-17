#!/usr/bin/env python3
"""Faz 0 — GDELT DOC 2.0 kapsam sondası.

Üç soru:
  A. Bilinen jeopolitik olayın YAYINI (başlangıç/zirve/bitiş) veriyor mu?
  B. Türkiye enerji kapsamı var mı, ne yoğunlukta?
  C. Türkiye'ye özgü yerel bir olayı görüyor mu?

KISITLAR (ölçüldü, 2 Eyl 2026):
  - Hız sınırı: 5 saniyede 1 istek. İhlal → 429, ardından bağlantı düşer
    (SSL UNEXPECTED_EOF). Sıralı sorguda RATE_S bekleme ZORUNLU.
  - Sorgu sözdizimi: parantez YALNIZCA OR'lu ifadeler için. "(a b)" hata verir,
    "(a OR b)" ve boşluklu "a b" (örtük AND) geçerli.
  - Doğru aralıkla istek başına 3-5 sn.

    .venv/bin/python experiments/scripts/probe_gdelt.py
"""
from __future__ import annotations
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
TIMEOUT, RATE_S = 90, 6.0
_last = [0.0]


def q(**params):
    wait = RATE_S - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    url = BASE + "?" + urllib.parse.urlencode(params)
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8", "replace")
    except Exception as e:
        _last[0] = time.time()
        print(f"    HATA {type(e).__name__}: {str(e)[:70]}  ({time.time()-t0:.1f}s)")
        return None
    _last[0] = time.time()
    if not raw.strip():
        print("    boş cevap")
        return None
    if raw.lstrip()[:1] not in "{[":
        print(f"    API mesajı: {raw.strip()[:110]}")
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    JSON hatası: {e}")
        return None


def timeline(label, query, start, end, show_arc=False):
    d = q(query=query, mode="timelinevolraw", startdatetime=start, enddatetime=end, format="json")
    if not d or not d.get("timeline"):
        print(f"  {label:32s} — sonuç yok")
        return
    pts = d["timeline"][0]["data"]
    tot = sum(p["value"] for p in pts)
    nz = sum(1 for p in pts if p["value"] > 0)
    print(f"  {label:32s} toplam={tot:>10.0f}  dolu={nz}/{len(pts)} gün")
    if show_arc and tot:
        peak = max(pts, key=lambda p: p["value"])
        print(f"    ZİRVE {peak['date'][:8]} = {peak['value']:.0f}")
        for p in pts:
            pct = p["value"] / peak["value"] * 100 if peak["value"] else 0
            print(f"      {p['date'][:8]} {p['value']:>8.0f} {'█' * int(pct/4)}")


print("=" * 70)
print("A. OLAY YAYI — 'iran israel', 1 Haz → 15 Tem 2025")
print("=" * 70)
timeline("iran israel", "iran israel", "20250601000000", "20250715000000", show_arc=True)

print("\n" + "=" * 70)
print("B. TÜRKİYE ENERJİ KAPSAMI — 2025 boyunca")
print("=" * 70)
for lab, query in [
    ("elektrik (Türkçe kaynak)", "elektrik sourcelang:turkish"),
    ("doğalgaz (Türkçe kaynak)", "doğalgaz sourcelang:turkish"),
    ("turkey electricity (EN)",  "turkey electricity"),
    ("botas OR epias OR epdk",   "(botas OR epias OR epdk)"),
]:
    timeline(lab, query, "20250101000000", "20251231000000")

print("\n" + "=" * 70)
print("C. YEREL OLAY — İran→Türkiye gaz kesintisi, Oca-Şub 2022")
print("=" * 70)
d = q(query="turkey iran gas", mode="artlist", maxrecords="25",
      startdatetime="20220110000000", enddatetime="20220215000000", format="json")
if d and d.get("articles"):
    arts = d["articles"]
    print(f"  {len(arts)} makale bulundu")
    for a in arts[:14]:
        print(f"    {a['seendate'][:8]} [{(a.get('sourcecountry') or '?')[:11]:<11}] "
              f"{(a.get('language') or '?')[:8]:<8} {a['title'][:60]}")
else:
    print("  sonuç yok")
