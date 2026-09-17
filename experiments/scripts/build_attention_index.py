#!/usr/bin/env python3
"""A — haber dikkat endeksi (günlük, konu bazında). OLAY_ANALIZI_V2_PLAN.md §5.1.

Etki DEĞİL dikkat: her gün her konu için haber korpusunun o konuya ne kadar
"baktığı". TF-IDF kosinüs benzerliği (gömü yükseltmesi sonraki tur — torch yok).

Konu başına iki ölçüm:
  attn_mean  — o günün haberlerinin konu tohumuna ort. benzerliği (hacme normalize)
  hit_share  — o gün "konuyla ilgili" (global %97 eşiği üstü) haber payı

Çıktı: silver.news_attention_daily

    .venv/bin/python experiments/scripts/build_attention_index.py [--write]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

TR_STOP = """acaba altmış altı ama ancak arada aslında ayrıca bana bazı belki ben benden beni benim
beri beş bile bin bir birçok biri birkaç birkez birşey birşeyi biz bizden bize bizi bizim böyle böylece
bu buna bunda bundan bunlar bunları bunların bunu bunun burada çok çünkü da daha dahi de defa değil diğer
diye doksan dokuz dolayı dolayısıyla dört edecek eden ederek edilecek ediliyor edilmesi ediyor eğer elli
en etmesi etti ettiği ettiğini gibi göre halen hangi hatta hem henüz hep hepsi her herhangi herkesin hiç
hiçbir için iki ile ilgili ise işte itibaren itibariyle kadar karşın katrilyon kendi kendilerine kendini
kendisi kendisine kendisini kez ki kim kimden kime kimi kimse kırk milyar milyon mu mü mı nasıl ne neden
nedenle nerede nereye niçin niye o olan olarak oldu olduğu olduğunu olduklarını olmadı olmadığı olmak olması
olmayan olmaz olsa olsun olup olur olursa oluyor on ona ondan onlar onlardan onları onların onu onun otuz
oysa öyle pek rağmen sadece sanki sekiz seksen sen senden seni senin sonra şey şeyden şeyi şeyler şu şuna
şunda şundan şunları şunu tarafından trilyon tüm üç üzere var vardı ve veya ya yani yapacak yapılan yapılması
yapıyor yapmak yaptı yaptığı yaptığını yaptıkları yedi yerine yetmiş yine yirmi yoksa yüz zaten""".split()

TOPICS = {
    "gas_supply": (
        "avrupa doğalgaz arz güvenliği gaz kesinti tedarik boru hattı akış azaldı durdu depo "
        "stok seviyesi lng ithalat rusya'dan gaz gazprom kuzey akım ttf spot gaz fiyatı rekor "
        "avrupa gaz krizi kışa hazırlık gaz talebi"),
    "geopolitics": (
        "savaş çatışma işgal saldırı yaptırım ambargo jeopolitik gerilim askeri operasyon "
        "füze hava saldırısı ateşkes rusya ukrayna israil iran abd ortadoğu nükleer tesis "
        "enerji arz güvenliği risk"),
    "regulation": (
        "azami fiyat limiti tavan fiyat epdk epiaş piyasa müdahale düzenleme yönetmelik karar "
        "azami uzlaştırma fiyatı auf gelir tavanı teklif tavanı elektrik fiyatına sınır "
        "botaş tarife zam indirim yekdem destekleme"),
    "hydro_weather": (
        "baraj doluluk oranı yağış kar erimesi kuraklık su seviyesi hidroelektrik üretim rekor "
        "hidrolik santral akarsu debisi mevsim normalleri sıcaklık dalgası soğuk hava kış "
        "yenilenebilir pay güneş rüzgar üretimi"),
    "oil": (
        "petrol fiyatı brent ham petrol opec üretim kısıntı varil dolar jeopolitik risk primi "
        "hürmüz boğazı tanker arz talep dengesi rafineri ithalat akaryakıt"),
}


def load_news(eng):
    df = pd.read_sql(text("""
        SELECT article_id, published_at, section, title,
               coalesce(description,'') d, coalesce(body,'') b
        FROM bronze.news_raw
        WHERE title NOT LIKE 'Spot elektrik fiyatı%%'
        ORDER BY published_at"""), eng)
    df["published_at"] = pd.to_datetime(df.published_at, utc=True).dt.tz_convert("Europe/Istanbul")
    df["day"] = df.published_at.dt.tz_localize(None).dt.normalize()
    df["txt"] = (df.title.fillna("") + ". " + df.d + ". " + df.b).str.lower()
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    eng = get_db_engine()

    df = load_news(eng)
    print(f"{len(df)} haber  {df.day.min().date()} → {df.day.max().date()}")
    daily_n = df.groupby("day").size()

    vec = TfidfVectorizer(stop_words=TR_STOP, ngram_range=(1, 2), min_df=5, max_df=0.4,
                          sublinear_tf=True, max_features=120_000)
    X = vec.fit_transform(df.txt)
    print(f"TF-IDF {X.shape}")

    cal = pd.date_range(df.day.min(), df.day.max(), freq="D")
    out = pd.DataFrame(index=cal)
    out.index.name = "d"
    out["n_articles"] = daily_n.reindex(cal).fillna(0).astype(int)

    for name, seed in TOPICS.items():
        q = vec.transform([seed.lower()])
        sim = cosine_similarity(X, q).ravel()
        thr = np.quantile(sim, 0.97)
        g = pd.DataFrame({"sim": sim, "day": df.day.values, "hit": (sim > thr).astype(int)})
        by = g.groupby("day")
        attn_mean = by.sim.mean().reindex(cal)
        hit_share = (by.hit.sum() / daily_n).reindex(cal).fillna(0)
        out[f"{name}_attn"] = attn_mean.interpolate(limit_direction="both")
        out[f"{name}_hit"] = hit_share
        # 30g z-skor (rejim-göreli sıçrama tespiti için)
        s = out[f"{name}_attn"]
        out[f"{name}_z"] = ((s - s.rolling(90, min_periods=30).mean())
                            / s.rolling(90, min_periods=30).std())

    print("\nkonu bazında ort. dikkat (attn_mean) ve en yüksek 3 gün:")
    for name in TOPICS:
        s = out[f"{name}_attn"]
        top = s.nlargest(3)
        print(f"  {name:14s} ort {s.mean():.4f}  zirveler: " +
              ", ".join(f"{d.date()}({v:.3f})" for d, v in top.items()))

    # duman testi: bilinen epizotlarda z sıçraması
    print("\nbilinen epizot kontrolü (konu_z, olay penceresi maks):")
    checks = [("geopolitics", "2022-02-15", "2022-03-15", "Rusya-Ukrayna"),
              ("geopolitics", "2025-06-10", "2025-06-30", "ABD-İran 2025"),
              ("gas_supply", "2021-09-01", "2021-12-31", "2021 Avrupa gaz"),
              ("hydro_weather", "2026-02-01", "2026-05-31", "2026 hidro"),
              ("oil", "2026-02-25", "2026-03-31", "Hürmüz 2026")]
    for topic, a, b, lab in checks:
        w = out.loc[a:b, f"{topic}_z"]
        print(f"  {lab:18s} {topic:14s} maks z = {w.max():+.1f}")

    if args.write:
        with eng.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS silver.news_attention_daily"))
            cols_sql = ", ".join(f'"{c2}" numeric' for c2 in out.columns if c2 != "n_articles")
            c.execute(text(f"""CREATE TABLE silver.news_attention_daily (
                d date PRIMARY KEY, n_articles int, {cols_sql},
                method text DEFAULT 'tfidf_v1', created_at timestamptz DEFAULT now())"""))
            rec = out.reset_index()
            for _, r in rec.iterrows():
                keys = [k for k in out.columns]
                c.execute(text(f"""INSERT INTO silver.news_attention_daily
                    (d, {", ".join(f'"{k}"' for k in keys)})
                    VALUES (:d, {", ".join(f':{k}' for k in keys)})"""),
                    {"d": r["d"].date(), **{k: (None if pd.isna(r[k]) else float(r[k])) for k in keys}})
        print(f"\n→ silver.news_attention_daily ({len(out)} gün, {len(TOPICS)} konu)")

    out.to_pickle(ROOT / "experiments/notebooks/05_crisis_analysis/_attention_index.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
