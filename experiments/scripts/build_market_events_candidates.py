#!/usr/bin/env python3
"""silver.market_events — ADAY üretimi.

Otomatik tespit DEĞİL (haberden günlük atfetme 3 testte çürüdü,
`05_crisis_analysis/04_event_attribution_attempt.ipynb`). Bu script yalnız
ADAYLARI üretir ve her adayı haber arşiviyle kaynaklar. İnsan (Opus) sonra
`label / mechanism / direction / description / confidence / chosen_article_ids`
alanlarını doldurur — iki mevcut silver serisiyle (`price_cap_official`,
`gas_tariff_electricity`) aynı iş bölümü: retrieval otomatik, seçim/yazım insan.

Aday kaynakları:
  A. Regülasyon  — silver.price_cap_official + silver.gas_tariff_electricity
                   basamak değişimleri (drift değil)
  B. Fiyat hareketi — raw_mcp_hourly aylık |Δ| > EŞİK, ardışık aynı-yön aylar birleşik
  C. Dış şok — elle tarih listesi (savaşlar, deprem, kur şoku, mekanizma başlangıçları)

Her aday için: MCP öncesi/sırası/sonrası ort., % değişim, crisis_cf_v5 kalıntı
(gerçek − fundamental), tavan/tarife değeri + TF-IDF ile pencere içi ilk N haber.

Çıktı:
  experiments/notebooks/05_crisis_analysis/market_events_candidates.json  (Opus doldurur)
  experiments/notebooks/05_crisis_analysis/market_events_candidates.csv   (hızlı tarama)

    .venv/bin/python experiments/scripts/build_market_events_candidates.py
"""
from __future__ import annotations
import json
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

OUT_DIR = ROOT / "experiments/notebooks/05_crisis_analysis"
CF_MODEL, CF_VARIANT = "crisis_cf_v5", "fundamental"

PRICE_MOVE_PCT = 20.0     # aylık |Δ%| eşiği (USD)
CAP_STEP_PCT = 8.0        # tavan basamak eşiği
TARIFF_STEP_PCT = 10.0    # tarife basamak eşiği
TOP_NEWS = 10             # aday başına döndürülen haber
NEWS_PRE_DAYS, NEWS_POST_DAYS = 21, 14   # haber arama penceresi payı

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

SEED_REGULATION = ("tavan fiyat afl azami uzlaştırma fiyatı auf epdk epiaş düzenleme piyasa müdahale "
                   "teklif tavanı elektrik fiyatına sınır yönetmelik karar")
SEED_TARIFF = ("botaş doğalgaz tarife elektrik üretim santral gaz fiyatı zam indirim ithalat "
               "spot gaz maliyeti")
SEED_PRICEMOVE = ("elektrik spot fiyatı ptf yükseldi düştü rekor tavan sıfır hidro termal yenilenebilir "
                  "arz talep üretim payı gaz maliyeti")

# --- C: dış şok adayları (tarih + arama tohumu) ---
EXTERNAL = [
    dict(start="2021-12-15", end=None, hint="Kur şoku / KKM açıklaması",
         seed="dolar kur tl döviz kkm kur korumalı mevduat merkez bankası rekor değer kaybı maliyet"),
    dict(start="2022-01-18", end="2022-02-05", hint="İran doğalgaz kesintisi",
         seed="iran doğalgaz kesinti sevkiyat azaltma boru hattı arz gazdan üretim düşüş kısıntı sanayi"),
    dict(start="2022-02-24", end="2022-12-31", hint="Rusya-Ukrayna savaşı (yörünge)",
         seed="rusya ukrayna savaş işgal yaptırım gazprom kuzey akım avrupa enerji krizi ttf spot gaz "
              "doğalgaz akışı kesinti rekor fiyat"),
    dict(start="2022-04-01", end=None, hint="AUF mekanizması + AFL artışı (1 Nisan 2022)",
         seed="azami uzlaştırma fiyatı auf gelir tavanı teknoloji bazlı afl artış 1 nisan mekanizma "
              "gaz kömür yenilenebilir tavan"),
    dict(start="2023-02-06", end="2023-02-28", hint="Kahramanmaraş depremi",
         seed="deprem kahramanmaraş enerji altyapı hasar elektrik kesinti doğalgaz iletim talep düşüş "
              "bölge sanayi durdu"),
    dict(start="2025-06-12", end="2025-06-26", hint="ABD-İsrail-İran savaşı",
         seed="israil iran saldırı savaş hürmüz boğazı petrol brent jeopolitik gerilim abd nükleer tesis "
              "füze ortadoğu ham petrol sıçradı arz güvenliği"),
]


def _to_ist(s):
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)


def load_frames(eng):
    news = pd.read_sql(text("""
        SELECT article_id, published_at, section, title,
               coalesce(description,'') d, coalesce(body,'') b, url
        FROM bronze.news_raw
        WHERE title NOT LIKE 'Spot elektrik fiyatı%%'          -- günlük fiyat serisi (olay değil, verinin kendisi)
        ORDER BY published_at"""), eng)
    news["published_at"] = _to_ist(news.published_at)
    news = news.reset_index(drop=True)
    news["txt"] = (news.title.fillna("") + ". " + news.d + ". " + news.b).str.lower()

    mcp = pd.read_sql(text("SELECT ts, price_usd, price_try FROM raw_mcp_hourly"), eng)
    mcp["ts"] = _to_ist(mcp.ts)
    mcp = mcp.set_index("ts").sort_index()

    cf = pd.read_sql(text("""
        SELECT ts, residual_usd, at_cap FROM gold.crisis_counterfactual
        WHERE model_name=:m AND variant=:v"""), eng, params={"m": CF_MODEL, "v": CF_VARIANT})
    cf["ts"] = _to_ist(cf.ts)
    cf = cf.set_index("ts").sort_index()

    cap = pd.read_sql(text("SELECT effective_from, cap_try, source_article_id FROM silver.price_cap_official "
                           "ORDER BY effective_from"), eng)
    tar = pd.read_sql(text("SELECT effective_from, price_try_1000m3, source, source_ref "
                           "FROM silver.gas_tariff_electricity ORDER BY effective_from"), eng)
    return news, mcp, cf, cap, tar


def window_context(mcp, cf, start, end):
    end = end or start
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    pre = mcp.loc[s - pd.Timedelta(days=30): s - pd.Timedelta(seconds=1), "price_usd"]
    dur = mcp.loc[s: e + pd.Timedelta(days=1), "price_usd"]
    post = mcp.loc[e + pd.Timedelta(days=1): e + pd.Timedelta(days=31), "price_usd"]
    cfw = cf.loc[s: e + pd.Timedelta(days=1)]
    f = lambda x: round(float(x.mean()), 1) if len(x) and not np.isnan(x.mean()) else None
    pct = None
    if len(pre) and len(dur) and pre.mean():
        pct = round(100 * (dur.mean() / pre.mean() - 1), 1)
    return dict(
        mcp_usd_before=f(pre), mcp_usd_during=f(dur), mcp_usd_after=f(post), pct_change=pct,
        cf_residual_usd=(round(float(cfw.residual_usd.astype(float).mean()), 1) if len(cfw) else None),
        cf_at_cap_share=(round(float(cfw.at_cap.mean()), 2) if len(cfw) else None),
        n_hours=int(len(dur)))


def gen_regulation(cap, tar):
    out = []
    cap = cap.copy()
    cap["pct"] = cap.cap_try.pct_change() * 100
    for _, r in cap.iterrows():
        if pd.notna(r.pct) and abs(r.pct) >= CAP_STEP_PCT:
            out.append(dict(source_type="A_cap", start_date=str(r.effective_from), end_date=None,
                            seed=SEED_REGULATION, hint=f"AFL tavanı {r.pct:+.0f}% → {r.cap_try:.0f} TL",
                            silver_ref=dict(cap_try=float(r.cap_try),
                                            source_article_id=int(r.source_article_id) if pd.notna(r.source_article_id) else None)))
    tar = tar.copy()
    tar["pct"] = tar.price_try_1000m3.pct_change() * 100
    for _, r in tar.iterrows():
        big = (pd.notna(r.pct) and abs(r.pct) >= TARIFF_STEP_PCT) or r.source == "news_absolute"
        if big and pd.notna(r.pct):
            out.append(dict(source_type="A_tariff", start_date=str(r.effective_from), end_date=None,
                            seed=SEED_TARIFF,
                            hint=f"BOTAŞ elektrik gaz tarifesi {r.pct:+.0f}% → {r.price_try_1000m3:.0f} TL/1000m³",
                            silver_ref=dict(price_try_1000m3=float(r.price_try_1000m3),
                                            source=r.source, source_ref=r.source_ref)))
    return out


def gen_price_moves(mcp):
    m = mcp.price_usd.resample("MS").mean()
    pct = m.pct_change() * 100
    flagged = pct[pct.abs() >= PRICE_MOVE_PCT].dropna()
    # ardışık ve aynı-yön ayları birleştir
    events, cur = [], None
    for mo, p in flagged.items():
        sign = np.sign(p)
        if cur and sign == cur["sign"] and (mo - cur["last"]).days <= 40:
            cur["last"] = mo
            cur["pcts"].append(round(p, 1))
        else:
            if cur:
                events.append(cur)
            cur = dict(first=mo, last=mo, sign=sign, pcts=[round(p, 1)])
    if cur:
        events.append(cur)
    out = []
    for ev in events:
        end = (ev["last"] + pd.offsets.MonthEnd(0)).normalize()
        arrow = "yükseliş" if ev["sign"] > 0 else "düşüş"
        out.append(dict(source_type="B_pricemove", start_date=str(ev["first"].date()),
                        end_date=str(end.date()), seed=SEED_PRICEMOVE,
                        hint=f"MCP {arrow}, aylık {'/'.join(f'{x:+.0f}%' for x in ev['pcts'])}",
                        silver_ref=None))
    return out


def gen_external():
    return [dict(source_type="C_external", start_date=e["start"], end_date=e["end"],
                 seed=e["seed"], hint=e["hint"], silver_ref=None) for e in EXTERNAL]


def attach_news(cand, news, vec, X):
    s = pd.Timestamp(cand["start_date"]) - pd.Timedelta(days=NEWS_PRE_DAYS)
    e = pd.Timestamp(cand["end_date"] or cand["start_date"]) + pd.Timedelta(days=NEWS_POST_DAYS)
    mask = (news.published_at >= s) & (news.published_at <= e)
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        return []
    q = vec.transform([cand["seed"].lower()])
    sim = cosine_similarity(X[idx], q).ravel()
    order = idx[np.argsort(-sim)][:TOP_NEWS]
    sim_map = dict(zip(idx, sim))
    rows = []
    for i in order:
        r = news.iloc[i]
        rows.append(dict(article_id=int(r.article_id),
                         published_at=r.published_at.strftime("%Y-%m-%d"),
                         similarity=round(float(sim_map[i]), 3),
                         section=r.section, title=r.title, url=r.url))
    return rows


def main():
    eng = get_db_engine()
    print("veri yükleniyor…")
    news, mcp, cf, cap, tar = load_frames(eng)
    print(f"  haber {len(news)}  ·  MCP {mcp.index.min().date()}→{mcp.index.max().date()}  ·  cf {len(cf)}")

    print("TF-IDF…")
    vec = TfidfVectorizer(stop_words=TR_STOP, ngram_range=(1, 2), min_df=5, max_df=0.4,
                          sublinear_tf=True, max_features=120_000)
    X = vec.fit_transform(news.txt)

    cands = gen_regulation(cap, tar) + gen_price_moves(mcp) + gen_external()
    cands.sort(key=lambda c: c["start_date"])

    records = []
    for k, c in enumerate(cands, 1):
        ctx = window_context(mcp, cf, c["start_date"], c["end_date"])
        newsrows = attach_news(c, news, vec, X)
        records.append({
            "candidate_id": k,
            "source_type": c["source_type"],
            "start_date": c["start_date"],
            "end_date": c["end_date"],
            "auto_hint": c["hint"],
            "silver_ref": c["silver_ref"],
            "auto_context": ctx,
            "candidate_news": newsrows,
            # ---- OPUS DOLDURACAK ----
            "label": None,
            "mechanism": None,        # maliyet | arz | regülasyon | jeopolitik | talep | kur
            "direction": None,        # yukari | asagi | karisik
            "description": None,      # 1-2 cümle, seçilen haberlere dayanarak
            "confidence": None,       # kesin | muhtemel
            "chosen_article_ids": [],
            "drop": False,            # aday geçersizse True (ör. dedupe, gürültü)
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "market_events_candidates.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = []
    for r in records:
        flat.append({
            "candidate_id": r["candidate_id"], "source_type": r["source_type"],
            "start_date": r["start_date"], "end_date": r["end_date"],
            "auto_hint": r["auto_hint"],
            **{f"ctx_{k}": v for k, v in r["auto_context"].items()},
            "top_news": " | ".join(f'[{n["article_id"]}] {n["published_at"]} {n["title"][:60]}'
                                   for n in r["candidate_news"][:3]),
        })
    pd.DataFrame(flat).to_csv(OUT_DIR / "market_events_candidates.csv", index=False)

    print(f"\n{len(records)} aday →")
    print(f"  {OUT_DIR/'market_events_candidates.json'}")
    print(f"  {OUT_DIR/'market_events_candidates.csv'}")
    print("\nkaynak dağılımı:", pd.Series([r['source_type'] for r in records]).value_counts().to_dict())
    print("\nözet:")
    for r in records:
        c = r["auto_context"]
        print(f"  #{r['candidate_id']:2d} {r['source_type']:12s} {r['start_date']}"
              f"{'..'+r['end_date'] if r['end_date'] else '':13s}  "
              f"Δ{str(c['pct_change'])+'%':>8s}  kalıntı {str(c['cf_residual_usd']):>6s}  "
              f"tavan-pay {c['cf_at_cap_share']}  | {r['auto_hint'][:52]}")


if __name__ == "__main__":
    main()
