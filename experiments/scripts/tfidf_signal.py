#!/usr/bin/env python3
"""TF-IDF hızlı sinyal kontrolü — Rus-Ukrayna (2022) + ABD-İsrail-İran (2025).

Soru: olay tohumuna TF-IDF kosinüs benzerliğinden kurulan günlük 'anlatı yoğunluğu',
MCP fiyatı / karşı-olgusal kalıntı ile zaman-hizalı bir hareket gösteriyor mu?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/Users/beratkaratasoglu/etkb_intern_project/electricity_price_forecasting_in_turkish_day_ahead_market")
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine
from sqlalchemy import text

OUT = ROOT / "experiments/notebooks/05_crisis_analysis"
eng = get_db_engine()

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

print("haberler çekiliyor...")
df = pd.read_sql(text("""
    SELECT article_id, published_at, title, coalesce(description,'') d, coalesce(body,'') b
    FROM bronze.news_raw ORDER BY published_at
"""), eng)
df["published_at"] = pd.to_datetime(df.published_at, utc=True).dt.tz_convert("Europe/Istanbul")
df["day"] = df.published_at.dt.tz_localize(None).dt.normalize()
df["txt"] = (df.title.fillna("") + ". " + df.d + ". " + df.b).str.lower()
print(f"  {len(df)} haber, {df.day.min().date()} → {df.day.max().date()}")

print("TF-IDF...")
vec = TfidfVectorizer(stop_words=TR_STOP, ngram_range=(1, 2), min_df=5, max_df=0.4,
                      sublinear_tf=True, max_features=120_000)
X = vec.fit_transform(df.txt)
print(f"  matris {X.shape}")

EVENTS = {
    "Rus-Ukrayna savaşı (2022)": dict(
        seed=("rusya ukrayna savaş işgal saldırı doğalgaz gaz kesinti gazprom kuzey akım boru hattı "
              "avrupa enerji krizi yaptırım ttf spot gaz fiyat rekor natural gas rusya'dan gaz"),
        a="2021-10-01", b="2023-03-01", war="2022-02-24"),
    "ABD-İsrail-İran savaşı (2025)": dict(
        seed=("israil iran saldırı savaş çatışma hürmüz boğazı petrol brent fiyat jeopolitik gerilim "
              "abd iran nükleer tesis füze ortadoğu enerji arz güvenliği ham petrol sıçradı"),
        a="2025-04-01", b="2025-09-01", war="2025-06-13"),
}

# --- fiyat + kalıntı ---
mcp = pd.read_sql(text("SELECT ts, price_usd FROM raw_mcp_hourly WHERE ts >= '2021-09-01'"), eng)
mcp["ts"] = pd.to_datetime(mcp.ts, utc=True).dt.tz_convert("Europe/Istanbul")
mcp_d = mcp.set_index("ts").price_usd.resample("D").mean()
mcp_d.index = mcp_d.index.tz_localize(None)

try:
    res = pd.read_sql(text("""SELECT ts, residual_usd FROM gold.crisis_counterfactual
        WHERE model_name='crisis_cf_v5' AND variant='fundamental'"""), eng)
    res["ts"] = pd.to_datetime(res.ts, utc=True).dt.tz_convert("Europe/Istanbul")
    res_d = res.set_index("ts").residual_usd.astype(float).resample("D").mean()
    res_d.index = res_d.index.tz_localize(None)
except Exception as e:
    print("kalıntı yok:", e); res_d = None

daily_n = df.groupby("day").size()

fig, axes = plt.subplots(len(EVENTS), 1, figsize=(13, 5 * len(EVENTS)))
for ax, (name, cfg) in zip(np.atleast_1d(axes), EVENTS.items()):
    q = vec.transform([cfg["seed"].lower()])
    sim = cosine_similarity(X, q).ravel()
    s = pd.Series(sim, index=df.day)
    win = (df.day >= cfg["a"]).to_numpy() & (df.day < cfg["b"]).to_numpy()
    sub = df.loc[win]
    sim_sub = pd.Series(sim[win], index=sub.index)

    # günlük yoğunluk metrikleri
    g = pd.DataFrame({"sim": sim_sub.values, "day": sub.day.values})
    inten_mean = g.groupby("day").sim.mean()                      # hacme doğal normalize
    thr = np.quantile(sim, 0.97)                                  # global 97. persentil = 'ilgili haber'
    inten_hit = g.groupby("day").sim.apply(lambda x: (x > thr).sum())
    hit_share = inten_hit / daily_n.reindex(inten_hit.index)

    idx = pd.date_range(cfg["a"], pd.Timestamp(cfg["b"]) - pd.Timedelta(days=1), freq="D")
    im = inten_mean.reindex(idx).interpolate().rolling(7, center=True, min_periods=1).mean()
    hs = hit_share.reindex(idx).fillna(0).rolling(7, center=True, min_periods=1).mean()
    price = mcp_d.reindex(idx).rolling(7, center=True, min_periods=1).mean()

    z = lambda x: (x - x.mean()) / x.std()
    ax.plot(idx, z(im), label="anlatı yoğ. (ort. benzerlik, 7g)", color="#c0392b", lw=1.6)
    ax.plot(idx, z(hs), label="anlatı yoğ. (ilgili haber payı, 7g)", color="#e67e22", lw=1.1, alpha=.8)
    ax.plot(idx, z(price), label="MCP $ (7g)", color="#2c3e50", lw=1.6)
    if res_d is not None:
        rr = res_d.reindex(idx).rolling(7, center=True, min_periods=1).mean()
        ax.plot(idx, z(rr), label="kontrafaktüel kalıntı (7g)", color="#16a085", lw=1.2, ls="--")
    ax.axvline(pd.Timestamp(cfg["war"]), color="k", lw=1, ls=":")
    ax.set_title(name); ax.legend(fontsize=8, ncol=2); ax.set_ylabel("z-skor")

    # lead-lag: yoğunluk (mean) vs MCP günlük değişim
    d_price = mcp_d.reindex(idx).diff()
    base = pd.DataFrame({"inten": inten_mean.reindex(idx).interpolate(),
                         "dP": d_price, "P": mcp_d.reindex(idx)}).dropna()
    print(f"\n=== {name} ===")
    print(f"  pencere {cfg['a']} → {cfg['b']}, {int(win.sum())} haber")
    print(f"  top-benzerlik haberler:")
    for _, r in sub.assign(sim=sim_sub.values).nlargest(6, "sim")[["published_at", "sim", "title"]].iterrows():
        print(f"    {r.published_at.date()}  {r.sim:.3f}  {r.title[:70]}")
    for lag in (-14, -7, -3, 0, 3, 7, 14):
        c = base["inten"].corr(base["P"].shift(-lag))
        print(f"  corr(yoğunluk_t , MCP_t{lag:+d})  = {c:+.3f}")
    if res_d is not None:
        br = pd.DataFrame({"inten": inten_mean.reindex(idx).interpolate(),
                           "res": res_d.reindex(idx)}).dropna()
        for lag in (0, 3, 7, 14):
            c = br["inten"].corr(br["res"].shift(-lag))
            print(f"  corr(yoğunluk_t , kalıntı_t{lag:+d}) = {c:+.3f}")

plt.tight_layout()
plt.savefig(OUT / "tfidf_signal.png", dpi=110)
print(f"\n→ {OUT/'tfidf_signal.png'}")
