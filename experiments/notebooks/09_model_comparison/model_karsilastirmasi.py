# %% [markdown]
# # Model Varyantları — Kronolojik Karşılaştırma
#
# **Türkiye GÖP fiyat tahmini, 2021–2026**
#
# Bu notebook, projede denenen model varyantlarından **kronolojik olarak önemli**
# olanları seçer ve her açıdan karşılaştırır: genel doğruluk, **farklı rejimlerdeki**
# performans, **farklı fiyat aralıklarındaki** performans, güven aralığı kalitesi,
# **her varyantı neyi çözmek için geliştirdiğimiz**, ve iki referans yönteme
# (**LEAR** kanonik EPF, **EPNet** derin öğrenme) karşı kafa kafaya kıyas.
#
# ## Varyant soyağacı
#
# ```
# Eski Log1p  ──►  LightGBM_v1 (pre-lag0)  ──►  lgb_lag0_v2  ──►  ensemble_v1
#  (log1p,        (log1p BIRAKILDI,           (lag0 yenilenebilir   (3× LightGBM
#   sıkıştırma)    3-başlı native quantile)    baskı oranları,       farklı pencere,
#                                              min_child 20→10)      + konformal band)
#                                            CANLI 13 Ağu 2026     CANLI 2 Eyl 2026
#
# REFERANS YÖNTEMLER (yenildi):  LEAR ensemble (kanonik LAGO / epftoolbox) ·
#                               EPNet CNN-LSTM · Hybrid routing · Paper stratejileri
# ```
#
# `lgb_cqr_v2` bir **güven-aralığı kalibrasyon deneyi** (CQR); nokta tahmin varyantı
# değil, ayrı araç — §6'da kısa değinilir, ana karşılaştırmaya girmez.
#
# Kaynak: `EXPERIMENT_REPORT.md`, `07_lago_protocol/`, `logs/epnet_*`,
# `gold.ptf_predictions_experimental`, `gold.ptf_predictions_daily`.

# %%
import sys
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/beratkaratasoglu/etkb_intern_project/electricity_price_forecasting_in_turkish_day_ahead_market")
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine
from sqlalchemy import text

eng = get_db_engine()
OUT = ROOT / "experiments/notebooks/09_model_comparison"
LAGO = ROOT / "experiments/notebooks/07_lago_protocol"
pd.set_option("display.width", 220)
plt.rcParams["figure.figsize"] = (11, 4)

# %% [markdown]
# ---
# # 1. Kronoloji ve "neyi niye geliştirdik"
#
# | # | Varyant | Tarih | Ne değişti | **Neden** | Sonuç |
# |---|---|---|---|---|---|
# | 0 | **Eski Log1p** | ~2024 | `log1p(fiyat)` hedefi, tek tahmin | İlk model; çarpık fiyat dağılımını normalleştirme | Notebook WAPE %16,5 / MAE $8,4. log1p yüksek fiyatı sıkıştırıp aşağı yanlı tahmin veriyor |
# | 1 | **LightGBM_v1** (pre-lag0) | ~Tem 2026 | log1p **bırakıldı**; 3-başlı **native quantile** (P10/P50/P90); `min_child=20` | log1p sapması + dürüst güven aralığı isteği | Canlı; 2026 bahar çöküşünde ucuz saatleri aşırı tahmin ediyor |
# | 2 | **lgb_lag0_v2** | **13 Ağu 2026** (canlı) | + `net_load_lag0`, `renewable_pressure_ratio_lag0`, `zero_price_risk_score` vb.; `min_child` 20→10 | 2026 bahar **sıfır-fiyat rejimi** — hidro bolluğu, model ucuz saatleri $16 aşırı tahmin ediyordu | **Kısmi**: tek-gün testi WAPE %28→%16; ama Şub-Mar 2026 dilimi boyunca MAE hâlâ ~$12-13 |
# | 3 | **ensemble_v1** | **2 Eyl 2026** (canlı) | P50 = ort(base + roll90 + roll150); 3× LightGBM farklı eğitim pencereleri; + konformal band | Çöküş **körlüğü** — tek model rejim kaymasında aşırı tahmin; farklı pencereli modellerin ortalaması bunu azaltır (LEAR ensemble mantığı) | Çöküş MAE $10,5→$9,6, BIAS +3,8→+1,5; DM p<0,0001 |
#
# **Not (`lgb_cqr_v2`):** aynı P50, P10/P90 CQR (conformalized quantile regression)
# ile kalibre. Native quantile gerçek kapsaması ~%72, hedef %80. Deneysel kaldı; §6.

# %% [markdown]
# ---
# # 2. Veri — ortak test penceresi
#
# Per-saat tahmin serisi olan 3 nokta-tahmin varyantı: `lgb_baseline_v1`,
# `lgb_lag0_v2` (deney tablosu, 2-yıl walk-forward), `ensemble_v1` (canlı tablo).
# `LightGBM_v1` içeriği = `lgb_lag0_v2` (etiket yanıltıcı, atlandı).

# %%
def series(tbl, mn):
    d = pd.read_sql(text(f"""SELECT target_ts ts, predicted_mcp_usd p50,
        predicted_mcp_usd_p10 p10, predicted_mcp_usd_p90 p90
        FROM gold.{tbl} WHERE model_name=:m"""), eng, params={"m": mn})
    d["ts"] = pd.to_datetime(d.ts, utc=True)
    return d.set_index("ts")

MODELS = {
    "lgb_baseline_v1": ("ptf_predictions_experimental", "pre-lag0 baseline"),
    "lgb_lag0_v2":     ("ptf_predictions_experimental", "lag0 (canlı 13 Ağu)"),
    "ensemble_v1":     ("ptf_predictions_daily",        "ensemble (canlı 2 Eyl)"),
}
raw = {m: series(tbl, m) for m, (tbl, _) in MODELS.items()}

act = pd.read_sql(text("SELECT ts, price_usd, at_cap FROM silver.mcp_with_cap WHERE ts>='2024-08-01'"), eng)
act["ts"] = pd.to_datetime(act.ts, utc=True)
act = act.set_index("ts")

P = pd.DataFrame({m: raw[m].p50 for m in MODELS}).join(
    act.rename(columns={"price_usd": "actual"})).dropna(subset=["actual", *MODELS])
print(f"ortak pencere: {P.index.min()} → {P.index.max()}  ({len(P):,} saat)")

naive = act.price_usd.reindex(P.index - pd.Timedelta(hours=168))
naive.index = P.index
P["naive"] = naive
P = P.dropna(subset=["naive"])

# %% [markdown]
# ---
# # 3. Genel doğruluk
#
# MAE, WAPE, RMSE, BIAS (ort. hata — pozitif = aşırı tahmin), rMAE (MAE / naive MAE;
# <1 = naive'i geçiyor). `METRICS.md`: düşük fiyatta WAPE paydası çöküyor, **BIAS +
# rMAE** kullan.

# %%
def metrics(pred, actual, naive):
    err = pred - actual
    return pd.Series({
        "MAE": err.abs().mean(),
        "WAPE%": 100 * err.abs().sum() / actual.abs().sum(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "BIAS": err.mean(),
        "rMAE": err.abs().mean() / (naive - actual).abs().mean(),
    })

overall = pd.DataFrame({m: metrics(P[m], P.actual, P.naive) for m in MODELS}).T.round(3)
overall["etiket"] = [MODELS[m][1] for m in overall.index]
print(overall.to_string())

# %% [markdown]
# **Okuma:**
# - `baseline → lag0_v2`: MAE $7,47 → $7,34, BIAS +1,78 → +1,53. Küçük ama tutarlı.
# - `lag0_v2 → ensemble_v1`: MAE $7,34 → $7,15, **BIAS +1,53 → +0,92** (yarıya). En
#   büyük kazanç sapmada — çöküş körlüğü tam da aşırı-tahmin (pozitif BIAS) sorunuydu.
# - Üçü de naive'i açık geçiyor (rMAE 0,63–0,65).

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
x = np.arange(len(MODELS))
ax.bar(x - 0.2, overall.MAE, 0.4, label="MAE", color="#2c3e50")
ax.bar(x + 0.2, overall.BIAS, 0.4, label="BIAS", color="#e67e22")
ax.set_xticks(x); ax.set_xticklabels([MODELS[m][1] for m in MODELS], rotation=12, ha="right")
ax.axhline(0, color="k", lw=0.6); ax.legend(); ax.set_ylabel("$/MWh")
ax.set_title("Genel MAE ve BIAS")
fig.savefig(OUT / "g1_overall.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ---
# # 4. Rejim bazında performans
#
# Ortak pencere 2024-09'da başladığı için 2021-2022 kriz rejimi kapsam dışı.

# %%
REGIMES = {
    "normal (2024-09…2025-12)": ("2024-09-01", "2026-01-01"),
    "çöküş (2026-02…2026-05)":   ("2026-02-01", "2026-06-01"),
    "toparlanma (2026-06…2026-08)": ("2026-06-01", "2026-08-13"),
}
rows = []
for name, (a, b) in REGIMES.items():
    seg = P.loc[(P.index >= a) & (P.index < b)]
    for m in MODELS:
        mm = metrics(seg[m], seg.actual, seg.naive)
        rows.append({"rejim": name, "model": MODELS[m][1], "n": len(seg),
                     "MAE": round(mm.MAE, 2), "BIAS": round(mm.BIAS, 2), "rMAE": round(mm.rMAE, 3)})
reg = pd.DataFrame(rows)
print(reg.pivot(index="rejim", columns="model", values="MAE").to_string())
print("\nBIAS:")
print(reg.pivot(index="rejim", columns="model", values="BIAS").to_string())

# %% [markdown]
# **Okuma (ÖNEMLİ):**
# - **Normal rejim:** üç varyant birbirine yakın (~$6,3). İyileştirmeler burada
#   nötr — sorun normal rejimde değildi.
# - **Çöküş rejimi:** `baseline` MAE $10,87 / BIAS +5,0. **`lag0_v2` bu rejimde
#   baseline'ı GEÇMİYOR** ($10,89 / +5,2) — 2-yıl backfill'de lag0 oranlarının çöküş
#   katkısı ~sıfır. `EXPERIMENT_REPORT.md`'nin "tek günde yardımcı, rejim boyunca
#   yetersiz" bulgusu doğrulanıyor. Çöküşü asıl indiren **`ensemble_v1`**: MAE $9,91,
#   BIAS +2,98 (baseline'a göre −%9 MAE, BIAS ~yarı).
# - **Toparlanma:** ensemble BIAS −1,6; MAE'ler yakın, örneklem küçük.
#
# → **lag0_v2 canlıya alma kararı tek-gün testine dayanıyordu; 2-yıl perspektifinde
#   kazancı marjinal ve genel MAE'de. Rejim sorununu ensemble çözdü.**

# %%
fig, ax = plt.subplots(figsize=(9, 3.5))
reg.pivot(index="rejim", columns="model", values="MAE").plot(kind="bar", ax=ax)
ax.set_ylabel("MAE $/MWh"); ax.set_title("Rejim bazında MAE")
ax.set_xticklabels(ax.get_xticklabels(), rotation=12, ha="right")
fig.savefig(OUT / "g2_regime.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ---
# # 5. Fiyat aralığı bazında performans
#
# Gerçekleşen fiyata göre dilim. `model-shrinkage-low-prices`: ucuz dilimde model
# yukarı kayıyor (pozitif BIAS). lag0_v2 bunu hedefledi.

# %%
BANDS = [(-1, 10), (10, 30), (30, 60), (60, 120), (120, 1e9)]
rows = []
for lo, hi in BANDS:
    seg = P[(P.actual >= lo) & (P.actual < hi)]
    if len(seg) < 30:
        continue
    for m in MODELS:
        err = seg[m] - seg.actual
        rows.append({"dilim": f"${max(lo,0):.0f}-{hi:.0f}" if hi < 1e8 else f"${lo:.0f}+",
                     "model": MODELS[m][1], "n": len(seg),
                     "MAE": round(err.abs().mean(), 2), "BIAS": round(err.mean(), 2)})
band = pd.DataFrame(rows)
print("n (dilim başına):")
print(band[band.model == band.model.iloc[0]][["dilim", "n"]].to_string(index=False))
print("\nMAE:"); print(band.pivot(index="dilim", columns="model", values="MAE").to_string())
print("\nBIAS (+ = aşırı tahmin):")
print(band.pivot(index="dilim", columns="model", values="BIAS").to_string())

# %% [markdown]
# **Okuma:**
# - **$0-10 dilim:** en büyük BIAS burada — üç varyant da ucuz saatleri **aşırı
#   tahmin** ediyor (+$7-8). `ensemble` en iyi ama hâlâ pozitif. Sıfır-fiyat rejimi
#   tam çözülmedi.
# - **$30-60 dilim** ("normal"): hepsi iyi.
# - **$60-120 dilim:** negatif BIAS — pahalı uçları alttan tahmin.
# - Genel şekil: **modeller ortaya çekiyor** (shrinkage) — ucuzu yukarı, pahalıyı aşağı.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
order = ["$0-10", "$10-30", "$30-60", "$60-120", "$120+"]
for ax, val in zip(axes, ["MAE", "BIAS"]):
    piv = band.pivot(index="dilim", columns="model", values=val)
    piv.reindex([o for o in order if o in piv.index]).plot(kind="bar", ax=ax, legend=(val == "MAE"))
    ax.set_title(f"Fiyat dilimi — {val}"); ax.axhline(0, color="k", lw=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.savefig(OUT / "g3_price_bands.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ---
# # 6. Güven aralığı kalitesi (P10–P90) — cqr_v2 dahil
#
# Hedef %80 kapsama. Burada `lgb_cqr_v2` de var (kalibrasyon deneyi olduğu için).

# %%
IV = dict(MODELS)
IV["lgb_cqr_v2"] = ("ptf_predictions_experimental", "cqr_v2 (deneysel)")
raw_iv = {**raw, "lgb_cqr_v2": series("ptf_predictions_experimental", "lgb_cqr_v2")}
rows = []
for m in IV:
    d = pd.DataFrame({"a": act.price_usd, "p10": raw_iv[m].p10, "p90": raw_iv[m].p90}).dropna()
    d = d[(d.index >= P.index.min()) & (d.index <= P.index.max())]
    cov = ((d.a >= d.p10) & (d.a <= d.p90)).mean()
    rows.append({"model": IV[m][1], "kapsama%": round(100 * cov, 1),
                 "ort. genişlik $": round((d.p90 - d.p10).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# **Okuma:**
# - `baseline` / `lag0_v2`: native quantile — kapsama **~%72** (hedef %80'in altında).
# - `ensemble_v1`: konformal band — kapsama **%79,5** (hedefe en yakın).
# - `cqr_v2`: DB'deki bu serinin hem kapsaması (**%54**) hem P50 nokta tahmini
#   (MAE ~$14) bozuk — CQR'ın *yükseltmesi* gereken kapsamayı düşürmüş.
#   **`gold.ptf_predictions_experimental`'daki `lgb_cqr_v2` içeriği şüpheli**, ayrı
#   incelenmeli. CQR yönteminin kendisi geçerli (`05_confidence_intervals/`).

# %% [markdown]
# ---
# # 7. İstatistiksel anlamlılık — Diebold-Mariano
#
# Mutlak-hata farkının testi (DM, h=1, Newey-West varyans).

# %%
from scipy import stats
def dm_test(e1, e2):
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    g0 = np.var(d, ddof=0)
    g1 = np.cov(d[:-1], d[1:])[0, 1]
    dm = d.mean() / np.sqrt((g0 + 2 * g1) / n)
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))

for a, b in [("lgb_baseline_v1", "lgb_lag0_v2"), ("lgb_lag0_v2", "ensemble_v1"),
             ("lgb_baseline_v1", "ensemble_v1")]:
    dm, p = dm_test((P[a] - P.actual).values, (P[b] - P.actual).values)
    better = MODELS[b][1] if dm > 0 else MODELS[a][1]
    print(f"  {MODELS[a][1]:22s} vs {MODELS[b][1]:22s}  DM={dm:+.2f}  p={p:.2e}  → {better} daha iyi")

# %% [markdown]
# ---
# # 8. Referans yöntem 1 — LEAR ensemble (kanonik LAGO / epftoolbox)
#
# **LEAR** = *Lasso Estimated AutoRegressive* — EPF literatürünün açık-erişim referans
# modeli (Lago ve ark. 2021, `epftoolbox`). LASSO ile seçilen 240+ terimli lineer
# oto-regresif model, birden çok kalibrasyon penceresinin (56g/84g/3y/4y) günlük
# ortalaması = "LEAR ensemble".
#
# **Bizim çalışmamız** (`07_lago_protocol/`): makalenin **gerçek `epftoolbox` kodu**
# (GitHub'dan izole `numpy<2` venv'e kurulup çalıştırıldı — bizim reimplementasyonumuz
# değil), Türkiye GÖP verisinde, **dondurulmuş protokolle** (`src/eval/lago_protocol.py`).
#
# Referans kodun kendi **iki kırılganlığı** bulunup düzeltildi (bizim hatamız değil,
# epftoolbox'ın kendisi):
# 1. `LassoLarsIC(criterion='aic')` n≤p'de matematiksel tanımsız (sklearn PR #21481 bug
#    ilan etti) — 1.1'in `noise_variance` parametresiyle çözüldü.
# 2. `MedianScaler` 2026 çöküşünde MAD=0 → sıfıra bölüm (167/1344 saat tam $0) —
#    MAD=0 ise ölçekleme atlanarak çözüldü.
#
# **Bilinçli olarak LEAR'a sızıntı avantajı verildi:** besleme SQL'i hedef günün
# gerçekleşen `load_forecast_mw`/`kgup_total_mw`'ını çekiyor; canlı LightGBM bunları
# T+1 için görmüyor (pre-forecast kullanıyor). LEAR'a haksız bilgi avantajı — buna
# rağmen kaybetti, yani rapor edilen fark bir **alt sınır**.

# %%
# LEAR ailesi — CSV'lerden MAE (tr_epf_ext.csv actual'a karşı)
act_lago = pd.read_csv(LAGO / "tr_epf_ext.csv", index_col=0, parse_dates=True)["Price"]
actd = {d: g.values[:24] for d, g in act_lago.groupby(act_lago.index.date) if len(g) >= 24}

def lear_mae(fname):
    df = pd.read_csv(LAGO / fname, index_col=0)
    df.index = pd.to_datetime(df.index).date
    e = [np.abs(r.values.astype(float) - actd[d]) for d, r in df.iterrows() if d in actd]
    return np.concatenate(e).mean() if e else np.nan

lear = {f.split("/")[-1].replace("lago_ref_lear_", "").replace(".csv", ""): lear_mae(f.split("/")[-1])
        for f in sorted(glob.glob(str(LAGO / "lago_ref_lear_cw*.csv")))}
print("Referans LEAR (epftoolbox), kalibrasyon penceresine göre MAE (3-yıl span):")
for k, v in sorted(lear.items(), key=lambda x: x[1]):
    print(f"  {k:8s}  MAE ${v:.2f}")

# %% [markdown]
# ## Kafa kafaya — dondurulmuş kanonik protokol (`04_canonical_lago_benchmark.ipynb`)
#
# ### Uzun pencere (3 yıl, 1096 gün)
#
# | Model | MAE | rMAE(n2) |
# |---|---:|---:|
# | **LightGBM ensemble (90/150/tüm)** | **$7,30** | **0,630** |
# | LightGBM WF (tek, canlı politika) | $7,44 | 0,642 |
# | LEAR ensemble (4 pencere) | $8,00 | 0,690 |
# | LEAR cw56 / cw84 / cw1456 / cw1092 | $8,72 / $8,79 / $8,84 / $8,98 | 0,75–0,78 |
# | naive-3 / naive-2 | $10,14 / $11,59 | |
#
# LightGBM ensemble, LEAR ensemble'ı **DM p = 2,9×10⁻¹³** ile geçiyor.
#
# ### Birincil pencere (728 gün) + rejim katmanları
#
# | Model | genel | normal (520g) | **çöküş** (147g) | toparlanma (58g) |
# |---|---:|---:|---:|---:|
# | **LightGBM ensemble** | **$7,15** | 6,32 | **9,65** | **8,20** |
# | LightGBM (canlı = lag0_v2) | $7,35 | **6,30** | 10,54 | 8,63 |
# | LEAR ensemble | $8,08 | 6,56 | 12,98 | 9,30 |
# | LEAR cw1092 / cw1456 | $9,35 / $9,15 | 7,21 / 7,11 | **17,13 / 16,40** | 8,85 / 8,98 |
#
# **Bulgular:**
# 1. **LightGBM her katmanda LEAR'ı geçiyor** — canlı tek-model bile LEAR ensemble'ı
#    DM p = 1,9×10⁻⁹ ile geçiyordu; LightGBM ensemble onun da üstüne (p = 1,4×10⁻¹⁴).
# 2. **Lago'nun 3-4 yıllık penceresi Türkiye'ye transfer olmuyor** — `cw1092`/`cw1456`
#    çöküşte $16-17 (normal örnekler kriz rejimini seyreltiyor). TR'de kısa pencere
#    (`cw56`) uzunu eziyor.
# 3. Aynı prensip **LightGBM ensemble'ın kısa üyelerinin (90g/150g) çöküşü neden
#    düzelttiğini** açıklıyor — "çok-pencere" fikri Lago'dan, ama LightGBM tabanıyla.
# 4. Tam beslenmiş 511-özellikli LEAR de geride ($9,34, `03`).

# %% [markdown]
# ---
# # 9. Referans yöntem 2 — EPNet (CNN-LSTM derin öğrenme)
#
# EPF literatüründen CNN-LSTM hibrit ağ. `logs/epnet_*` altında üç grid koşusu.

# %%
def parse_epnet(md_path, key_cols=("Strategy Name", "1M MAE", "12M WAPE", "12M MAE")):
    lines = [l for l in Path(md_path).read_text().splitlines() if l.startswith("|") and "---" not in l]
    hdr = [c.strip().strip("`*") for c in lines[0].strip("|").split("|")]
    out = []
    for l in lines[1:]:
        cells = [c.strip().strip("`*") for c in l.strip("|").split("|")]
        row = dict(zip(hdr, cells))
        out.append(row)
    return pd.DataFrame(out)

grid = parse_epnet(ROOT / "logs/epnet_experiments/epnet_grid_summary.md")
cols = [c for c in grid.columns if c in ("Strategy Name", "Lookback", "1M MAE", "12M WAPE", "12M MAE")]
print("EPNet klasik grid (CNN-LSTM), en iyi 4 (12M WAPE'ye göre):")
g = grid[cols].copy()
g["_w"] = pd.to_numeric(g["12M WAPE"].str.replace("%", "").str.replace(",", "."), errors="coerce")
print(g.sort_values("_w").head(4).drop(columns="_w").to_string(index=False))

robust = parse_epnet(ROOT / "logs/epnet_fast_robust_experiments/epnet_robust_summary.md")
rc = [c for c in robust.columns if c in ("Strategy Name", "1M MAE", "12M WAPE", "12M MAE")]
print("\nEPNet zero-price robust varyantları:")
print(robust[rc].to_string(index=False))

# %% [markdown]
# **Bulgular:**
# - **En iyi EPNet** (STRAT_03, 6 ay geriye bakış): 12M WAPE **%32,5** / MAE **$10,10**.
#   Çoğu strateji %33–99 WAPE — kısa pencereli olanlar tamamen çöküyor.
# - **Zero-price robust varyantları:** 12M WAPE ~%39–42, 1M (Temmuz 2026 çöküş) WAPE
#   %25–28. Huber loss + floor eklemek yetmedi.
# - **Kıyas:** LightGBM ailesi genel WAPE **~%12,5**, çöküş MAE **~$10**. EPNet 12M
#   MAE'si (~$10) yakın görünüyor ama WAPE 2,5–3 kat kötü — EPNet ortalama seviyeyi
#   tutturuyor, saatlik şekli tutturamıyor.
# - **`EXPERIMENT_REPORT.md` §3:** EPNet **bahar rejim kaymasında çöktü**; hybrid
#   routing (EPNet+LightGBM harmanı) LightGBM tek başına'dan **kötü** (%32,7 vs %26,9).
#
# **Sonuç:** derin öğrenme (bu veri ölçeğinde, bu rejim oynaklığında) LightGBM'in
# çok gerisinde. Literatürdeki "DNN > LightGBM" bulgusu (çoğu Avrupa piyasası,
# durağan rejim) Türkiye'ye transfer olmuyor.

# %% [markdown]
# ---
# # 10. Sentez
#
# ## Ne öğrendik — nokta tahmin evrimi
#
# 1. **log1p → native quantile** (v0→v1): log1p yüksek fiyatı sıkıştırıp sistematik
#    aşağı sapma veriyordu; dürüst P10/P90 elde edildi.
# 2. **lag0 yenilenebilir oranları** (v1→v2): genel MAE $7,47→$7,34 (DM p=1,2e-3),
#    ama **hedeflediği çöküş rejiminde katkısı sıfır** ($10,89 vs $10,87). Canlıya
#    alma tek-gün testine dayanıyordu. **Ders: tek-gün testine güvenme.**
# 3. **Çok-pencereli ensemble** (v2→ensemble): **asıl kazanç.** Genel MAE $7,34→$7,15
#    (DM p=4e-7), çöküş MAE −%9, çöküş BIAS +5,2→+3,0, kapsama %72→%79,5. Fikir LEAR
#    ensemble'dan (Lago), taban LightGBM.
#
# ## Referans yöntemlere karşı
#
# | Yöntem | Genel MAE | Çöküş MAE | LightGBM'e karşı |
# |---|---:|---:|---|
# | **LightGBM ensemble** (canlı) | **$7,15** | **$9,65** | — |
# | LEAR ensemble (kanonik epftoolbox) | $8,08 | $12,98 | DM p=1,4e-14 kaybetti (sızıntı avantajına rağmen) |
# | EPNet en iyi (CNN-LSTM) | ~$10 (12M) | çöktü | WAPE 2,5-3× kötü |
# | naive-2 (sezonsal) | $11,59 | $16,00 | — |
#
# ## Sürücü tema
#
# Model geliştirmesinin tamamı **2026 bahar çöküşü / düşük-fiyat rejimi** etrafında
# döndü. Normal rejimde varyantlar ayrışmıyor (~$6,3). Çöküş hâlâ tam çözülmedi:
# **her varyant $0-10 diliminde fiyatı +$7-8 aşırı tahmin ediyor** (shrinkage).
# LEAR'ın Türkiye'ye transfer olmayan uzun penceresi, çok-pencereli ensemble'ın
# neden işe yaradığını da açıklıyor: kısa pencere kriz rejimini seyreltmeden görüyor.
#
# ## Açık
#
# - `lgb_cqr_v2` DB serisi bozuk (§6) — deney tablosuna ne yazıldığı netleşmeli.
# - **2021-2022 kriz rejiminde varyant karşılaştırması yok** (backfill 2024-08'de
#   başlıyor) — en zorlu rejim ölçüm dışı.
# - EPNet için per-saat seri DB'de yok; `logs/` özet tabloları farklı backtest kurulumu.
