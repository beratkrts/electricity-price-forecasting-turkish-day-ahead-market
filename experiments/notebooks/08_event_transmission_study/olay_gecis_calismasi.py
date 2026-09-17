# %% [markdown]
# # Tavanlı Bir Elektrik Piyasasında Dışsal Şokların Fiyata Geçişi
#
# **Türkiye Gün Öncesi Piyasası, 2021–2026**
#
# Bu notebook uçtan uca çalışır: veri çekimi → haber dikkat endeksi → olay
# ayrıştırması → yapısal geçiş modeli → sentez. Ağır adımlar (plasebo, Tobit MLE)
# `_*.pkl` önbelleğinden okunur; `RECOMPUTE = True` ile yeniden hesaplanır.
#
# ## Araştırma sorusu
#
# > Bağlayıcı fiyat tavanı ve idari olarak belirlenen yakıt fiyatı olan bir
# > elektrik piyasasında, dışsal şoklar (savaş, gaz kesintisi, petrol) toptan
# > fiyata **nasıl** geçer, ve gözlenen fiyat değişkenliğinin ne kadarını tavan
# > bastırır?
#
# ## Neden yeni bir çalışma
#
# İlk katalog (`silver.market_events`, 23 olay) düz bir tabloydu: her satıra bir
# `% değişim` ve bir kontrafaktüel kalıntı. Üç sorun (bkz. `reports/OLAY_HATTI_METODOLOJI.md`):
#
# 1. "Olay" tanımlı bir birim değildi — regülasyon basamağı, arz şoku, savaş
#    yörüngesi ve mevsimsel talep aynı 5 kolonla yan yanaydı.
# 2. `% değişim` penceredeki her şeyi (trend, mevsim, kur, örtüşen olaylar) içeriyordu.
# 3. 23 olayın en fazla 2'si kontrafaktüelin gürültü bandını aşıyordu — yani çıktı
#    büyük ölçüde **null**, ama tablo formatı bunu "her satır bir ölçüm" gibi gösteriyordu.
#
# Bu çalışma olayı **dışsal, tarihlenebilir, bir temele şok** olarak tanımlar
# (tavan/tarife/AUF değişiklikleri olay değil, *kontrol*), ve etkiyi üç yoldan ölçer:
# haber dikkat endeksi (A), iki-kontrafaktüel ayrıştırması (B), yapısal geçiş modeli (C).

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/beratkaratasoglu/etkb_intern_project/electricity_price_forecasting_in_turkish_day_ahead_market")
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine
from sqlalchemy import text

SCR = ROOT / "experiments/scripts"
CACHE = ROOT / "experiments/notebooks/05_crisis_analysis"
RECOMPUTE = False
eng = get_db_engine()
pd.set_option("display.width", 200)
plt.rcParams["figure.figsize"] = (12, 4)

# %% [markdown]
# ---
# # 1. Veri çekimi
#
# Beş kaynak. Hepsi aynı PostgreSQL'e bağlı (canlı sistemle paylaşımlı DB, ayrı git).

# %% [markdown]
# ## 1.1 Fiyat + sansür — `silver.mcp_with_cap`
#
# `MAX(price)` ile tavan çıkarmak yanlış (ay-ortası değişiklikleri kaçırır, fiyatın
# tavana hiç değmediği aylarda tamamen yanılır). `silver.price_cap_official`
# 29 resmî yürürlük kaydı, her satır kaynak haberli. `at_cap` = o saat tavanda mı.

# %%
mcp = pd.read_sql(text("""
    SELECT ts, price_usd, price_try, cap_try, at_cap
    FROM silver.mcp_with_cap ORDER BY ts"""), eng)
mcp["ts"] = pd.to_datetime(mcp.ts, utc=True).dt.tz_convert("Europe/Istanbul")
mcp = mcp.set_index("ts")
print(f"{len(mcp):,} saat  {mcp.index.min().date()} → {mcp.index.max().date()}")
print(f"tavanda geçen saat payı: %{100*mcp.at_cap.mean():.1f}")
mcp_d = mcp.price_usd.resample("D").mean()
mcp_d.index = mcp_d.index.tz_localize(None)  # grafik/join için tz-naive

fig, ax = plt.subplots()
ax.plot(mcp_d.index, mcp_d, lw=0.7, color="#2c3e50")
capshare_m = mcp.at_cap.resample("MS").mean()
ax2 = ax.twinx()
ax2.fill_between(capshare_m.index, capshare_m, alpha=0.15, color="#c0392b")
ax.set_ylabel("MCP $/MWh"); ax2.set_ylabel("tavan payı (kırmızı)", color="#c0392b")
ax.set_title("Gün öncesi fiyatı ve tavanda geçen saat oranı, 2021–2026")
fig.savefig(ROOT / "experiments/notebooks/08_event_transmission_study/f1_price_cap.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# 2021–2022 tavan sık sık bağlıyor (Mart 2022 saatlerin %65'i); 2024 sonrası
# fiyat düşük ve tavandan uzak; 2026 baharında hidro bolluğuyla sıfıra yakın.

# %% [markdown]
# ## 1.2 Yukarı-akış maliyet çıpaları — `silver.upstream_drivers`
#
# TTF (Avrupa gaz referansı), Brent, EUR/TRY, USD/TRY. TTF **olay dedektörü değil**
# (Türkiye'ye özgü olaylar TTF'de yok, FAZ0_BULGULAR.md), dışsal maliyet çıpası.

# %%
if RECOMPUTE:
    import subprocess
    subprocess.run([sys.executable, str(SCR / "build_upstream_drivers.py"), "--write"], check=True)

up = pd.read_sql(text("SELECT * FROM silver.upstream_drivers ORDER BY d"), eng)
up["d"] = pd.to_datetime(up.d); up = up.set_index("d")
print(up[["ttf_eur_mwh", "brent_usd_bbl", "usd_try", "eur_try"]].describe().round(1).to_string())

# %% [markdown]
# ## 1.3 İdari yakıt fiyatı — `silver.gas_tariff_electricity`
#
# BOTAŞ'ın elektrik üretimine uyguladığı doğal gaz tarifesi — marjinal santralin
# **fiilen ödediği** fiyat. Literatür genelde TTF/spot gaz kullanır; Türkiye'de bu
# fiyat idari olarak belirleniyor ve piyasadan ayrışıyor. 30 basamak, boşluksuz.

# %%
tar = pd.read_sql(text("""
    SELECT ts, price_try_1000m3 FROM silver.gas_cost_hourly ORDER BY ts"""), eng)
tar["ts"] = pd.to_datetime(tar.ts, utc=True).dt.tz_convert("Europe/Istanbul")
tar_m = tar.set_index("ts").price_try_1000m3.resample("MS").mean()
tar_m.index = tar_m.index.tz_localize(None)
# USD/MWh_th (10.646 MWh_th / 1000 Sm³)
tar_usd_th = (tar_m / 10.646) / up.usd_try.resample("MS").mean()
print("BOTAŞ elektrik gaz tarifesi (TL/1000m³), basamaklar:")
print(tar_m.groupby(tar_m).head(1).to_string())

# %% [markdown]
# ## 1.4 Temeller — üretim kırılımı, yük, hava
#
# `raw_kgup_hourly` (kaynak bazında planlanan üretim), `raw_load_forecast_hourly`,
# `raw_weather_hourly` (26 bölge ağırlıklı sıcaklık). Bunlar kontrafaktüel modelin girdisi.

# %%
fund = pd.read_sql(text("""
    SELECT k.ts, k.total_mw, k.natural_gas_mw,
           (k.natural_gas_mw+k.import_coal_mw+k.lignite_mw+k.black_coal_mw) thermal_mw,
           (k.dammed_hydro_mw+k.river_hydro_mw) hydro_mw, (k.wind_mw+k.solar_mw) vre_mw,
           l.load_forecast_mw
    FROM raw_kgup_hourly k JOIN raw_load_forecast_hourly l ON k.ts=l.ts ORDER BY k.ts"""), eng)
fund["ts"] = pd.to_datetime(fund.ts, utc=True).dt.tz_convert("Europe/Istanbul")
fund = fund.set_index("ts")
fund["thermal_share"] = fund.thermal_mw / fund.total_mw
fund["hydro_share"] = fund.hydro_mw / fund.total_mw
print(fund[["thermal_share", "hydro_share"]].resample("YS").mean().round(3).to_string())

# %% [markdown]
# Termal pay 2021'de %66 → 2026'da %40. Hidro pay 2026 baharında %28 → %50.
# **Fiyatı belirleyen son santral genelde termal** — bu yüzden termal pay yıllar
# arası taşınıyor, hidro pay taşınmıyor.

# %% [markdown]
# ## 1.5 Haber korpusu — `bronze.news_raw`
#
# enerjigunlugu.net arşivi, 28.102 haber, 2021-01-01 → 2026-08-16. Tek kaynak.
# Kör noktalar ölçüldü: kur şoku ("KKM" → ~0 sonuç) ve Türkiye-özgü gaz kesintileri
# (İran olayı "İran" adıyla değil "sanayiye gaz kısıtı" diye geçiyor).

# %%
news = pd.read_sql(text("""
    SELECT date(published_at) d, count(*) n FROM bronze.news_raw GROUP BY 1"""), eng)
news["d"] = pd.to_datetime(news.d)
print(f"{news.n.sum():,} haber, {len(news)} gün, ort {news.n.mean():.1f}/gün")

# %% [markdown]
# ## 1.6 Kontrafaktüel model — `gold.crisis_counterfactual` (`crisis_cf_v5`)
#
# Ayrı bir LightGBM: **sadece sansürsüz saatlerde** eğitilir (`at_cap=false`), her
# saate tahmin üretir → "tavan olmasaydı fiyat ne olurdu". Fiyat türevli feature
# yok (fundamental varyant). Yakıt maliyeti = BOTAŞ tarifesi. Eğitim 2021'den
# (canlı model 2023'ten — karıştırma). Kod: `src/crisis/analysis_model.py`.
#
# **Kalıntı = gerçek − kontrafaktüel = fiyatın temellerle açıklanamayan kısmı.**
# Boş bandı: p5 ≈ −$4, p95 ≈ +$12 (30 günlük hareketli ort. kalıntı).

# %%
cf = pd.read_sql(text("""
    SELECT ts, actual_usd, counterfactual_usd, residual_usd, at_cap
    FROM gold.crisis_counterfactual
    WHERE model_name='crisis_cf_v5' AND variant='fundamental' ORDER BY ts"""), eng)
cf["ts"] = pd.to_datetime(cf.ts, utc=True).dt.tz_convert("Europe/Istanbul")
cf = cf.set_index("ts")
cf_resid_m = cf.residual_usd.astype(float).resample("MS").mean()
cf_resid_m.index = cf_resid_m.index.tz_localize(None)
print(f"{len(cf):,} saat  {cf.index.min().date()} → {cf.index.max().date()}")
print("aylık ort. kalıntı, |>8|:")
print(cf_resid_m[cf_resid_m.abs() > 8].round(1).to_string())

# %% [markdown]
# 2022'nin çoğu ayında kalıntı +$8…+15 — modelin **açıklayamadığı** bir yükseklik
# (cf_v5 yazarları Eki 2022 / Oca 2023 sapmasını çözemedi). Bu, aşağıdaki
# ayrıştırmada 2022 kalıntı kanalını yorumlarken kritik.

# %% [markdown]
# ---
# # 2. Olay evreni (dondurulmuş)
#
# `OLAY_EVRENI.md` v1. Seçim üç kanaldan: (K1) sabit şok-tipi listesi, (K2) yukarı-akış
# temelinde büyük hareket, (K3) haber dikkat sıçraması (§3'te eklendi). Kurallar:
# dışsal mı → ≥2 kaynakla tarihlenebilir mi → temel veride hareket etti mi → örtüşme.

# %%
EVENTS = pd.DataFrame([
    ("G1", "2021 Avrupa gaz tırmanışı",      "2021-09-01", "TTF",       "fuel_cost", "bağlıyor"),
    ("G2", "Rusya-Ukrayna işgali",            "2022-02-24", "TTF",       "fuel_cost", "bağlıyor"),
    ("G3", "Freeport LNG patlaması",           "2022-06-08", "TTF",       "fuel_cost", "bağlıyor"),
    ("G4", "Nord Stream akış durması",         "2022-09-02", "TTF",       "fuel_cost", "bağlıyor"),
    ("G5", "2022-23 Avrupa gaz normalleşmesi", "2022-10-15", "TTF",       "fuel_cost", "bağlıyor→gevşek"),
    ("G6", "2026 Hürmüz / hat tehdidi",        "2026-02-25", "TTF+Brent", "fuel_cost", "gevşek"),
    ("S1", "İran gaz kesintisi",               "2022-01-19", "gaz MW",    "gas_supply", "bağlıyor %73"),
    ("O1", "ABD-İsrail-İran çatışması",         "2025-06-13", "Brent",     "fuel_cost", "gevşek %8"),
    ("K1", "Kasım-Aralık 2021 lira çöküşü",    "2021-11-18", "USD/TRY",   "fx",        "bağlıyor"),
    ("D1", "Kahramanmaraş depremi",             "2023-02-06", "yük",       "demand",    "gevşek %15"),
    ("H1", "2026 hidro bolluğu / fiyat çöküşü", "2026-02-01", "hidro payı", "hydro",    "gevşek→çok gevşek"),
    ("H2", "2026 hidro normalleşmesi",          "2026-06-01", "hidro payı", "hydro",    "çok gevşek"),
], columns=["id", "label", "onset", "affected", "channel", "regime"])
EVENTS

# %% [markdown]
# **Kontroller** (olay değil): tüm AFL değişiklikleri, BOTAŞ tarife basamakları,
# AUF (1 Nis 2022–), YEKDEM, mevsim/takvim, termal pay rejimi.
#
# **Reddedilenler** (gerekçe `OLAY_EVRENI.md` §6): TANAP azalması, Mavi Akım bakımları
# (planlı → dışsal değil), CBRT başkan değişimi (küçük), Norveç grevi (1 günde çözüldü),
# Süveyş 2021 (gaza etki yok), Groningen (kademeli).

# %% [markdown]
# ---
# # 3. Faz A — Haber dikkat endeksi
#
# **Etki değil dikkat.** Her gün her konu için korpusun o konuya ne kadar baktığı.
# TF-IDF kosinüs benzerliği (gömü modeli sonraki tur — torch yok). Kod:
# `experiments/scripts/build_attention_index.py`.

# %%
if RECOMPUTE:
    import subprocess
    subprocess.run([sys.executable, str(SCR / "build_attention_index.py"), "--write"], check=True)

att = pd.read_sql(text("SELECT * FROM silver.news_attention_daily ORDER BY d"), eng)
att["d"] = pd.to_datetime(att.d); att = att.set_index("d")
TOPICS = ["gas_supply", "geopolitics", "regulation", "hydro_weather", "oil"]
print("konu bazında ort. dikkat ve zirve günü:")
for t in TOPICS:
    s = att[f"{t}_attn"]
    print(f"  {t:14s} ort {s.mean():.4f}  zirve {s.idxmax().date()} ({s.max():.3f})")

# %% [markdown]
# ## 3.1 Endeks doğrulaması — bilinen epizotlar
#
# Beş bilinen epizotun beşi de **doğru konuda** net z-skoru sıçraması veriyor.

# %%
checks = [("geopolitics", "2022-02-15", "2022-03-15", "Rusya-Ukrayna (Şub 2022)"),
          ("geopolitics", "2025-06-10", "2025-06-30", "ABD-İsrail-İran (Haz 2025)"),
          ("gas_supply", "2021-09-01", "2021-12-31", "2021 Avrupa gaz krizi"),
          ("hydro_weather", "2026-02-01", "2026-05-31", "2026 hidro çöküşü"),
          ("oil", "2026-02-25", "2026-03-31", "Hürmüz / hat tehdidi (Mar 2026)")]
print(f"{'epizot':<32}{'konu':<15}{'pencere maks z'}")
for topic, a, b, lab in checks:
    print(f"{lab:<32}{topic:<15}{att.loc[a:b, f'{topic}_z'].max():+.1f}")

# %%
fig, ax = plt.subplots()
z = lambda s: (s - s.mean()) / s.std()
ax.plot(att.index, z(att.geopolitics_attn).rolling(14).mean(), label="jeopolitik dikkat (14g)", color="#c0392b")
ax.plot(att.index, z(mcp_d.reindex(att.index)).rolling(14).mean(), label="MCP $ (14g)", color="#2c3e50")
for _, r in EVENTS[EVENTS.channel.isin(["fuel_cost"])].iterrows():
    ax.axvline(pd.Timestamp(r.onset), color="grey", ls=":", lw=0.8)
ax.legend(); ax.set_title("Jeopolitik haber dikkati vs MCP (z-skor)")
ax.set_ylabel("z-skor")
fig.savefig(ROOT / "experiments/notebooks/08_event_transmission_study/f2_attention.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# geopolitics zirvesi **2025** (z ≈ +6,7) ≫ **2022** (z ≈ +3,7): 2025'te anlatı
# yoğunluğu daha yüksek ama — aşağıda göreceğiz — fiyat geçişi yok.
#
# Endeksin kendiliğinden çıkardığı yeni adaylar: **İsrail-Lübnan Eyl 2024**
# (geopolitics tüm-zaman zirvesi) ve **AUF yargı freni Oca 2023** (regulation zirvesi).

# %% [markdown]
# ## 3.2 Geçiş gecikmesi — local projection
#
# `Y_{t+h} − Y_{t−1} = α + β·dikkat_z_t + kontroller`, aylık, HAC SE, h = 0..6 ay.
# Kod: `experiments/scripts/attention_local_projection.py`.

# %%
import subprocess
lp_out = subprocess.run([sys.executable, str(SCR / "attention_local_projection.py")],
                        capture_output=True, text=True).stdout
print(lp_out)

# %% [markdown]
# **Sinyal yok.** n=68 ay ile dikkat → tarife/MCP/kalıntı hiçbiri anlamlı (|t| < 2).
# Tek "anlamlı" (hidro-hava → MCP, +$25 t=2.2) konu-tanımı artefaktı: aynı konu hem
# "hidro bolluğu" (fiyat ↓) hem "hava stresi" (fiyat ↑) haberini topluyor.
#
# **Yorum:** Dikkat endeksi olay **tarihlemek/doğrulamak** için iyi (5/5 epizot),
# geçiş **öncelemek** için değil. İdari vana politika takvimiyle açılıyor, haber
# takvimiyle değil — haber hızlı, vana yavaş, aralarında ölçülebilir bir kurşun yok.
# Bu, "vana politikayla açılır" iddiasının **negatif kanıtı** — tezi güçlendiriyor.

# %% [markdown]
# ---
# # 4. Faz B — Olay ayrıştırması (iki kontrafaktüel)
#
# Her olay için model, **olay penceresi + 8 gün tampon dışındaki sansürsüz
# saatlerle** eğitilir. İki tahmin:
#
# | | tanım |
# |---|---|
# | **CF2** | pencere için düz tahmin |
# | **CF1** | olayın vurduğu temel (tarife+Brent, ya da gaz üretimi) olay-öncesi seviyeye **sabitlenmiş** |
#
# **toplam = gerçek − CF1** (şok olmasaydı) · **mekanik kanal = CF2 − CF1** (temellere
# geçen) · **kalıntı kanal = gerçek − CF2** (temellerin ötesi)
#
# Plasebo boş dağılımı: sakin pencerelerde (tavan payı <%5, olaydan uzak) aynı üçlü,
# 4 pencere-uzunluğu kovası × 80. Kod: `experiments/scripts/event_decomposition.py`.

# %%
if RECOMPUTE:
    subprocess.run([sys.executable, str(SCR / "event_decomposition.py"),
                    "--placebo", "80", "--write"], check=True)

dec = pd.read_sql(text("""
    SELECT event_id, channel, horizon, win_days, actual_usd, cf1_usd, cf2_usd,
           total_usd, mechanical_usd, residual_usd, at_cap_share, is_lower_bound,
           total_pctile, mech_pctile, resid_pctile
    FROM gold.event_decomposition_v2 WHERE model_name='event_decomp_v1'
    ORDER BY event_id, horizon"""), eng)
dec

# %% [markdown]
# ## 4.1 Merkez sonuç — jeopolitik şokun geçişi
#
# **G2 (Rusya-Ukrayna) vs O1 (ABD-İran) vs G6 (Hürmüz)** — aynı sınıf şok, üç sonuç.

# %%
mid = dec[(dec.event_id.isin(["G2", "O1", "G6"])) & (dec.horizon == "uzun")]
print(mid[["event_id", "win_days", "actual_usd", "cf1_usd", "total_usd",
           "mechanical_usd", "residual_usd", "at_cap_share"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.5))
labels = {"G2": "Rusya-Ukrayna\n2022 (311g)", "O1": "ABD-İran\n2025 (60g)", "G6": "Hürmüz\n2026 (90g)"}
x = np.arange(3)
for i, ev in enumerate(["G2", "O1", "G6"]):
    r = mid[mid.event_id == ev].iloc[0]
    ax.bar(i - 0.2, r.mechanical_usd, 0.4, color="#e67e22", label="mekanik kanal" if i == 0 else "")
    ax.bar(i + 0.2, r.residual_usd, 0.4, color="#16a085", label="kalıntı kanal" if i == 0 else "")
ax.set_xticks(x); ax.set_xticklabels([labels[e] for e in ["G2", "O1", "G6"]])
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("$/MWh"); ax.legend()
ax.set_title("Jeopolitik şokun MCP'ye geçişi: aynı sınıf, üç sonuç")
fig.savefig(ROOT / "experiments/notebooks/08_event_transmission_study/f3_war_contrast.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# - **G2 (2022):** savaş MCP'yi **~$45/MWh** yakıt-maliyeti kanalından yükseltti
#   (idari BOTAŞ tarifesi + Brent). Kalıntı +$26 — ama 2022'nin açıklanamayan
#   model yüksekliğiyle çakışıyor (§1.6), büyüklüğü belirsiz.
# - **O1 (2025):** benzer büyüklükte jeopolitik gerilim, **~$0.** Mekanik ve kalıntı
#   ikisi de plasebo dağılımının ortasında. TTF tepkisiz, BOTAŞ tarifesi kıpırdamadı.
# - **G6 (2026):** TTF Mart 2026'da +%63 sıçradı ama düşük-fiyat rejiminde mekanik
#   kanal +$1,4 — tarife vanası yine açılmadı.
#
# **Tez cümlesi:** geçiş şokun büyüklüğüne değil, **idari yakıt fiyatı kanalının
# açılıp açılmamasına** bağlı.

# %% [markdown]
# ## 4.2 Yakıt-maliyeti kanalının temiz çalıştığı örnek — G3 (Freeport LNG)
#
# ABD LNG kapasitesi −%20 → TTF → BOTAŞ tarifesi → MCP **+$23 mekanik**, kalıntı ~0.
# Kanalın "temiz" çalıştığı en iyi örnek.
#
# ## 4.3 Aşağı yönlü yapışkanlık — G5 (2022-23 gaz düşüşü)
#
# Yakıt maliyeti düştü → mekanik kanal **−$9** ("düşmeliydi") ama fiyat düşmedi.
# G1/G3'teki yukarı geçişle birlikte **asimetri** işareti (§5'te yapısal olarak ölçülüyor).

# %% [markdown]
# ## 4.4 S1 (İran) — arz-MW kanalı ve neden çalışmadığı
#
# CF1 = "kesinti olmasaydı gaz üretimi talep-yoğunluğuyla ölçeklenmiş seviyede kalırdı".

# %%
s1 = dec[dec.event_id == "S1"]
print(s1[["horizon", "win_days", "actual_usd", "cf1_usd", "cf2_usd",
          "total_usd", "mechanical_usd", "residual_usd", "at_cap_share"]].to_string(index=False))

# %% [markdown]
# **Güvenilir sonuç = kalıntı: +$12-13/MWh, ve bir ALT SINIR** (saatlerin %59-75'i
# tavanda). İran kesintisi sırasında MCP, azaltılmış gaz üretimini *gören* temel
# modelin öngördüğünün $12-13 üstündeydi.
#
# **Mekanik kanal GÜVENİLİR DEĞİL (−$6).** Gaz üretim-MW'ı eğitim verisinde **içsel**
# (sistem stresliyken gaz yüksek). Model "yüksek gaz ↔ yüksek fiyat" öğrenmiş.
# Plasebo bunu doğruluyor: sakin pencerelerde de gas_supply "mekanik" medyanı negatif.
# Gaz *miktar* şokunun nedensel etkisi ancak yapısal modelle ölçülebilir.

# %% [markdown]
# ## 4.5 Dürüst kısıtlar (Faz B)
#
# 1. **Güvenilir bileşen = mekanik kanal** (aynı modelin iki tahmininin farkı, ortak
#    katkısal sapma iptal). Kalıntı 2022'de model yüksekliği taşıyor.
# 2. CF1 uzun pencerede kaba: G2-uzunda tarife "savaş olmasaydı 10 ay donardı"
#    varsayımı → mekanik = **üst sınır**.
# 3. cf_v5 tarifeyi feature kullanıyor → "mekanik kanal" = modelin tarifeye atfettiği
#    geçiş, yapısal ölçüm değil (§5).
# 4. Arz-MW ayrıştırması içsellik yüzünden çalışmıyor (§4.4).

# %% [markdown]
# ---
# # 5. Faz C — Yapısal geçiş modeli
#
# ```
# TTF (EUR/MWh) --Link 1--> BOTAŞ tarifesi --÷verim--> marjinal gaz SRMC --Link 3--> MCP
#                (idari, gecikmeli, asimetrik)          (mekanik)         [tavan = sansür]
# ```
# Kod: `experiments/scripts/passthrough_model.py`, `tobit_hourly.py`.

# %%
pt_out = subprocess.run([sys.executable, str(SCR / "passthrough_model.py")],
                        capture_output=True, text=True).stdout
print(pt_out)

# %% [markdown]
# ## 5.1 Link 1 — TTF → BOTAŞ tarifesi: ASİMETRİK
#
# NARDL (asimetrik dağıtılmış gecikme + hata düzeltme), aylık.
#
# | | değer |
# |---|---:|
# | kümülatif geçiş **YUKARI** | **+0,36** (TTF €1 ↑ → tarife ~€0,36 ↑, 3 ay içinde) |
# | kümülatif geçiş **AŞAĞI** | **+0,07** (TTF €1 ↓ → tarife ~€0,07 ↓) |
# | asimetri Wald | **p = 0,008** |
# | gecikme profili | zirve **+1 ay** (Faz 0 ile tutarlı) |
#
# İdari yakıt fiyatı kanalı **asimetrik** — küresel gaz fiyatını yukarı ~5 kat daha
# güçlü izliyor. G5 (2022-23) aşağı-yapışkanlığının yapısal karşılığı.
#
# ## 5.2 Link 3 — marjinal gaz SRMC → MCP
#
# | örneklem | n | **δ (geçiş)** |
# |---|---:|---:|
# | tüm aylar | 69 | 1,72 ± 0,08 |
# | **tavan bağlamıyor (<%15)** | 40 | **1,77 ± 0,07** |
#
# $1/MWh_th yakıt maliyeti artışı → **~$1,77/MWh_e** MCP. Fizik: %55 verimli marjinal
# gaz santrali katsayı 1,8 verir → **neredeyse tam geçiş**. Memory'deki kontrolsüz
# 1,45'i düzeltir.

# %% [markdown]
# ## 5.3 Sansür — saatlik iki-limitli Tobit
#
# `y* = Xβ + ε`, gözlenen `y = min(max(y*, $0), AFL)`. Censored-normal MLE, 49k saat.

# %%
th_out = subprocess.run([sys.executable, str(SCR / "tobit_hourly.py")],
                        capture_output=True, text=True).stdout
print(th_out)

# %% [markdown]
# | yıl | gözlenen std | gizli std | bastırma |
# |---|---:|---:|---:|
# | 2021 | 19,6 | 31,4 | **%61** |
# | 2022 | 58,3 | 52,4 | %−24 *(ölçülemez, bkz. aşağı)* |
# | 2023 | 42,5 | 46,2 | %16 |
# | 2024 | 19,5 | 30,9 | **%60** |
# | 2025 | 19,8 | 35,3 | **%69** |
# | 2026 | 29,8 | 35,5 | %29 |
#
# **Bulgu 1 — normal/düşük rejimde tavan büyük:** saatlik fiyat varyansının
# **~%30-69'unu** bastırıyor. Fiyat çoğu saat serbestçe oluşuyor ama tepe saatler
# (%15-20) tavana çarpıp kırpılıyor; kırpma ort. **$20/MWh**.
#
# **Bulgu 2 — kriz yılı bu yöntemle ÖLÇÜLEMEZ:** 2022'de gizli std < gözlenen std.
# Sansür varyansı azaltır, ters çıkması → 2022'de gözlenen saatlik oynaklık herhangi
# bir temel-fiyat modelinin üretebileceğinden fazla. Kaynak: **idari aygıtın kendisi**
# (tavan Oca-Nis 2022 ayda bir 1.345→2.500 TL adımladı, her seviyede plato). Kukla
# değişkenlerle denendi, düzelmedi. Tez-ilgili: kriz döneminde tavan sadece
# *sınırlamadı*, basamak-ayar süreci fiyatı *daha oynak* yaptı.

# %% [markdown]
# ---
# # 6. Sentez
#
# ## Dört bulgu birbirini destekliyor
#
# | Faz | Bulgu |
# |---|---|
# | **B** olay ayrıştırma | G2 savaşı 2022'de MCP'yi **+$45 yakıt kanalından** yükseltti; O1/G6 (2025/26) **~$0** |
# | **C** Link 1 | İdari vana **asimetrik**: TTF'yi yukarı +0,36, aşağı +0,07 (p=0,008) |
# | **C** Link 3 | Yakıt → MCP geçişi **1,77** — neredeyse tam |
# | **C** Tobit | Tavan düşük rejimde saatlik varyansın **%30-69'unu** bastırıyor; kriz rejiminde *ekliyor* |
# | **A** dikkat endeksi | Haber dikkati 5/5 epizodu yakalıyor ama geçişi **öncelemiyor** — vana politikayla açılır |
# | **S1** İran | Gaz kesintisi MCP'yi temellerin **≥$12-13 üstüne** çıkardı (tavan alt sınır) |
#
# ## Tek cümle
#
# > Tavanlı, idari yakıt fiyatlı bir elektrik piyasasında dışsal şoklar toptan
# > fiyata ancak **idari vana (BOTAŞ tarifesi) açıldığında** geçiyor; vana yukarı
# > hızlı, aşağı yapışkan (0,36 vs 0,07); ve tavan, fiyatın serbestçe oluştuğu
# > saatlerde varyansın yarıdan fazlasını bastırıyor — kriz döneminde ise
# > basamak-ayarıyla fiyatı daha oynak yapıyor.
#
# ## Katkı (literatürdeki boşluk)
#
# Türkiye'de fiyat tavanının etkisini ölçen tek çalışma (Energy Policy 2022,
# 2016-2018 verisi) **etkiyi sıfır** buluyor — çünkü o dönemde tavan bağlamıyordu.
# Bu çalışma **tavanın fiilen bağladığı rejimde** aynı soruyu soruyor ve üç
# mekanizmayı (bağlayıcı tavan + idari gaz fiyatı + büyük hidro salınımı) birlikte
# ölçülmüş serilerle ele alıyor.
#
# ## Bilinen eksikler / v2
#
# 1. Faz C: heteroskedastik σ, kriz rejimi ayrı model, Link 2'nin (tarife→SRMC)
#    açık ölçümü, AUF serisi kontrolü.
# 2. Faz A: gömü modeli (torch), konu ayrımı (bolluk vs stres), günlük LP.
# 3. Faz B: cf_v5 sapmasının 2022 kalıntısından ayrıştırılması; K1 (kur) ve D1
#    (deprem) kanalları.
# 4. İkinci haber kaynağı (GDELT hedefli, kur şoku için).
# 5. Yeni adaylar: İsrail-Lübnan Eyl 2024.
