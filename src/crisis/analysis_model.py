"""
Analiz modeli — kriz/olay etkisi için kontrafaktüel üretici.

CRISIS_ANALYSIS_PLAN.md §3.4 ve §8 adım 6.

CANLIYA ASLA GİRMEZ. Canlı fiyat modelinden (`scripts/predict_daily_pipeline.py`,
`lgb_lag0_v2`) üç yerde bilinçli olarak ayrılır:

1. EĞİTİM PENCERESİ 2021'DEN. Canlı model 2023'ten eğitiliyor çünkü 2021-2022
   azami fiyat limiti rejimiydi ve canlı davranışı bozardı. Analiz modelinin işi
   tam olarak o dönem hakkında konuşmak, dolayısıyla o veriye ihtiyacı var.

2. SADECE SANSÜRSÜZ SAATLERDE EĞİTİLİR (`at_cap = false`, silver.mcp_with_cap).
   Tavandaki saatler eğitime girerse model "sıkışıklıkta fiyat tavandır" öğrenir,
   kalıntı sıfırlanır ve tam da ölçmek istediğimiz şey görünmez olur.
   Tahmin ise TÜM saatler için üretilir — tavandaki saatlerde kontrafaktüel
   "tavan olmasaydı fiyat ne olurdu"yu söyler.

3. LAG0 FUNDAMENTALLER KULLANILIR. Canlı model tahmin anında (D günü 04:00)
   yarının KGÜP/yükünü bilemez, o yüzden pre-forecast'lara dayanır. Analiz modeli
   geçmişe bakıyor; o saatin GERÇEKLEŞEN üretim kırılımını görebilir.
   Bu bilinçli bir seçim: ölçtüğümüz şey "sürpriz" değil "AÇIKLANAMAYAN".
   Soru "gaz düşüşü fiyat yükselişini açıklıyor mu" — modelin gaz düşüşünü
   görmesi gerekiyor ki cevap anlamlı olsun. Kalıntı ≈ 0 ise "evet, gaz
   düşüşü verilince fiyat normal"; kalıntı > 0 ise "gaz düşüşünün ötesinde
   bir şey var".

   Bu yüzden `gold.kgup_load_pre_forecasts` join'i BİLİNÇLİ OLARAK YOK.
   Ayrıca pratik bir zorunluluk: pre-forecast'lar 2024-08-13'te başlıyor.
   Join edilseydi feature'ın anlamı serinin ortasında sessizce değişirdi
   (2021-2024 gerçekleşen, 2024+ tahmin) — dönemler arası kalıntı
   karşılaştırması anlamsızlaşırdı.

İKİ VARYANT — hangi soruyu sorduğuna göre:

  'fundamental'    : fiyat türevli hiçbir feature yok (mcp lag/rolling, rejim
                     bayrakları, SMF lag). Olay etkisi ölçümü için DOĞRU olan bu.
                     Çok günlük bir krizde `mcp_usd_lag_24` modele dünkü şok
                     fiyatını verir; model ikinci günden itibaren şoku "tahmin
                     eder" ve kalıntı çöker. 10 günlük İran epizodunun 9 günü
                     görünmez olur.
  'autoregressive' : canlı modelin feature seti + lag0 blok. Nokta doğruluğu
                     yüksektir ama yukarıdaki nedenle olay ölçemez.
                     Karşılaştırma/teşhis için tutuluyor.

KALICI SINIR — TAVANDAKİ KONTRAFAKTÜEL ALT SINIRDIR.
Ağaç modeli eğitimde gördüğü maksimum hedefin üstünü tahmin edemez. Model
sansürsüz saatlerde eğitildiği için tavan üstü fiyat hiç görmedi. Dolayısıyla
tavandaki saatler için ürettiği kontrafaktüel gerçek değerin ALT SINIRIDIR;
"fiyat tavan olmasaydı en az bu kadardı" denebilir, "tam olarak bu kadardı"
denemez. Bu, yöntemin düzeltilebilir bir hatası değil, doğasında olan bir kısıt.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import get_db_engine
from src.features.feature_engineering import build_robust_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster

logger = logging.getLogger("CrisisAnalysisModel")

# Analiz modelinin eğitim tabanı. Canlı modelin TRAINING_DATA_START'ı (2023-01-01)
# ile KARIŞTIRMA — bunlar kasıtlı olarak farklı ve farklı kalmalı.
ANALYSIS_DATA_START = "2021-01-01"

# Canlı şampiyonun (lgb_lag0_v2) hiperparametreleri. Aynı tutuluyor ki kalıntı
# farkı feature/pencere kararlarından gelsin, ayar farkından değil.
ANALYSIS_PARAMS: Dict = {
    "objective": "quantile",
    "alpha": 0.50,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 63,
    "min_child_samples": 10,
    "verbose": -1,
    "random_state": 42,
}

# GRF tabanlı yakıt maliyeti sütunları. 'tariff' kaynağı seçildiğinde bunlar
# feature setinden çıkarılır ve yerlerine gas_tariff_usd_mwh_lag0 girer.
# Gerekçe: GRF, EPİAŞ'ın gaz piyasası REFERANS fiyatı; santrallerin fiilen ödediği
# BOTAŞ tarifesi. İkisi normalde örtüşüyor ama Kas 2021 - Haz 2023 arasında
# ayrışıyorlar (ort |oran-1| = 0,17) ve kalıntı tam orada bozuluyor (r = 0,81).
GRF_COST_COLS = ["natural_gas_grf_usd_lag0", "natural_gas_grf_lag_48"]
TARIFF_COST_COLS = ["gas_tariff_usd_mwh_lag0"]

# Geçmiş fiyattan türeyen her şey. 'fundamental' varyantta bunlar dışarıda —
# gerekçesi modül docstring'inde.
PRICE_DERIVED_COLS = [
    "mcp_usd_lag_24", "mcp_usd_lag_48", "mcp_usd_lag_168",
    "mcp_usd_roll_mean_24h", "mcp_usd_roll_std_24h", "mcp_usd_roll_mean_7d",
    "is_low_price_regime", "is_zero_price_hour", "price_volatility_24h",
    "smp_usd_lag_48",
]

# O saatin gerçekleşen arz/talep kırılımı. Canlı modelde YOK (tahmin anında
# bilinemez), analiz modelinin çekirdeği.
LAG0_FUNDAMENTAL_COLS = [
    "kgup_total_lag0", "kgup_gas_lag0", "kgup_coal_lag0",
    "kgup_hydro_lag0", "kgup_wind_lag0", "kgup_solar_lag0",
    "load_forecast_lag0", "actual_cons_lag0", "actual_gen_lag0",
    "supply_demand_gap_lag0", "reserve_margin_lag0",
    "gas_share_lag0", "coal_share_lag0", "thermal_share_lag0",
    "renewable_share_lag0", "hydro_share_lag0",
    "natural_gas_grf_usd_lag0", "brent_oil_usd_lag0", "usd_try_lag0",
    "hydro_water_energy_lag0",
]

# Yakıt maliyetinin GEÇMİŞİ. v1'de yoktu ve ölçülebilir bir hataya yol açtı:
# model gaz maliyetinin elektrik fiyatına ANINDA geçtiğini varsayıyordu. Gerçekte
# geçiş gecikmeli: gaz Eki 2022–Mar 2023 arasında %64 düşerken fiyat o hızda
# düşmedi, kalıntı +$13 şişti. Tüm seride kalıntı ile GRF'nin 3 aylık değişimi
# arasındaki korelasyon -0,43.
#
# ÖLÇÜLMÜŞ TAKAS — hangi setin seçileceği ideolojik değil, deneysel bir karar:
#   'long'  (v2) mutlak seviye sütunları içeriyor (ma_30d, ma_90d $/MWh olarak).
#           Eki22-Mar23 sapmasını düzeltti AMA İran sinyalini yuttu (+$16,5 -> +$0,2):
#           90 günlük ortalama "anlık maliyet"ten çok "hangi dönemdeyiz" demek, ve
#           dönemi tanıyan model o dönemde ne olduysa onu da açıklıyor.
#   'short' (v3) sadece 7 günlük pencere. Dönem kodlamaz ama gaz günden güne çok
#           yavaş hareket ettiği için lag0'a yeni bilgi de katmayabilir.
#   'ratio' (v4) uzun pencere AMA sadece birimsiz sütunlar (oran ve % değişim).
#           Geçiş gecikmesini görür, dönem seviyesini taşımaz — hipotez bu.
#   None    (v1) hiçbiri.
FUEL_DYNAMICS_SETS = {
    "long": [
        "gas_usd_ma_30d", "gas_usd_ma_90d", "gas_usd_vs_ma90",
        "gas_usd_chg_30d", "gas_usd_chg_90d",
        "brent_usd_ma_30d", "brent_usd_vs_ma90",
    ],
    "short": [
        "gas_usd_ma_7d", "gas_usd_vs_ma7", "gas_usd_chg_7d",
        "brent_usd_vs_ma7",
    ],
    "ratio": [
        "gas_usd_vs_ma7", "gas_usd_vs_ma30", "gas_usd_vs_ma90",
        "gas_usd_chg_7d", "gas_usd_chg_30d", "gas_usd_chg_90d",
        "brent_usd_vs_ma30", "brent_usd_vs_ma90",
    ],
}

TARGET_COL = "mcp_price_usd"


def load_analysis_data(start: str = ANALYSIS_DATA_START) -> pd.DataFrame:
    """
    2021'den itibaren ham seriyi + resmî tavan bayrağını yükler.

    Canlı `load_all_historical_data()`'dan iki farkı var:
      - `gold.kgup_load_pre_forecasts` join'i YOK (bkz. modül docstring'i)
      - `silver.mcp_with_cap` join'i VAR: `cap_try` ve `at_cap` geliyor
    """
    engine = get_db_engine()
    master_sql = text("""
        SELECT
            m.ts,
            m.price_usd AS mcp_price_usd,
            m.price_try AS mcp_price_try,
            cap.cap_try,
            cap.at_cap,
            s.system_marginal_price_try AS smp_price_try,
            l.load_forecast_mw,
            k.total_mw AS kgup_total_mw,
            k.natural_gas_mw AS kgup_gas_mw,
            k.wind_mw AS kgup_wind_mw,
            k.solar_mw AS kgup_solar_mw,
            k.dammed_hydro_mw + k.river_hydro_mw AS kgup_hydro_mw,
            k.import_coal_mw + k.lignite_mw + k.black_coal_mw AS kgup_coal_mw,
            g.total_mw AS actual_gen_total_mw,
            c.consumption_mw AS actual_cons_mw,
            w.turkey_weighted_temperature_c AS temperature_c,
            wf.turkey_weighted_temperature_forecast_c AS temperature_forecast_c,
            mc.usd_try,
            mc.brent_oil_usd,
            ng.gas_reference_price_try AS natural_gas_grf_try,
            gt.price_try_1000m3 AS gas_tariff_try_1000m3,
            gt.tariff_is_gap AS gas_tariff_is_gap,
            wp.hydro_water_energy_mwh
        FROM raw_mcp_hourly m
        JOIN silver.mcp_with_cap cap ON m.ts = cap.ts
        LEFT JOIN raw_smp_hourly s ON m.ts = s.ts
        LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
        LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
        LEFT JOIN raw_actual_generation_hourly g ON m.ts = g.ts
        LEFT JOIN raw_actual_consumption_hourly c ON m.ts = c.ts
        LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
        LEFT JOIN raw_weather_forecast_hourly wf ON m.ts = wf.ts
        LEFT JOIN raw_macro_daily mc ON DATE(m.ts) = mc.entry_date
        LEFT JOIN raw_natural_gas_daily ng ON DATE(m.ts) = ng.entry_date
        LEFT JOIN silver.gas_cost_hourly gt ON m.ts = gt.ts
        LEFT JOIN (
            SELECT DATE(date_time) AS entry_date, SUM(water_energy_provision_mwh) AS hydro_water_energy_mwh
            FROM raw_master_water_energy_provision
            GROUP BY DATE(date_time)
        ) wp ON DATE(m.ts) = wp.entry_date
        WHERE m.ts >= :analysis_start
        ORDER BY m.ts ASC;
    """)

    with engine.connect() as conn:
        df_raw = pd.read_sql(master_sql, conn, params={"analysis_start": start})

    ts_series = pd.to_datetime(df_raw["ts"])
    if ts_series.dt.tz is None:
        df_raw["ts"] = ts_series.dt.tz_localize("Europe/Istanbul")
    else:
        df_raw["ts"] = ts_series.dt.tz_convert("Europe/Istanbul")
    df_raw = df_raw.set_index("ts").sort_index()

    for col in ["usd_try", "brent_oil_usd", "natural_gas_grf_try",
                "hydro_water_energy_mwh", "gas_tariff_try_1000m3"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].ffill().bfill()

    # at_cap NULL olamaz: view LEFT JOIN LATERAL kullanıyor, tavan kaydı yoksa
    # (2021-01, ilk yürürlükten önce) at_cap NULL döner. Sansürsüz say — o ayda
    # bilinen bir tavan yok, yani fiyat serbest kabul edilir.
    if "at_cap" in df_raw.columns:
        n_null = int(df_raw["at_cap"].isna().sum())
        if n_null:
            logger.warning(
                "%d saatte tavan kaydı yok (ilk yürürlük öncesi) — sansürsüz sayılıyor.", n_null
            )
        df_raw["at_cap"] = df_raw["at_cap"].fillna(False).astype(bool)

    return df_raw


def build_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canlı `build_robust_features()` + lag0 gerçekleşen fundamental bloğu.

    Not: `df`'te pre-forecast sütunları olmadığı için robust hattındaki
    `predicted_*_lag0` fallback'leri devreye girer ve gerçekleşen değerlere
    (load_forecast_mw, kgup_solar_mw, kgup_wind_mw) eşitlenir. Yani
    `net_load_lag0`, `renewable_pressure_ratio_lag0` vb. burada zaten
    "gerçekleşen" anlamındadır — ayrıca yeniden türetmiyoruz.
    """
    df_feat = build_robust_features(df)

    def col(name: str) -> pd.Series:
        return df_feat[name].astype(float) if name in df_feat.columns else pd.Series(np.nan, index=df_feat.index)

    kgup_total = col("kgup_total_mw")
    kgup_gas = col("kgup_gas_mw")
    kgup_coal = col("kgup_coal_mw")
    kgup_hydro = col("kgup_hydro_mw")
    kgup_wind = col("kgup_wind_mw")
    kgup_solar = col("kgup_solar_mw")
    load_fc = col("load_forecast_mw")

    df_feat["kgup_total_lag0"] = kgup_total
    df_feat["kgup_gas_lag0"] = kgup_gas
    df_feat["kgup_coal_lag0"] = kgup_coal
    df_feat["kgup_hydro_lag0"] = kgup_hydro
    df_feat["kgup_wind_lag0"] = kgup_wind
    df_feat["kgup_solar_lag0"] = kgup_solar
    df_feat["load_forecast_lag0"] = load_fc
    df_feat["actual_cons_lag0"] = col("actual_cons_mw")
    df_feat["actual_gen_lag0"] = col("actual_gen_total_mw")

    total_safe = kgup_total.replace(0, np.nan)
    load_safe = load_fc.replace(0, np.nan)

    df_feat["supply_demand_gap_lag0"] = load_fc - kgup_total
    df_feat["reserve_margin_lag0"] = ((kgup_total - load_fc) / load_safe).astype(float)
    df_feat["gas_share_lag0"] = (kgup_gas / total_safe).astype(float)
    df_feat["coal_share_lag0"] = (kgup_coal / total_safe).astype(float)
    df_feat["thermal_share_lag0"] = ((kgup_gas + kgup_coal) / total_safe).astype(float)
    df_feat["renewable_share_lag0"] = ((kgup_wind + kgup_solar + kgup_hydro) / total_safe).astype(float)
    df_feat["hydro_share_lag0"] = (kgup_hydro / total_safe).astype(float)

    # Yakıt maliyeti EŞ ZAMANLI (48h lag'li değil). Canlıda 48h gecikme sızıntı
    # koruması; burada açıklayıcı model kuruyoruz, o saatin maliyeti serbest.
    usd_try = col("usd_try")
    fx_safe = usd_try.replace(0, np.nan)
    df_feat["usd_try_lag0"] = usd_try
    df_feat["natural_gas_grf_usd_lag0"] = (col("natural_gas_grf_try") / fx_safe).astype(float)
    df_feat["brent_oil_usd_lag0"] = col("brent_oil_usd")
    df_feat["hydro_water_energy_lag0"] = col("hydro_water_energy_mwh")

    # BOTAŞ elektrik üretim tarifesi -> $/MWh(gaz). GRF'nin yerine geçen aday.
    # 1000 Sm3 = 10,646 MWh (9155 Kcal/Sm3 üst ısıl değer, BOTAŞ tarife dipnotu).
    # Santral verimi sabit olduğu için modele ayrıca verilmiyor; ağaç öğrenir.
    df_feat["gas_tariff_usd_mwh_lag0"] = (
        (col("gas_tariff_try_1000m3") / 10.646) / fx_safe
    ).astype(float)

    # --- Yakıt maliyeti dinamiği (bkz. FUEL_DYNAMICS_SETS gerekçesi) ---
    # Tümü üretiliyor, seçimi get_analysis_feature_columns yapıyor. Pencereler
    # saat cinsinden: 7g=168, 30g=720, 90g=2160. Hepsi geriye dönük (trailing),
    # o saati de içeriyor — açıklayıcı model, sızıntı sorunu yok.
    gas_usd = df_feat["natural_gas_grf_usd_lag0"]
    brent = df_feat["brent_oil_usd_lag0"]

    # Sütunlar tek seferde eklenir: döngü içinde tek tek atamak çerçeveyi
    # parçalıyor ve pandas PerformanceWarning basıyor (notebook çıktısını kirletir).
    yakit = {}
    for src, name in ((gas_usd, "gas"), (brent, "brent")):
        for days, hours, min_p in ((7, 168, 48), (30, 720, 168), (90, 2160, 336)):
            ma = src.rolling(hours, min_periods=min_p).mean()
            yakit[f"{name}_usd_ma_{days}d"] = ma
            yakit[f"{name}_usd_vs_ma{days}"] = (src / ma.replace(0, np.nan)).astype(float)
            yakit[f"{name}_usd_chg_{days}d"] = src.pct_change(hours, fill_method=None).astype(float)

    return pd.concat([df_feat, pd.DataFrame(yakit, index=df_feat.index)], axis=1)


def get_analysis_feature_columns(
    variant: str = "fundamental",
    df: Optional[pd.DataFrame] = None,
    fuel_dynamics: Optional[str] = None,
    gas_cost: str = "grf",
) -> List[str]:
    """
    Varyanta göre feature listesi. Bkz. modül docstring'i.

    `fuel_dynamics`: None (v1, varsayılan) | 'long' (v2) | 'short' (v3) | 'ratio' (v4).
    Varsayılanın None olması ölçülmüş bir karar: 'long' İran sinyalini yutuyor.
    """
    if variant not in ("fundamental", "autoregressive"):
        raise ValueError(f"Bilinmeyen varyant: {variant!r} (fundamental | autoregressive)")

    cols = list(get_feature_columns("robust")) + list(LAG0_FUNDAMENTAL_COLS)
    if gas_cost == "tariff":
        cols = [c for c in cols if c not in GRF_COST_COLS] + list(TARIFF_COST_COLS)
    elif gas_cost != "grf":
        raise ValueError(f"Bilinmeyen gaz maliyeti kaynağı: {gas_cost!r} (grf | tariff)")
    if fuel_dynamics:
        if fuel_dynamics not in FUEL_DYNAMICS_SETS:
            raise ValueError(f"Bilinmeyen yakıt seti: {fuel_dynamics!r} ({list(FUEL_DYNAMICS_SETS)})")
        cols += list(FUEL_DYNAMICS_SETS[fuel_dynamics])
    if variant == "fundamental":
        cols = [c for c in cols if c not in PRICE_DERIVED_COLS]

    # Sıra korunarak tekilleştir
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)

    if df is not None:
        out = [c for c in out if c in df.columns]
    return out


def prepare_model_frame(df_feat: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Eğitim/tahmin için kullanılabilir satırlar. dropna SADECE kullanılan sütunlarda."""
    needed = [c for c in feature_cols + [TARGET_COL, "at_cap"] if c in df_feat.columns]
    df_model = df_feat.dropna(subset=needed).copy()
    return df_model


def fit_counterfactual(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    params: Optional[Dict] = None,
) -> LightGBMForecaster:
    """
    Kontrafaktüel modeli eğitir. GİRDİ ZATEN MASKELENMİŞ OLMALI —
    maskeyi burada uygulamıyoruz ki çağıran ne yaptığını görsün.
    """
    forecaster = LightGBMForecaster(params=dict(params or ANALYSIS_PARAMS))
    forecaster.fit(df_train[feature_cols], df_train[TARGET_COL].values, feature_columns=feature_cols)
    return forecaster


def uncensored_mask(df: pd.DataFrame) -> pd.Series:
    """Eğitim maskesi: sadece tavana değmemiş saatler."""
    return ~df["at_cap"].astype(bool)


def blocked_oof_counterfactual(
    df_model: pd.DataFrame,
    feature_cols: List[str],
    block_days: int = 28,
    buffer_days: int = 8,
    params: Optional[Dict] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Bloklu örneklem-dışı kontrafaktüel.

    Her blok sırayla dışarı alınır; model bloğun DIŞINDAKİ sansürsüz saatlerle
    eğitilir, blok içindeki TÜM saatler için tahmin üretir.

    Neden ileri yönlü walk-forward değil: genişleyen pencere Ocak 2022 için
    sadece 2021'i görür ve ağaç modeli eğitimde gördüğü maksimumun üstünü
    tahmin edemez — serinin ilk büyük şoku sistematik olarak "açıklanamayan"
    çıkardı, bu yöntem artefaktı olurdu. Bloklu şema hem öncesini hem
    sonrasını kullanır; olay penceresi yine de örneklem dışıdır.

    `buffer_days` bloğun iki yanında eğitimden çıkarılan tampon. 168 saatlik
    lag'ler var; tamponsuz, bloğun hemen ardındaki eğitim satırları blok
    fiyatlarını lag feature'ı olarak taşır ve blok dolaylı yoldan sızar.
    """
    if progress:
        logger.info(
            "Bloklu OOF: %d satır, %d feature, blok=%dg, tampon=%dg",
            len(df_model), len(feature_cols), block_days, buffer_days,
        )

    idx = df_model.index
    start, end = idx.min().normalize(), idx.max().normalize()
    block_edges = pd.date_range(start=start, end=end + pd.Timedelta(days=block_days), freq=f"{block_days}D")

    train_ok = uncensored_mask(df_model)
    buffer = pd.Timedelta(days=buffer_days)
    results = []

    for fold_id, (b_start, b_end) in enumerate(zip(block_edges[:-1], block_edges[1:])):
        in_block = (idx >= b_start) & (idx < b_end)
        if not in_block.any():
            continue

        outside = ((idx < b_start - buffer) | (idx >= b_end + buffer)) & train_ok.values
        n_train = int(outside.sum())
        if n_train < 2000:
            logger.warning(
                "Blok %s: eğitim satırı yetersiz (%d), atlanıyor.", b_start.date(), n_train
            )
            continue

        forecaster = fit_counterfactual(df_model.loc[outside], feature_cols, params=params)
        preds = forecaster.predict(df_model.loc[in_block])

        block_df = df_model.loc[in_block, [TARGET_COL, "at_cap"]].copy()
        block_df["counterfactual_usd"] = preds
        block_df["fold_id"] = fold_id
        block_df["n_train"] = n_train
        block_df["train_max_usd"] = float(df_model.loc[outside, TARGET_COL].max())
        results.append(block_df)

        if progress and fold_id % 10 == 0:
            logger.info("  blok %3d  %s  eğitim=%d", fold_id, b_start.date(), n_train)

    if not results:
        raise RuntimeError("Hiçbir blok üretilemedi — veri aralığını kontrol et.")

    oof = pd.concat(results).sort_index()
    oof["residual_usd"] = oof[TARGET_COL] - oof["counterfactual_usd"]
    # Tavandaki saatlerde kontrafaktüel alt sınırdır: model eğitimde tavan üstü
    # fiyat görmedi, o yüzden oradaki kalıntı da alt sınır (gerçekte daha büyük).
    oof["is_lower_bound"] = oof["at_cap"].astype(bool)
    return oof
