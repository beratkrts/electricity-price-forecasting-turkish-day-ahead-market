#!/usr/bin/env python3
"""C — yapısal geçiş modeli. OLAY_ANALIZI_V2_PLAN.md §5.3.

Zincir:  TTF (EUR/MWh) → BOTAŞ elektrik gaz tarifesi → marjinal gaz SRMC → MCP
Tavan = sağdan sansür (AFL). Aylık panel, 2021-01 → 2026-09 (69 ay).

Link 1 — TTF → tarife:  NARDL (asimetrik dağıtılmış gecikme + ECM).
                        "İdari vana" — Faz 0'da eşzamanlı 0 / +1 ay +0,58 ölçülmüştü.
Link 3 — SRMC → MCP:    Kontrollü OLS, Newey-West (HAC) SE. Üç örneklem:
                        tümü / tavan payı <%15 (bağlamıyor) / >%30 (ağır sansür).
                        Alt-örneklem farkı = sansür zayıflatması.
Sansür  — iki-limitli Tobit (aylık, cap_share ağırlıklı sezgisel). Tam saatlik
          Tobit sonraki tur.
Varyans — Link 3 (bağlamayan) katsayılarıyla gizli MCP simüle, gözlenen (sansürlü)
          ile karşılaştır → "tavan varyansın ne kadarını bastırıyor".

    .venv/bin/python experiments/scripts/passthrough_model.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

MJ_PER_1000M3 = 10.646  # MWh_th / 1000 Sm³ (BOTAŞ tarife dipnotu, 9155 kcal/Sm³)
PANEL = ROOT / "experiments/notebooks/05_crisis_analysis/_passthrough_panel.pkl"

PANEL_SQL = """
WITH m AS (
  SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') mon,
         avg(price_usd) mcp_usd, avg(price_try) mcp_try,
         avg(CASE WHEN at_cap THEN 1 ELSE 0 END) cap_share,
         avg(price_usd) FILTER (WHERE NOT at_cap) mcp_usd_unc,
         max(cap_try) cap_try
  FROM silver.mcp_with_cap GROUP BY 1),
g AS (SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') mon,
         avg(price_try_1000m3) tariff_try FROM silver.gas_cost_hourly GROUP BY 1),
u AS (SELECT date_trunc('month', d::timestamp) mon,
         avg(ttf_eur_mwh) ttf, avg(brent_usd_bbl) brent,
         avg(usd_try) usd_try, avg(eur_try) eur_try FROM silver.upstream_drivers GROUP BY 1),
k AS (SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') mon,
         avg((natural_gas_mw+import_coal_mw+lignite_mw+black_coal_mw)/NULLIF(total_mw,0)) thermal_share,
         avg((dammed_hydro_mw+river_hydro_mw)/NULLIF(total_mw,0)) hydro_share,
         avg(total_mw) gen FROM raw_kgup_hourly GROUP BY 1),
l AS (SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') mon,
         avg(load_forecast_mw) load FROM raw_load_forecast_hourly GROUP BY 1)
SELECT m.mon, m.mcp_usd, m.mcp_try, m.mcp_usd_unc, m.cap_share, m.cap_try,
       g.tariff_try, u.ttf, u.brent, u.usd_try, u.eur_try,
       k.thermal_share, k.hydro_share, k.gen, l.load
FROM m JOIN g USING(mon) JOIN u USING(mon) JOIN k USING(mon) JOIN l USING(mon)
ORDER BY m.mon
"""


def load_panel() -> pd.DataFrame:
    df = pd.read_sql(text(PANEL_SQL), get_db_engine())
    df["mon"] = pd.to_datetime(df["mon"])
    df = df.set_index("mon")
    # tarife → EUR/MWh_th (TTF ile aynı birim) ve USD/MWh_th
    df["tariff_eur_th"] = (df.tariff_try / MJ_PER_1000M3) / df.eur_try
    df["tariff_usd_th"] = (df.tariff_try / MJ_PER_1000M3) / df.usd_try
    df["cap_usd"] = df.cap_try / df.usd_try
    df["ln_gen"] = np.log(df.gen)
    return df


# ---------------------------------------------------------------- OLS + HAC

def ols_hac(y: np.ndarray, X: np.ndarray, names: list[str], L: int = 3) -> pd.DataFrame:
    """OLS + Newey-West (HAC) standart hatalar."""
    import statsmodels.api as sm
    m = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": L})
    return pd.DataFrame({"coef": m.params, "se": m.bse, "t": m.tvalues, "p": m.pvalues},
                        index=names), m


# ---------------------------------------------------------------- Link 1: NARDL

def link1_nardl(df: pd.DataFrame, K: int = 3):
    import statsmodels.api as sm
    d = df[["tariff_eur_th", "ttf", "eur_try"]].dropna().copy()
    d["dtar"] = d.tariff_eur_th.diff()
    d["dttf"] = d.ttf.diff()
    d["dttf_p"] = d.dttf.clip(lower=0)
    d["dttf_n"] = d.dttf.clip(upper=0)
    # ECM terimi: önceki ay tarife − uzun-vade TTF ilişkisi
    lr = sm.OLS(d.tariff_eur_th, sm.add_constant(d.ttf), missing="drop").fit()
    d["ecm"] = (d.tariff_eur_th - lr.predict(sm.add_constant(d.ttf))).shift(1)

    cols = {"const": 1.0, "ecm": d.ecm}
    for k in range(0, K + 1):
        cols[f"dttf_p_{k}"] = d.dttf_p.shift(k)
        cols[f"dttf_n_{k}"] = d.dttf_n.shift(k)
    Xd = pd.DataFrame(cols, index=d.index).dropna()
    yy = d.dtar.reindex(Xd.index)
    res, m = ols_hac(yy.values, Xd.values, list(Xd.columns))

    bp = res.loc[[f"dttf_p_{k}" for k in range(K + 1)], "coef"].sum()
    bn = res.loc[[f"dttf_n_{k}" for k in range(K + 1)], "coef"].sum()
    # asimetri Wald testi
    Rp = np.array([[1.0 if n.startswith("dttf_p") else 0.0 for n in Xd.columns]])
    Rn = np.array([[1.0 if n.startswith("dttf_n") else 0.0 for n in Xd.columns]])
    wald = m.t_test(Rp - Rn)
    return res, dict(cum_up=bp, cum_down=bn, lr_slope=float(lr.params.iloc[1]),
                     asym_p=float(wald.pvalue), n=int(len(Xd)))


# ---------------------------------------------------------------- Link 3: SRMC → MCP

CONTROLS = ["thermal_share", "hydro_share", "ln_gen"]


def link3(df: pd.DataFrame, mask=None, label=""):
    import statsmodels.api as sm
    d = df.copy()
    if mask is not None:
        d = d[mask]
    d = d[["mcp_usd", "mcp_usd_unc", "tariff_usd_th", *CONTROLS, "cap_share"]].dropna()
    y = d.mcp_usd_unc.fillna(d.mcp_usd).values   # sansürsüz saat ort., yoksa genel
    X = sm.add_constant(d[["tariff_usd_th", *CONTROLS]]).values
    res, m = ols_hac(y, X, ["const", "srmc", *CONTROLS])
    return res, dict(label=label, n=int(len(d)), srmc=float(res.loc["srmc", "coef"]),
                     srmc_se=float(res.loc["srmc", "se"]), r2=float(m.rsquared))


# ---------------------------------------------------------------- Tobit (aylık, sezgisel)

def tobit_monthly(df: pd.DataFrame):
    """İki-limitli Tobit yaklaşımı: her ay, tavanda geçen saat oranı kadar sansürlü
    kabul. Gizli aylık ort. = gözlenen + sansür düzeltmesi (Mills oranı sezgisel).
    Tam saatlik Tobit sonraki tur — bu bir ÜST-SINIR düzeltmesi."""
    from scipy.stats import norm
    d = df[["mcp_usd", "mcp_usd_unc", "cap_share", "cap_usd", "tariff_usd_th", *CONTROLS]].dropna().copy()
    # sansürsüz aylardan gizli süreç (OLS)
    import statsmodels.api as sm
    clean = d[d.cap_share < 0.05]
    X = sm.add_constant(clean[["tariff_usd_th", *CONTROLS]])
    mdl = sm.OLS(clean.mcp_usd, X).fit()
    sigma = np.sqrt(mdl.scale)
    Xall = sm.add_constant(d[["tariff_usd_th", *CONTROLS]])
    d["latent_mean"] = mdl.predict(Xall)
    # sansürlü aylarda gerçek gizli ort.: gözlenen alt kısım + tavan üstü kuyruk
    # E[y] = Φ(z)·E[y|y<cap] + (1-Φ(z))·(cap + σ·λ),  z=(cap-μ)/σ
    z = (d.cap_usd - d.latent_mean) / sigma
    lam = norm.pdf(z) / np.clip(1 - norm.cdf(z), 1e-6, None)
    d["latent_adj"] = np.where(
        d.cap_share > 0.02,
        norm.cdf(z) * d.mcp_usd + (1 - norm.cdf(z)) * (d.cap_usd + sigma * lam),
        d.mcp_usd)
    return d, dict(sigma=float(sigma), n_clean=int(len(clean)))


# ---------------------------------------------------------------- main

def main() -> int:
    df = load_panel()
    print(f"panel: {len(df)} ay  {df.index.min().date()} → {df.index.max().date()}")

    print("\n" + "=" * 70)
    print("LINK 1 — TTF → BOTAŞ elektrik gaz tarifesi (NARDL, EUR/MWh_th)")
    print("=" * 70)
    res1, s1 = link1_nardl(df)
    print(res1.round(3).to_string())
    print(f"\n  kümülatif geçiş  YUKARI: {s1['cum_up']:+.3f}   AŞAĞI: {s1['cum_down']:+.3f}")
    print(f"  uzun-vade eğim (tarife/TTF): {s1['lr_slope']:.3f}")
    print(f"  asimetri (yukarı=aşağı?) Wald p = {s1['asym_p']:.3f}   n={s1['n']}")

    print("\n" + "=" * 70)
    print("LINK 3 — marjinal gaz SRMC (tarife USD/MWh_th) → MCP  [OLS + Newey-West]")
    print("=" * 70)
    for mask, lab in [(None, "tüm aylar"),
                      (df.cap_share < 0.15, "tavan bağlamıyor (<%15)"),
                      (df.cap_share > 0.30, "ağır sansür (>%30)")]:
        res3, s3 = link3(df, mask, lab)
        print(f"\n[{lab}]  n={s3['n']}  R²={s3['r2']:.2f}")
        print(res3.round(3).to_string())
        print(f"  → SRMC geçiş katsayısı δ = {s3['srmc']:.3f} ± {s3['srmc_se']:.3f}")

    print("\n" + "=" * 70)
    print("SANSÜR — iki-limitli Tobit (aylık sezgisel) + varyans bastırma")
    print("=" * 70)
    d, st = tobit_monthly(df)
    obs, lat = d.mcp_usd, d.latent_adj
    print(f"  sansürsüz-ay gizli süreç σ = {st['sigma']:.1f}  (n_temiz={st['n_clean']})")
    print(f"  gözlenen MCP:  ort {obs.mean():.1f}  std {obs.std():.1f}  maks {obs.max():.1f}")
    print(f"  gizli   MCP:   ort {lat.mean():.1f}  std {lat.std():.1f}  maks {lat.max():.1f}")
    supp = 1 - obs.std() ** 2 / lat.std() ** 2
    print(f"  → tavan gözlenen varyansın ~%{100*supp:.0f}'ini bastırıyor (aylık ort. düzeyinde)")
    top = d.nlargest(6, "latent_adj")[["mcp_usd", "latent_adj", "cap_share", "cap_usd"]]
    print("\n  en yüksek gizli aylar (gözlenen vs gizli):")
    print(top.round(1).to_string())

    d.to_pickle(ROOT / "experiments/notebooks/05_crisis_analysis/_passthrough_fit.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
