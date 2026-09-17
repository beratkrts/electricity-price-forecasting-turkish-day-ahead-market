#!/usr/bin/env python3
"""C — saatlik iki-limitli Tobit: tavanın bastırdığı fiyat varyansı.

OLAY_ANALIZI_V2_PLAN.md §5.3 RQ1: "Gözlenen fiyat değişkenliğinin ne kadarı tavan
tarafından bastırılıyor? Bastırılmasa fiyat ne olurdu?"

Model:  y* = Xβ + ε,  ε ~ N(0, σ²)
        gözlenen y = min(max(y*, taban), tavan)
        taban ≈ 0 (2026 sıfır-fiyat saatleri), tavan = AFL (zamanla değişen)

MLE (censored-normal, iki limit). Gizli süreç Xβ + ε'nin dağılımı ile gözlenen
(sansürlü) y karşılaştırılır → bastırma oranı.

Sürücüler yapısal + eşzamanlı (kontrafaktüel felsefesi): SRMC, arz karması, yük,
saat/ay sabit etkileri. Fiyat lag'i YOK.

NOT (3 Eyl 2026): heteroskedastik-σ v2 taslağı yazıldı ama oturum durduruldu, test
edilmedi ve geri alındı. Notebook 08 ve reports/OLAY_YAPISAL_GECIS_BULGULARI.md
BU (v1 / homoskedastik) sonuçları yansıtıyor. v2 planı bulgu dosyasının "Sıradaki"
bölümünde.

    .venv/bin/python experiments/scripts/tobit_hourly.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

MJ = 10.646

SQL = """
SELECT m.ts, m.price_usd, m.at_cap, (m.cap_try / u.usd_try) AS cap_usd,
       (g.price_try_1000m3 / :mj / u.usd_try) AS srmc_usd_th,
       k.natural_gas_mw, k.total_mw,
       (k.natural_gas_mw + k.import_coal_mw + k.lignite_mw + k.black_coal_mw) AS thermal_mw,
       (k.dammed_hydro_mw + k.river_hydro_mw) AS hydro_mw,
       (k.wind_mw + k.solar_mw) AS vre_mw,
       l.load_forecast_mw
FROM silver.mcp_with_cap m
JOIN raw_kgup_hourly k ON m.ts = k.ts
JOIN raw_load_forecast_hourly l ON m.ts = l.ts
LEFT JOIN silver.gas_cost_hourly g ON m.ts = g.ts
LEFT JOIN LATERAL (
   SELECT usd_try FROM silver.upstream_drivers
   WHERE d <= (m.ts AT TIME ZONE 'Europe/Istanbul')::date
   ORDER BY d DESC LIMIT 1) u ON true
ORDER BY m.ts
"""


def load() -> pd.DataFrame:
    df = pd.read_sql(text(SQL), get_db_engine(), params={"mj": MJ})
    df["ts"] = pd.to_datetime(df.ts, utc=True).dt.tz_convert("Europe/Istanbul")
    df = df.set_index("ts")
    tot = df.total_mw.replace(0, np.nan)
    df["thermal_share"] = df.thermal_mw / tot
    df["hydro_share"] = df.hydro_mw / tot
    df["vre_share"] = df.vre_mw / tot
    df["net_load_k"] = (df.load_forecast_mw - df.vre_mw - df.hydro_mw) / 1000.0
    df["srmc_lag1"] = df.srmc_usd_th.shift(24 * 30)  # ~1 ay gecikme (Link 1: geçiş +1ayda zirve)
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["floor_usd"] = 0.0
    df = df.dropna(subset=["price_usd", "cap_usd", "srmc_usd_th", "srmc_lag1", "thermal_share",
                           "hydro_share", "net_load_k", "load_forecast_mw"])
    # üst sansür bayrağı: at_cap ya da fiyat tavanın %99.5'i üstünde
    df["cens_hi"] = df.at_cap.astype(bool) | (df.price_usd >= 0.995 * df.cap_usd)
    df["cens_lo"] = df.price_usd <= 0.5  # fiili sıfır
    return df


def design(df: pd.DataFrame):
    # Kolinearite: paylar ~1'e toplandığı için hepsini koymak intercept kaydırıyor.
    # Sadece termal pay (fiyatı belirleyen kaynak) + net yük + SRMC (eş + 1ay gecikme).
    cols = ["srmc_usd_th", "srmc_lag1", "thermal_share", "net_load_k", "hydro_share"]
    X = [np.ones(len(df))] + [df[c].values for c in cols]
    names = ["const"] + cols
    for h in range(1, 24):
        X.append((df.hour == h).values.astype(float)); names.append(f"h{h}")
    for mo in range(2, 13):
        X.append((df.month == mo).values.astype(float)); names.append(f"m{mo}")
    return np.column_stack(X), names


def neg_ll(theta, X, y, lo, hi, m_lo, m_hi, m_un):
    beta, ls = theta[:-1], theta[-1]
    sig = np.exp(ls)
    mu = X @ beta
    ll = np.empty(len(y))
    z_un = (y[m_un] - mu[m_un]) / sig
    ll[m_un] = norm.logpdf(z_un) - ls
    ll[m_hi] = norm.logsf((hi[m_hi] - mu[m_hi]) / sig)
    ll[m_lo] = norm.logcdf((lo[m_lo] - mu[m_lo]) / sig)
    return -np.sum(ll)


def main() -> int:
    df = load()
    print(f"{len(df)} saat  {df.index.min().date()} → {df.index.max().date()}")
    print(f"  üst-sansürlü %{100*df.cens_hi.mean():.1f}  ·  alt-sansürlü %{100*df.cens_lo.mean():.1f}")

    X, names = design(df)
    y = df.price_usd.values.astype(float)
    lo = df.floor_usd.values.astype(float)
    hi = df.cap_usd.values.astype(float)
    m_hi, m_lo = df.cens_hi.values, df.cens_lo.values
    m_un = ~(m_hi | m_lo)

    # başlangıç: sansürsüzde OLS
    b0, *_ = np.linalg.lstsq(X[m_un], y[m_un], rcond=None)
    resid = y[m_un] - X[m_un] @ b0
    theta0 = np.append(b0, np.log(resid.std()))

    print("MLE (L-BFGS-B)…")
    res = optimize.minimize(neg_ll, theta0, args=(X, y, lo, hi, m_lo, m_hi, m_un),
                            method="L-BFGS-B", options={"maxiter": 500})
    beta, sig = res.x[:-1], np.exp(res.x[-1])
    print(f"  yakınsama: {res.success}  negLL={res.fun:.0f}  σ={sig:.1f}")

    coef = pd.Series(beta, index=names)
    print("\nyapısal katsayılar (saat/ay sabit etkileri gizli):")
    print(coef[["const", "srmc_usd_th", "srmc_lag1", "thermal_share", "net_load_k", "hydro_share"]].round(2).to_string())

    mu = X @ beta
    latent_var = np.var(mu) + sig ** 2
    obs_var = np.var(y)
    print(f"\n  gözlenen MCP:  ort {y.mean():.1f}  std {np.sqrt(obs_var):.1f}")
    print(f"  gizli   MCP:   ort {mu.mean():.1f}  std {np.sqrt(latent_var):.1f}")
    print(f"  → tavan+taban SAATLİK varyansın ~%{100*(1 - obs_var/latent_var):.0f}'ini bastırıyor")

    df["mu"] = mu
    print("\nyıl bazında (gözlenen std → gizli std, üst-sansür payı):")
    for yr, g in df.groupby(df.index.year):
        gv = np.var(g.mu.values) + sig ** 2
        ov = np.var(g.price_usd.values)
        print(f"  {yr}: {np.sqrt(ov):5.1f} → {np.sqrt(gv):5.1f}   "
              f"sansür %{100*g.cens_hi.mean():4.1f}   bastırma %{100*(1-ov/gv):3.0f}")

    hc = df[df.cens_hi]
    z = (hc.cap_usd.values - hc.mu.values) / sig
    lam = norm.pdf(z) / np.clip(norm.sf(z), 1e-9, None)
    exp_latent = hc.mu.values + sig * lam
    clip = exp_latent - hc.cap_usd.values
    print(f"\n  üst-sansürlü saatlerde ort. kırpılan miktar: ${clip.mean():.1f}/MWh "
          f"(n={len(hc)}, saatlerin %{100*len(hc)/len(df):.1f}'i)")

    df[["price_usd", "cap_usd", "mu", "cens_hi", "cens_lo"]].to_pickle(
        ROOT / "experiments/notebooks/05_crisis_analysis/_tobit_hourly.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
