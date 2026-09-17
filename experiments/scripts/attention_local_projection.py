#!/usr/bin/env python3
"""A — dikkat şoklarının geçiş gecikmesi (local projection). Plan §5.1.

Soru: bir konu dikkat sıçraması → {BOTAŞ tarifesi, MCP, kontrafaktüel kalıntı}
hangi gecikmeyle, ne kadar hareket ediyor? Beklenti (Faz 0'dan): jeopolitik/gaz
dikkati → tarife (~1-2 ay) → MCP (~3-5 ay); kalıntı tepkisi ~0.

Aylık panel. Local projection: Y_{t+h} − Y_{t−1} = α_h + β_h · shock_t + kontroller + ε
HAC (Newey-West) SE. h = 0..6 ay.

    .venv/bin/python experiments/scripts/attention_local_projection.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

TOPICS = ["gas_supply", "geopolitics", "regulation", "hydro_weather", "oil"]
H = [0, 1, 2, 3, 4, 5, 6]


def panel() -> pd.DataFrame:
    eng = get_db_engine()
    att = pd.read_sql(text("SELECT * FROM silver.news_attention_daily ORDER BY d"), eng)
    att["d"] = pd.to_datetime(att.d)
    att = att.set_index("d")
    m = pd.DataFrame(index=pd.PeriodIndex(att.index, freq="M").unique().to_timestamp())
    for t in TOPICS:
        m[f"{t}_z"] = att[f"{t}_z"].resample("MS").mean()
        m[f"{t}_attn"] = att[f"{t}_attn"].resample("MS").mean()

    mcp = pd.read_sql(text("""
        SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') mon,
               avg(price_usd) mcp, avg(CASE WHEN at_cap THEN 1 ELSE 0 END) cap_share
        FROM silver.mcp_with_cap GROUP BY 1"""), eng)
    tar = pd.read_sql(text("""
        SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') mon,
               avg(price_try_1000m3) tariff FROM silver.gas_cost_hourly GROUP BY 1"""), eng)
    ttf = pd.read_sql(text("""
        SELECT date_trunc('month', d::timestamp) mon, avg(ttf_eur_mwh) ttf,
               avg(usd_try) fx FROM silver.upstream_drivers GROUP BY 1"""), eng)
    cf = pd.read_sql(text("""
        SELECT date_trunc('month', ts AT TIME ZONE 'Europe/Istanbul') mon,
               avg(residual_usd) resid FROM gold.crisis_counterfactual
        WHERE model_name='crisis_cf_v5' AND variant='fundamental' GROUP BY 1"""), eng)
    for x in (mcp, tar, ttf, cf):
        x["mon"] = pd.to_datetime(x["mon"])
        x.set_index("mon", inplace=True)
    m = m.join(mcp).join(tar).join(ttf).join(cf)
    m["tariff_usd"] = (m.tariff / 10.646) / m.fx
    m["d_ttf"] = m.ttf.pct_change()
    return m.dropna(subset=["mcp", "tariff_usd"])


def lp(m: pd.DataFrame, shock: str, y: str, transform="level"):
    """Local projection: h-ay sonrası Y değişimi ~ shock_t + kontrol."""
    rows = []
    yv = m[y]
    for h in H:
        if transform == "level":
            dep = yv.shift(-h) - yv.shift(1)
        else:  # pct
            dep = yv.shift(-h) / yv.shift(1) - 1
        X = pd.DataFrame({
            "shock": m[shock],
            "y_lag": yv.shift(1),
            "d_ttf": m.d_ttf,
            "cap_share": m.cap_share,
        })
        d = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        if len(d) < 20:
            rows.append((h, np.nan, np.nan)); continue
        res = sm.OLS(d.dep, sm.add_constant(d[["shock", "y_lag", "d_ttf", "cap_share"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append((h, res.params["shock"], res.bse["shock"]))
    return pd.DataFrame(rows, columns=["h", "beta", "se"])


def main() -> int:
    m = panel()
    print(f"panel {len(m)} ay  {m.index.min().date()} → {m.index.max().date()}\n")

    specs = [
        ("geopolitics_z", "tariff_usd", "level", "Jeopolitik dikkat → BOTAŞ tarifesi (USD/MWh_th)"),
        ("geopolitics_z", "mcp",        "level", "Jeopolitik dikkat → MCP (USD/MWh)"),
        ("geopolitics_z", "resid",      "level", "Jeopolitik dikkat → kontrafaktüel kalıntı"),
        ("gas_supply_z",  "tariff_usd", "level", "Gaz-arz dikkat → BOTAŞ tarifesi"),
        ("gas_supply_z",  "mcp",        "level", "Gaz-arz dikkat → MCP"),
        ("oil_z",         "mcp",        "level", "Petrol dikkat → MCP"),
        ("regulation_z",  "mcp",        "level", "Regülasyon dikkat → MCP"),
        ("hydro_weather_z","mcp",       "level", "Hidro-hava dikkat → MCP"),
    ]
    for shock, y, tr, label in specs:
        r = lp(m, shock, y, tr)
        print(f"── {label}")
        line = "   h(ay): " + "  ".join(f"{h:>2d}" for h in r.h)
        bet = "   β    : " + "  ".join(f"{b:+.1f}" if not np.isnan(b) else "  —" for b in r.beta)
        sig = "   t    : " + "  ".join(
            f"{b/s:+.1f}" if (not np.isnan(b) and s and s > 0) else "  —"
            for b, s in zip(r.beta, r.se))
        print(line); print(bet); print(sig)
        peak = r.loc[r.beta.abs().idxmax()] if r.beta.notna().any() else None
        if peak is not None:
            print(f"   → zirve h={int(peak.h)} ay, β={peak.beta:+.1f} (t={peak.beta/peak.se:+.1f})\n")

    print("Okuma: β_h = shock_t'nin bir standart sapması, Y'yi h ay sonra kaç birim "
          "hareket ettiriyor (y_lag, ΔTTF, tavan payı kontrol). |t|≳2 anlamlı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
