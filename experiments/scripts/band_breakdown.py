#!/usr/bin/env python3
"""Ensemble konformal band — fiyat seviyesi + saat kırılımı."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path("/Users/beratkaratasoglu/etkb_intern_project/electricity_price_forecasting_in_turkish_day_ahead_market")
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

H = ROOT / "experiments/notebooks/07_lago_protocol"
COLS = [f"h{h:02d}" for h in range(24)]
FLOOR = 0.0


def dblock(s):
    s = s.dropna(); idx = s.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Europe/Istanbul").tz_localize(None)
    w = pd.DataFrame({"v": s.to_numpy()}, index=idx)
    w["d"] = w.index.normalize(); w["h"] = w.index.hour
    return w.pivot_table(index="d", columns="h", values="v", aggfunc="first").reindex(
        columns=range(24)).set_axis(COLS, axis=1)


def wf(tag):
    d = pd.read_csv(H / f"wf_lgbm_{tag}.csv", index_col=0, parse_dates=True); d.columns = COLS
    return d


y = dblock(pd.read_csv(H / "tr_epf_ext.csv", index_col=0, parse_dates=True)["Price"])
W = {"base": wf("bt2y_v2_base"), 90: wf("bt2y_v2_roll90"), 150: wf("bt2y_v2_roll150")}
idx = pd.DatetimeIndex(sorted(set.intersection(*[set(v.index) for v in W.values()])))
for k in W:
    W[k] = W[k].reindex(idx)
ens = (W["base"] + W[90] + W[150]) / 3
disagree = pd.concat([W["base"], W[90], W[150]]).groupby(level=0).std()


def bands(p50, N=60, w=0.5, alpha=0.20):
    err = (p50 - y.reindex(p50.index)).to_numpy()
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    L = pd.DataFrame(index=p50.index, columns=COLS, dtype=float); U = L.copy()
    dv = disagree.reindex(p50.index).to_numpy()
    for i in range(len(p50.index)):
        j0 = max(0, i - N)
        if i - j0 < 20:
            continue
        cal = err[j0:i]
        L.iloc[i] = np.maximum(p50.iloc[i].to_numpy() - np.nanquantile(cal, hi_q, axis=0) - w * dv[i], FLOOR)
        U.iloc[i] = p50.iloc[i].to_numpy() - np.nanquantile(cal, lo_q, axis=0) + w * dv[i]
    return L, U


L, U = bands(ens)
A0, B0 = pd.Timestamp("2024-11-01"), pd.Timestamp("2026-08-27")
k = y.index[(y.index >= A0) & (y.index <= B0)].intersection(L.dropna(how="all").index)

yy = y.reindex(k).to_numpy().ravel()
ll = L.reindex(k).to_numpy().ravel()
uu = U.reindex(k).to_numpy().ravel()
pp = ens.reindex(k).to_numpy().ravel()
hh = np.tile(np.arange(24), len(k))
ok = ~np.isnan(yy) & ~np.isnan(ll) & ~np.isnan(uu)
yy, ll, uu, pp, hh = yy[ok], ll[ok], uu[ok], pp[ok], hh[ok]

inside = (yy >= ll) & (yy <= uu)
below = yy < ll
above = yy > uu
width = uu - ll


def show(mask, label):
    n = int(mask.sum())
    if n == 0:
        print(f"  {label:16s}  n=0")
        return
    print(f"  {label:16s}  n={n:6d}  kaps {100*inside[mask].mean():5.1f}%  "
          f"band ${width[mask].mean():6.2f}  P10 ${ll[mask].mean():6.2f}  P90 ${uu[mask].mean():6.2f}  "
          f"alt-kaçış {100*below[mask].mean():4.1f}%  üst-kaçış {100*above[mask].mean():4.1f}%")


print("=== GERÇEK FİYAT SEVİYESİ kırılımı (N=60, w=0.5, FLOOR=$0) ===")
edges = [0, 5, 20, 40, 60, 100, 1e9]
names = ["$0–5", "$5–20", "$20–40", "$40–60", "$60–100", "$100+"]
for lo, hi, nm in zip(edges[:-1], edges[1:], names):
    show((yy >= lo) & (yy < hi), nm)
show(yy == 0, "(tam $0)")

print("\n=== TAHMİN (P50) SEVİYESİ kırılımı ===")
for lo, hi, nm in zip(edges[:-1], edges[1:], names):
    show((pp >= lo) & (pp < hi), nm)

print("\n=== SAAT kırılımı ===")
for grp, hrs in [("gece 00–06", range(0, 6)), ("sabah 06–10", range(6, 10)),
                 ("gündüz 10–17", range(10, 17)), ("akşam-puant 17–23", range(17, 23)),
                 ("gece-geç 23–24", [23])]:
    show(np.isin(hh, list(hrs)), grp)

print("\n=== REJİM kırılımı ===")
kk = k.repeat(1)
dts = np.repeat(k.values, 24)[ok]
seg = {"normal": (dts >= np.datetime64("2024-11-01")) & (dts <= np.datetime64("2026-01-31")),
       "çöküş": (dts >= np.datetime64("2026-02-01")) & (dts <= np.datetime64("2026-06-30")),
       "toparlanma": (dts >= np.datetime64("2026-07-01"))}
for nm, m in seg.items():
    show(m, nm)
