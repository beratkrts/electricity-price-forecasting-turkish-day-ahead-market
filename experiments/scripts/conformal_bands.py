#!/usr/bin/env python3
"""
C.5 — P10/P90 için yuvarlanan konformal band testi.

P50 = C.5.3' ensemble (base + roll90 + roll150). Band = ensemble'ın KENDİ son N günlük
saat-bazlı hata dağılımından (nedensel split-conformal):
    L(d,h) = P50 − q_{0.90}(err[d-N:d-1, h]) ,  U(d,h) = P50 − q_{0.10}(...)
    err = tahmin − gerçek  (model fazla tahmin ediyorsa band aşağı kayar + genişler)

Canlı `gold.ptf_predictions_daily` P10/P90'a karşı: kapsama + band genişliği, rejim bazlı.

    .venv/bin/python experiments/scripts/conformal_bands.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from db.connection import get_db_engine

H = ROOT / "experiments/notebooks/07_lago_protocol"
COLS = [f"h{h:02d}" for h in range(24)]
SEG = {"TÜM": ("2024-11-01", "2026-08-27"), "normal": ("2024-11-01", "2026-01-31"),
       "çöküş": ("2026-02-01", "2026-06-30"), "toparlanma": ("2026-07-01", "2026-08-27")}


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


def conformal(p50, y, N, lo=0.10, hi=0.90):
    """Nedensel yuvarlanan konformal band, saat bazlı."""
    err = (p50 - y.reindex(p50.index))       # gün × 24, tam p50 index'i
    idx = p50.index
    L = pd.DataFrame(index=idx, columns=COLS, dtype=float)
    U = pd.DataFrame(index=idx, columns=COLS, dtype=float)
    ev = err.to_numpy()
    for i in range(len(idx)):
        j0 = max(0, i - N)
        if i - j0 < 20:
            continue
        cal = ev[j0:i]                       # (win, 24)
        qlo = np.nanquantile(cal, hi, axis=0)   # U için q_hi(err)
        qhi = np.nanquantile(cal, lo, axis=0)   # L için q_lo(err)
        L.iloc[i] = p50.iloc[i].to_numpy() - qlo
        U.iloc[i] = p50.iloc[i].to_numpy() - qhi
    return L, U


def report(name, y, L, U, p50=None):
    for seg, (a, b) in SEG.items():
        k = y.index[(y.index >= pd.Timestamp(a)) & (y.index <= pd.Timestamp(b))]
        k = k.intersection(L.dropna(how="all").index)
        yy = y.reindex(k).to_numpy().ravel()
        ll = L.reindex(k).to_numpy().ravel(); uu = U.reindex(k).to_numpy().ravel()
        ok = ~np.isnan(yy) & ~np.isnan(ll) & ~np.isnan(uu)
        yy, ll, uu = yy[ok], ll[ok], uu[ok]
        cov = 100 * np.mean((yy >= ll) & (yy <= uu))
        wid = np.mean(uu - ll)
        extra = ""
        if p50 is not None:
            pp = p50.reindex(k).to_numpy().ravel()[ok]
            extra = f"  P50-MAE {np.mean(np.abs(pp - yy)):.2f}"
        print(f"    {seg:11s} n={int(ok.sum()/24):3d}g  kapsama {cov:5.1f}%  band ${wid:5.1f}{extra}")


def main():
    eng = get_db_engine()
    px = pd.read_csv(H / "tr_epf_ext.csv", index_col=0, parse_dates=True)["Price"]
    y = dblock(px)

    W = {"base": wf("bt2y_v2_base"), 90: wf("bt2y_v2_roll90"), 150: wf("bt2y_v2_roll150")}
    idx = pd.DatetimeIndex(sorted(set.intersection(*[set(v.index) for v in W.values()])))
    for k in W:
        W[k] = W[k].reindex(idx)
    ens = (W["base"] + W[90] + W[150]) / 3

    # canlı P10/P90
    q = pd.read_sql(text("SELECT target_ts ts, predicted_mcp_usd p50, predicted_mcp_usd_p10 p10, "
                         "predicted_mcp_usd_p90 p90 FROM gold.ptf_predictions_daily "
                         "WHERE predicted_mcp_usd_p10 IS NOT NULL"), eng)
    q["ts"] = pd.to_datetime(q.ts)
    liveL = dblock(q.set_index("ts").p10); liveU = dblock(q.set_index("ts").p90)
    liveP = dblock(q.set_index("ts").p50)

    print("=== CANLI (3-head quantile, gold.ptf_predictions_daily) ===")
    report("canlı", y, liveL, liveU, liveP)

    for N in (45, 60, 75, 90):
        L, U = conformal(ens, y, N)
        print(f"\n=== KONFORMAL band, ensemble P50, pencere={N}g ===")
        report(f"conf{N}", y, L, U, ens)

    # ek: konformal band + ensemble-anlaşmazlığı genişletmesi
    disagree = pd.concat([W["base"], W[90], W[150]]).groupby(level=0).std()
    L, U = conformal(ens, y, 60)
    L2 = L - 0.5 * disagree.reindex(L.index)
    U2 = U + 0.5 * disagree.reindex(U.index)
    print("\n=== KONFORMAL 60g + 0.5·ensemble-std genişletme ===")
    report("conf60+dis", y, L2, U2, ens)


if __name__ == "__main__":
    main()
