#!/usr/bin/env python3
"""
P10/P90 konformal band — final parametre taraması + doğrulama.

P50 = C.5.3' ensemble (bt2y_v2: base + roll90 + roll150).
Band = ensemble P50'nin KENDİ son-N-gün saat-bazlı hata dağılımı (nedensel split-conformal):
    L = P50 − q_{1-α/2}(err)  −  w·std({base,r90,r150})
    U = P50 − q_{α/2}(err)    +  w·std({base,r90,r150})
    err = tahmin − gerçek ;  L = max(L, FLOOR)
Hedef kapsama = %80 (α=0.20).
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
       "çöküş": ("2026-02-01", "2026-06-30"), "toparlanma": ("2026-07-01", "2026-08-27"),
       "son 90g": ("2026-05-29", "2026-08-27"), "son 30g": ("2026-07-28", "2026-08-27")}
FLOOR = 0.0   # TR MCP tabanı $0 (hiç negatif olmamış, 547 saat tam $0)


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


def bands(p50, y, disagree, N, w, alpha=0.20):
    err = (p50 - y.reindex(p50.index)).to_numpy()
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    L = pd.DataFrame(index=p50.index, columns=COLS, dtype=float); U = L.copy()
    dv = disagree.reindex(p50.index).to_numpy()
    for i in range(len(p50.index)):
        j0 = max(0, i - N)
        if i - j0 < 20:
            continue
        cal = err[j0:i]
        qhi = np.nanquantile(cal, hi_q, axis=0)   # U için: L = P50 - qhi
        qlo = np.nanquantile(cal, lo_q, axis=0)
        L.iloc[i] = np.maximum(p50.iloc[i].to_numpy() - qhi - w * dv[i], FLOOR)
        U.iloc[i] = p50.iloc[i].to_numpy() - qlo + w * dv[i]
    return L, U


def cov_width(y, L, U, a, b):
    k = y.index[(y.index >= pd.Timestamp(a)) & (y.index <= pd.Timestamp(b))]
    k = k.intersection(L.dropna(how="all").index)
    yy = y.reindex(k).to_numpy().ravel()
    ll = L.reindex(k).to_numpy().ravel(); uu = U.reindex(k).to_numpy().ravel()
    ok = ~np.isnan(yy) & ~np.isnan(ll) & ~np.isnan(uu)
    yy, ll, uu = yy[ok], ll[ok], uu[ok]
    return 100 * np.mean((yy >= ll) & (yy <= uu)), np.mean(uu - ll), int(ok.sum() / 24)


def main():
    y = dblock(pd.read_csv(H / "tr_epf_ext.csv", index_col=0, parse_dates=True)["Price"])
    W = {"b": wf("bt2y_v2_base"), 90: wf("bt2y_v2_roll90"), 150: wf("bt2y_v2_roll150")}
    idx = pd.DatetimeIndex(sorted(set.intersection(*[set(v.index) for v in W.values()])))
    for k in W:
        W[k] = W[k].reindex(idx)
    ens = (W["b"] + W[90] + W[150]) / 3
    disagree = pd.concat([W["b"], W[90], W[150]]).groupby(level=0).std()

    q = pd.read_sql(text("SELECT target_ts ts, predicted_mcp_usd_p10 p10, predicted_mcp_usd_p90 p90 "
                         "FROM gold.ptf_predictions_daily WHERE predicted_mcp_usd_p10 IS NOT NULL"),
                    get_db_engine())
    q["ts"] = pd.to_datetime(q.ts)
    liveL, liveU = dblock(q.set_index("ts").p10), dblock(q.set_index("ts").p90)

    print("=== CANLI 3-head quantile ===")
    for seg, (a, b) in SEG.items():
        c, wd, n = cov_width(y, liveL, liveU, a, b)
        print(f"  {seg:11s} n={n:3d}g  kapsama {c:5.1f}%  band ${wd:5.1f}")

    print("\n=== KONFORMAL taraması (N gün, w = anlaşmazlık ağırlığı) ===")
    for N in (45, 60, 90):
        for w in (0.0, 0.3, 0.5, 0.7):
            L, U = bands(ens, y, disagree, N, w)
            r = {seg: cov_width(y, L, U, a, b) for seg, (a, b) in SEG.items()}
            tag = f"N={N:2d} w={w:.1f}"
            covs = "  ".join(f"{s}:{r[s][0]:.0f}%" for s in ["TÜM", "normal", "çöküş", "toparlanma", "son 90g"])
            print(f"  {tag}   {covs}   bandTÜM ${r['TÜM'][1]:.1f}")

    print("\n=== SEÇİM: N=60, w=0.5 — rejim + yakın dönem ===")
    L, U = bands(ens, y, disagree, 60, 0.5)
    for seg, (a, b) in SEG.items():
        c, wd, n = cov_width(y, L, U, a, b)
        cl, wl, _ = cov_width(y, liveL, liveU, a, b)
        print(f"  {seg:11s} n={n:3d}g   konformal {c:5.1f}% / ${wd:5.1f}   (canlı {cl:5.1f}% / ${wl:5.1f})")

    # negatif P10 / mantıksızlık kontrolü
    L60, _ = bands(ens, y, disagree, 60, 0.5)
    neg = (L60 < 0).sum().sum(); tot = L60.notna().sum().sum()
    print(f"\n  P10 < $0 saat: {neg}/{tot} ({100*neg/tot:.1f}%)  — FLOOR ${FLOOR}")
    print(f"  P10 < FLOOR (kırpılan): {(L60 <= FLOOR).sum().sum()}")


if __name__ == "__main__":
    main()
