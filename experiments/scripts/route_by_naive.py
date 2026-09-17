#!/usr/bin/env python3
"""
C.5.4 ön test: naive-2 eşiğiyle LEAR<->LightGBM routing.

Kural: her (gün, saat) için naive-2(d,h) <= eşik ise LEAR, değilse LightGBM.
naive-2 = Lago naive (hafta içi d-1, hafta sonu d-7) — 04:00'te bilinen.

Çöküş dilimi 2026-02-01 -> 06-30. Saf LightGBM / saf LEAR / router karşılaştırması.

    .venv/bin/python experiments/scripts/route_by_naive.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import lago_protocol as lp

H = ROOT / "experiments/notebooks/07_lago_protocol"
A, B = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-06-30")
TR4 = [56, 180, 1092, 1456]
COLS = [f"h{h:02d}" for h in range(24)]


def dblock(s):
    s = s.dropna(); idx = s.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Europe/Istanbul").tz_localize(None)
    w = pd.DataFrame({"v": s.to_numpy()}, index=idx)
    w["d"] = w.index.normalize(); w["h"] = w.index.hour
    return w.pivot_table(index="d", columns="h", values="v", aggfunc="first").reindex(
        columns=range(24)).set_axis(COLS, axis=1)


def adaptive(dd, subset, y, lb=7):
    cols = list(subset); mats = {c: dd[c].reindex(y.index) for c in cols}
    ya = y.to_numpy(); out = np.full((len(y.index), 24), np.nan); ep = None
    for i in range(len(y.index)):
        pr = np.stack([mats[c].to_numpy()[i] for c in cols])
        if i < lb or ep is None or np.isnan(ep).any():
            w = np.ones(len(cols)) / len(cols)
        else:
            inv = 1 / np.clip(ep, 1e-3, None); w = inv / inv.sum()
        out[i] = np.nansum(pr * w[:, None], axis=0)
        if not np.isnan(ya[i]).any():
            ep = np.nanmean(np.abs(pr - ya[i][None, :]), axis=1)
    return pd.DataFrame(out, index=y.index, columns=COLS)


def metrics(pred, y, n2, tag):
    k = y.index[(y.index >= A) & (y.index <= B)]
    k = k.intersection(pred.dropna(how="all").index)
    yy = y.reindex(k).to_numpy().ravel()
    pp = pred.reindex(k).to_numpy().ravel()
    n2v = n2.reindex(k).to_numpy().ravel()
    ok = ~np.isnan(pp) & ~np.isnan(yy) & (np.abs(pp) < 500)
    yy, pp, n2v = yy[ok], pp[ok], n2v[ok]
    e = pp - yy
    z = yy <= 1
    return dict(model=tag, MAE=round(float(np.mean(np.abs(e))), 2),
                BIAS=round(float(np.mean(e)), 2),
                rMAE=round(float(np.mean(np.abs(e)) / np.mean(np.abs(n2v - yy))), 3),
                sifir_MAE=round(float(np.mean(np.abs(e[z]))), 2) if z.any() else None,
                LEAR_pay=None)


def main():
    px = pd.read_csv(H / "tr_epf_ext.csv", index_col=0, parse_dates=True)["Price"]
    y = dblock(px); n2 = dblock(lp.naive_forecast(px, 2)); n3 = dblock(lp.naive_forecast(px, 3))

    lgb = pd.read_csv(H / "wf_lgbm_cokus_tr2023.csv", index_col=0, parse_dates=True)
    lgb.columns = COLS
    our = {int(f.split("cw")[1].split(".")[0]): pd.read_csv(H / f, index_col=0, parse_dates=True)
           for f in __import__("os").listdir(H)
           if f.startswith("ours_lear_cw") and "_spike" not in f and "cw28" not in f}
    lear = adaptive(our, TR4, y)

    # hizala
    idx = y.index[(y.index >= A) & (y.index <= B)]
    idx = idx.intersection(lgb.index).intersection(lear.dropna(how="all").index)
    yv = y.reindex(idx); n2v = n2.reindex(idx)
    lgbv = lgb.reindex(idx); learv = lear.reindex(idx)

    print(f"çöküş dilimi {A.date()} -> {B.date()}  ({len(idx)} gün)\n")
    base = []
    base.append(metrics(lgbv, y, n2, "saf LightGBM"))
    base.append(metrics(learv, y, n2, "saf LEAR TR-4 adaptif"))
    base.append(metrics(n2, y, n2, "naive-2"))
    base.append(metrics(n3, y, n2, "naive-3"))
    print(pd.DataFrame(base).set_index("model").to_string())

    print("\n--- router: naive-2(d,h) <= eşik ise LEAR, değilse LightGBM ---")
    rows = []
    for thr in [5, 10, 15, 20, 25, 30, 40, 50]:
        use_lear = n2v <= thr
        routed = lgbv.where(~use_lear, learv)
        m = metrics(routed, y, n2, f"eşik ${thr}")
        m["LEAR_pay"] = round(100 * float(use_lear.to_numpy().mean()), 1)
        rows.append(m)
    print(pd.DataFrame(rows).set_index("model").to_string())

    print("\n--- oracle üst sınır: her saat gerçekten daha iyi olanı seç ---")
    e_lgb = (lgbv - yv).abs(); e_lear = (learv - yv).abs()
    oracle = lgbv.where(e_lgb <= e_lear, learv)
    print(pd.DataFrame([metrics(oracle, y, n2, "oracle (hile)")]).set_index("model").to_string())

    # ayrıca: routing sinyali olarak LightGBM'in KENDİ tahmini
    print("\n--- alternatif router: LightGBM tahmini <= eşik ise LEAR ---")
    rows = []
    for thr in [10, 20, 30, 40, 50]:
        use_lear = lgbv <= thr
        routed = lgbv.where(~use_lear, learv)
        m = metrics(routed, y, n2, f"LGB-tahmin <= ${thr}")
        m["LEAR_pay"] = round(100 * float(use_lear.to_numpy().mean()), 1)
        rows.append(m)
    print(pd.DataFrame(rows).set_index("model").to_string())

    # --- TAM birincil pencere: canlı LightGBM (gold) vs LEAR, router normal rejimi bozuyor mu ---
    print("\n\n=== TAM birincil pencere (canlı gold LightGBM) — router normal rejimi bozar mı ===")
    from sqlalchemy import text
    from db.connection import get_db_engine
    g = pd.read_sql(text("SELECT target_ts ts, predicted_mcp_usd v FROM gold.ptf_predictions_daily "
                         "WHERE predicted_mcp_usd IS NOT NULL"), get_db_engine())
    g["ts"] = pd.to_datetime(g.ts)
    live = dblock(g.set_index("ts").v)
    strata = {"normal": ("2024-08-30", "2026-01-31"), "cokus": ("2026-02-01", "2026-06-30"),
              "toparlanma": ("2026-07-01", "2026-08-27")}
    for name, (sa, sb) in strata.items():
        ii = y.index[(y.index >= sa) & (y.index <= sb)]
        ii = ii.intersection(live.index).intersection(lear.dropna(how="all").index)
        if len(ii) < 10:
            continue
        lv = live.reindex(ii); lr = lear.reindex(ii); n2i = n2.reindex(ii)
        def m2(pred, tag):
            yy = y.reindex(ii).to_numpy().ravel(); pp = pred.to_numpy().ravel()
            nn = n2i.to_numpy().ravel(); ok = ~np.isnan(pp) & ~np.isnan(yy) & (np.abs(pp) < 500)
            e = pp[ok] - yy[ok]
            return dict(model=tag, MAE=round(float(np.mean(np.abs(e))), 2),
                        BIAS=round(float(np.mean(e)), 2),
                        rMAE=round(float(np.mean(np.abs(e)) / np.mean(np.abs(nn[ok] - yy[ok]))), 3))
        rr = lv.where(~(n2i <= 40), lr)
        print(f"\n  [{name}]  {len(ii)} gün")
        print(pd.DataFrame([m2(lv, "canlı LightGBM"), m2(lr, "LEAR TR-4 adaptif"),
                            m2(rr, "router (naive2<=40)")]).set_index("model").to_string())


if __name__ == "__main__":
    main()
