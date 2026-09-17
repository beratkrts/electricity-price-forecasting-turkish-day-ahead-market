#!/usr/bin/env python3
"""Read-only: canlı pipeline'in AYNI mantığıyla (aynı immutable pre-forecast
lag0 değerleri, aynı hiperparametreler) son N günü walk-forward yeniden üretir
ve gold.ptf_predictions_daily'deki canlı tahminle karşılaştırır. DB'ye YAZMAZ.

NOT: Bu script kasıtlı olarak canlı repodaki koda (predict_daily_pipeline,
feature_engineering) import ediyor — kendi kopyasını tutmak yerine tek doğru
kaynağı (canlı kod) kullanmak için. Yol sabit: ../enerji_fiyat_tahmini kardeş
repo olarak mevcut olmalı.
"""
import sys
from pathlib import Path

ROOT = Path("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import get_db_engine
from src.features.feature_engineering import build_robust_features, get_feature_columns
from src.models.lightgbm_model import LightGBMForecaster
from predict_daily_pipeline import load_all_historical_data

N_DAYS = 7
LIVE_PARAMS = dict(objective='quantile', alpha=0.50, n_estimators=300,
                    learning_rate=0.03, max_depth=8, num_leaves=63,
                    min_child_samples=10, verbose=-1, random_state=42)

print("veri yükleniyor (canlı ile birebir aynı sorgu, immutable pre-forecast dahil)...")
df_raw = load_all_historical_data()
df_feat = build_robust_features(df_raw)
feature_cols = get_feature_columns('robust', df_feat)
target_col = 'mcp_price_usd'
df_model = df_feat.dropna(subset=feature_cols + [target_col]).copy()
print(f"model verisi: {len(df_model):,} saat, {df_model.index.min()} -> {df_model.index.max()}")

max_day = df_model.index.normalize().max()
target_days = pd.date_range(max_day - pd.Timedelta(days=N_DAYS - 1), max_day, freq="D")

rows = []
for day in target_days:
    day_start, day_end = day, day + pd.Timedelta(hours=23)
    train_end = day_start - pd.Timedelta(hours=1)
    tr = df_model.loc[:train_end]
    te = df_model.loc[day_start:day_end]
    if len(tr) < 1000 or len(te) == 0:
        print(f"{day.date()}: yetersiz veri (train={len(tr)}, test={len(te)}), atlandı")
        continue
    m = LightGBMForecaster(params=LIVE_PARAMS)
    m.fit(tr[feature_cols], tr[target_col].values)
    pred = m.predict(te[feature_cols])
    for ts, p, y in zip(te.index, pred, te[target_col].values):
        rows.append({"target_ts": ts, "replay_pred": p, "actual": y})
    print(f"{day.date()}: {len(te)} saat replay edildi")

replay = pd.DataFrame(rows).set_index("target_ts")

eng = get_db_engine()
live_sql = text("""
    SELECT target_ts, predicted_mcp_usd AS live_pred
    FROM gold.ptf_predictions_daily
    WHERE target_ts >= :start
""")
live = pd.read_sql(live_sql, eng, params={"start": target_days[0]})
live['target_ts'] = pd.to_datetime(live['target_ts']).dt.tz_convert('Europe/Istanbul')
live = live.set_index('target_ts')

cmp = replay.join(live, how='left')
cmp['live_err'] = cmp.live_pred - cmp.actual
cmp['replay_err'] = cmp.replay_pred - cmp.actual
cmp['replay_vs_live'] = cmp.replay_pred - cmp.live_pred

pd.set_option('display.width', 160)
print("\n=== günlük özet: canlı hata vs replay hata (MAE) ===")
cmp['day'] = cmp.index.normalize()
summary = cmp.groupby('day').apply(lambda g: pd.Series({
    'live_MAE': g.live_err.abs().mean(),
    'replay_MAE': g.replay_err.abs().mean(),
    'live_BIAS': g.live_err.mean(),
    'replay_BIAS': g.replay_err.mean(),
    'mean_abs_replay_vs_live': g.replay_vs_live.abs().mean(),
}))
print(summary.round(2))

print("\n=== örnek saatler (ilk 15) ===")
print(cmp[['actual', 'live_pred', 'replay_pred', 'live_err', 'replay_err']].head(15).round(2))

out = Path(__file__).parent / "scratch_replay_comparison.csv"
cmp.to_csv(out)
print(f"\ntam tablo: {out}")
