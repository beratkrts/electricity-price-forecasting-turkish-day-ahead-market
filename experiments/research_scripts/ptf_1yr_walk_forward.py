import sys, pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path.cwd()))
from src.features.feature_engineering import build_robust_features, get_feature_columns
from scripts.predict_daily_pipeline import load_all_historical_data

def wape(y_t, y_p):
    return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100

print("Veriler Yükleniyor...")
df_raw = load_all_historical_data()
df_feat = build_robust_features(df_raw)
target_col = 'mcp_price_usd'

new_feat_cols = get_feature_columns('robust', df_feat)
df_new = df_feat.dropna(subset=new_feat_cols + [target_col])

test_days = 365
max_ts = df_new.index.max()

actuals = []
preds = []
timestamps = []

print("Walk-Forward Başlıyor (365 Gün)...")
for day_idx in tqdm(range(test_days - 1, -1, -1)):
    test_end = max_ts - pd.Timedelta(days=day_idx)
    test_start = test_end - pd.Timedelta(hours=23)
    train_end = test_start - pd.Timedelta(hours=1)
    
    tr_df = df_new.loc[:train_end]
    te_df = df_new.loc[test_start:test_end]
    
    if len(tr_df) < 1000 or len(te_df) < 12:
        continue
        
    lgb_new = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
    lgb_new.fit(tr_df[new_feat_cols], tr_df[target_col].values)
    
    p = lgb_new.predict(te_df[new_feat_cols])
    actuals.extend(te_df[target_col].values)
    preds.extend(p)
    timestamps.extend(te_df.index)

df_res = pd.DataFrame({'ts': timestamps, 'actual': actuals, 'pred': preds}).set_index('ts')

months = [1, 3, 6, 9, 12]
print("\n--- YENİ MODEL (Ön-Tahminler Dahil) WALK-FORWARD WAPE SONUÇLARI ---")
for m in months:
    cutoff = df_res.index.max() - pd.DateOffset(months=m)
    df_sub = df_res[df_res.index >= cutoff]
    w = wape(df_sub['actual'], df_sub['pred'])
    print(f"Son {m} Ay WAPE: {w:.2f}%")
