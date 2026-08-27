import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_lightgbm_standalone_backtest import load_master_dataset
from src.features.feature_engineering import build_robust_features, get_feature_columns

from src.models.lightgbm_model import LightGBMForecaster

def run_strategies_backtest(max_days=30):
    print("📦 Master Veri Seti Yükleniyor...")
    df_raw = load_master_dataset()
    
    # MWA Lags and Computation
    print("🧮 Optimal MWA Hesaplanıyor...")
    mwa_lags = [24, 48, 72, 96, 120, 144, 168]
    optimal_weights = np.array([0.412, 0.057, 0.058, 0.006, 0.0, 0.061, 0.406])
    
    for lag in mwa_lags:
        df_raw[f'mwa_lag_{lag}'] = df_raw['mcp_price_usd'].shift(lag)
        
    df_raw = df_raw.dropna(subset=[f'mwa_lag_{lag}' for lag in mwa_lags])
    
    mwa_pred = np.zeros(len(df_raw))
    for i, lag in enumerate(mwa_lags):
        mwa_pred += df_raw[f'mwa_lag_{lag}'] * optimal_weights[i]
        
    df_raw['mwa_pred_usd'] = mwa_pred
    df_raw['residual_usd'] = df_raw['mcp_price_usd'] - df_raw['mwa_pred_usd']

    print("⚙️ Feature Engineering Uygulanıyor...")
    df_feat = build_robust_features(df_raw)
    df_model = df_feat.dropna().copy()
    
    base_features = get_feature_columns('robust', df_model)
    target_col = 'mcp_price_usd'
    
    # Canlıdaki Birebir Aynı Parametreler
    lgbm_params = {
        'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 63,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'min_child_samples': 20, 'verbose': -1, 'random_state': 42,
        'deterministic': True, 'force_col_wise': True
    }

    log_dir = project_root / "logs" / "lightgbm_experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_file = log_dir / f"LGBM_MWA_Strategies_{max_days}d.csv"
    
    print(f"\\n🏃 {max_days} Günlük Strateji Backtesti Başlatılıyor...\\n")
    
    daily_results = []
    max_ts = df_model.index.max()
    
    for day_idx in range(max_days - 1, -1, -1):
        test_end = max_ts - pd.Timedelta(days=day_idx)
        test_start = test_end - pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)
        
        tr_df = df_model.loc[:train_end]
        te_df = df_model.loc[test_start:test_end]
        
        if len(tr_df) < 1000 or len(te_df) < 24:
            continue
            
        y_true = te_df[target_col].values
        mwa_true = te_df['mwa_pred_usd'].values
        sum_y_true = np.sum(y_true)
        if sum_y_true == 0: sum_y_true = 1.0 # avoid div by zero
        
        daily_metrics = {'tarih': test_start.strftime('%Y-%m-%d'), 'day_offset': day_idx + 1}
        
        # --- 0. MWA SADECE ---
        mae_mwa = np.mean(np.abs(y_true - mwa_true))
        wape_mwa = np.sum(np.abs(y_true - mwa_true)) / sum_y_true * 100
        daily_metrics['mwa_mae'] = mae_mwa
        daily_metrics['mwa_wape'] = wape_mwa

        # --- 1. BASELINE LIGHTGBM (Canlı Model Sınıfı - log1p) ---
        X_tr, y_tr = tr_df[base_features], tr_df[target_col]
        X_te = te_df[base_features]
        model_base = LightGBMForecaster(params=lgbm_params).fit(X_tr, y_tr)
        preds_base = model_base.predict(X_te)
        
        daily_metrics['base_mae'] = np.mean(np.abs(y_true - preds_base))
        daily_metrics['base_wape'] = np.sum(np.abs(y_true - preds_base)) / sum_y_true * 100
        
        # --- 2. STRAT 1: MWA as Feature (Canlı Model Sınıfı - log1p) ---
        features_s1 = base_features + ['mwa_pred_usd']
        X_tr_s1 = tr_df[features_s1]
        X_te_s1 = te_df[features_s1]
        model_s1 = LightGBMForecaster(params=lgbm_params).fit(X_tr_s1, y_tr)
        preds_s1 = model_s1.predict(X_te_s1)
        
        daily_metrics['strat1_feat_mae'] = np.mean(np.abs(y_true - preds_s1))
        daily_metrics['strat1_feat_wape'] = np.sum(np.abs(y_true - preds_s1)) / sum_y_true * 100
        
        # --- 3. STRAT 2: Residual Learning (Log1p YAPILAMAZ çünkü hedef negatif olabilir, bu yüzden lgb kullanıldı) ---
        y_tr_res = tr_df['residual_usd']
        model_s2 = lgb.LGBMRegressor(**lgbm_params).fit(X_tr, y_tr_res)
        preds_s2_res = model_s2.predict(X_te)
        preds_s2 = np.maximum(preds_s2_res + mwa_true, 0.0)
        
        daily_metrics['strat2_res_mae'] = np.mean(np.abs(y_true - preds_s2))
        daily_metrics['strat2_res_wape'] = np.sum(np.abs(y_true - preds_s2)) / sum_y_true * 100
        
        # --- 4. STRAT 3: Hybrid Blending (20% MWA + 80% Base LGBM) ---
        preds_s3 = 0.20 * mwa_true + 0.80 * preds_base
        
        daily_metrics['strat3_blend_mae'] = np.mean(np.abs(y_true - preds_s3))
        daily_metrics['strat3_blend_wape'] = np.sum(np.abs(y_true - preds_s3)) / sum_y_true * 100
        
        # --- 5. STRAT 4: Feature Selection (Canlı Model Sınıfı - log1p) ---
        features_s4 = [f for f in base_features if f not in ['mcp_usd_lag_48', 'mcp_usd_lag_72', 'mcp_usd_lag_96', 'mcp_usd_lag_120', 'mcp_usd_lag_144']]
        X_tr_s4 = tr_df[features_s4]
        X_te_s4 = te_df[features_s4]
        model_s4 = LightGBMForecaster(params=lgbm_params).fit(X_tr_s4, y_tr)
        preds_s4 = model_s4.predict(X_te_s4)
        
        daily_metrics['strat4_select_mae'] = np.mean(np.abs(y_true - preds_s4))
        daily_metrics['strat4_select_wape'] = np.sum(np.abs(y_true - preds_s4)) / sum_y_true * 100
        
        daily_results.append(daily_metrics)
        
        if len(daily_results) % 5 == 0 or len(daily_results) == max_days:
            print(f"İşlenen Gün: {len(daily_results)}/{max_days} | "
                  f"Base: %{daily_metrics['base_wape']:.2f} | "
                  f"S1(Feat): %{daily_metrics['strat1_feat_wape']:.2f} | "
                  f"S2(Res): %{daily_metrics['strat2_res_wape']:.2f}")

    res_df = pd.DataFrame(daily_results)
    res_df.to_csv(csv_file, index=False)
    print(f"\\n💾 Tüm sonuçlar kaydedildi: {csv_file}")
    
    # Calculate Overall Volumetric WAPE correctly
    print("\\n" + "="*60)
    print(f"🏆 {max_days} GÜNLÜK STRATEJİ KARŞILAŞTIRMASI (VOLÜMETRİK WAPE)")
    print("="*60)
    
    cols = ['mwa', 'base', 'strat1_feat', 'strat2_res', 'strat3_blend', 'strat4_select']
    names = ['MWA (Optimal)', 'Baseline LGBM', 'Strat 1: MWA as Feature', 'Strat 2: Residual Learning', 'Strat 3: Blending (20/80)', 'Strat 4: Feature Select']
    
    for c, n in zip(cols, names):
        # We can approximate Volumetric WAPE since we have daily MAE
        # total_MAE_hours = daily_MAE.mean() is wrong, we should do:
        overall_mae = res_df[f'{c}_mae'].mean()
        # Sum of absolute errors = sum(daily_mae * 24)
        sum_abs_err = (res_df[f'{c}_mae'] * 24).sum()
        # Sum of actuals = sum((daily_mae * 24) / (daily_wape / 100))
        sum_actuals = ((res_df[f'{c}_mae'] * 24) / (res_df[f'{c}_wape'] / 100)).sum()
        overall_wape = (sum_abs_err / sum_actuals) * 100
        
        print(f"{n:<28} | MAE: ${overall_mae:5.2f} | WAPE: %{overall_wape:5.2f}")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Test edilecek gün sayısı")
    args = parser.parse_args()
    run_strategies_backtest(args.days)
