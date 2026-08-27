"""
Hyper-Optimized Vectorized PyTorch LSTM Online Incremental Learning Test

Location: experiments/scripts/fast_lstm_online_test.py
Batches sequences and runs 365-day online learning evaluation in under 20 seconds.
Logs cleanly to both console and `logs/lstm_online_experiment.log`.
"""

import argparse
import gc
import sys
import time
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Ensure logs directory exists
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "lstm_online_experiment.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LSTM_Online_Exp")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns

class VectorizedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=24, dropout=0.2):
        super(VectorizedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

def run_fast_lstm_experiment(device_str: str = "cpu"):
    logger.info("=" * 80)
    logger.info("🧠 PYTORCH LSTM ONLINE INCREMENTAL LEARNING EXPERIMENT (365 DAYS)")
    logger.info("=" * 80)
    
    start_time = time.time()
    logger.info("📥 Loading historical raw data from PostgreSQL...")
    df_raw = load_all_historical_data()
    
    logger.info("⚙️ Building robust features including Net Load & Lag0 Renewable Ratios...")
    df_feat = build_robust_features(df_raw)
    
    feature_cols = get_feature_columns('robust', df_feat)
    target_col = 'mcp_price_usd'
    
    df_model = df_feat.dropna(subset=feature_cols + [target_col]).copy()
    
    # Split 365 Days Test Window (13.08.2025 -> 12.08.2026)
    test_days = 365
    eval_dates = df_model.index.normalize().unique()[-test_days:]
    
    pretrain_end = eval_dates[0] - pd.Timedelta(hours=1)
    train_pretrain = df_model.loc[:pretrain_end].copy()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_pretrain[feature_cols])
    y_train_scaled = scaler_y.fit_transform(train_pretrain[[target_col]])
    
    seq_len = 168
    
    def build_fast_sequences(X, y, seq_len=168):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len - 24 + 1):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len : i + seq_len + 24, 0])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
        
    X_pre, y_pre = build_fast_sequences(X_train_scaled, y_train_scaled, seq_len)
    
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
        
    logger.info(f"⚡ Device: {device} | Pre-train Window: {train_pretrain.index.min().strftime('%Y-%m-%d')} -> {train_pretrain.index.max().strftime('%Y-%m-%d')} ({X_pre.shape[0]} sequences)")
    logger.info(f"🎯 Test Window: {eval_dates.min().strftime('%Y-%m-%d')} -> {eval_dates.max().strftime('%Y-%m-%d')} ({len(eval_dates)} days / 8,760 hours)")
    
    def test_scenario(scen_name, update_epochs=1, lr=1e-4):
        t0 = time.time()
        logger.info(f"\n🚀 Starting Scenario: {scen_name}...")
        
        model = VectorizedLSTM(input_dim=len(feature_cols)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.L1Loss()
        
        # Memory-safe batch loading (keep dataset on CPU RAM, move batches to device)
        ds = TensorDataset(torch.from_numpy(X_pre), torch.from_numpy(y_pre))
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        
        model.train()
        for ep in range(12):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                
        logger.info(f"  ✅ Base Pre-training complete (Final Pre-train Loss: {loss.item():.4f})")
        
        # Batched Online Test Loop
        online_opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        preds_all = []
        acts_all = []
        
        test_df = df_model.loc[eval_dates[0] - pd.Timedelta(hours=seq_len):].copy()
        test_X_scaled = scaler_X.transform(test_df[feature_cols])
        
        test_dates_list = eval_dates
        total_test_days = len(test_dates_list)
        
        for idx, d in enumerate(test_dates_list):
            t_start = d
            t_end = d + pd.Timedelta(hours=23)
            
            sub_hist = test_df.loc[:t_start - pd.Timedelta(hours=1)]
            if len(sub_hist) < seq_len:
                continue
                
            input_seq = test_X_scaled[len(sub_hist) - seq_len : len(sub_hist)]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                pred_scaled = model(input_tensor).cpu().numpy()[0]
                pred_usd = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                
            act_usd = test_df.loc[t_start:t_end, target_col].values
            if len(act_usd) == 24:
                preds_all.extend(pred_usd)
                acts_all.extend(act_usd)
                
            # Online fine-tune step (1 batch)
            y_target_scaled = scaler_y.transform(act_usd.reshape(-1, 1)).T
            bx_tune = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            by_tune = torch.tensor(y_target_scaled, dtype=torch.float32).to(device)
            
            model.train()
            for _ in range(update_epochs):
                online_opt.zero_grad()
                l_tune = criterion(model(bx_tune), by_tune)
                l_tune.backward()
                online_opt.step()
                
            if (idx + 1) % 90 == 0 or (idx + 1) == total_test_days:
                logger.info(f"  ⏳ Progress: {idx + 1}/{total_test_days} test days ({(idx + 1)/total_test_days*100:.1f}%) processed...")
                if device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
                
        t_elapsed = time.time() - t0
        y_a = np.array(acts_all)
        y_p = np.array(preds_all)
        
        mae = np.mean(np.abs(y_a - y_p))
        rmse = np.sqrt(np.mean((y_a - y_p)**2))
        wape = (np.sum(np.abs(y_a - y_p)) / np.sum(y_a)) * 100
        
        logger.info(f"  🏁 {scen_name} Completed in {t_elapsed:.1f}s | WAPE: %{wape:.2f} | MAE: ${mae:.2f} | RMSE: ${rmse:.2f}")
        
        # Cleanup
        del model, optimizer, online_opt
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

        return {'Senaryo': scen_name, 'WAPE (%)': round(wape, 2), 'MAE ($)': round(mae, 2), 'RMSE ($)': round(rmse, 2), 'Süre (s)': round(t_elapsed, 1)}

    res1 = test_scenario("Senaryo A (1 Epoch Online)", update_epochs=1, lr=1e-4)
    res2 = test_scenario("Senaryo B (3 Epoch Online)", update_epochs=3, lr=1e-4)
    res3 = test_scenario("Senaryo C (5 Epoch Online)", update_epochs=5, lr=5e-4)
    
    df_res = pd.DataFrame([res1, res2, res3])
    
    # Save CSV results
    out_csv = project_root / "experiments" / "lstm_online_365d_results.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"🏆 PYTORCH LSTM ONLINE LEARNING SON 1 YIL (365 GÜN) PERFORMANS TABLOSU")
    logger.info("=" * 80)
    logger.info("\n" + df_res.to_string(index=False))
    logger.info(f"\n💾 Results saved to: {out_csv}")
    logger.info(f"📝 Full log saved to: {log_file}")
    logger.info(f"⏱️ Total Experiment Runtime: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-Optimized Vectorized PyTorch LSTM Online Incremental Learning Test")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"], help="Device to run PyTorch model on (default: cpu for stability on Apple Silicon)")
    args = parser.parse_args()
    
    run_fast_lstm_experiment(device_str=args.device)

