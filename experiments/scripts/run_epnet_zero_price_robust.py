"""EPNet Zero-Price Robust Backtest Benchmark Engine.

Focuses specifically on handling zero/near-zero price regimes (May-June hydro/renewable surges)
using 3 targeted Zero-Price Robust strategies:
1. ROBUST_01: Renewable & Hydro Pressure Ratios + Huber Loss
2. ROBUST_02: Cyclic Seasonality (Sin/Cos Month & DayOfYear) + Huber Loss
3. ROBUST_03: Combined Pressure + Cyclic + Zero-Floor Thresholding ($0.01) + Huber Loss

Evaluates 12-month (365 days) walk-forward backtests with 3-seed ensemble.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from db.connection import get_db_engine
from sqlalchemy import text
from src.features.feature_engineering import build_robust_features as base_build_robust_features


# --- Output Directories & Log Files ---
def setup_log_environment(mode="fast"):
    if mode == "fast":
        log_dir = project_root / "logs" / "epnet_fast_robust_experiments"
        checkpoint_file = log_dir / "fast_robust_checkpoint.json"
        summary_csv = log_dir / "epnet_fast_robust_summary.csv"
        summary_md = log_dir / "epnet_fast_robust_summary.md"
    else:
        log_dir = project_root / "logs" / "epnet_robust_experiments"
        checkpoint_file = log_dir / "robust_checkpoint.json"
        summary_csv = log_dir / "epnet_robust_summary.csv"
        summary_md = log_dir / "epnet_robust_summary.md"
        
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, checkpoint_file, summary_csv, summary_md

LOG_DIR, CHECKPOINT_FILE, SUMMARY_CSV_FILE, SUMMARY_MD_FILE = setup_log_environment("fast")


def set_seed(seed=42):
    """Sets deterministic seed across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- EPNet Architecture ---

class EPNet_Robust(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=32, lstm_hidden_dim=32, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, batch_first=True)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.relu3(lstm_out[:, -1, :])
        out = self.fc(out)
        return out.squeeze(-1)


# --- Data Loading & Feature Engineering ---

def load_master_dataset():
    """Fetches master silver layer data from PostgreSQL database."""
    engine = get_db_engine()
    master_sql = text("""
        SELECT 
            m.ts,
            m.price_usd AS mcp_price_usd,
            m.price_try AS mcp_price_try,
            s.system_marginal_price_try AS smp_price_try,
            l.load_forecast_mw,
            k.total_mw AS kgup_total_mw,
            k.natural_gas_mw AS kgup_gas_mw,
            k.wind_mw AS kgup_wind_mw,
            k.solar_mw AS kgup_solar_mw,
            k.dammed_hydro_mw + k.river_hydro_mw AS kgup_hydro_mw,
            k.import_coal_mw + k.lignite_mw + k.black_coal_mw AS kgup_coal_mw,
            g.total_mw AS actual_gen_total_mw,
            c.consumption_mw AS actual_cons_mw,
            w.turkey_weighted_temperature_c AS temperature_c,
            mc.usd_try,
            mc.brent_oil_usd,
            ng.gas_reference_price_try AS natural_gas_grf_try,
            wp.hydro_water_energy_mwh
        FROM raw_mcp_hourly m
        LEFT JOIN raw_smp_hourly s ON m.ts = s.ts
        LEFT JOIN raw_load_forecast_hourly l ON m.ts = l.ts
        LEFT JOIN raw_kgup_hourly k ON m.ts = k.ts
        LEFT JOIN raw_actual_generation_hourly g ON m.ts = g.ts
        LEFT JOIN raw_actual_consumption_hourly c ON m.ts = c.ts
        LEFT JOIN raw_weather_hourly w ON m.ts = w.ts
        LEFT JOIN raw_macro_daily mc ON DATE(m.ts) = mc.entry_date
        LEFT JOIN raw_natural_gas_daily ng ON DATE(m.ts) = ng.entry_date
        LEFT JOIN (
            SELECT DATE(date_time) AS entry_date, SUM(water_energy_provision_mwh) AS hydro_water_energy_mwh
            FROM raw_master_water_energy_provision
            GROUP BY DATE(date_time)
        ) wp ON DATE(m.ts) = wp.entry_date
        ORDER BY m.ts ASC;
    """)

    with engine.connect() as conn:
        df_raw = pd.read_sql(master_sql, conn)

    df_raw['ts'] = pd.to_datetime(df_raw['ts']).dt.tz_convert('Europe/Istanbul')
    df_raw = df_raw.set_index('ts').sort_index()

    df_raw['usd_try'] = df_raw['usd_try'].ffill().bfill()
    df_raw['brent_oil_usd'] = df_raw['brent_oil_usd'].ffill().bfill()
    df_raw['natural_gas_grf_try'] = df_raw['natural_gas_grf_try'].ffill().bfill()
    return df_raw


def build_robust_features(df):
    """Engineers Zero-Price Robust features including Pressure Ratios and Cyclic Seasonality."""
    df_feat = base_build_robust_features(df)
    df_model = df_feat.dropna().copy()
    
    # Feature Matrices
    exclude_base = ['mcp_price_usd', 'mcp_price_try', 'smp_price_try', 'actual_gen_total_mw', 'actual_cons_mw', 'load_forecast_mw', 'kgup_total_mw', 'kgup_gas_mw', 'kgup_wind_mw', 'kgup_solar_mw', 'kgup_hydro_mw', 'kgup_coal_mw', 'dayofyear', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy']
    base_full = [c for c in df_model.columns if c not in exclude_base]
    
    matrix_pressure = base_full + ['hydro_pressure_ratio', 'renewable_pressure_ratio', 'solar_peak_pressure_ratio']
    matrix_cyclic = base_full + ['sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy']
    matrix_combined = base_full + ['hydro_pressure_ratio', 'renewable_pressure_ratio', 'solar_peak_pressure_ratio', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy']
    
    feature_matrices = {
        'Matrix_Pressure': matrix_pressure,
        'Matrix_Cyclic': matrix_cyclic,
        'Matrix_Combined': matrix_combined
    }
    
    return df_model, feature_matrices


# --- Metrics ---

def calculate_safe_mape(y_true, y_pred):
    safe_denom = np.maximum(y_true, 1.0)
    return np.mean(np.abs((y_true - y_pred) / safe_denom)) * 100


def calculate_wape(y_true, y_pred):
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100


# --- Training Engine with Zero-Floor Protection ---

def train_predict_robust(tr_df, te_df, features, target='mcp_price_usd', floor_threshold=0.01, seq_len=24, epochs=25, batch_size=64, lr=0.001, n_seeds=3):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(tr_df[features].values)
    
    # Zero-floor clipping to protect against log(0) singularities
    y_tr_clipped = np.maximum(tr_df[target].values, floor_threshold)
    y_tr_log = np.log1p(y_tr_clipped)
    
    X_te_s = scaler.transform(te_df[features].values)
    
    X_tr_seq, y_tr_seq = [], []
    for i in range(len(X_tr_s) - seq_len + 1):
        X_tr_seq.append(X_tr_s[i:i+seq_len])
        y_tr_seq.append(y_tr_log[i+seq_len-1])
        
    X_tr_t = torch.tensor(np.array(X_tr_seq), dtype=torch.float32)
    y_tr_t = torch.tensor(np.array(y_tr_seq), dtype=torch.float32)
    
    X_full_s = np.vstack([X_tr_s[-(seq_len-1):], X_te_s])
    X_te_seq = []
    for i in range(len(X_te_s)):
        X_te_seq.append(X_full_s[i:i+seq_len])
    X_te_t = torch.tensor(np.array(X_te_seq), dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [42, 123, 999, 2024, 777][:n_seeds]
    all_seed_preds = []
    
    # Adaptive batch size scaling to ensure sufficient gradient updates
    effective_batch_size = min(batch_size, max(16, len(X_tr_seq) // 8))
    
    for seed in seeds:
        set_seed(seed)
        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        
        model = EPNet_Robust(input_dim=len(features)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Huber Loss (Smooth L1) for zero/spike robustness
        criterion = nn.SmoothL1Loss(beta=0.1)
            
        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            preds_log = model(X_te_t.to(device)).cpu().numpy()
            
        preds_seed = np.expm1(preds_log)
        all_seed_preds.append(np.maximum(preds_seed, 0.0))
        
    return np.mean(all_seed_preds, axis=0)


# --- Resumable State Checkpoint ---

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_strategies": [], "results": {}}


def save_checkpoint(completed_strategies, results_dict):
    data = {
        "completed_strategies": completed_strategies,
        "results": results_dict,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def generate_markdown_report(results_dict):
    md_content = "# 🛡️ EPNet Zero-Price Robust Strategies Benchmark Summary\n\n"
    md_content += f"*Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md_content += "| Strategy ID | Strategy Name | Matrix | Lookback | Loss | Floor | 1M WAPE | 1M MAE | 3M WAPE | 6M WAPE | 9M WAPE | 12M WAPE | 12M MAE | 12M MAPE |\n"
    md_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    for s_id, s in results_dict.items():
        m = s["metrics"]
        w1 = f"%{m.get('wape_1m', 0):.2f}" if 'wape_1m' in m else "-"
        mae1 = f"${m.get('mae_1m', 0):.2f}" if 'mae_1m' in m else "-"
        w3 = f"%{m.get('wape_3m', 0):.2f}" if 'wape_3m' in m else "-"
        w6 = f"%{m.get('wape_6m', 0):.2f}" if 'wape_6m' in m else "-"
        w9 = f"%{m.get('wape_9m', 0):.2f}" if 'wape_9m' in m else "-"
        w12 = f"%{m.get('wape_12m', 0):.2f}" if 'wape_12m' in m else "-"
        mae12 = f"${m.get('mae_12m', 0):.2f}" if 'mae_12m' in m else "-"
        mape12 = f"%{m.get('mape_12m', 0):.2f}" if 'mape_12m' in m else "-"
        
        md_content += f"| **{s_id}** | `{s['name']}` | {s['matrix']} | {s['lookback_name']} | {s['loss']} | {s['floor']} | **{w1}** | **{mae1}** | {w3} | {w6} | {w9} | **{w12}** | {mae12} | {mape12} |\n"
        
    with open(SUMMARY_MD_FILE, "w") as f:
        f.write(md_content)


# --- 3 Zero-Price Robust Strategies ---

ROBUST_STRATEGIES = [
    {
        "id": "ROBUST_01",
        "name": "Pressure_Ratios_Huber",
        "matrix": "Matrix_Pressure",
        "lookback_hours": 0,
        "lookback_name": "Tüm Geçmiş (2024-2026)",
        "loss": "Huber",
        "floor": 0.01
    },
    {
        "id": "ROBUST_02",
        "name": "Cyclic_Seasonality_Huber",
        "matrix": "Matrix_Cyclic",
        "lookback_hours": 0,
        "lookback_name": "Tüm Geçmiş (2024-2026)",
        "loss": "Huber",
        "floor": 0.01
    },
    {
        "id": "ROBUST_03",
        "name": "Combined_Floor_Huber",
        "matrix": "Matrix_Combined",
        "lookback_hours": 0,
        "lookback_name": "Tüm Geçmiş (2024-2026)",
        "loss": "Huber",
        "floor": 0.01
    }
]


def run_robust_fast_sequential_benchmark(total_days=365, n_seeds=3, initial_epochs=25, update_epochs=1):
    """Fast Sequential Incremental Benchmark for Zero-Price Robust Strategies.
    
    1. Pre-trains ensemble models using Huber Loss & Zero-Floor Clipping on all historical data up to backtest start.
    2. Sequentially evaluates each day chronologically (day 1 to day N).
    3. Performs 1-epoch warm-start weight update after each daily prediction using low LR (0.0002).
    """
    print("=" * 95)
    print("⚡ EPNET SIFIR FİYAT HIZLI SIRA-ARDIŞIK ROBUST BENCHMARK MOTORU BAŞLATILIYOR")
    print(f"📊 Odak Strateji Sayısı: {len(ROBUST_STRATEGIES)} adet | Backtest Süresi: {total_days} gün (12 Ay)")
    print(f"⚙️ Mod: Hızlı Sıra-Ardışık Warm-Start | Initial Epochs: {initial_epochs} | Daily Update: {update_epochs} Epoch (LR=0.0002)")
    print("=" * 95)
    
    print("\n📦 EPİAŞ Veritabanından Veriler Yükleniyor...")
    df_raw = load_master_dataset()
    df_model, feature_matrices = build_robust_features(df_raw)
    
    checkpoint_state = load_checkpoint()
    completed_strategies = set(checkpoint_state.get("completed_strategies", []))
    results_dict = checkpoint_state.get("results", {})
    
    start_grid_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_ts = df_model.index.max()
    min_ts = df_model.index.min()
    backtest_start = max_ts - pd.Timedelta(days=total_days)
    
    init_tr_start = min_ts
    init_tr_end = backtest_start - pd.Timedelta(hours=1)
    init_tr_df = df_model.loc[init_tr_start:init_tr_end]
    
    for s_idx, s in enumerate(ROBUST_STRATEGIES):
        s_id = s["id"]
        s_name = s["name"]
        
        if s_id in completed_strategies:
            print(f"⏩ [{s_idx+1:02d}/{len(ROBUST_STRATEGIES):02d}] {s_id} ({s_name}) DAHA ÖNCE TAMAMLANIŞ. ATLANIYOR...")
            continue
            
        lookback_h = s.get("lookback_hours", 4380)
        if lookback_h and lookback_h > 0:
            init_tr_start = max(min_ts, backtest_start - pd.Timedelta(hours=lookback_h))
        else:
            init_tr_start = min_ts
            
        init_tr_end = backtest_start - pd.Timedelta(hours=1)
        init_tr_df = df_model.loc[init_tr_start:init_tr_end]
        
        print("\n" + "-" * 95)
        print(f"🔥 [{s_idx+1:02d}/{len(ROBUST_STRATEGIES):02d}] YENİ HIZLI ROBUST STRATEJİ BAŞLATILIYOR: {s_id} ({s_name})")
        print(f"🛠️ Matrix: {s['matrix']} | Lookback: {s['lookback_name']} ({init_tr_start.strftime('%Y-%m-%d')} -> {init_tr_end.strftime('%Y-%m-%d')}, {len(init_tr_df):,} Saat) | Loss: {s['loss']} | Floor: ${s['floor']}")
        print("-" * 95)
        
        features = feature_matrices[s["matrix"]]
        strat_start_time = time.time()
        
        # Pre-train ensemble
        scaler = StandardScaler()
        X_init_tr_s = scaler.fit_transform(init_tr_df[features].values)
        y_init_clipped = np.maximum(init_tr_df['mcp_price_usd'].values, s['floor'])
        y_init_log = np.log1p(y_init_clipped)
        
        seq_len = 24
        X_init_seq, y_init_seq = [], []
        for i in range(len(X_init_tr_s) - seq_len + 1):
            X_init_seq.append(X_init_tr_s[i:i+seq_len])
            y_init_seq.append(y_init_log[i+seq_len-1])
            
        X_init_t = torch.tensor(np.array(X_init_seq), dtype=torch.float32)
        y_init_t = torch.tensor(np.array(y_init_seq), dtype=torch.float32)
        
        seeds = [42, 123, 999, 2024, 777][:n_seeds]
        ensemble_models = []
        optimizers = []
        
        for seed in seeds:
            set_seed(seed)
            dataset = TensorDataset(X_init_t, y_init_t)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            model = EPNet_Robust(input_dim=len(features)).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=initial_epochs)
            criterion = nn.SmoothL1Loss(beta=0.1)
            
            model.train()
            for epoch in range(initial_epochs):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                
            ensemble_models.append(model)
            optimizers.append(optimizer)
            
        daily_records = []
        eval_start_time = time.time()
        last_seq_buffer = X_init_tr_s[-(seq_len-1):]
        
        for day_idx in range(total_days):
            day_start = backtest_start + pd.Timedelta(days=day_idx)
            day_end = day_start + pd.Timedelta(hours=23)
            
            day_df = df_model.loc[day_start:day_end]
            if len(day_df) < 24:
                continue
                
            X_day_s = scaler.transform(day_df[features].values)
            y_day_true = day_df['mcp_price_usd'].values
            
            X_full_s = np.vstack([last_seq_buffer, X_day_s])
            X_day_seq = []
            for i in range(len(X_day_s)):
                X_day_seq.append(X_full_s[i:i+seq_len])
            X_day_t = torch.tensor(np.array(X_day_seq), dtype=torch.float32).to(device)
            
            seed_preds = []
            for model in ensemble_models:
                model.eval()
                with torch.no_grad():
                    pred_log = model(X_day_t).cpu().numpy()
                pred_val = np.expm1(pred_log)
                seed_preds.append(np.maximum(pred_val, 0.0))
                
            day_preds = np.mean(seed_preds, axis=0)
            
            mae = mean_absolute_error(y_day_true, day_preds)
            wape = calculate_wape(y_day_true, day_preds)
            mape = calculate_safe_mape(y_day_true, day_preds)
            
            daily_records.append({
                'day_offset': total_days - day_idx,
                'tarih': day_start.strftime('%Y-%m-%d'),
                'mae': mae,
                'wape': wape,
                'mape': mape
            })
            
            # Incremental Huber update with 1 Epoch (lr=0.0002)
            y_day_clipped = np.maximum(y_day_true, s['floor'])
            y_day_log = np.log1p(y_day_clipped)
            y_day_seq_target = [y_day_log[i] for i in range(len(X_day_s))]
            
            X_up_t = torch.tensor(np.array(X_day_seq), dtype=torch.float32).to(device)
            y_up_t = torch.tensor(np.array(y_day_seq_target), dtype=torch.float32).to(device)
            
            criterion = nn.SmoothL1Loss(beta=0.1)
            for model, opt in zip(ensemble_models, optimizers):
                model.train()
                for param_group in opt.param_groups:
                    param_group['lr'] = 0.0002
                for _ in range(update_epochs):
                    opt.zero_grad()
                    loss = criterion(model(X_up_t), y_up_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    
            last_seq_buffer = X_day_s[-(seq_len-1):]
            
            if (day_idx + 1) % 10 == 0 or (day_idx + 1) == total_days:
                elapsed_sec = time.time() - eval_start_time
                avg_sec = elapsed_sec / (day_idx + 1)
                eta_sec = (total_days - day_idx - 1) * avg_sec
                pct = ((day_idx + 1) / total_days) * 100
                print(f"⏳ Robust [{s_id}] [{day_idx+1:03d}/{total_days:03d}] (%{pct:5.1f}) | Tarih: {day_start.strftime('%Y-%m-%d')} | MAE: ${mae:5.2f}/MWh | WAPE: %{wape:5.2f} | Kalan ETA: {eta_sec:4.0f}s")
                
        strat_df = pd.DataFrame(daily_records)
        strat_df.to_csv(LOG_DIR / f"{s_id}_{s_name}_daily.csv", index=False)
        
        m_1m = strat_df.tail(30)
        m_3m = strat_df[strat_df['day_offset'] <= 90] if 'day_offset' in strat_df.columns else strat_df.head(90)
        m_6m = strat_df[strat_df['day_offset'] <= 180] if 'day_offset' in strat_df.columns else strat_df.head(180)
        m_9m = strat_df[strat_df['day_offset'] <= 270] if 'day_offset' in strat_df.columns else strat_df.head(270)
        m_12m = strat_df
        
        metrics = {
            'wape_1m': m_1m['wape'].mean() if len(m_1m) > 0 else strat_df['wape'].mean(),
            'mae_1m': m_1m['mae'].mean() if len(m_1m) > 0 else strat_df['mae'].mean(),
            'mape_1m': m_1m['mape'].mean() if len(m_1m) > 0 else strat_df['mape'].mean(),
            'wape_3m': m_3m['wape'].mean() if len(m_3m) > 0 else strat_df['wape'].mean(),
            'wape_6m': m_6m['wape'].mean() if len(m_6m) > 0 else strat_df['wape'].mean(),
            'wape_9m': m_9m['wape'].mean() if len(m_9m) > 0 else strat_df['wape'].mean(),
            'wape_12m': m_12m['wape'].mean(),
            'mae_12m': m_12m['mae'].mean(),
            'mape_12m': m_12m['mape'].mean()
        }
        
        s_summary = {**s, "metrics": metrics, "eval_days_count": len(strat_df)}
        results_dict[s_id] = s_summary
        completed_strategies.add(s_id)
        
        save_checkpoint(list(completed_strategies), results_dict)
        generate_markdown_report(results_dict)
        
        summary_rows = []
        for s_k, s_v in results_dict.items():
            row = {**s_v, **s_v['metrics']}
            del row['metrics']
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV_FILE, index=False)
        
        print(f"✅ [{s_idx+1:02d}/{len(ROBUST_STRATEGIES):02d}] {s_id:10s} ({s_name:25s}) | 12M WAPE: %{metrics['wape_12m']:5.2f} | MAE: ${metrics['mae_12m']:5.2f} | Süre: {(time.time() - strat_start_time)/60:4.1f} dk")


def run_robust_benchmark(step_days=2, total_days=365, n_seeds=3):
    print("=" * 95)
    print("🛡️ EPNET SIFIR FİYAT ROBUST BENCHMARK MOTORU BAŞLATILIYOR")
    print(f"📊 Odak Strateji Sayısı: {len(ROBUST_STRATEGIES)} adet | Backtest Süresi: {total_days} gün (12 Ay)")
    print(f"⚙️ Test Stride: {step_days} günde 1 evaluation checkpoint | Seed Ensemble: {n_seeds} Model/Gün")
    print("=" * 95)
    
    print("\n📦 EPİAŞ Veritabanından Veriler Yükleniyor...")
    df_raw = load_master_dataset()
    df_model, feature_matrices = build_robust_features(df_raw)
    
    checkpoint_state = load_checkpoint()
    completed_strategies = set(checkpoint_state.get("completed_strategies", []))
    results_dict = checkpoint_state.get("results", {})
    
    start_grid_time = time.time()
    
    for s_idx, s in enumerate(ROBUST_STRATEGIES):
        s_id = s["id"]
        s_name = s["name"]
        
        if s_id in completed_strategies:
            print(f"⏩ [{s_idx+1:02d}/{len(ROBUST_STRATEGIES):02d}] {s_id} ({s_name}) DÖNEMİ DAHA ÖNCE TAMAMLANIŞ. ATLANIYOR...")
            continue
            
        print("\n" + "-" * 95)
        print(f"🔥 [{s_idx+1:02d}/{len(ROBUST_STRATEGIES):02d}] YENİ ROBUST STRATEJİ BAŞLATILIYOR: {s_id} ({s_name})")
        print(f"🛠️ Matrix: {s['matrix']} | Lookback: {s['lookback_name']} | Loss: {s['loss']} | Floor: ${s['floor']}")
        print("-" * 95)
        
        features = feature_matrices[s["matrix"]]
        daily_records = []
        strat_start_time = time.time()
        
        eval_indices = list(range(0, total_days, step_days))
        
        for e_idx, day_offset in enumerate(eval_indices):
            test_end = df_model.index.max() - pd.Timedelta(days=day_offset)
            test_start = test_end - pd.Timedelta(hours=23)
            train_end = test_start - pd.Timedelta(hours=1)
            train_start = train_end - pd.Timedelta(hours=s["lookback_hours"])
            
            tr_df = df_model.loc[train_start:train_end]
            te_df = df_model.loc[test_start:test_end]
            
            if len(tr_df) < 150 or len(te_df) < 12:
                continue
                
            preds = train_predict_robust(
                tr_df, te_df, features, target='mcp_price_usd',
                floor_threshold=s['floor'], epochs=25, batch_size=64, lr=0.001, n_seeds=n_seeds
            )
            
            y_true = te_df['mcp_price_usd'].values
            mae = mean_absolute_error(y_true, preds)
            mape = calculate_safe_mape(y_true, preds)
            wape = calculate_wape(y_true, preds)
            
            daily_records.append({
                'day_offset': day_offset,
                'tarih': test_start.strftime('%Y-%m-%d'),
                'mae': mae,
                'mape': mape,
                'wape': wape
            })
            
            if (e_idx + 1) % 5 == 0 or (e_idx + 1) == len(eval_indices):
                elapsed_s = time.time() - strat_start_time
                progress_pct = ((e_idx + 1) / len(eval_indices)) * 100
                sys.stdout.write(f"\r   ⏳ Robust [{s_idx+1:02d}/{len(ROBUST_STRATEGIES):02d}] ({s_id}) %{progress_pct:5.1f} | Gün: {test_start.strftime('%Y-%m-%d')} | Anlık MAE: ${mae:5.2f} | WAPE: %{wape:5.2f} | {elapsed_s:3.0f}s")
                sys.stdout.flush()

        sys.stdout.write('\r' + ' ' * 110 + '\r')
        sys.stdout.flush()

        strat_df = pd.DataFrame(daily_records)
        strat_df.to_csv(LOG_DIR / f"{s_id}_{s_name}_daily.csv", index=False)
        
        m_1m = strat_df.tail(30)
        m_3m = strat_df[strat_df['day_offset'] <= 90]
        m_6m = strat_df[strat_df['day_offset'] <= 180]
        m_9m = strat_df[strat_df['day_offset'] <= 270]
        m_12m = strat_df
        
        metrics = {
            'wape_1m': m_1m['wape'].mean() if len(m_1m) > 0 else strat_df['wape'].mean(),
            'mae_1m': m_1m['mae'].mean() if len(m_1m) > 0 else strat_df['mae'].mean(),
            'mape_1m': m_1m['mape'].mean() if len(m_1m) > 0 else strat_df['mape'].mean(),
            'wape_3m': m_3m['wape'].mean() if len(m_3m) > 0 else strat_df['wape'].mean(),
            'wape_6m': m_6m['wape'].mean() if len(m_6m) > 0 else strat_df['wape'].mean(),
            'wape_9m': m_9m['wape'].mean() if len(m_9m) > 0 else strat_df['wape'].mean(),
            'wape_12m': m_12m['wape'].mean(),
            'mae_12m': m_12m['mae'].mean(),
            'mape_12m': m_12m['mape'].mean()
        }
        
        s_summary = {**s, "metrics": metrics, "eval_days_count": len(strat_df)}
        results_dict[s_id] = s_summary
        completed_strategies.add(s_id)
        
        save_checkpoint(list(completed_strategies), results_dict)
        generate_markdown_report(results_dict)
        
        summary_rows = []
        for s_k, s_v in results_dict.items():
            row = {**s_v, **s_v['metrics']}
            del row['metrics']
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV_FILE, index=False)
        
        strat_duration = time.time() - strat_start_time
        overall_elapsed = time.time() - start_grid_time
        completed_count = len(completed_strategies)
        remaining_count = len(ROBUST_STRATEGIES) - completed_count
        avg_strat_time = overall_elapsed / max(completed_count, 1)
        eta_hours = (remaining_count * avg_strat_time) / 3600.0
        
        print(f"✅ [{s_idx+1:02d}/{len(ROBUST_STRATEGIES):02d}] {s_id:10s} ({s_name:25s}) | 12M WAPE: %{metrics['wape_12m']:5.2f} | MAE: ${metrics['mae_12m']:5.2f} | Süre: {strat_duration/60:4.1f} dk | Kalan ETA: {eta_hours:3.1f}h")
        
    total_grid_duration = time.time() - start_grid_time
    print("\n" + "=" * 95)
    print(f"🎉 TÜM SIFIR FİYAT ROBUST BENCHMARK STRATEJİLERİ BAŞARIYLA TAMAMLANDI!")
    print(f"⏱️ Toplam Çalışma Süresi: {total_grid_duration/3600:.2f} saat")
    print(f"📄 Özet Markdown Raporu: {SUMMARY_MD_FILE}")
    print(f"📊 Özet CSV Raporu: {SUMMARY_CSV_FILE}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPNet Zero-Price Robust Benchmark Engine")
    parser.add_argument("--step-days", type=int, default=2, help="Walk-forward test adımı (gün cinsinden, Varsayılan: 2 - Klasik mod için)")
    parser.add_argument("--total-days", type=int, default=365, help="Toplam test gün sayısı (Varsayılan: 365 gün)")
    parser.add_argument("--seeds", type=int, default=3, help="Ensemble için gün başına seed sayısı (Varsayılan: 3)")
    parser.add_argument("--mode", "-m", type=str, default="fast", choices=["fast", "classic"], help="Backtest modu: 'fast' (Hızlı Sıralı Warm-Start) veya 'classic' (Klasik Her Gün Sıfırdan)")
    parser.add_argument("--reset", action="store_true", help="Önceki robust checkpoint verilerini sıfırlayıp baştan başlar")
    args = parser.parse_args()
    
    LOG_DIR, CHECKPOINT_FILE, SUMMARY_CSV_FILE, SUMMARY_MD_FILE = setup_log_environment(args.mode)
    
    if args.reset:
        if CHECKPOINT_FILE.exists():
            os.remove(CHECKPOINT_FILE)
            print("🧹 Önceki robust checkpoint sıfırlandı, baştan başlatılıyor...")
            
    if args.mode == "fast":
        run_robust_fast_sequential_benchmark(total_days=args.total_days, n_seeds=args.seeds)
    else:
        run_robust_benchmark(step_days=args.step_days, total_days=args.total_days, n_seeds=args.seeds)
