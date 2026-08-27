"""EPNet Overnight Grid Search Backtest Engine with Resumable Checkpoints.

Runs a comprehensive set of EPNet strategies over a 12-month (365-day) walk-forward backtest.
Saves daily prediction logs, updates summary Markdown/CSV reports, and supports RESUME
if interrupted.
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
from src.features.feature_engineering import (
    build_base_features, 
    build_core_features, 
    build_full_features, 
    build_robust_features,
    get_feature_columns
)

# --- Output Directories & Checkpoint Files ---
def setup_log_environment(mode="fast"):
    if mode == "fast":
        log_dir = project_root / "logs" / "epnet_fast_grid_experiments"
        checkpoint_file = log_dir / "fast_grid_checkpoint.json"
        summary_csv = log_dir / "epnet_fast_grid_summary.csv"
        summary_md = log_dir / "epnet_fast_grid_summary.md"
    else:
        log_dir = project_root / "logs" / "epnet_experiments"
        checkpoint_file = log_dir / "grid_checkpoint.json"
        summary_csv = log_dir / "epnet_grid_summary.csv"
        summary_md = log_dir / "epnet_grid_summary.md"
        
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


# --- Model Architectures ---

class EPNet_LSTM(nn.Module):
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


class EPNet_GRU(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=32, gru_hidden_dim=32, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=gru_hidden_dim, batch_first=True)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(gru_hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        out = self.relu3(gru_out[:, -1, :])
        out = self.fc(out)
        return out.squeeze(-1)


class EPNet_BiLSTM(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=64, lstm_hidden_dim=64, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.bilstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
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


def build_feature_matrices(df):
    """Engineers 4 distinct feature sets (Set_Pure_PTF, Set_Core, Set_Full, Set_Cyclic)."""
    df_robust = build_robust_features(df)
    df_model = df_robust.dropna().copy()
    
    # Define Column Sets
    set_pure_ptf = ['hour', 'dayofweek', 'is_weekend', 'mcp_usd_lag_24', 'mcp_usd_lag_48', 'mcp_usd_lag_168', 'mcp_usd_roll_mean_24h', 'mcp_usd_roll_std_24h', 'mcp_usd_roll_mean_7d']
    
    set_core = set_pure_ptf + ['load_lag_24', 'load_lag_48', 'kgup_lag_24', 'kgup_lag_48', 'supply_demand_gap_mw']
    
    exclude_all = ['mcp_price_usd', 'mcp_price_try', 'smp_price_try', 'actual_gen_total_mw', 'actual_cons_mw', 'load_forecast_mw', 'kgup_total_mw', 'kgup_gas_mw', 'kgup_wind_mw', 'kgup_solar_mw', 'kgup_hydro_mw', 'kgup_coal_mw', 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow']
    set_full = [c for c in df_model.columns if c not in exclude_all]
    
    set_cyclic = set_full + ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow']
    
    feature_sets = {
        'Set_Pure_PTF': set_pure_ptf,
        'Set_Core': set_core,
        'Set_Full': set_full,
        'Set_Cyclic': set_cyclic
    }
    
    return df_model, feature_sets


# --- Metric Calculation Helpers ---

def calculate_safe_mape(y_true, y_pred):
    safe_denom = np.maximum(y_true, 1.0)
    return np.mean(np.abs((y_true - y_pred) / safe_denom)) * 100


def calculate_wape(y_true, y_pred):
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100


# --- Training Engine ---

def train_predict_epnet(tr_df, te_df, features, target='mcp_price_usd', model_arch='LSTM', loss_type='MSE', seq_len=24, epochs=25, batch_size=256, lr=0.001, n_seeds=3):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(tr_df[features].values)
    y_tr_log = np.log1p(np.maximum(tr_df[target].values, 0.0))
    
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
    
    for seed in seeds:
        set_seed(seed)
        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if model_arch == 'GRU':
            model = EPNet_GRU(input_dim=len(features)).to(device)
        elif model_arch == 'BiLSTM':
            model = EPNet_BiLSTM(input_dim=len(features)).to(device)
        else:
            model = EPNet_LSTM(input_dim=len(features)).to(device)
            
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        if loss_type == 'Huber':
            criterion = nn.SmoothL1Loss(beta=0.1)
        else:
            criterion = nn.MSELoss()
            
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


# --- Resumable Checkpoint State Management ---

def load_checkpoint():
    """Loads checkpoint state if it exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_strategies": [], "results": {}}


def save_checkpoint(completed_strategies, results_dict):
    """Persists completed strategy names and summary metrics."""
    data = {
        "completed_strategies": completed_strategies,
        "results": results_dict,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def generate_markdown_report(results_dict):
    """Generates a clean Markdown summary table showing 3m, 6m, 9m, and 12m metrics."""
    md_content = "# 🏆 EPNet Grid Search Experiment Results Summary\n\n"
    md_content += f"*Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md_content += "| Strategy ID | Strategy Name | Arch | Matrix | Lookback | Loss | 3M WAPE | 6M WAPE | 9M WAPE | 12M WAPE | 12M MAE | 12M MAPE |\n"
    md_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    for s_id, s in results_dict.items():
        m = s["metrics"]
        w3 = f"%{m.get('wape_3m', 0):.2f}" if 'wape_3m' in m else "-"
        w6 = f"%{m.get('wape_6m', 0):.2f}" if 'wape_6m' in m else "-"
        w9 = f"%{m.get('wape_9m', 0):.2f}" if 'wape_9m' in m else "-"
        w12 = f"%{m.get('wape_12m', 0):.2f}" if 'wape_12m' in m else "-"
        mae12 = f"${m.get('mae_12m', 0):.2f}" if 'mae_12m' in m else "-"
        mape12 = f"%{m.get('mape_12m', 0):.2f}" if 'mape_12m' in m else "-"
        
        md_content += f"| **{s_id}** | `{s['name']}` | {s['arch']} | {s['matrix']} | {s['lookback_name']} | {s['loss']} | {w3} | {w6} | {w9} | **{w12}** | {mae12} | {mape12} |\n"
        
    with open(SUMMARY_MD_FILE, "w") as f:
        f.write(md_content)


# --- Strategy Definitions (14 Diverse Configurations) ---

STRATEGIES = [
    # Pure PTF Benchmarks
    {"id": "STRAT_01", "name": "PurePTF_1Week", "arch": "CNN-LSTM", "matrix": "Set_Pure_PTF", "lookback_hours": 168, "lookback_name": "1 Hafta", "loss": "MSE"},
    {"id": "STRAT_02", "name": "PurePTF_1Month", "arch": "CNN-LSTM", "matrix": "Set_Pure_PTF", "lookback_hours": 720, "lookback_name": "1 Ay", "loss": "MSE"},
    {"id": "STRAT_03", "name": "PurePTF_6Month", "arch": "CNN-LSTM", "matrix": "Set_Pure_PTF", "lookback_hours": 4380, "lookback_name": "6 Ay", "loss": "MSE"},
    
    # Core Market Feature Benchmarks
    {"id": "STRAT_04", "name": "Core_1Month", "arch": "CNN-LSTM", "matrix": "Set_Core", "lookback_hours": 720, "lookback_name": "1 Ay", "loss": "MSE"},
    {"id": "STRAT_05", "name": "Core_3Month", "arch": "CNN-LSTM", "matrix": "Set_Core", "lookback_hours": 2160, "lookback_name": "3 Ay", "loss": "MSE"},
    {"id": "STRAT_06", "name": "Core_6Month", "arch": "CNN-LSTM", "matrix": "Set_Core", "lookback_hours": 4380, "lookback_name": "6 Ay", "loss": "MSE"},
    
    # Full Feature Base & Lookback Variations
    {"id": "STRAT_07", "name": "Full_1Month", "arch": "CNN-LSTM", "matrix": "Set_Full", "lookback_hours": 720, "lookback_name": "1 Ay", "loss": "MSE"},
    {"id": "STRAT_08", "name": "Full_3Month", "arch": "CNN-LSTM", "matrix": "Set_Full", "lookback_hours": 2160, "lookback_name": "3 Ay", "loss": "MSE"},
    {"id": "STRAT_09", "name": "Full_6Month_Base", "arch": "CNN-LSTM", "matrix": "Set_Full", "lookback_hours": 4380, "lookback_name": "6 Ay", "loss": "MSE"},
    {"id": "STRAT_10", "name": "Full_12Month", "arch": "CNN-LSTM", "matrix": "Set_Full", "lookback_hours": 8760, "lookback_name": "12 Ay", "loss": "MSE"},
    
    # Architectural & Loss Variants
    {"id": "STRAT_11", "name": "CNN_GRU_6Month", "arch": "CNN-GRU", "matrix": "Set_Full", "lookback_hours": 4380, "lookback_name": "6 Ay", "loss": "MSE"},
    {"id": "STRAT_12", "name": "Deep_BiLSTM_6Month", "arch": "CNN-BiLSTM", "matrix": "Set_Full", "lookback_hours": 4380, "lookback_name": "6 Ay", "loss": "MSE"},
    {"id": "STRAT_13", "name": "Huber_Loss_6Month", "arch": "CNN-LSTM", "matrix": "Set_Full", "lookback_hours": 4380, "lookback_name": "6 Ay", "loss": "Huber"},
    {"id": "STRAT_14", "name": "Cyclic_SinCos_6Month", "arch": "CNN-LSTM", "matrix": "Set_Cyclic", "lookback_hours": 4380, "lookback_name": "6 Ay", "loss": "MSE"},
]


# --- Main Grid Search Execution Loop ---

def run_fast_sequential_grid(total_days=365, n_seeds=3, initial_epochs=25, update_epochs=1):
    """Fast Sequential Incremental Warm-Start Engine for EPNet Overnight Grid Search.
    
    All hyper-parameters match the zero-price robust engine 100%:
    - Initial pre-training on all database history up to backtest start.
    - initial_epochs = 25, batch_size = 64, Adam lr = 0.001.
    - Daily warm-start update = 1 epoch with Adam lr = 0.0002.
    - Supports CNN-LSTM, CNN-GRU, CNN-BiLSTM, MSE & Huber Loss functions.
    """
    print("=" * 95)
    print("⚡ EPNET GECE STRATEJİ HIZLI SIRA-ARDIŞIK (FAST SEQUENTIAL) GRID SEARCH MOTORU BAŞLATILIYOR")
    print(f"📊 Toplam Strateji Sayısı: {len(STRATEGIES)} adet | Backtest Süresi: {total_days} gün (12 Ay)")
    print(f"⚙️ Mod: Hızlı Sıra-Ardışık Warm-Start | Initial Epochs: {initial_epochs} | Daily Update: {update_epochs} Epoch (LR=0.0002)")
    print("=" * 95)
    
    print("\n📦 EPİAŞ Veritabanından Veriler Yükleniyor...")
    df_raw = load_master_dataset()
    df_model, feature_matrices = build_feature_matrices(df_raw)
    
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
    
    arch_map = {
        "CNN-LSTM": EPNet_LSTM,
        "CNN-GRU": EPNet_GRU,
        "CNN-BiLSTM": EPNet_BiLSTM
    }
    
    for s_idx, s in enumerate(STRATEGIES):
        s_id = s["id"]
        s_name = s["name"]
        
        if s_id in completed_strategies:
            print(f"⏩ [{s_idx+1:02d}/{len(STRATEGIES):02d}] {s_id} ({s_name}) DAHA ÖNCE TAMAMLANIŞ. ATLANIYOR...")
            continue
            
        # Calculate strategy-specific initial lookback start date
        lookback_h = s.get("lookback_hours", 4380)
        if lookback_h and lookback_h > 0:
            init_tr_start = max(min_ts, backtest_start - pd.Timedelta(hours=lookback_h))
        else:
            init_tr_start = min_ts
            
        init_tr_end = backtest_start - pd.Timedelta(hours=1)
        init_tr_df = df_model.loc[init_tr_start:init_tr_end]
        
        print("\n" + "-" * 95)
        print(f"🔥 [{s_idx+1:02d}/{len(STRATEGIES):02d}] YENİ HIZLI GRID STRATEJİ BAŞLATILIYOR: {s_id} ({s_name})")
        print(f"🛠️ Arch: {s['arch']} | Matrix: {s['matrix']} | Lookback: {s['lookback_name']} ({init_tr_start.strftime('%Y-%m-%d')} -> {init_tr_end.strftime('%Y-%m-%d')}, {len(init_tr_df):,} Saat) | Loss: {s['loss']}")
        print("-" * 95)
        features = feature_matrices[s["matrix"]]
        strat_start_time = time.time()
        
        # Pre-train ensemble
        scaler = StandardScaler()
        X_init_tr_s = scaler.fit_transform(init_tr_df[features].values)
        y_init_log = np.log1p(np.maximum(init_tr_df['mcp_price_usd'].values, 0.01))
        
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
        
        ModelClass = arch_map.get(s["arch"], EPNet_LSTM)
        
        for seed in seeds:
            set_seed(seed)
            dataset = TensorDataset(X_init_t, y_init_t)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            model = ModelClass(input_dim=len(features)).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=initial_epochs)
            
            if s["loss"] == "Huber":
                criterion = nn.SmoothL1Loss(beta=0.1)
            else:
                criterion = nn.MSELoss()
                
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
            
            # Incremental update with 1 Epoch (lr=0.0002)
            y_day_log = np.log1p(np.maximum(y_day_true, 0.01))
            y_day_seq_target = [y_day_log[i] for i in range(len(X_day_s))]
            
            X_up_t = torch.tensor(np.array(X_day_seq), dtype=torch.float32).to(device)
            y_up_t = torch.tensor(np.array(y_day_seq_target), dtype=torch.float32).to(device)
            
            if s["loss"] == "Huber":
                criterion = nn.SmoothL1Loss(beta=0.1)
            else:
                criterion = nn.MSELoss()
                
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
                print(f"⏳ Grid [{s_id}] [{day_idx+1:03d}/{total_days:03d}] (%{pct:5.1f}) | Tarih: {day_start.strftime('%Y-%m-%d')} | MAE: ${mae:5.2f}/MWh | WAPE: %{wape:5.2f} | Kalan ETA: {eta_sec:4.0f}s")
                
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
        
        print(f"✅ [{s_idx+1:02d}/{len(STRATEGIES):02d}] {s_id:8s} ({s_name:20s}) | 12M WAPE: %{metrics['wape_12m']:5.2f} | MAE: ${metrics['mae_12m']:5.2f} | Süre: {(time.time() - strat_start_time)/60:4.1f} dk")


def run_grid_search(step_days=2, total_days=365, n_seeds=3):
    print("=" * 95)
    print("🚀 EPNET GECE STRATEJİ GRID SEARCH BACKTEST MOTORU (RESUMABLE CHECKPOINT)")
    print(f"📊 Toplam Strateji Sayısı: {len(STRATEGIES)} adet | Backtest Süresi: {total_days} gün (12 Ay)")
    print(f"⚙️ Test Stride: {step_days} günde 1 evaluation checkpoint | Seed Ensemble: {n_seeds} Model/Gün")
    print("=" * 95)
    
    print("\n📦 EPİAŞ Veritabanından Veriler Yükleniyor...")
    df_raw = load_master_dataset()
    df_model, feature_matrices = build_feature_matrices(df_raw)
    
    checkpoint_state = load_checkpoint()
    completed_strategies = set(checkpoint_state.get("completed_strategies", []))
    results_dict = checkpoint_state.get("results", {})
    
    start_grid_time = time.time()
    
    for s_idx, s in enumerate(STRATEGIES):
        s_id = s["id"]
        s_name = s["name"]
        
        if s_id in completed_strategies:
            print(f"⏩ [{s_idx+1:02d}/{len(STRATEGIES):02d}] {s_id} ({s_name}) DÖNEMİ DAHA ÖNCE TAMAMLANIŞ. ATLANIYOR...")
            continue
            
        print("\n" + "-" * 95)
        print(f"🔥 [{s_idx+1:02d}/{len(STRATEGIES):02d}] YENİ STRATEJİ BAŞLATILIYOR: {s_id} ({s_name})")
        print(f"🛠️ Mimari: {s['arch']} | Input Matrix: {s['matrix']} | Lookback: {s['lookback_name']} | Loss: {s['loss']}")
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
                
            preds = train_predict_epnet(
                tr_df, te_df, features, target='mcp_price_usd',
                model_arch=s['arch'], loss_type=s['loss'],
                epochs=25, batch_size=256, lr=0.001, n_seeds=n_seeds
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
                sys.stdout.write(f"\r   ⏳ Strateji [{s_idx+1:02d}/{len(STRATEGIES):02d}] ({s_id}) %{progress_pct:5.1f} | Gün: {test_start.strftime('%Y-%m-%d')} | Anlık MAE: ${mae:5.2f} | WAPE: %{wape:5.2f} | {elapsed_s:3.0f}s")
                sys.stdout.flush()

        # Clear in-place progress line
        sys.stdout.write('\r' + ' ' * 110 + '\r')
        sys.stdout.flush()

        strat_df = pd.DataFrame(daily_records)
        strat_df.to_csv(LOG_DIR / f"{s_id}_{s_name}_daily.csv", index=False)
        
        # Calculate sub-period metrics (3M: 0-90 days, 6M: 0-180 days, 9M: 0-270 days, 12M: 0-365 days)
        m_1m = strat_df[strat_df['day_offset'] <= 30]
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
        
        # Persist checkpoint & Markdown report
        save_checkpoint(list(completed_strategies), results_dict)
        generate_markdown_report(results_dict)
        
        # Export CSV summary
        summary_rows = []
        for s_k, s_v in results_dict.items():
            row = {**s_v, **s_v['metrics']}
            del row['metrics']
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV_FILE, index=False)
        
        strat_duration = time.time() - strat_start_time
        overall_elapsed = time.time() - start_grid_time
        completed_count = len(completed_strategies)
        remaining_count = len(STRATEGIES) - completed_count
        avg_strat_time = overall_elapsed / max(completed_count, 1)
        eta_hours = (remaining_count * avg_strat_time) / 3600.0
        
        print(f"✅ [{s_idx+1:02d}/{len(STRATEGIES):02d}] {s_id:8s} ({s_name:20s}) | 12M WAPE: %{metrics['wape_12m']:5.2f} | MAE: ${metrics['mae_12m']:5.2f} | Süre: {strat_duration/60:4.1f} dk | Kalan ETA: {eta_hours:3.1f}h")
        
    total_grid_duration = time.time() - start_grid_time
    print("\n" + "=" * 95)
    print(f"🎉 TÜM GRID SEARCH STRATEJİLERİ BAŞARIYLA TAMAMLANDI!")
    print(f"⏱️ Toplam Çalışma Süresi: {total_grid_duration/3600:.2f} saat")
    print(f"📄 Özet Markdown Raporu: {SUMMARY_MD_FILE}")
    print(f"📊 Özet CSV Raporu: {SUMMARY_CSV_FILE}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPNet Resumable Grid Search Backtest Engine")
    parser.add_argument("--step-days", type=int, default=2, help="Walk-forward test adımı (gün cinsinden, Varsayılan: 2 - Klasik mod için)")
    parser.add_argument("--total-days", type=int, default=365, help="Toplam test gün sayısı (Varsayılan: 365 gün / 12 Ay)")
    parser.add_argument("--seeds", type=int, default=3, help="Ensemble için gün başına seed sayısı (Varsayılan: 3)")
    parser.add_argument("--mode", "-m", type=str, default="fast", choices=["fast", "classic"], help="Backtest modu: 'fast' (Hızlı Sıralı Warm-Start) veya 'classic' (Klasik Her Gün Sıfırdan)")
    parser.add_argument("--reset", action="store_true", help="Önceki checkpoint verilerini sıfırlayıp baştan başlar")
    args = parser.parse_args()
    
    LOG_DIR, CHECKPOINT_FILE, SUMMARY_CSV_FILE, SUMMARY_MD_FILE = setup_log_environment(args.mode)
    
    if args.reset:
        if CHECKPOINT_FILE.exists():
            os.remove(CHECKPOINT_FILE)
            print("🧹 Önceki checkpoint sıfırlandı, grid search sıfırdan başlatılıyor...")
            
    if args.mode == "fast":
        run_fast_sequential_grid(total_days=args.total_days, n_seeds=args.seeds)
    else:
        run_grid_search(step_days=args.step_days, total_days=args.total_days, n_seeds=args.seeds)
