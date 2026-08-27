"""EPNet (CNN-LSTM Hybrid PyTorch Paper Model) Standalone Backtest Script.

Allows configuring backtest duration (--days / -d) and outputs live progress logging
with percent complete, daily MAE/WAPE metrics, and ETA timer.
"""

import argparse
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
from src.features.feature_engineering import build_full_features, get_feature_columns



def set_seed(seed=42):
    """Sets deterministic seed across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- EPNet PyTorch Architecture (Optimized CNN-LSTM Hybrid) ---
class EPNet(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=32, lstm_hidden_dim=32, dropout=0.1):
        super(EPNet, self).__init__()
        # 1D-CNN Feature Extraction (Preserve 24 timesteps for full hourly resolution)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=cnn_out_channels)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # LSTM Sequential Forecasting
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, batch_first=True)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(lstm_hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim] -> permute to [batch_size, input_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.dropout(self.relu2(self.bn2(self.conv2(x))))
        # permute back to [batch_size, seq_len, cnn_out_channels] for LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.relu3(lstm_out[:, -1, :])
        out = self.fc(out)
        return out.squeeze(-1)


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
    if 'hydro_water_energy_mwh' in df_raw.columns:
        df_raw['hydro_water_energy_mwh'] = df_raw['hydro_water_energy_mwh'].ffill().bfill()

    return df_raw


def calculate_safe_mape(y_true, y_pred):
    safe_denom = np.maximum(y_true, 1.0)
    return np.mean(np.abs((y_true - y_pred) / safe_denom)) * 100


def calculate_wape(y_true, y_pred):
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100


def train_predict_epnet_paper(train_df, test_df, features, target, seq_len=24, epochs=25, batch_size=64, lr=0.001, n_seeds=3):
    """Trains EPNet using Seed Averaging Ensemble, Gradient Clipping, and Cosine Annealing."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(train_df[features].values)
    y_tr_log = np.log1p(np.maximum(train_df[target].values, 0.0))
    
    X_te_s = scaler.transform(test_df[features].values)
    
    # Create sequences
    X_tr_seq, y_tr_seq = [], []
    for i in range(len(X_tr_s) - seq_len + 1):
        X_tr_seq.append(X_tr_s[i:i+seq_len])
        y_tr_seq.append(y_tr_log[i+seq_len-1])
        
    X_tr_t = torch.tensor(np.array(X_tr_seq), dtype=torch.float32)
    y_tr_t = torch.tensor(np.array(y_tr_seq), dtype=torch.float32)
    
    # Test sequence (pad with last train sequence)
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
        
        model = EPNet(input_dim=len(features)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        
    final_preds = np.mean(all_seed_preds, axis=0)
    return final_preds


def run_epnet_fast_sequential_backtest(df_data, feature_list, target_name, max_days=365, initial_epochs=25, update_epochs=1, n_seeds=3):
    """Fast Sequential Incremental Backtest Engine.
    
    1. Pre-trains ensemble models once for initial_epochs on ALL historical data up to backtest start.
    2. Sequentially evaluates each day chronologically (day 1 to day N).
    3. Performs 1-epoch warm-start weight update after each daily prediction using low LR (0.0002).
    """
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_ts = df_data.index.max()
    min_ts = df_data.index.min()
    backtest_start = max_ts - pd.Timedelta(days=max_days)
    
    # Use ALL historical data in database up to backtest start
    init_tr_start = min_ts
    init_tr_end = backtest_start - pd.Timedelta(hours=1)
    
    init_tr_df = df_data.loc[init_tr_start:init_tr_end]
    
    print("\n" + "=" * 95)
    print("⚡ EPNET HIZLI SIRA-ARDIŞIK (FAST SEQUENTIAL) BACKTEST MOTORU BAŞLATILIYOR")
    print(f"🎯 Toplam Backtest Süresi: {max_days} Gün (Kronolojik Sıralı) | Seed Ensemble: {n_seeds} Model")
    print(f"⚙️ İlk Eğitim Veri Kapsamı: {init_tr_start.strftime('%Y-%m-%d')} -> {init_tr_end.strftime('%Y-%m-%d')} ({len(init_tr_df):,} Saat)")
    print(f"⚙️ Parametreler: İlk Eğitim={initial_epochs} Epoch | Günlük Güncelleme={update_epochs} Epoch (LR=0.0002)")
    print("=" * 95)
    
    print(f"\n🧠 [1/2] Başlangıç Modeli Veritabanındaki Tüm Geçmişle Eğitiliyor...")
    
    scaler = StandardScaler()
    X_init_tr_s = scaler.fit_transform(init_tr_df[feature_list].values)
    y_init_tr_log = np.log1p(np.maximum(init_tr_df[target_name].values, 0.0))
    
    seq_len = 24
    X_init_seq, y_init_seq = [], []
    for i in range(len(X_init_tr_s) - seq_len + 1):
        X_init_seq.append(X_init_tr_s[i:i+seq_len])
        y_init_seq.append(y_init_tr_log[i+seq_len-1])
        
    X_init_t = torch.tensor(np.array(X_init_seq), dtype=torch.float32)
    y_init_t = torch.tensor(np.array(y_init_seq), dtype=torch.float32)
    
    seeds = [42, 123, 999, 2024, 777][:n_seeds]
    ensemble_models = []
    optimizers = []
    
    # Pre-train ensemble
    for seed in seeds:
        set_seed(seed)
        dataset = TensorDataset(X_init_t, y_init_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        model = EPNet(input_dim=len(feature_list)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=initial_epochs)
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
        
    print(f"✅ Başlangıç Eğitimi Tamamlandı ({time.time() - start_time:.1f} saniye). Günlük Tahmin Döngüsüne Geçiliyor...\n")
    
    # Sequential Daily Loop
    daily_results = []
    eval_start_time = time.time()
    
    last_seq_buffer = X_init_tr_s[-(seq_len-1):]
    
    for day_idx in range(max_days):
        day_start = backtest_start + pd.Timedelta(days=day_idx)
        day_end = day_start + pd.Timedelta(hours=23)
        
        day_df = df_data.loc[day_start:day_end]
        if len(day_df) < 24:
            continue
            
        X_day_s = scaler.transform(day_df[feature_list].values)
        y_day_true = day_df[target_name].values
        
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
        
        daily_results.append({
            'day_idx': day_idx,
            'tarih': day_start.strftime('%Y-%m-%d'),
            'mae': mae,
            'wape': wape,
            'mape': mape
        })
        
        y_day_log = np.log1p(np.maximum(y_day_true, 0.0))
        y_day_seq_target = []
        for i in range(len(X_day_s)):
            y_day_seq_target.append(y_day_log[i])
            
        X_up_t = torch.tensor(np.array(X_day_seq), dtype=torch.float32).to(device)
        y_up_t = torch.tensor(np.array(y_day_seq_target), dtype=torch.float32).to(device)
        
        criterion = nn.MSELoss()
        for m_idx, (model, opt) in enumerate(zip(ensemble_models, optimizers)):
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
        
        elapsed_sec = time.time() - eval_start_time
        avg_sec = elapsed_sec / (day_idx + 1)
        eta_sec = (max_days - day_idx - 1) * avg_sec
        pct = ((day_idx + 1) / max_days) * 100
        
        if (day_idx + 1) % 5 == 0 or (day_idx + 1) == max_days:
            print(f"⏳ Hızlı Sıralı [{day_idx+1:03d}/{max_days:03d}] (%{pct:5.1f}) | "
                  f"Tarih: {day_start.strftime('%Y-%m-%d')} | "
                  f"MAE: ${mae:5.2f}/MWh | WAPE: %{wape:5.2f} | MAPE: %{mape:5.2f} | "
                  f"Geçen: {elapsed_sec:4.0f}s | Kalan ETA: {eta_sec:4.0f}s")
                  
    res_df = pd.DataFrame(daily_results)
    
    print("\n" + "=" * 95)
    print("🏆 EPNET HIZLI SIRA-ARDIŞIK NİHAİ BACKTEST SONUÇLARI ÖZETİ")
    print("=" * 95)
    print(f"📊 Toplam Değerlendirilen Gün Sayısı: {len(res_df)} gün")
    print(f"💵 Ortalama MAE : ${res_df['mae'].mean():.2f}/MWh")
    print(f"🎯 Ortalama WAPE: %{res_df['wape'].mean():.2f}")
    print(f"📈 Ortalama MAPE: %{res_df['mape'].mean():.2f}")
    print(f"⏱️ Toplam Süre  : {(time.time() - start_time)/60:.2f} dakika")
    print("=" * 95)
    
    return res_df


def run_epnet_walk_forward(df_data, feature_list, target_name, max_days=7, epochs=25, n_seeds=3):
    """Klasik Walk-Forward Backtest (Her gün için sıfırdan tüm geçmişle sıfırdan eğitir)."""
    daily_results = []
    start_time = time.time()
    min_ts = df_data.index.min()
    
    print("\n" + "=" * 95)
    print(f"🚀 EPNET KLASİK HER-GÜN-SIFIRDAN WALK-FORWARD BACKTEST BAŞLATILIYOR")
    print(f"🎯 Hedef Gün Sayısı: {max_days} gün | Seed Ensemble: {n_seeds} Model / Gün")
    print(f"⚙️ Veri Kapsamı: Veritabanındaki En Eski Kayıttan ({min_ts.strftime('%Y-%m-%d')}) İtibaren Eğitilir")
    print("=" * 95)
    
    for day_idx in range(max_days):
        test_end = df_data.index.max() - pd.Timedelta(days=day_idx)
        test_start = test_end - pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)
        train_start = min_ts  # Veritabanındaki en eski kayıttan itibaren eğit
        
        tr_df = df_data.loc[train_start:train_end]
        te_df = df_data.loc[test_start:test_end]
        
        if len(tr_df) < 1000 or len(te_df) < 12:
            continue
            
        preds = train_predict_epnet_paper(tr_df, te_df, feature_list, target_name, epochs=epochs, n_seeds=n_seeds)
        
        y_true = te_df[target_name].values
        mae = mean_absolute_error(y_true, preds)
        mape = calculate_safe_mape(y_true, preds)
        wape = calculate_wape(y_true, preds)
        
        daily_results.append({
            'day_offset': day_idx,
            'tarih': test_start.strftime('%Y-%m-%d'),
            'mae': mae,
            'mape': mape,
            'wape': wape
        })
        
        elapsed_sec = time.time() - start_time
        avg_sec_per_day = elapsed_sec / (day_idx + 1)
        remaining_days = max_days - (day_idx + 1)
        eta_sec = remaining_days * avg_sec_per_day
        progress_pct = ((day_idx + 1) / max_days) * 100
        
        print(f"⏳ Klasik [{day_idx+1:02d}/{max_days:02d}] ({progress_pct:5.1f}%) | "
              f"Tarih: {test_start.strftime('%Y-%m-%d')} | "
              f"MAE: ${mae:5.2f}/MWh | WAPE: %{wape:5.2f} | MAPE: %{mape:5.2f} | "
              f"Geçen: {elapsed_sec:4.0f}s | Kalan ETA: {eta_sec:4.0f}s")
        
    res_df = pd.DataFrame(daily_results)
    
    print("\n" + "=" * 95)
    print("🏆 EPNET KLASİK NİHAİ BACKTEST SONUÇLARI ÖZETİ")
    print("=" * 95)
    print(f"📊 Toplam Değerlendirilen Gün Sayısı: {len(res_df)} gün")
    print(f"💵 Ortalama MAE : ${res_df['mae'].mean():.2f}/MWh")
    print(f"🎯 Ortalama WAPE: %{res_df['wape'].mean():.2f}")
    print(f"📈 Ortalama MAPE: %{res_df['mape'].mean():.2f}")
    print("=" * 95)
    
    return res_df


def main():
    parser = argparse.ArgumentParser(description="EPNet (CNN-LSTM Hybrid) EPİAŞ PTF Backtest Script")
    parser.add_argument("--days", "-d", type=int, default=365, help="Tahmin edilecek backtest gün sayısı (Örn: 7, 30, 365. Varsayılan: 365 gün)")
    parser.add_argument("--epochs", "-e", type=int, default=25, help="İlk eğitim epoch sayısı (Varsayılan: 25)")
    parser.add_argument("--seeds", "-s", type=int, default=3, help="Ensemble için seed sayısı (Varsayılan: 3)")
    parser.add_argument("--mode", "-m", type=str, default="fast", choices=["fast", "classic"], help="Backtest modu: 'fast' (Hızlı Sıralı Warm-Start) veya 'classic' (Klasik Her Gün Sıfırdan)")
    args = parser.parse_args()
    
    print("📦 EPİAŞ Veritabanından Veriler Yükleniyor...")
    df_raw = load_master_dataset()
    df_feat = build_full_features(df_raw)
    df_model = df_feat.dropna().copy()
    
    feature_cols = get_feature_columns('full', df_model)
    target_col = 'mcp_price_usd'
    
    if args.mode == "fast":
        run_epnet_fast_sequential_backtest(df_model, feature_cols, target_col, max_days=args.days, initial_epochs=args.epochs, n_seeds=args.seeds)
    else:
        run_epnet_walk_forward(df_model, feature_cols, target_col, max_days=args.days, epochs=args.epochs, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
