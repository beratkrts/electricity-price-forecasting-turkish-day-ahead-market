import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 PyTorch LSTM + Online Incremental Learning (Son 1 Yıllık Canlı Test ve 3 Senaryo Deneyi)\n",
            "\n",
            "Bu notebook, **IISE PG&E 2024 Dünya Şampiyonu (Team 1 - University of Miami)** metodolojisini temel alarak tasarlanmıştır.\n",
            "\n",
            "### 🎯 Veri Bölünmesi ve 1 Yıllık Canlı Test Metodolojisi:\n",
            "- **Toplam Veri:** 1 Ocak 2023 -> 12 Ağustos 2026 (3.6 Yıl / 31.680 Saat)\n",
            "- **Ön Eğitim Penceresi (Base Pre-training):** 1 Ocak 2023 -> 12 Ağustos 2025 (İlk 2.6 Yıl / 22.900 Saat)\n",
            "- **Canlı Test Penceresi (Online Test):** **13 Ağustos 2025 -> 12 Ağustos 2026 (Son 1 Yıl / 365 Gün / 8.760 Saat)**\n",
            "\n",
            "### 🧪 3 Online Adaptasyon Senaryosu:\n",
            "1. **Senaryo A (Muhafazakar):** Sabit 1 Epoch Online Güncelleme ($lr = 1\\times 10^{-4}$)\n",
            "2. **Senaryo B (Esnek - Önerilen):** Dünün MAE hatasına göre Dinamik 1-3 Epoch Güncelleme ($lr = 1\\times 10^{-4}$)\n",
            "3. **Senaryo C (Aggressive):** Sabit 5 Epoch Online Güncelleme ($lr = 5\\times 10^{-4}$)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "project_root = Path(\"..\").resolve()\n",
            "if str(project_root) not in sys.path:\n",
            "    sys.path.insert(0, str(project_root))\n",
            "\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import time\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
            "plt.rcParams[\"font.sans-serif\"] = [\"DejaVu Sans\", \"Arial\"]\n",
            "%matplotlib inline\n",
            "print(\"✅ Ortam kütüphaneleri yüklendi.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. 🗄️ Veri Hazırlama & Özellik Seçimi (1 Ocak 2023 - 12 Ağustos 2026)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from scripts.predict_daily_pipeline import load_all_historical_data\n",
            "from src.features.feature_engineering import build_robust_features, get_feature_columns\n",
            "\n",
            "# Veritabanından yükle ve öznitelikleri türet\n",
            "df_raw = load_all_historical_data()\n",
            "df_feat = build_robust_features(df_raw)\n",
            "\n",
            "# Özellik sütunları ve hedef değişken\n",
            "feature_cols = get_feature_columns('robust', df_feat)\n",
            "target_col = 'mcp_price_usd'\n",
            "\n",
            "df_model = df_feat.dropna(subset=feature_cols + [target_col]).copy()\n",
            "print(f\"📊 Toplam İşlenebilir Veri Boyutu: {len(df_model)} saat ({df_model.index.min().strftime('%Y-%m-%d')} -> {df_model.index.max().strftime('%Y-%m-%d')})\")\n",
            "print(f\"🔹 Kullanılan Öznitelik Sayısı: {len(feature_cols)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. 🧬 PyTorch Multi-Output LSTM Model Mimarisi Tanımı"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "try:\n",
            "    import torch\n",
            "    import torch.nn as nn\n",
            "    import torch.optim as optim\n",
            "    from torch.utils.data import DataLoader, TensorDataset\n",
            "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "    print(f\"🚀 PyTorch Donanım Hızlandırma Hazır: {device}\")\n",
            "except ImportError:\n",
            "    print(\"⚠️ PyTorch ortam kontrolü yapılıyor...\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class PyTorchLSTMForecaster(nn.Module):\n",
            "    \"\"\"\n",
            "    IISE 2024 Şampiyonu Multi-Output LSTM Sinir Ağı Mimarisi\n",
            "    \"\"\"\n",
            "    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=24, dropout=0.2):\n",
            "        super(PyTorchLSTMForecaster, self).__init__()\n",
            "        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)\n",
            "        self.fc = nn.Sequential(\n",
            "            nn.Linear(hidden_dim, 64),\n",
            "            nn.ReLU(),\n",
            "            nn.Dropout(0.1),\n",
            "            nn.Linear(64, output_dim)\n",
            "        )\n",
            "        \n",
            "    def forward(self, x):\n",
            "        out, _ = self.lstm(x)\n",
            "        last_hidden = out[:, -1, :]\n",
            "        preds = self.fc(last_hidden)\n",
            "        return preds\n",
            "\n",
            "print(\"✅ LSTM Sinir Ağı Sınıfı Tanımlandı.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. 🧪 3 Online Senaryonun Son 1 Yıl (365 Gün) İçin Çalıştırılması"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Pre-training ve Test Dönemlerinin Ayrılması (Son 1 Yıl / 365 Gün Test)\n",
            "test_days = 365  # Son 1 Yıl Canlı Test\n",
            "eval_dates = df_model.index.normalize().unique()[-test_days:]\n",
            "\n",
            "pretrain_end = eval_dates[0] - pd.Timedelta(hours=1)\n",
            "train_pretrain = df_model.loc[:pretrain_end].copy()\n",
            "\n",
            "scaler_X = StandardScaler()\n",
            "scaler_y = StandardScaler()\n",
            "\n",
            "X_train_scaled = scaler_X.fit_transform(train_pretrain[feature_cols])\n",
            "y_train_scaled = scaler_y.fit_transform(train_pretrain[[target_col]])\n",
            "\n",
            "# 24 saatlik sequence veri hazırlığı (Lookback window = 7 gün / 168 saat)\n",
            "seq_len = 168\n",
            "def create_sequences(X, y, seq_len=168):\n",
            "    X_seq, y_seq = [], []\n",
            "    for i in range(len(X) - seq_len - 24 + 1):\n",
            "        X_seq.append(X[i : i + seq_len])\n",
            "        y_seq.append(y[i + seq_len : i + seq_len + 24, 0])\n",
            "    return np.array(X_seq), np.array(y_seq)\n",
            "\n",
            "X_seq_pre, y_seq_pre = create_sequences(X_train_scaled, y_train_scaled, seq_len=seq_len)\n",
            "print(f\"📦 Pre-training Dönemi: {train_pretrain.index.min().strftime('%Y-%m-%d')} -> {train_pretrain.index.max().strftime('%Y-%m-%d')} ({X_seq_pre.shape[0]} Dizilim)\")\n",
            "print(f\"🎯 Son 1 Yıl Canlı Test Dönemi: {eval_dates.min().strftime('%Y-%m-%d')} -> {eval_dates.max().strftime('%Y-%m-%d')} ({len(eval_dates)} Gün / 8.760 Saat)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Simülasyon Fonksiyonu (Son 365 Gün İçin)\n",
            "def run_online_scenario(scenario_name, epochs_mode='fixed_1', base_lr=1e-4):\n",
            "    start_t = time.time()\n",
            "    print(f\"\\n🚀 {scenario_name} Başlatılıyor...\")\n",
            "    \n",
            "    model = PyTorchLSTMForecaster(input_dim=len(feature_cols)).to(device)\n",
            "    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n",
            "    criterion = nn.L1Loss()\n",
            "    \n",
            "    dataset = TensorDataset(torch.tensor(X_seq_pre, dtype=torch.float32), torch.tensor(y_seq_pre, dtype=torch.float32))\n",
            "    loader = DataLoader(dataset, batch_size=64, shuffle=True)\n",
            "    \n",
            "    model.train()\n",
            "    for epoch in range(15):\n",
            "        for bx, by in loader:\n",
            "            bx, by = bx.to(device), by.to(device)\n",
            "            optimizer.zero_grad()\n",
            "            out = model(bx)\n",
            "            loss = criterion(out, by)\n",
            "            loss.backward()\n",
            "            optimizer.step()\n",
            "            \n",
            "    print(f\"  ✅ Base Pre-training Tamamlandı. (Ön Eğitim Loss: {loss.item():.4f})\")\n",
            "    \n",
            "    online_optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)\n",
            "    predictions = []\n",
            "    actuals = []\n",
            "    last_mae = 0.0\n",
            "    \n",
            "    for idx, d in enumerate(eval_dates):\n",
            "        target_start = d\n",
            "        target_end = d + pd.Timedelta(hours=23)\n",
            "        \n",
            "        hist_till_d = df_model.loc[:target_start - pd.Timedelta(hours=1)]\n",
            "        if len(hist_till_d) < seq_len:\n",
            "            continue\n",
            "            \n",
            "        input_seq_raw = hist_till_d[feature_cols].tail(seq_len)\n",
            "        input_seq_scaled = scaler_X.transform(input_seq_raw)\n",
            "        input_tensor = torch.tensor(input_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)\n",
            "        \n",
            "        model.eval()\n",
            "        with torch.no_grad():\n",
            "            pred_scaled = model(input_tensor).cpu().numpy()\n",
            "            pred_usd = scaler_y.inverse_transform(pred_scaled)[0]\n",
            "            \n",
            "        act_usd = df_model.loc[target_start:target_end, target_col].values\n",
            "        if len(act_usd) == 24:\n",
            "            predictions.extend(pred_usd)\n",
            "            actuals.extend(act_usd)\n",
            "            last_mae = np.mean(np.abs(act_usd - pred_usd))\n",
            "            \n",
            "        if epochs_mode == 'fixed_1':\n",
            "            num_epochs = 1\n",
            "        elif epochs_mode == 'dynamic':\n",
            "            num_epochs = 1 if last_mae <= 5.0 else (3 if last_mae <= 10.0 else 5)\n",
            "        elif epochs_mode == 'fixed_5':\n",
            "            num_epochs = 5\n",
            "            \n",
            "        y_seq_new = scaler_y.transform(df_model.loc[target_start:target_end, [target_col]]).T\n",
            "        bx_new = torch.tensor(input_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)\n",
            "        by_new = torch.tensor(y_seq_new, dtype=torch.float32).to(device)\n",
            "        \n",
            "        model.train()\n",
            "        for _ in range(num_epochs):\n",
            "            online_optimizer.zero_grad()\n",
            "            out_new = model(bx_new)\n",
            "            l_new = criterion(out_new, by_new)\n",
            "            l_new.backward()\n",
            "            online_optimizer.step()\n",
            "            \n",
            "        if (idx + 1) % 60 == 0 or (idx + 1) == len(eval_dates):\n",
            "            print(f\"  ⏳ Processed {idx + 1}/{len(eval_dates)} test days ({(idx + 1)/len(eval_dates)*100:.1f}%)...\")\n",
            "            \n",
            "    elapsed = time.time() - start_t\n",
            "    y_act = np.array(actuals)\n",
            "    y_pred = np.array(predictions)\n",
            "    \n",
            "    mae = np.mean(np.abs(y_act - y_pred))\n",
            "    rmse = np.sqrt(np.mean((y_act - y_pred)**2))\n",
            "    wape = (np.sum(np.abs(y_act - y_pred)) / np.sum(y_act)) * 100\n",
            "    \n",
            "    print(f\"  🏁 {scenario_name} Tamamlandı. (Süre: {elapsed:.1f}s | MAE: ${mae:.2f} | WAPE: %{wape:.2f})\")\n",
            "    return {'Senaryo': scenario_name, 'MAE ($)': round(mae, 2), 'RMSE ($)': round(rmse, 2), 'WAPE (%)': round(wape, 2), 'Süre (s)': round(elapsed, 1)}, y_pred, y_act"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. 📊 3 Senaryonun Son 1 Yıl (365 Gün) Performans Tablosu"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "res_a, p_a, y_act = run_online_scenario('Senaryo A: Muhafazakar (Sabit 1 Epoch)', epochs_mode='fixed_1', base_lr=1e-4)\n",
            "res_b, p_b, _     = run_online_scenario('Senaryo B: Esnek Dinamik (1-3 Epoch)', epochs_mode='dynamic', base_lr=1e-4)\n",
            "res_c, p_c, _     = run_online_scenario('Senaryo C: Aggressive (Sabit 5 Epoch)', epochs_mode='fixed_5', base_lr=5e-4)\n",
            "\n",
            "df_results = pd.DataFrame([res_a, res_b, res_c])\n",
            "print(\"\\n=========================================================================\")\n",
            "print(\"🏆 SON 1 YIL (365 GÜN) PYTORCH LSTM ONLINE LEARNING SENARYO KARŞILAŞTIRMASI\")\n",
            "print(\"=========================================================================\")\n",
            "display(df_results)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. 📈 Senaryoların Son 1 Yıldaki Tahmin Performansı Görselleştirmesi"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(15, 6))\n",
            "view_hours = 168  # Son 7 Gün (168 Saat)\n",
            "x = np.arange(view_hours)\n",
            "\n",
            "plt.plot(x, y_act[-view_hours:], 'o-', label='Gerçekleşen PTF ($)', color='black', linewidth=2.5)\n",
            "plt.plot(x, p_a[-view_hours:], '--', label=f\"Senaryo A (WAPE: %{res_a['WAPE (%)']})\", color='#e74c3c', alpha=0.8)\n",
            "plt.plot(x, p_b[-view_hours:], '-', label=f\"Senaryo B (WAPE: %{res_b['WAPE (%)']})\", color='#2ecc71', linewidth=2)\n",
            "plt.plot(x, p_c[-view_hours:], ':', label=f\"Senaryo C (WAPE: %{res_c['WAPE (%)']})\", color='#9b59b6', alpha=0.8)\n",
            "\n",
            "plt.title('Son 1 Yıl Canlı Test: LSTM Online Learning Senaryoları Tahmin Kıyaslaması', fontsize=14, fontweight='bold')\n",
            "plt.xlabel('Saat')\n",
            "plt.ylabel('Fiyat ($/MWh)')\n",
            "plt.legend()\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("/Users/beratkaratasoglu/etkb_intern_project/enerji_fiyat_tahmini/eda/lstm_online_learning_experiment.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("✅ Updated Notebook with full 1-year test: eda/lstm_online_learning_experiment.ipynb")
