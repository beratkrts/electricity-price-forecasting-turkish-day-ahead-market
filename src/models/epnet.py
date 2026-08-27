"""
[EXPERIMENTAL - Canlı pipeline'da kullanılmıyor]
EPNet (Energy Price Network) - CNN+LSTM tabanlı derin öğrenme modeli.
Walk-forward backtestlerde LightGBM'e göre belirgin şekilde daha kötü performans
göstermiş (12 ay WAPE: EPNet %39.67 vs LightGBM %26.88). Canlıya alınmamıştır.
"""

import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EPNet_LSTM(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=32, lstm_hidden_dim=64, lstm_layers=1, dropout=0.2):
        super(EPNet_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) -> transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2) # (batch_size, seq_len, cnn_out_channels)
        
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)

class EPNet_GRU(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=32, gru_hidden_dim=64, gru_layers=1, dropout=0.2):
        super(EPNet_GRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        
        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=gru_hidden_dim, 
                           num_layers=gru_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out.squeeze(-1)

class EPNet_BiLSTM(nn.Module):
    def __init__(self, input_dim, cnn_out_channels=32, lstm_hidden_dim=64, lstm_layers=1, dropout=0.2):
        super(EPNet_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        
        self.bilstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, 
                              num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.bilstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)
