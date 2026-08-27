"""
[EXPERIMENTAL - Canlı pipeline'da kullanılmıyor]
Piyasa rejimi tespit modülü. Fiyat verilerinden düşük/yüksek fiyat rejimlerini tespit eder.
Model router ile birlikte kullanılmak üzere tasarlanmıştır ancak canlıda aktif değildir.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union


class RegimeDetector:
    """
    Piyasa koşullarını (NORMAL, CRASH, VOLATILE) tespit eden rejim belirleyici sınıf.
    """
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            'crash_mean_price': 15.0,
            'volatile_mean_price': 30.0,
            'crash_volatility': 1.0,
            'volatile_volatility': 0.5,
            'crash_zero_count': 4,
            'volatile_zero_count': 1,
            'crash_renewable_ratio': 0.55,
            'volatile_renewable_ratio': 0.40,
        }

    def compute_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Girdi DataFrame'inden rejim tespit sinyallerini hesaplar.
        df index'inin DatetimeIndex ve sıralı olduğu varsayılır.
        """
        # shifted price (son 24h/48h sinyalleri)
        if 'mcp_price_usd' not in df.columns:
            raise ValueError("mcp_price_usd column is required to compute regime signals.")
            
        mcp_usd = df['mcp_price_usd']
        
        # Son 48 saatlik ortalama (leakage önlemek için shift(24) kullanılır)
        shifted_price = mcp_usd.shift(24)
        rolling_mean_48h = shifted_price.rolling(window=48, min_periods=24).mean().iloc[-1]
        rolling_mean_24h = shifted_price.rolling(window=24, min_periods=12).mean().iloc[-1]
        rolling_std_24h = shifted_price.rolling(window=24, min_periods=12).std().iloc[-1]
        
        mean_safe = rolling_mean_24h if (not pd.isna(rolling_mean_24h) and rolling_mean_24h > 0) else 1.0
        price_volatility_24h = (rolling_std_24h / mean_safe) if not pd.isna(rolling_std_24h) else 0.0
        
        # Sıfır/yakın sıfır fiyat sayısı (<= $0.5)
        zero_price_count_24h = int((shifted_price.tail(24) <= 0.5).sum())
        
        # Yenilenebilir & Hidro baskı oranları
        renewable_pressure_ratio = 0.0
        hydro_pressure_ratio = 0.0
        
        if 'load_forecast_mw' in df.columns and 'load_lag_24' in df.columns:
            load_val = df['load_lag_24'].iloc[-1]
            if pd.isna(load_val) or load_val <= 0:
                load_val = df['load_forecast_mw'].shift(24).iloc[-1]
                
            if not pd.isna(load_val) and load_val > 0:
                hydro_val = df['kgup_hydro_lag_24'].iloc[-1] if 'kgup_hydro_lag_24' in df.columns else 0
                wind_val = df['kgup_wind_lag_24'].iloc[-1] if 'kgup_wind_lag_24' in df.columns else 0
                solar_val = df['kgup_solar_lag_24'].iloc[-1] if 'kgup_solar_lag_24' in df.columns else 0
                
                hydro_val = 0 if pd.isna(hydro_val) else hydro_val
                wind_val = 0 if pd.isna(wind_val) else wind_val
                solar_val = 0 if pd.isna(solar_val) else solar_val
                
                hydro_pressure_ratio = float(hydro_val / load_val)
                renewable_pressure_ratio = float((wind_val + solar_val + hydro_val) / load_val)

        return {
            'rolling_mean_48h': float(rolling_mean_48h) if not pd.isna(rolling_mean_48h) else 50.0,
            'price_volatility_24h': float(price_volatility_24h),
            'zero_price_count_24h': zero_price_count_24h,
            'renewable_pressure_ratio': float(renewable_pressure_ratio),
            'hydro_pressure_ratio': float(hydro_pressure_ratio)
        }

    def detect_regime(self, signals: Dict[str, float]) -> str:
        """
        Hesaplanan sinyallere göre rejim sınıflandırması yapar: 'CRASH', 'VOLATILE', veya 'NORMAL'.
        """
        crash_score = 0
        volatile_score = 0

        # Fiyat seviyesi
        mean_p = signals.get('rolling_mean_48h', 50.0)
        if mean_p < self.thresholds['crash_mean_price']:
            crash_score += 3
        elif mean_p < self.thresholds['volatile_mean_price']:
            volatile_score += 2

        # Volatilite
        vol = signals.get('price_volatility_24h', 0.0)
        if vol > self.thresholds['crash_volatility']:
            crash_score += 2
        elif vol > self.thresholds['volatile_volatility']:
            volatile_score += 1

        # Sıfır fiyat saati sayısı
        zero_cnt = signals.get('zero_price_count_24h', 0)
        if zero_cnt >= self.thresholds['crash_zero_count']:
            crash_score += 3
        elif zero_cnt >= self.thresholds['volatile_zero_count']:
            volatile_score += 2

        # Yenilenebilir baskısı
        ren_ratio = signals.get('renewable_pressure_ratio', 0.0)
        if ren_ratio > self.thresholds['crash_renewable_ratio']:
            crash_score += 1
        elif ren_ratio > self.thresholds['volatile_renewable_ratio']:
            volatile_score += 1

        # Sınıflandırma Mantığı
        if crash_score >= 3:
            return 'CRASH'
        elif volatile_score >= 2 or crash_score >= 2:
            return 'VOLATILE'
        else:
            return 'NORMAL'
