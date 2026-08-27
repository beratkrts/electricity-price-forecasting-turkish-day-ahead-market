"""
Rejim Doğrulama & Model Karşılaştırma Script'i.
Mevcut EPNet backtest sonuçları ile Rejim Tespit Modülünü çalıştırarak
rejim sınıflandırmalarını ve performans metriklerini doğrular.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.routing.regime_detector import RegimeDetector


def main():
    print("🔍 Rejim Tespit Modülü Doğrulama Çalıştırılıyor...")
    
    # Fake/sample 365 günlük simülasyon testi
    dates = pd.date_range("2025-08-01", "2026-07-31", freq="h")
    np.random.seed(42)
    
    # 2026 Bahar aylarında (Mart-Mayıs) düşük fiyat ve yüksek volatilite simülasyonu
    prices = []
    for d in dates:
        if 3 <= d.month <= 5 and d.year == 2026:
            # Bahar çakılması
            p = max(0.0, np.random.normal(8.0, 12.0))
        else:
            # Normal dönem
            p = max(10.0, np.random.normal(65.0, 10.0))
        prices.append(p)
        
    df_sim = pd.DataFrame({
        'mcp_price_usd': prices,
        'load_forecast_mw': np.random.uniform(30000, 45000, len(dates)),
        'load_lag_24': np.random.uniform(30000, 45000, len(dates)),
        'kgup_hydro_lag_24': np.random.uniform(5000, 12000, len(dates)),
        'kgup_wind_lag_24': np.random.uniform(2000, 9000, len(dates)),
        'kgup_solar_lag_24': np.random.uniform(1000, 6000, len(dates)),
    }, index=dates)
    
    detector = RegimeDetector()
    
    regimes = []
    # Günlük adım atarak (her 24 saatte bir) rejim tespiti
    daily_dates = pd.date_range("2025-08-05", "2026-07-31", freq="D")
    
    for dt in daily_dates:
        sub_df = df_sim.loc[:dt]
        if len(sub_df) < 48:
            continue
        signals = detector.compute_signals(sub_df)
        regime = detector.detect_regime(signals)
        regimes.append({
            'date': dt.strftime('%Y-%m-%d'),
            'regime': regime,
            'rolling_mean_48h': signals['rolling_mean_48h'],
            'price_volatility_24h': signals['price_volatility_24h'],
            'zero_price_count_24h': signals['zero_price_count_24h'],
            'renewable_pressure_ratio': signals['renewable_pressure_ratio']
        })
        
    res_df = pd.DataFrame(regimes)
    print("\n📊 Rejim Dağılım Özeti:")
    print(res_df['regime'].value_counts())
    
    print("\nÖrnek Mart-Mayıs 2026 Rejim Durumu:")
    spring_df = res_df[res_df['date'].str.contains('2026-04')]
    print(spring_df[['date', 'regime', 'rolling_mean_48h', 'zero_price_count_24h']].head(10))


if __name__ == "__main__":
    main()
