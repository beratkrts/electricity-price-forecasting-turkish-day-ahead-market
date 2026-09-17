import pandas as pd
import numpy as np

def add_holiday_features(df):
    """
    Ekler:
    - is_holiday: 0 veya 1
    - days_to_nearest_holiday: int
    """
    df_feat = df.copy()
    
    # Sabit resmi tatiller (Ay, Gün)
    fixed_holidays = [
        (1, 1),   # Yılbaşı
        (4, 23),  # Ulusal Egemenlik ve Çocuk Bayramı
        (5, 1),   # Emek ve Dayanışma Günü
        (5, 19),  # Atatürk'ü Anma, Gençlik ve Spor Bayramı
        (7, 15),  # Demokrasi ve Milli Birlik Günü
        (8, 30),  # Zafer Bayramı
        (10, 29)  # Cumhuriyet Bayramı
    ]
    
    # Hareketli dini bayramlar (Ramazan ve Kurban Bayramı - 2024-2026 için)
    # Arefe günleri ve bayram günleri yaklaşık (veya net) tarihleri
    movable_holidays = [
        # 2024
        "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12", # Ramazan
        "2024-06-15", "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19", # Kurban
        # 2025
        "2025-03-29", "2025-03-30", "2025-03-31", "2025-04-01", # Ramazan
        "2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09", # Kurban
        # 2026
        "2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22", # Ramazan
        "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30"  # Kurban
    ]
    
    # Index üzerinde çalışıyorsak, DatetimeIndex yapıp date'e çevirelim
    idx_dt = pd.to_datetime(df_feat.index, utc=True).tz_convert('Europe/Istanbul')
    dates = pd.Series(idx_dt.date)
    
    is_fixed = dates.apply(lambda x: (x.month, x.day) in fixed_holidays)
    is_movable = dates.isin([pd.to_datetime(d).date() for d in movable_holidays])
    
    df_feat['is_holiday'] = (is_fixed | is_movable).astype(int).values
    
    # Calculate days to nearest holiday
    # Geçmiş ve gelecek tüm tatilleri datetime listesi olarak alalım
    all_holidays_dt = []
    
    # 2023-2027 arası sabit tatilleri listeye ekle
    for year in range(2023, 2028):
        for month, day in fixed_holidays:
            all_holidays_dt.append(pd.Timestamp(year=year, month=month, day=day))
            
    for md in movable_holidays:
        all_holidays_dt.append(pd.Timestamp(md))
        
    all_holidays_dt = sorted(pd.Series(all_holidays_dt).dt.date.unique())
    
    # Calculate distance for each day
    # Optimizasyon için unique günler üzerinden mesafe hesaplayalım
    unique_dates = dates.unique()
    dist_map = {}
    
    for d in unique_dates:
        # En yakın tatili bul (geçmiş veya gelecek)
        distances = [abs((d - hd).days) for hd in all_holidays_dt]
        dist_map[d] = min(distances) if distances else 0
        
    df_feat['days_to_nearest_holiday'] = dates.map(dist_map).values
    
    return df_feat
