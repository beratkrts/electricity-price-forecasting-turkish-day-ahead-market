import pandas as pd
import numpy as np
from src.features.holidays import add_holiday_features

def build_base_features(df):
    """
    Takvim + PTF lagları + rolling istatistikler + tatil özellikleri.
    """
    df_feat = df.copy()
    
    # Çapraz platform ve tz-aware uyumluluğu için DatetimeIndex garantisi
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat.index = pd.to_datetime(df_feat.index, utc=True).tz_convert('Europe/Istanbul')
    
    # Calendar features
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    if 'quarter' not in df_feat.columns:
        df_feat['quarter'] = df_feat.index.quarter
    if 'dayofyear' not in df_feat.columns:
        df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['is_weekend'] = (df_feat.index.dayofweek >= 5).astype(int)
    df_feat['is_peak_hour'] = df_feat['hour'].isin([17, 18, 19, 20, 21]).astype(int)
    
    # Holiday features
    df_feat = add_holiday_features(df_feat)
    
    # MCP (PTF) lags
    for lag in [24, 48, 168]:
        df_feat[f'mcp_usd_lag_{lag}'] = df_feat['mcp_price_usd'].shift(lag)
        
    # Rolling stats
    df_feat['mcp_usd_roll_mean_24h'] = df_feat['mcp_price_usd'].shift(24).rolling(window=24).mean()
    df_feat['mcp_usd_roll_std_24h'] = df_feat['mcp_price_usd'].shift(24).rolling(window=24).std()
    df_feat['mcp_usd_roll_mean_7d'] = df_feat['mcp_price_usd'].shift(24).rolling(window=168).mean()
    
    return df_feat


def build_core_features(df):
    """
    base + yük/KGÜP lagları + arz-talep farkı + yenilenebilir oranları + baraj doluluk.
    """
    df_feat = build_base_features(df)
    
    # Lags for Load and KGUP
    for lag in [24, 48, 168]:
        if 'load_forecast_mw' in df_feat.columns:
            df_feat[f'load_lag_{lag}'] = df_feat['load_forecast_mw'].shift(lag)
        if 'kgup_total_mw' in df_feat.columns:
            df_feat[f'kgup_lag_{lag}'] = df_feat['kgup_total_mw'].shift(lag)
        if 'kgup_wind_mw' in df_feat.columns:
            df_feat[f'kgup_wind_lag_{lag}'] = df_feat['kgup_wind_mw'].shift(lag)
        if 'kgup_solar_mw' in df_feat.columns:
            df_feat[f'kgup_solar_lag_{lag}'] = df_feat['kgup_solar_mw'].shift(lag)
        if 'kgup_hydro_mw' in df_feat.columns:
            df_feat[f'kgup_hydro_lag_{lag}'] = df_feat['kgup_hydro_mw'].shift(lag)
        if 'kgup_gas_mw' in df_feat.columns:
            df_feat[f'kgup_gas_lag_{lag}'] = df_feat['kgup_gas_mw'].shift(lag)
            
    # Supply-Demand Gap
    if 'load_lag_24' in df_feat.columns and 'kgup_lag_24' in df_feat.columns:
        df_feat['supply_demand_gap_mw'] = df_feat['load_lag_24'] - df_feat['kgup_lag_24']
        
        total_kgup_safe = df_feat['kgup_lag_24'].replace(0, np.nan)
        if 'kgup_wind_lag_24' in df_feat.columns:
            df_feat['renewable_ratio'] = (df_feat['kgup_wind_lag_24'] + df_feat['kgup_solar_lag_24'] + df_feat['kgup_hydro_lag_24']) / total_kgup_safe
            df_feat['renewable_ratio'] = df_feat['renewable_ratio'].fillna(0)
            df_feat['solar_wind_ratio'] = (df_feat['kgup_wind_lag_24'] + df_feat['kgup_solar_lag_24']) / total_kgup_safe
            df_feat['solar_wind_ratio'] = df_feat['solar_wind_ratio'].fillna(0)
            
    # Baraj doluluk (hydro_water_energy_mwh)
    if 'hydro_water_energy_mwh' in df_feat.columns:
        df_feat['hydro_water_energy_lag_24'] = df_feat['hydro_water_energy_mwh'].shift(24)
        
    return df_feat


def build_full_features(df):
    """
    core + gerçekleşen üretim/tüketim lagları.
    """
    df_feat = build_core_features(df)
    
    # Lags for actual generation and consumption
    for lag in [48, 168]:
        if 'actual_gen_total_mw' in df_feat.columns:
            df_feat[f'actual_gen_lag_{lag}'] = df_feat['actual_gen_total_mw'].shift(lag)
        if 'actual_cons_mw' in df_feat.columns:
            df_feat[f'actual_cons_lag_{lag}'] = df_feat['actual_cons_mw'].shift(lag)
            
    return df_feat


def build_robust_features(df):
    """
    full + pressure ratios + cyclic features + fiyat rejimi flagleri.
    """
    df_feat = build_full_features(df)
    
    # Cyclic Seasonality
    df_feat['sin_hour'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['cos_hour'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['sin_dow'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7.0)
    df_feat['cos_dow'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7.0)
    df_feat['sin_month'] = np.sin(2 * np.pi * df_feat['month'] / 12.0)
    df_feat['cos_month'] = np.cos(2 * np.pi * df_feat['month'] / 12.0)
    df_feat['sin_doy'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365.25)
    df_feat['cos_doy'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365.25)
    
    # Zero-Price Supply Pressure Ratios
    if 'load_lag_24' in df_feat.columns:
        load_safe = df_feat['load_lag_24'].replace(0, np.nan)
        df_feat['hydro_pressure_ratio'] = (df_feat['kgup_hydro_lag_24'] / load_safe).fillna(0)
        df_feat['renewable_pressure_ratio'] = ((df_feat['kgup_wind_lag_24'] + df_feat['kgup_solar_lag_24'] + df_feat['kgup_hydro_lag_24']) / load_safe).fillna(0)
        df_feat['solar_peak_pressure_ratio'] = ((df_feat['kgup_solar_lag_24']) / load_safe).fillna(0)

    # Fiyat Rejimi Flagleri
    # is_low_price_regime: son 24 saatteki ortalama fiyat < $10 ise 1, değilse 0
    # is_zero_price_hour: son 24 saatte sıfır fiyat sayısı (0-24 arası int)
    # price_volatility_24h: son 24 saatlik fiyat standart sapması / ortalama
    
    # Calculate these using a rolling window over shifted prices to prevent leakage
    shifted_price = df_feat['mcp_price_usd'].shift(24)
    
    rolling_mean_24 = shifted_price.rolling(window=24).mean()
    rolling_std_24 = shifted_price.rolling(window=24).std()
    
    df_feat['is_low_price_regime'] = (rolling_mean_24 < 10).astype(int)
    df_feat['is_zero_price_hour'] = (shifted_price <= 0).rolling(window=24).sum().fillna(0).astype(int)
    
    # avoid division by zero
    mean_safe = rolling_mean_24.replace(0, np.nan)
    df_feat['price_volatility_24h'] = (rolling_std_24 / mean_safe).fillna(0)
    
    # 48h Lags for Exogenous Variables (SMF, Temperature, Brent Oil, Natural Gas GRF) to prevent leakage
    if 'smp_price_try' in df_feat.columns and 'usd_try' in df_feat.columns:
        df_feat['smp_usd_lag_48'] = (df_feat['smp_price_try'] / df_feat['usd_try']).shift(48)
    if 'temperature_c' in df_feat.columns:
        df_feat['temperature_lag_48'] = df_feat['temperature_c'].shift(48)
    if 'brent_oil_usd' in df_feat.columns:
        df_feat['brent_oil_lag_48'] = df_feat['brent_oil_usd'].shift(48)
    if 'natural_gas_grf_try' in df_feat.columns and 'usd_try' in df_feat.columns:
        df_feat['natural_gas_grf_lag_48'] = (df_feat['natural_gas_grf_try'] / df_feat['usd_try']).shift(48)

    # 🌡️ Target-Day Temperature Forecast Features (Lag 0) & Degree-Hour Cooling/Heating Loads
    # For future forecast rows, use temperature_forecast_c; for historical backfill rows, fill missing values with temperature_c
    if 'temperature_forecast_c' in df_feat.columns and 'temperature_c' in df_feat.columns:
        temp_target = df_feat['temperature_forecast_c'].fillna(df_feat['temperature_c'])
    elif 'temperature_forecast_c' in df_feat.columns:
        temp_target = df_feat['temperature_forecast_c']
    elif 'temperature_c' in df_feat.columns:
        temp_target = df_feat['temperature_c']
    else:
        temp_target = None

    if temp_target is not None:
        df_feat['temp_forecast_lag0'] = temp_target.astype(float)
        df_feat['cdh_cooling_load'] = np.maximum(df_feat['temp_forecast_lag0'] - 18.0, 0.0).astype(float)
        df_feat['hdh_heating_load'] = np.maximum(18.0 - df_feat['temp_forecast_lag0'], 0.0).astype(float)
        if 'temperature_lag_48' in df_feat.columns:
            df_feat['temp_diff_from_yesterday'] = (df_feat['temp_forecast_lag0'] - df_feat['temperature_lag_48'].astype(float)).astype(float)

    # 🔮 Pre-Forecast Features Fallback for Historical Rows to Prevent Training Sample Truncation
    if 'predicted_load_lag0' in df_feat.columns and 'load_forecast_mw' in df_feat.columns:
        df_feat['predicted_load_lag0'] = df_feat['predicted_load_lag0'].fillna(df_feat['load_forecast_mw'])
    elif 'load_forecast_mw' in df_feat.columns:
        df_feat['predicted_load_lag0'] = df_feat['load_forecast_mw']

    if 'predicted_solar_lag0' in df_feat.columns and 'kgup_solar_mw' in df_feat.columns:
        df_feat['predicted_solar_lag0'] = df_feat['predicted_solar_lag0'].fillna(df_feat['kgup_solar_mw'])
    elif 'kgup_solar_mw' in df_feat.columns:
        df_feat['predicted_solar_lag0'] = df_feat['kgup_solar_mw']

    if 'predicted_wind_lag0' in df_feat.columns and 'kgup_wind_mw' in df_feat.columns:
        df_feat['predicted_wind_lag0'] = df_feat['predicted_wind_lag0'].fillna(df_feat['kgup_wind_mw'])
    elif 'kgup_wind_mw' in df_feat.columns:
        df_feat['predicted_wind_lag0'] = df_feat['kgup_wind_mw']

    # ☀️ Zero-Price & Pressure Ratio Lag0 Features (Net Load, Ramps, Ratios)
    if 'predicted_load_lag0' in df_feat.columns and 'predicted_solar_lag0' in df_feat.columns and 'predicted_wind_lag0' in df_feat.columns:
        load_lag0_safe = df_feat['predicted_load_lag0'].replace(0, np.nan)
        hydro_lag24 = df_feat['kgup_hydro_lag_24'].fillna(0) if 'kgup_hydro_lag_24' in df_feat.columns else 0.0
        
        df_feat['net_load_lag0'] = (df_feat['predicted_load_lag0'] - (df_feat['predicted_solar_lag0'] + df_feat['predicted_wind_lag0'])).astype(float)
        df_feat['renewable_pressure_ratio_lag0'] = (((df_feat['predicted_solar_lag0'] + df_feat['predicted_wind_lag0'] + hydro_lag24) / load_lag0_safe).fillna(0)).astype(float)
        df_feat['solar_peak_pressure_ratio_lag0'] = (((df_feat['predicted_solar_lag0']) / load_lag0_safe).fillna(0)).astype(float)
        df_feat['solar_ramp_rate_lag0'] = df_feat['predicted_solar_lag0'].diff().fillna(0).astype(float)
        df_feat['renewable_ramp_rate_lag0'] = (df_feat['predicted_solar_lag0'] + df_feat['predicted_wind_lag0']).diff().fillna(0).astype(float)
        df_feat['zero_price_risk_score'] = (df_feat['renewable_pressure_ratio_lag0'] * (1.0 - (df_feat['hour'].isin([17,18,19,20,21])).astype(float))).astype(float)

    return df_feat


def get_feature_columns(set_name='full', df=None):
    """
    İlgili set için kullanılacak column listesini döndürür.
    set_name: 'base', 'core', 'full', 'robust'
    """
    base_cols = [
        'hour', 'dayofweek', 'month', 'quarter', 'is_weekend', 'is_peak_hour',
        'is_holiday', 'days_to_nearest_holiday',
        'mcp_usd_lag_24', 'mcp_usd_lag_48', 'mcp_usd_lag_168',
        'mcp_usd_roll_mean_24h', 'mcp_usd_roll_std_24h', 'mcp_usd_roll_mean_7d'
    ]
    
    core_cols = base_cols + [
        'load_lag_24', 'load_lag_48', 'load_lag_168',
        'kgup_lag_24', 'kgup_lag_48', 'kgup_lag_168',
        'kgup_wind_lag_24', 'kgup_wind_lag_48', 'kgup_wind_lag_168',
        'kgup_solar_lag_24', 'kgup_solar_lag_48', 'kgup_solar_lag_168',
        'kgup_hydro_lag_24', 'kgup_hydro_lag_48', 'kgup_hydro_lag_168',
        'kgup_gas_lag_24', 'kgup_gas_lag_48', 'kgup_gas_lag_168',
        'supply_demand_gap_mw', 'renewable_ratio', 'solar_wind_ratio',
        'hydro_water_energy_lag_24'
    ]
    
    full_cols = core_cols + [
        'actual_gen_lag_48', 'actual_gen_lag_168',
        'actual_cons_lag_48', 'actual_cons_lag_168'
    ]
    
    robust_cols = full_cols + [
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'sin_month', 'cos_month', 'sin_doy', 'cos_doy',
        'hydro_pressure_ratio', 'renewable_pressure_ratio', 'solar_peak_pressure_ratio',
        'is_low_price_regime', 'is_zero_price_hour', 'price_volatility_24h',
        'smp_usd_lag_48', 'temperature_lag_48', 'brent_oil_lag_48', 'natural_gas_grf_lag_48',
        'temp_forecast_lag0', 'cdh_cooling_load', 'hdh_heating_load', 'temp_diff_from_yesterday',
        'predicted_load_lag0', 'predicted_solar_lag0', 'predicted_wind_lag0',
        'net_load_lag0', 'renewable_pressure_ratio_lag0', 'solar_peak_pressure_ratio_lag0',
        'solar_ramp_rate_lag0', 'renewable_ramp_rate_lag0', 'zero_price_risk_score'
    ]
    
    set_map = {
        'base': base_cols,
        'core': core_cols,
        'full': full_cols,
        'robust': robust_cols
    }
    
    selected_cols = set_map.get(set_name, full_cols)
    
    if df is not None:
        return [c for c in selected_cols if c in df.columns]
    
    return selected_cols
