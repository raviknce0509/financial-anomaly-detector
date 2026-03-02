import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Price-based features
    df['daily_return']     = df['Close'].pct_change()
    df['log_return']       = np.log(df['Close'] / df['Close'].shift(1))
    df['price_range']      = (df['High'] - df['Low']) / df['Close']
    df['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open']
    
    # Rolling statistics (20-day window)
    df['rolling_mean_20']  = df['Close'].rolling(20).mean()
    df['rolling_std_20']   = df['Close'].rolling(20).std()
    df['rolling_vol_20']   = df['Volume'].rolling(20).mean()
    
    # Bollinger Bands
    df['bb_upper'] = df['rolling_mean_20'] + (2 * df['rolling_std_20'])
    df['bb_lower'] = df['rolling_mean_20'] - (2 * df['rolling_std_20'])
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (
        df['bb_upper'] - df['bb_lower']
    )
    
    # Volume spike
    df['volume_spike'] = df['Volume'] / df['rolling_vol_20']
    
    # Z-score of returns (key anomaly signal)
    df['return_zscore'] = (
        df['daily_return'] - df['daily_return'].rolling(20).mean()
    ) / df['daily_return'].rolling(20).std()
    
    # Drop NaN rows from rolling calculations
    df.dropna(inplace=True)
    
    return df