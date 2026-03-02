import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = [
    'daily_return', 'log_return', 'price_range',
    'close_open_ratio', 'volume_spike', 'return_zscore', 'bb_position'
]

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05):
    """
    Use Isolation Forest to detect anomalous trading days.
    contamination = expected % of anomalies (5% default)
    """
    df = df.copy()
    
    X = df[FEATURES].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    df['anomaly'] = model.fit_predict(X_scaled)
    df['anomaly_score'] = model.score_samples(X_scaled)
    
    # Convert: IsolationForest returns -1 for anomaly, 1 for normal
    df['is_anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    
    # Risk classification
    df['risk_level'] = 'NORMAL'
    df.loc[df['is_anomaly'] == 1, 'risk_level'] = 'ANOMALY'
    df.loc[df['return_zscore'].abs() > 3, 'risk_level'] = 'EXTREME'
    
    return df, model, scaler

def get_anomaly_summary(df: pd.DataFrame) -> dict:
    total       = len(df)
    anomalies   = df['is_anomaly'].sum()
    extreme     = (df['risk_level'] == 'EXTREME').sum()
    
    return {
        'total_days':       total,
        'anomaly_days':     int(anomalies),
        'extreme_days':     int(extreme),
        'anomaly_rate':     f"{(anomalies/total)*100:.1f}%",
        'latest_signal':    df['risk_level'].iloc[-1],
        'latest_return':    f"{df['daily_return'].iloc[-1]*100:.2f}%",
        'latest_zscore':    f"{df['return_zscore'].iloc[-1]:.2f}"
    }