import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker.
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    print(f"Fetching data for {ticker}...", flush=True)
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)  # remove timezone
    
    print(f"✅ Fetched {len(df)} rows for {ticker}", flush=True)
    return df