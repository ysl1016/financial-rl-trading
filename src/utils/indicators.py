import numpy as np
import pandas as pd

def calculate_technical_indicators(data):
    """
    Calculate various technical indicators for the given price data
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    df = pd.DataFrame(index=data.index)
    
    # Extract price and volume data
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    delta = close.diff()
    
    # RSI
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Force Index
    df['ForceIndex2'] = close.diff(2) * volume
    
    # Stochastic Oscillator
    low_min = low.rolling(window=14).min()
    high_max = high.rolling(window=14).max()
    df['%K'] = 100 * (close - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACDSignal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    window = 20
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    df['SMA20'] = sma
    df['BBUpper'] = sma + (std * 2)
    df['BBLower'] = sma - (std * 2)
    df['BBWidth'] = (df['BBUpper'] - df['BBLower']) / df['SMA20']
    
    # Average True Range (ATR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Volume Price Trend (VPT)
    df['VPT'] = volume * ((close - close.shift(1)) / close.shift(1).replace(0, np.nan))
    df['VPT_MA'] = df['VPT'].rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (volume * (np.sign(delta))).cumsum()
    
    # Price Rate of Change (ROC)
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    
    # Remove missing values
    df = df.dropna()
    
    # Normalize indicators (Z-score)
    normalize_cols = ['RSI', 'ForceIndex2', '%K', '%D', 'MACD', 'MACDSignal',
                     'BBWidth', 'ATR', 'VPT', 'VPT_MA', 'OBV', 'ROC']
    for col in normalize_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_norm'] = (df[col] - mean) / (std + 1e-9)
    
    return df