import yfinance as yf
import pandas as pd
from ..utils.indicators import calculate_technical_indicators

def download_stock_data(symbol, start_date=None, end_date=None):
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    print(f"{symbol} 데이터 다운로드 중...")
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    print(f"다운로드 완료: {len(data)} 데이터 포인트\n")
    return data

def process_data(symbol, start_date=None, end_date=None):
    """
    Download and process stock data
    
    Args:
        symbol (str): Stock symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: Processed DataFrame with technical indicators
    """
    # Download data
    data = download_stock_data(symbol, start_date, end_date)
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(data)
    
    # Merge data with indicators and remove rows with missing values
    # Keep DatetimeIndex to preserve date alignment across assets
    processed_data = pd.concat([data, indicators], axis=1).dropna()

    return processed_data
