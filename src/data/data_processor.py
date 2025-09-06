import logging
import yfinance as yf
import pandas as pd
from ..utils.indicators import calculate_technical_indicators

logger = logging.getLogger(__name__)

def download_stock_data(symbol, start_date=None, end_date=None):
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data

    Raises:
        ValueError: If the data cannot be downloaded or is invalid.
    """
    try:
        logger.info("%s 데이터 다운로드 중...", symbol)
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
    except Exception as e:
        logger.error("yfinance 호출 실패: %s", e)
        raise ValueError(f"Failed to download data for {symbol}") from e

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if data.empty or not required_cols.issubset(data.columns):
        raise ValueError("Downloaded data is empty or missing required columns")

    logger.info("다운로드 완료: %d 데이터 포인트", len(data))
    return data

def process_data(symbol, start_date=None, end_date=None,
                 train_ratio=0.7, val_ratio=0.15):
    """
    Download stock data, calculate technical indicators, and split into
    train/validation/test sets. Normalization statistics are computed from
    the training period and reused for validation and test periods.

    Args:
        symbol (str): Stock symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        train_ratio (float, optional): Proportion of data used for training.
            Defaults to 0.7.
        val_ratio (float, optional): Proportion of data used for validation.
            Defaults to 0.15. The remainder is used for testing.

    Returns:
        pd.DataFrame: Processed DataFrame with technical indicators

    Raises:
        ValueError: Propagated from download_stock_data when data download fails.
    """
    # Download data
    data = download_stock_data(symbol, start_date, end_date)
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(data)
    
    # Merge data with indicators
    processed_data = pd.concat([data, indicators], axis=1)
    
    return processed_data
