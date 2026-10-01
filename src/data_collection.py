"""
Data Collection Module for Stock Price Prediction Pipeline.
Fetches historical stock market data using Yahoo Finance (yfinance).
"""

import os
import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str = "AAPL",
    start_date: str = "2021-01-01",
    end_date: str = None,
    save_path: str = "data/stock_data.csv"
) -> pd.DataFrame:
    """
    Fetch historical daily stock market data from Yahoo Finance.

    Parameters:
    -----------
    ticker : str
        Ticker symbol (default: 'AAPL').
    start_date : str
        Start date string YYYY-MM-DD.
    end_date : str or None
        End date string YYYY-MM-DD (default: current date).
    save_path : str
        Path to save CSV output.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume, Adj Close.
    """
    logger.info(f"Fetching {ticker} historical data from {start_date} to {end_date or 'latest'}...")
    
    # Download stock data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No stock data retrieved for ticker '{ticker}' from Yahoo Finance.")

    # Flatten multi-index columns if yfinance returns tuple headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to get Date as a column
    df = df.reset_index()

    # Ensure Date column is formatted cleanly as YYYY-MM-DD
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Select standard columns
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from yfinance output.")

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop any remaining NaN rows
    initial_len = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with NaN values.")

    # Save to file if directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(df)} records of {ticker} stock data to {save_path}")

    return df


if __name__ == "__main__":
    # Test script directly
    stock_df = fetch_stock_data()
    print("Sample stock data:")
    print(stock_df.head())
