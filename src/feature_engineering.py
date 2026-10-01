"""
Feature Engineering Module for Stock Price Prediction.
Computes technical indicators (MA7, MA21, RSI, MACD, Volatility, Daily Return),
merges stock data with daily sentiment features, and constructs the target variable.
Strictly prevents data leakage.
"""

import logging
import pandas as pd
import numpy as np
import ta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_technical_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical stock indicators using the 'ta' library and pandas.

    Parameters:
    -----------
    stock_df : pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume.

    Returns:
    --------
    pd.DataFrame
        DataFrame augmented with technical indicator columns.
    """
    df = stock_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Daily Return (percentage change of Close)
    df["Daily_Return"] = df["Close"].pct_change()

    # 2. Moving Averages
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA21"] = df["Close"].rolling(window=21).mean()

    # 3. Volatility (14-day rolling standard deviation of daily return)
    df["Volatility"] = df["Daily_Return"].rolling(window=14).std()

    # 4. RSI (14-period Relative Strength Index using 'ta' library)
    rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()

    # 5. MACD (Moving Average Convergence Divergence)
    macd_indicator = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_indicator.macd()
    df["MACD_Signal"] = macd_indicator.macd_signal()
    df["MACD_Diff"] = macd_indicator.macd_diff()

    # Format Date back to string
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df


def merge_stock_and_sentiment(stock_df: pd.DataFrame, daily_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock technical indicators with daily sentiment features on Date.
    Missing sentiment days are imputed with neutral defaults (0 sentiment, 0 counts).
    """
    df = pd.merge(stock_df, daily_sentiment_df, on="Date", how="left")

    # Fill days without news with neutral sentiment defaults
    df["Avg_Sentiment"] = df["Avg_Sentiment"].fillna(0.0)
    df["Pos_News_Count"] = df["Pos_News_Count"].fillna(0)
    df["Neg_News_Count"] = df["Neg_News_Count"].fillna(0)
    df["Neu_News_Count"] = df["Neu_News_Count"].fillna(0)
    df["Sentiment_Momentum"] = df["Sentiment_Momentum"].fillna(0.0)

    return df


def prepare_dataset_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct Next-Day Target Variable (Target = 1 if Next Day Close > Today Close else 0)
    and clean dataset of NaN rows resulting from rolling window indicators.

    Target = 1 -> Next Trading Day Closing Price > Current Trading Day Closing Price
    Target = 0 -> Next Trading Day Closing Price <= Current Trading Day Closing Price

    Data Leakage Prevention:
    - Features at row T only contain market and news sentiment information up to day T.
    - Target at row T is Close(T+1) > Close(T).
    - Drop row T_last because Close(T_last + 1) is unknown.
    """
    data = df.copy()

    # Construct Next Day Close and Target
    data["Next_Close"] = data["Close"].shift(-1)
    data["Target"] = (data["Next_Close"] > data["Close"]).astype(int)

    # Yesterday's trend direction for baseline comparison (Direction_T-1)
    data["Prev_Return"] = data["Daily_Return"].shift(1)
    data["Baseline_Prev_Direction"] = (data["Daily_Return"] > 0).astype(int)

    # Drop the very last row since Next_Close is NaN
    data = data.iloc[:-1].copy()

    # Drop early rows containing NaNs due to rolling window technical indicators (MA21, MACD, Volatility)
    feature_cols = [
        "Daily_Return", "MA7", "MA21", "Volatility", "RSI", "MACD",
        "Volume", "Avg_Sentiment", "Pos_News_Count", "Neg_News_Count",
        "Neu_News_Count", "Sentiment_Momentum"
    ]
    
    initial_rows = len(data)
    data = data.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info(f"Dataset prepared for ML: {len(data)} valid rows (dropped {initial_rows - len(data)} NaN burn-in rows).")

    return data


if __name__ == "__main__":
    from src.data_collection import fetch_stock_data
    from src.news_generator import generate_synthetic_apple_news
    from src.sentiment_analysis import process_news_sentiment, aggregate_daily_sentiment

    stock_raw = fetch_stock_data()
    stock_tech = compute_technical_indicators(stock_raw)

    news_raw = generate_synthetic_apple_news()
    news_sent = process_news_sentiment(news_raw)
    daily_sent = aggregate_daily_sentiment(news_sent, full_date_range=stock_tech["Date"])

    merged = merge_stock_and_sentiment(stock_tech, daily_sent)
    final_ml_df = prepare_dataset_for_ml(merged)

    print("Final ML Dataset Sample:")
    print(final_ml_df[["Date", "Close", "MA7", "RSI", "Avg_Sentiment", "Target"]].tail())
