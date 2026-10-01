"""
Master Pipeline & Prediction Module for Apple Inc. (AAPL) Stock Prediction.
Executes the full end-to-end pipeline:
Stock Data -> Technical Indicators -> Synthetic News -> LLM Sentiment ->
Data Merging -> ML Training & Evaluation -> Chart Visualizations -> Terminal Summary
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_collection import fetch_stock_data
from src.news_generator import generate_synthetic_apple_news
from src.sentiment_analysis import LLMSentimentAnalyzer, process_news_sentiment, aggregate_daily_sentiment
from src.feature_engineering import compute_technical_indicators, merge_stock_and_sentiment, prepare_dataset_for_ml
from src.model import StockTrendPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure dark/modern visualization theme
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_visualizations(full_df: pd.DataFrame, test_df: pd.DataFrame, eval_results: dict, predictor: StockTrendPredictor, output_dir: str = "visualizations"):
    """
    Generate and save professional charts:
    1. AAPL closing-price history
    2. Moving averages (MA7 & MA21)
    3. RSI Indicator
    4. MACD Indicator
    5. Daily sentiment scores
    6. Stock price vs sentiment
    7. Actual vs predicted trend
    8. Confusion matrix
    9. Feature importance
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating professional visualizations in '{output_dir}'...")

    full_df["Date_dt"] = pd.to_datetime(full_df["Date"])

    # 1. Closing Price History
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.plot(full_df["Date_dt"], full_df["Close"], color="#1f77b4", linewidth=1.8, label="AAPL Close Price")
    ax.set_title("Apple Inc. (AAPL) Historical Closing Price (3-Year)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_closing_price_history.png"))
    plt.close(fig)

    # 2. Moving Averages
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.plot(full_df["Date_dt"], full_df["Close"], color="#1f77b4", alpha=0.5, label="Close Price")
    ax.plot(full_df["Date_dt"], full_df["MA7"], color="#ff7f0e", linewidth=1.5, label="7-Day MA (MA7)")
    ax.plot(full_df["Date_dt"], full_df["MA21"], color="#2ca02c", linewidth=1.5, label="21-Day MA (MA21)")
    ax.set_title("AAPL Moving Averages (MA7 vs MA21)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_moving_averages.png"))
    plt.close(fig)

    # 3. RSI Indicator
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.plot(full_df["Date_dt"], full_df["RSI"], color="#9467bd", linewidth=1.5, label="RSI (14)")
    ax.axhline(70, color="crimson", linestyle="--", alpha=0.7, label="Overbought (70)")
    ax.axhline(30, color="forestgreen", linestyle="--", alpha=0.7, label="Oversold (30)")
    ax.fill_between(full_df["Date_dt"], 70, full_df["RSI"], where=(full_df["RSI"] >= 70), color="crimson", alpha=0.2)
    ax.fill_between(full_df["Date_dt"], 30, full_df["RSI"], where=(full_df["RSI"] <= 30), color="forestgreen", alpha=0.2)
    ax.set_title("AAPL Relative Strength Index (RSI 14)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("RSI Score", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_rsi_indicator.png"))
    plt.close(fig)

    # 4. MACD Indicator
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.plot(full_df["Date_dt"], full_df["MACD"], color="#1f77b4", linewidth=1.5, label="MACD Line")
    ax.plot(full_df["Date_dt"], full_df["MACD_Signal"], color="#ff7f0e", linestyle="--", linewidth=1.5, label="Signal Line")
    ax.bar(full_df["Date_dt"], full_df["MACD_Diff"], color=np.where(full_df["MACD_Diff"] >= 0, "green", "red"), alpha=0.4, label="Histogram")
    ax.set_title("AAPL MACD Indicator (12, 26, 9)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_macd_indicator.png"))
    plt.close(fig)

    # 5. Daily Sentiment Scores
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    sentiment_df = full_df[full_df["Avg_Sentiment"] != 0]
    ax.bar(sentiment_df["Date_dt"], sentiment_df["Avg_Sentiment"], color=np.where(sentiment_df["Avg_Sentiment"] > 0, "forestgreen", "crimson"), alpha=0.75, width=2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Daily News Sentiment Scores (-1.0 to +1.0)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Sentiment Score", fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_daily_sentiment_scores.png"))
    plt.close(fig)

    # 6. Stock Price vs Sentiment
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
    ax2 = ax1.twinx()
    ax1.plot(full_df["Date_dt"], full_df["Close"], color="#1f77b4", linewidth=1.8, label="AAPL Close Price")
    ax2.plot(full_df["Date_dt"], full_df["Sentiment_Momentum"], color="#d62728", linestyle="-.", linewidth=1.5, label="Sentiment Momentum (3D MA)")
    ax1.set_title("AAPL Stock Price vs Daily Sentiment Momentum", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Stock Price ($)", color="#1f77b4", fontsize=11)
    ax2.set_ylabel("Sentiment Momentum", color="#d62728", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_stock_vs_sentiment.png"))
    plt.close(fig)

    # 7. Actual vs Predicted Trend (Test Set)
    test_df_dt = test_df.copy()
    test_df_dt["Date_dt"] = pd.to_datetime(test_df_dt["Date"])
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    y_test_vals = eval_results["y_pred"]
    ax.plot(test_df_dt["Date_dt"], test_df_dt["Target"], label="Actual Trend (1=UP, 0=DOWN)", color="#1f77b4", marker="o", markersize=3, linestyle="None", alpha=0.6)
    ax.plot(test_df_dt["Date_dt"], y_test_vals, label="Predicted Trend (Random Forest)", color="#ff7f0e", marker="x", markersize=4, linestyle="None", alpha=0.8)
    ax.set_title("Test Set: Actual vs Predicted Next-Day Stock Trend", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Trend Direction (1=UP, 0=DOWN)", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["DOWN", "UP"])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_actual_vs_predicted_trend.png"))
    plt.close(fig)

    # 8. Confusion Matrix
    cm = eval_results["Random_Forest"]["Confusion_Matrix"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred DOWN", "Pred UP"],
                yticklabels=["Actual DOWN", "Actual UP"], ax=ax)
    ax.set_title("Confusion Matrix - Random Forest", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_confusion_matrix.png"))
    plt.close(fig)

    # 9. Feature Importance
    fi_df = predictor.get_feature_importance_df()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Random Forest Feature Importances", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Gini Importance", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_feature_importance.png"))
    plt.close(fig)

    logger.info("All 9 charts successfully saved to visualizations directory.")


def run_pipeline():
    """
    Run complete prediction pipeline and display formatted terminal summary.
    """
    logger.info("Starting AAPL Stock Price Prediction Pipeline...")

    # 1. Fetch stock data
    stock_df = fetch_stock_data(ticker="AAPL", start_date="2021-01-01", save_path="data/stock_data.csv")
    stock_df = compute_technical_indicators(stock_df)

    # 2. Synthetic news & LLM sentiment analysis
    news_df = generate_synthetic_apple_news(start_date="2021-01-01", save_path="data/synthetic_news.csv")
    analyzer = LLMSentimentAnalyzer()
    news_sentiment_df = process_news_sentiment(news_df, analyzer)
    daily_sentiment_df = aggregate_daily_sentiment(news_sentiment_df, full_date_range=stock_df["Date"])

    # 3. Merge stock + sentiment data and prepare ML target
    merged_df = merge_stock_and_sentiment(stock_df, daily_sentiment_df)
    ml_df = prepare_dataset_for_ml(merged_df)

    # 4. Model training with chronological split
    predictor = StockTrendPredictor(n_estimators=100, max_depth=5, random_state=42)
    X_train, X_test, y_train, y_test, train_df, test_df = predictor.chronological_split(ml_df, train_ratio=0.8)

    predictor.fit(X_train, y_train)
    eval_results = predictor.evaluate(X_test, y_test, test_df)

    # Save model
    predictor.save("models/random_forest.pkl")

    # Save visualizations
    generate_visualizations(merged_df, test_df, eval_results, predictor, output_dir="visualizations")

    # 5. Output Latest Market Prediction
    latest_row = merged_df.iloc[-1]
    latest_features = latest_row[predictor.feature_cols].to_frame().T

    pred_class = predictor.predict(latest_features)[0]
    pred_probas = predictor.predict_proba(latest_features)[0]

    prob_down = pred_probas[0] * 100
    prob_up = pred_probas[1] * 100
    trend_str = "UP 📈" if pred_class == 1 else "DOWN 📉"

    avg_sent = latest_row['Avg_Sentiment']
    sent_prefix = "+" if avg_sent >= 0 else ""

    rf_acc = eval_results['Random_Forest']['Accuracy'] * 100
    base_acc = eval_results['Baseline']['Accuracy'] * 100 if eval_results['Baseline'] else 0.0

    output_summary = f"""
=====================================================
          STOCK PRICE PREDICTION REPORT (AAPL)       
=====================================================
Stock: AAPL
Current Date: {latest_row['Date']}
Current Price: ${latest_row['Close']:.2f}

Market Signals:
  RSI: {latest_row['RSI']:.2f}
  MA7: ${latest_row['MA7']:.2f}
  MA21: ${latest_row['MA21']:.2f}
  MACD: {latest_row['MACD']:.2f}

News Sentiment:
  Average Sentiment: {sent_prefix}{avg_sent:.2f}
  Positive News: {int(latest_row['Pos_News_Count'])}
  Negative News: {int(latest_row['Neg_News_Count'])}
  Neutral News: {int(latest_row['Neu_News_Count'])}

Model Performance (Test Set):
  Random Forest Accuracy: {rf_acc:.2f}%
  Naive Baseline Accuracy: {base_acc:.2f}%
  Precision: {eval_results['Random_Forest']['Precision']:.4f}
  Recall: {eval_results['Random_Forest']['Recall']:.4f}
  F1-Score: {eval_results['Random_Forest']['F1-Score']:.4f}

Prediction:
  Next-Day Trend: {trend_str}
  Probability UP: {prob_up:.1f}%
  Probability DOWN: {prob_down:.1f}%
=====================================================
"""
    print(output_summary)
    return output_summary


if __name__ == "__main__":
    run_pipeline()
