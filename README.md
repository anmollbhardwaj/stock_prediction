# Stock Price Prediction using Machine Learning + LLM Sentiment Analysis (AAPL)

An end-to-end Machine Learning system that predicts **Apple Inc. (AAPL)** next-day stock price trend (UP / DOWN) by fusing **Yahoo Finance historical market data**, **technical indicators**, and **LLM news sentiment analysis**.

---

## 🌟 Architecture & Key Features

```text
Yahoo Finance (yfinance)          Synthetic Apple News Dataset
         │                                    │
   AAPL Price Data                    LLM Sentiment Analyzer
(Open, High, Low, Close, Vol)       (Positive / Neutral / Negative)
         │                                    │
  Technical Indicators                Daily Sentiment Features
 (MA7, MA21, RSI, MACD)            (Avg Score, Pos/Neg Counts)
         └──────────────────┬─────────────────┘
                            │
               Merged Anti-Leakage Features
                            │
              Chronological Train / Test Split
                            │
               Random Forest Classifier
                            │
                Next-Day UP/DOWN Prediction
```

1. **Market Data & Technical Indicators**: Collects daily AAPL stock market data via `yfinance` and engineers 7 core technical features: `Daily_Return`, `MA7`, `MA21`, `Volatility`, `RSI`, `MACD` line, signal line & histogram.
2. **LLM Sentiment Component**: Synthesizes realistic corporate/financial news articles covering 12 Apple topics (iPhone demand, AI developments, earnings, supply chain, regulatory issues, analyst upgrades/downgrades). Features a pluggable **LLM Sentiment Engine** supporting local mock mode, Google Gemini API, or OpenAI API.
3. **Anti-Leakage Engineering**: Uses a strict **chronological 80/20 train/test split** (no random shuffling). Features at day $T$ only contain information available by market close of day $T$ to predict day $T+1$ price movement.
4. **Model Training & Evaluation**: Trains a `RandomForestClassifier` with `StandardScaler` fitted strictly on training data. Evaluates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix against a Naive Baseline model.
5. **Visualizations & Interactive Web App**: Generates 9 high-resolution diagnostic charts saved to `visualizations/` and includes an interactive Streamlit Web Dashboard (`app.py`).

---

## 📁 Project Structure

```text
stock_prediction/
│
├── data/
│   ├── stock_data.csv          # AAPL historical market data
│   └── synthetic_news.csv      # Generated Apple news dataset with LLM sentiment
│
├── src/
│   ├── data_collection.py      # Yahoo Finance data fetching module
│   ├── news_generator.py       # Synthetic Apple news generator
│   ├── sentiment_analysis.py   # LLM Sentiment Analyzer (Mock / Gemini / OpenAI)
│   ├── feature_engineering.py  # Technical indicators & target alignment
│   ├── model.py                # Random Forest model & chronological split
│   └── prediction.py           # Master pipeline runner & CLI summary
│
├── app.py                      # Interactive Streamlit Web Dashboard
│
├── notebooks/
│   └── stock_prediction.ipynb  # Step-by-step walkthrough notebook
│
├── models/
│   └── random_forest.pkl       # Trained model & scaler checkpoint
│
├── visualizations/             # 9 Generated diagnostic PNG charts
│   ├── 01_closing_price_history.png
│   ├── 02_moving_averages.png
│   ├── 03_rsi_indicator.png
│   ├── 04_macd_indicator.png
│   ├── 05_daily_sentiment_scores.png
│   ├── 06_stock_vs_sentiment.png
│   ├── 07_actual_vs_predicted_trend.png
│   ├── 08_confusion_matrix.png
│   └── 09_feature_importance.png
│
├── .env.example                # LLM API configuration example
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start & Usage

### 1. Prerequisites & Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
# Initialize virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure LLM API Keys (Optional)

Copy `.env.example` to `.env`. By default, the system runs with the high-precision **local mock LLM engine** without requiring any external API key.

```ini
LLM_PROVIDER=mock
# GEMINI_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
```

### 3. Run the Complete Prediction Pipeline

Execute the master pipeline script to download stock data, analyze news sentiment, train the Random Forest model, output predictions, and generate all 9 visual charts:

```bash
python src/prediction.py
```

Sample CLI Output:

```text
=====================================================
          STOCK PRICE PREDICTION REPORT (AAPL)       
=====================================================
Stock: AAPL
Current Date: 2026-09-30
Current Price: $XXX.XX

Market Signals:
  RSI: 54.32
  MA7: $228.45
  MA21: $224.10
  MACD: 1.45

News Sentiment:
  Average Sentiment: +0.65
  Positive News: 2
  Negative News: 0
  Neutral News: 1

Model Performance (Test Set):
  Random Forest Accuracy: 62.50%
  Naive Baseline Accuracy: 48.10%
  Precision: 0.6315
  Recall: 0.6540
  F1-Score: 0.6426

Prediction:
  Next-Day Trend: UP 📈
  Probability UP: 68.4%
  Probability DOWN: 31.6%
=====================================================
```

---

## 📊 Interactive Web App (Streamlit)

Launch the Streamlit Web Application to interactively explore technical charts, sentiment breakdown, feature importances, and run real-time custom news sentiment simulations:

```bash
streamlit run app.py
```

---

## 📈 Visualizations Generated

The project automatically saves 9 diagnostic charts in the `visualizations/` folder:

1. `01_closing_price_history.png`: AAPL historical closing price over 3+ years.
2. `02_moving_averages.png`: Short-term (MA7) vs medium-term (MA21) moving averages.
3. `03_rsi_indicator.png`: 14-period RSI indicator with overbought/oversold boundaries.
4. `04_macd_indicator.png`: MACD line, signal line, and momentum histogram.
5. `05_daily_sentiment_scores.png`: Daily aggregated LLM sentiment scores.
6. `06_stock_vs_sentiment.png`: Stock price overlaid with 3-day sentiment momentum.
7. `07_actual_vs_predicted_trend.png`: Actual vs predicted stock direction on out-of-sample test set.
8. `08_confusion_matrix.png`: Confusion matrix highlighting true positives/negatives.
9. `09_feature_importance.png`: Ranked Gini feature importances across market & sentiment variables.

---

## 📜 License

Distributed under the Apache 2.0 License.
