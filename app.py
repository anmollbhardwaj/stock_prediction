"""
Streamlit Web Dashboard for AAPL Stock Price Prediction & LLM Sentiment Analysis.
Run with: streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Insert root directory into python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_collection import fetch_stock_data
from src.news_generator import generate_synthetic_apple_news
from src.sentiment_analysis import LLMSentimentAnalyzer, process_news_sentiment, aggregate_daily_sentiment
from src.feature_engineering import compute_technical_indicators, merge_stock_and_sentiment, prepare_dataset_for_ml
from src.model import StockTrendPredictor

# Page Config
st.set_page_config(
    page_title="AAPL Stock Trend & Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data_and_run_pipeline():
    # 1. Fetch Stock Data
    stock_df = fetch_stock_data(ticker="AAPL", start_date="2021-01-01", save_path="data/stock_data.csv")
    stock_df = compute_technical_indicators(stock_df)

    # 2. Synthetic News & Sentiment
    news_df = generate_synthetic_apple_news(start_date="2021-01-01", save_path="data/synthetic_news.csv")
    analyzer = LLMSentimentAnalyzer(provider="mock")
    news_sent_df = process_news_sentiment(news_df, analyzer)
    daily_sent_df = aggregate_daily_sentiment(news_sent_df, full_date_range=stock_df["Date"])

    # 3. Merge & Prepare Dataset
    merged_df = merge_stock_and_sentiment(stock_df, daily_sent_df)
    ml_df = prepare_dataset_for_ml(merged_df)

    # 4. Train Model
    predictor = StockTrendPredictor(n_estimators=100, max_depth=5, random_state=42)
    X_train, X_test, y_train, y_test, train_df, test_df = predictor.chronological_split(ml_df, train_ratio=0.8)
    predictor.fit(X_train, y_train)
    eval_res = predictor.evaluate(X_test, y_test, test_df)

    return merged_df, ml_df, test_df, eval_res, predictor, news_sent_df


# Header
st.markdown('<div class="main-header">🍏 Apple Inc. (AAPL) Stock Trend Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning (Random Forest) + LLM News Sentiment Analysis</div>', unsafe_allow_html=True)

with st.spinner("Fetching market data and running LLM sentiment pipeline..."):
    merged_df, ml_df, test_df, eval_res, predictor, news_sent_df = load_data_and_run_pipeline()

latest = merged_df.iloc[-1]
latest_features = latest[predictor.feature_cols].to_frame().T
pred_class = predictor.predict(latest_features)[0]
pred_probas = predictor.predict_proba(latest_features)[0]

# Top KPI Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Latest AAPL Close", f"${latest['Close']:.2f}", f"{latest['Daily_Return']*100:+.2f}%")
with col2:
    st.metric("RSI (14)", f"{latest['RSI']:.2f}", "Overbought > 70" if latest['RSI'] > 70 else ("Oversold < 30" if latest['RSI'] < 30 else "Neutral"))
with col3:
    st.metric("Daily Sentiment", f"{latest['Avg_Sentiment']:+.2f}", f"{int(latest['Pos_News_Count'])} Pos / {int(latest['Neg_News_Count'])} Neg")
with col4:
    st.metric("Next-Day Prediction", "UP 📈" if pred_class == 1 else "DOWN 📉", f"Conf: {max(pred_probas)*100:.1f}%")
with col5:
    st.metric("Model Accuracy", f"{eval_res['Random_Forest']['Accuracy']*100:.1f}%", f"vs Baseline {eval_res['Baseline']['Accuracy']*100:.1f}%")

st.divider()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market & Technical Charts", "📰 News & LLM Sentiment", "🤖 ML Model Evaluation", "🔮 Custom Prediction Simulator"])

with tab1:
    st.subheader("AAPL Historical Price & Technical Indicators")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.write("#### Price & Moving Averages (MA7 vs MA21)")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(pd.to_datetime(merged_df["Date"]), merged_df["Close"], label="Close", color="#1f77b4")
        ax.plot(pd.to_datetime(merged_df["Date"]), merged_df["MA7"], label="MA7", color="#ff7f0e")
        ax.plot(pd.to_datetime(merged_df["Date"]), merged_df["MA21"], label="MA21", color="#2ca02c")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

    with col_chart2:
        st.write("#### RSI (14) & MACD")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
        ax1.plot(pd.to_datetime(merged_df["Date"]), merged_df["RSI"], color="#9467bd")
        ax1.axhline(70, color="r", linestyle="--")
        ax1.axhline(30, color="g", linestyle="--")
        ax1.set_ylabel("RSI")
        
        ax2.plot(pd.to_datetime(merged_df["Date"]), merged_df["MACD"], color="#1f77b4", label="MACD")
        ax2.plot(pd.to_datetime(merged_df["Date"]), merged_df["MACD_Signal"], color="#ff7f0e", label="Signal")
        ax2.set_ylabel("MACD")
        ax2.legend()
        st.pyplot(fig)

with tab2:
    st.subheader("LLM Sentiment Analysis Breakdown")
    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        st.write("#### Daily Sentiment Score & Stock Price Momentum")
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax2 = ax1.twinx()
        ax1.plot(pd.to_datetime(merged_df["Date"]), merged_df["Close"], color="#1f77b4", label="Close Price")
        ax2.plot(pd.to_datetime(merged_df["Date"]), merged_df["Sentiment_Momentum"], color="red", linestyle="--", label="Sentiment Momentum")
        ax1.set_ylabel("Stock Price ($)", color="#1f77b4")
        ax2.set_ylabel("Sentiment Momentum", color="red")
        st.pyplot(fig)

    with col_s2:
        st.write("#### Sample Analyzed Apple News Articles")
        st.dataframe(
            news_sent_df[["Date", "Headline", "Sentiment", "Sentiment_Score", "Reasoning"]].head(10),
            use_container_width=True
        )

with tab3:
    st.subheader("Random Forest Model Performance vs Baseline")
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(eval_res["Random_Forest"]["Confusion_Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred DOWN", "Pred UP"], yticklabels=["Actual DOWN", "Actual UP"])
        st.pyplot(fig)

    with c_m2:
        st.write("#### Feature Importances")
        fi_df = predictor.get_feature_importance_df()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        st.pyplot(fig)

with tab4:
    st.subheader("Live Custom News Sentiment & Stock Trend Simulator")
    st.write("Enter a custom news headline and description to simulate how LLM sentiment affects next-day prediction:")
    
    custom_headline = st.text_input("Headline:", "Apple announces major AI breakthrough with new chip design")
    custom_desc = st.text_area("Description:", "The new processor quadruples AI inference speed, boosting iPhone upgrade cycle expectations.")
    
    if st.button("Run Sentiment & Predict Stock Impact"):
        analyzer = LLMSentimentAnalyzer(provider="mock")
        res = analyzer.analyze_article(custom_headline, custom_desc)
        
        st.write("### Sentiment Analysis Result:")
        st.info(f"**Sentiment:** {res['sentiment']} | **Score:** {res['sentiment_score']:+.2f}\n\n**Reasoning:** {res['reasoning']}")
        
        # Modify latest row sentiment and predict
        sim_features = latest_features.copy()
        sim_features["Avg_Sentiment"] = res["sentiment_score"]
        sim_features["Pos_News_Count"] = 1 if res["sentiment"] == "Positive" else 0
        sim_features["Neg_News_Count"] = 1 if res["sentiment"] == "Negative" else 0
        
        sim_pred = predictor.predict(sim_features)[0]
        sim_proba = predictor.predict_proba(sim_features)[0]
        
        st.success(f"**Predicted Next-Day Stock Direction:** {'UP 📈' if sim_pred == 1 else 'DOWN 📉'} (Confidence: {max(sim_proba)*100:.1f}%)")
