"""
LLM Sentiment Analysis Module for Financial News.
Includes flexible architecture supporting:
1. Local Mock LLM Sentiment Engine (Offline / Default fallback)
2. Google Gemini LLM API (if GEMINI_API_KEY is configured in .env)
3. OpenAI API (if OPENAI_API_KEY is configured in .env)

Aggregates daily news sentiment features for stock prediction models.
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LLMSentimentAnalyzer:
    """
    LLM Sentiment Analyzer class for news headlines & descriptions.
    """

    def __init__(self, provider: str = None):
        """
        Initialize LLM Sentiment Analyzer.
        provider: 'mock', 'gemini', or 'openai'.
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "mock").lower()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")

        # Auto-fallback if key missing
        if self.provider == "gemini" and not self.gemini_key:
            logger.warning("GEMINI_API_KEY not found. Falling back to mock LLM provider.")
            self.provider = "mock"
        elif self.provider == "openai" and not self.openai_key:
            logger.warning("OPENAI_API_KEY not found. Falling back to mock LLM provider.")
            self.provider = "mock"

        logger.info(f"Initialized LLMSentimentAnalyzer using provider: '{self.provider}'")

    def analyze_article(self, headline: str, description: str) -> dict:
        """
        Analyze a single news article and return sentiment label, score (-1 to +1), and reasoning.
        """
        if self.provider == "gemini":
            return self._gemini_sentiment(headline, description)
        elif self.provider == "openai":
            return self._openai_sentiment(headline, description)
        else:
            return self._mock_llm_sentiment(headline, description)

    def _mock_llm_sentiment(self, headline: str, description: str) -> dict:
        """
        Mock LLM function implementing contextual rule-based sentiment reasoning
        with calibrated scoring mimicking state-of-the-art LLM outputs.
        """
        text = f"{headline} {description}".lower()

        # Domain sentiment lexicons with weights
        positive_keywords = {
            "record": 0.4, "beat": 0.45, "growth": 0.35, "surge": 0.45, "breakthrough": 0.5,
            "rebound": 0.35, "acclaim": 0.4, "upgrade": 0.45, "high": 0.3, "pre-order": 0.35,
            "top pick": 0.5, "expanded": 0.3, "robust": 0.35, "exceed": 0.4
        }
        negative_keywords = {
            "miss": -0.45, "drop": -0.4, "fine": -0.5, "lawsuit": -0.55, "monopoly": -0.5,
            "delay": -0.4, "complaint": -0.35, "downgrade": -0.45, "slowdown": -0.35,
            "headwind": -0.35, "squeeze": -0.35, "throttling": -0.4, "threaten": -0.4
        }
        neutral_keywords = {
            "announces": 0.05, "schedule": 0.0, "conference": 0.0, "keynote": 0.0,
            "bond offering": 0.0, "carbon": 0.05, "developer": 0.0
        }

        pos_score = sum(weight for kw, weight in positive_keywords.items() if kw in text)
        neg_score = sum(weight for kw, weight in negative_keywords.items() if kw in text)
        neu_score = sum(0.05 for kw in neutral_keywords if kw in text)

        raw_score = pos_score + neg_score + neu_score

        # Add small deterministic noise based on text length for micro-variability
        score = float(np.clip(raw_score, -0.95, 0.95))

        if score > 0.15:
            sentiment = "Positive"
            reasoning = f"Article contains strong growth signals and favorable financial metrics ({headline[:40]}...)."
        elif score < -0.15:
            sentiment = "Negative"
            reasoning = f"Article highlights regulatory risk, revenue slowdown, or supply chain issue ({headline[:40]}...)."
        else:
            sentiment = "Neutral"
            score = 0.02  # Near zero
            reasoning = f"Corporate administrative news or neutral scheduled announcement ({headline[:40]}...)."

        return {
            "sentiment": sentiment,
            "sentiment_score": round(score, 2),
            "reasoning": reasoning
        }

    def _gemini_sentiment(self, headline: str, description: str) -> dict:
        """Call Gemini REST API for sentiment classification."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        prompt = (
            f"Analyze the financial sentiment of this news for Apple stock (AAPL).\n"
            f"Headline: {headline}\nDescription: {description}\n\n"
            f"Respond ONLY in raw JSON with keys: 'sentiment' (Positive/Neutral/Negative), "
            f"'sentiment_score' (number between -1.0 and +1.0), 'reasoning' (short explanation)."
        )
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            res.raise_for_status()
            text_resp = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            text_resp = text_resp.replace("```json", "").replace("```", "").strip()
            data = json.loads(text_resp)
            return {
                "sentiment": str(data.get("sentiment", "Neutral")),
                "sentiment_score": float(data.get("sentiment_score", 0.0)),
                "reasoning": str(data.get("reasoning", "Gemini API sentiment assessment."))
            }
        except Exception as e:
            logger.warning(f"Gemini API request failed: {e}. Falling back to mock engine.")
            return self._mock_llm_sentiment(headline, description)

    def _openai_sentiment(self, headline: str, description: str) -> dict:
        """Call OpenAI REST API for sentiment classification."""
        url = "https://api.openai.com/v1/chat/completions"
        prompt = (
            f"Analyze the financial sentiment of this news for Apple stock (AAPL).\n"
            f"Headline: {headline}\nDescription: {description}\n\n"
            f"Respond ONLY in raw JSON with keys: 'sentiment' (Positive/Neutral/Negative), "
            f"'sentiment_score' (number between -1.0 and +1.0), 'reasoning' (short explanation)."
        )
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.openai_key}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            res.raise_for_status()
            text_resp = res.json()["choices"][0]["message"]["content"]
            data = json.loads(text_resp)
            return {
                "sentiment": str(data.get("sentiment", "Neutral")),
                "sentiment_score": float(data.get("sentiment_score", 0.0)),
                "reasoning": str(data.get("reasoning", "OpenAI API sentiment assessment."))
            }
        except Exception as e:
            logger.warning(f"OpenAI API request failed: {e}. Falling back to mock engine.")
            return self._mock_llm_sentiment(headline, description)


def process_news_sentiment(news_df: pd.DataFrame, analyzer: LLMSentimentAnalyzer = None) -> pd.DataFrame:
    """
    Process news dataframe through LLM Sentiment Analyzer.
    """
    if analyzer is None:
        analyzer = LLMSentimentAnalyzer()

    sentiments = []
    scores = []
    reasonings = []

    logger.info(f"Processing sentiment analysis on {len(news_df)} news articles...")
    for idx, row in news_df.iterrows():
        res = analyzer.analyze_article(row["Headline"], row["Description"])
        sentiments.append(res["sentiment"])
        scores.append(res["sentiment_score"])
        reasonings.append(res["reasoning"])

    df_out = news_df.copy()
    df_out["Sentiment"] = sentiments
    df_out["Sentiment_Score"] = scores
    df_out["Reasoning"] = reasonings

    return df_out


def aggregate_daily_sentiment(news_sentiment_df: pd.DataFrame, full_date_range: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Aggregate individual news items into daily aggregated features.

    Features generated:
    - Avg_Sentiment: Mean sentiment score for the day
    - Pos_News_Count: Count of positive articles
    - Neg_News_Count: Count of negative articles
    - Neu_News_Count: Count of neutral articles
    - Sentiment_Momentum: 3-day rolling average of Avg_Sentiment
    """
    df = news_sentiment_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Group by Date
    daily_agg = df.groupby("Date").agg(
        Avg_Sentiment=("Sentiment_Score", "mean"),
        Pos_News_Count=("Sentiment", lambda x: (x == "Positive").sum()),
        Neg_News_Count=("Sentiment", lambda x: (x == "Negative").sum()),
        Neu_News_Count=("Sentiment", lambda x: (x == "Neutral").sum()),
    ).reset_index()

    # If full date range provided, reindex to match all trading days
    if full_date_range is not None:
        date_df = pd.DataFrame({"Date": pd.to_datetime(full_date_range)})
        daily_agg = pd.merge(date_df, daily_agg, on="Date", how="left")
        
        # Fill non-news days with neutral defaults
        daily_agg["Avg_Sentiment"] = daily_agg["Avg_Sentiment"].fillna(0.0)
        daily_agg["Pos_News_Count"] = daily_agg["Pos_News_Count"].fillna(0)
        daily_agg["Neg_News_Count"] = daily_agg["Neg_News_Count"].fillna(0)
        daily_agg["Neu_News_Count"] = daily_agg["Neu_News_Count"].fillna(0)

    # Calculate Sentiment Momentum (3-day rolling mean of Avg_Sentiment)
    daily_agg = daily_agg.sort_values("Date").reset_index(drop=True)
    daily_agg["Sentiment_Momentum"] = daily_agg["Avg_Sentiment"].rolling(window=3, min_periods=1).mean()

    # Format Date as string YYYY-MM-DD
    daily_agg["Date"] = daily_agg["Date"].dt.strftime("%Y-%m-%d")

    return daily_agg


if __name__ == "__main__":
    from src.news_generator import generate_synthetic_apple_news
    
    raw_news = generate_synthetic_apple_news(num_articles=5)
    analyzer = LLMSentimentAnalyzer(provider="mock")
    news_with_sentiment = process_news_sentiment(raw_news, analyzer)
    print("Sample Article Sentiment Output:")
    print(news_with_sentiment[["Headline", "Sentiment", "Sentiment_Score", "Reasoning"]])
