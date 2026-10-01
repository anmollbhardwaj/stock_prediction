"""
Synthetic News Generator for Apple Inc. (AAPL)
Generates a realistic set of financial and corporate news headlines + descriptions
covering 2021 through current dates for sentiment analysis integration.
"""

import os
import random
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logger = logging.getLogger(__name__)

# Template catalog of Apple news categories and templates
NEWS_TEMPLATES = [
    # Positive - Earnings & Revenue
    {
        "category": "Earnings",
        "sentiment_type": "Positive",
        "headline": "Apple reports record Q{quarter} revenue driven by strong iPhone sales",
        "description": "Quarterly earnings beat Wall Street estimates with net profit rising {pct}% year-over-year behind surge in premium smartphone demand."
    },
    {
        "category": "Earnings",
        "sentiment_type": "Positive",
        "headline": "Services division reaches all-time high revenue of ${val}B",
        "description": "App Store, iCloud, and Apple Pay subscriptions continue rapid expansion, boosting overall gross margins for Apple."
    },
    # Negative - Earnings & Revenue
    {
        "category": "Earnings",
        "sentiment_type": "Negative",
        "headline": "Apple Q{quarter} revenue misses estimates amid foreign exchange headwinds",
        "description": "Quarterly revenue fell short of consensus forecasts as strong US dollar and consumer spending slowdown impacted sales."
    },
    # Positive - AI & Technology
    {
        "category": "AI",
        "sentiment_type": "Positive",
        "headline": "Apple unveils breakthrough Apple Intelligence features across iOS",
        "description": "New generative AI features integrated deeply into iOS receive widespread acclaim from tech reviewers and industry analysts."
    },
    {
        "category": "AI",
        "sentiment_type": "Positive",
        "headline": "Apple expands partnership with leading AI research labs for next-gen Siri",
        "description": "Strategic collaboration aims to supercharge Siri capabilities with multi-modal LLM reasoning and privacy-first design."
    },
    # Negative - Regulatory & Legal
    {
        "category": "Regulatory",
        "sentiment_type": "Negative",
        "headline": "EU antitrust regulators slap Apple with major fine over App Store policies",
        "description": "European regulators order changes to anti-steering rules, potentially squeezing lucrative App Store fee margins."
    },
    {
        "category": "Regulatory",
        "sentiment_type": "Negative",
        "headline": "US Department of Justice files antitrust lawsuit against Apple",
        "description": "Federal prosecutors allege Apple holds an illegal monopoly over smartphone ecosystem, sparking investor caution."
    },
    # Negative - Supply Chain & China
    {
        "category": "Supply Chain",
        "sentiment_type": "Negative",
        "headline": "Foxconn factory delays impact iPhone Pro production ahead of holiday season",
        "description": "Supply chain disruptions in Asian manufacturing hubs threaten to disrupt Q4 delivery timelines for flagship devices."
    },
    {
        "category": "China Market",
        "sentiment_type": "Negative",
        "headline": "Apple iPhone shipments in China drop {pct}% amid intense local competition",
        "description": "Aggressive product offerings from Huawei and rival Chinese OEMs squeeze Apple market share in key Asian market."
    },
    # Positive - China & International Growth
    {
        "category": "China Market",
        "sentiment_type": "Positive",
        "headline": "Apple rebounds in China with aggressive promotional campaign and trade-ins",
        "description": "Surge in device upgrades during national shopping festival drives double-digit market share gains in mainland China."
    },
    # Positive - Product Launch & Hardware
    {
        "category": "Product Launch",
        "sentiment_type": "Positive",
        "headline": "Apple introduces next-generation M-series chips powering new Mac lineup",
        "description": "Benchmarking tests demonstrate industry-leading performance per watt, triggering strong upgrade cycle interest among enterprise users."
    },
    {
        "category": "Product Launch",
        "sentiment_type": "Positive",
        "headline": "Pre-orders for flagship iPhone 16 Pro exceed initial supply estimates",
        "description": "Lead times for high-end models stretch to 4 weeks, indicating robust consumer appetite for camera and battery upgrades."
    },
    # Negative - Product / Hardware Issue
    {
        "category": "Hardware Issue",
        "sentiment_type": "Negative",
        "headline": "Users report thermal throttling issue on newly launched device model",
        "description": "Early adopter complaints on social forums regarding device heating prompt Apple software team to prepare hotfix patch."
    },
    # Positive - Analyst Opinions & Upgrades
    {
        "category": "Analyst Upgrade",
        "sentiment_type": "Positive",
        "headline": "Wall Street firm upgrades Apple to Top Pick with ${target_price} target",
        "description": "Analysts cite multi-year hardware refresh cycle, expanding software margins, and strong balance sheet cash flow."
    },
    # Negative - Analyst Downgrade
    {
        "category": "Analyst Downgrade",
        "sentiment_type": "Negative",
        "headline": "Major investment bank downgrades Apple rating citing valuation concerns",
        "description": "Report notes slowing hardware upgrade cadence and high price-to-earnings multiple relative to historical averages."
    },
    # Neutral - Corporate & Strategy
    {
        "category": "Corporate",
        "sentiment_type": "Neutral",
        "headline": "Apple announces annual developer conference schedule for June",
        "description": "Worldwide Developers Conference (WWDC) set to focus on software OS updates, developer SDKs, and platform tools."
    },
    {
        "category": "Corporate",
        "sentiment_type": "Neutral",
        "headline": "Apple issues ${val}B multi-tranche bond offering for share buybacks",
        "description": "Corporate debt issuance structured to optimize capital allocation, dividend payouts, and routine share repurchases."
    },
    {
        "category": "Executive",
        "sentiment_type": "Neutral",
        "headline": "CEO Tim Cook highlights long-term R&D investments in renewable energy and recycling",
        "description": "Keynote address underscores progress toward 2030 carbon neutrality goals across global manufacturing supply chain."
    }
]


def generate_synthetic_apple_news(
    start_date: str = "2021-01-01",
    end_date: str = "2026-09-30",
    num_articles: int = 90,
    save_path: str = "data/synthetic_news.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic news dataset for Apple Inc. (AAPL).

    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    num_articles : int
        Target number of synthetic news items to generate.
    save_path : str
        Path to save CSV output.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Date, Headline, Description, Category, Target_Sentiment
    """
    random.seed(seed)
    np.random.seed(seed)

    # Generate dates range (sample business/trading days)
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    sampled_dates = sorted(random.sample(list(date_range), min(num_articles, len(date_range))))

    articles = []
    for d in sampled_dates:
        date_str = d.strftime("%Y-%m-%d")
        tmpl = random.choice(NEWS_TEMPLATES)
        
        # Fill template variables dynamically
        q = random.choice([1, 2, 3, 4])
        pct = random.randint(4, 18)
        val = random.randint(18, 28)
        tp = random.randint(220, 290)

        headline = tmpl["headline"].format(quarter=q, pct=pct, val=val, target_price=tp)
        description = tmpl["description"].format(quarter=q, pct=pct, val=val, target_price=tp)

        articles.append({
            "Date": date_str,
            "Headline": headline,
            "Description": description,
            "Category": tmpl["category"],
            "Target_Sentiment": tmpl["sentiment_type"]
        })

    df = pd.DataFrame(articles)
    df = df.sort_values("Date").reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Generated and saved {len(df)} synthetic news articles to {save_path}")

    return df


if __name__ == "__main__":
    news_df = generate_synthetic_apple_news()
    print("Sample generated news:")
    print(news_df.head())
