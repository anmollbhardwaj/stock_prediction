"""
Machine Learning Model Module for Stock Price Trend Prediction.
Implements Random Forest Classifier, chronological train/test split, feature scaling,
baseline comparison, and evaluation metrics reporting.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Daily_Return",
    "MA7",
    "MA21",
    "Volatility",
    "RSI",
    "MACD",
    "Volume",
    "Avg_Sentiment",
    "Pos_News_Count",
    "Neg_News_Count",
    "Neu_News_Count",
    "Sentiment_Momentum"
]


class StockTrendPredictor:
    """
    Random Forest Stock Price Next-Day Trend Predictor.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        self.feature_cols = FEATURE_COLUMNS
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False

    def chronological_split(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """
        Split dataset chronologically into Train and Test sets to avoid time-series data leakage.

        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset sorted by Date.
        train_ratio : float
            Proportion of data to use for training (default: 0.8 = 80% train, 20% test).

        Returns:
        --------
        X_train, X_test, y_train, y_test, train_df, test_df
        """
        df_sorted = df.sort_values("Date").reset_index(drop=True)
        split_idx = int(len(df_sorted) * train_ratio)

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()

        X_train = train_df[self.feature_cols]
        y_train = train_df["Target"]
        X_test = test_df[self.feature_cols]
        y_test = test_df["Target"]

        logger.info(f"Chronological Split: {len(train_df)} train samples ({train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]}) | {len(test_df)} test samples ({test_df['Date'].iloc[0]} to {test_df['Date'].iloc[-1]})")

        return X_train, X_test, y_train, y_test, train_df, test_df

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit Scaler and Random Forest Model on Training data ONLY.
        """
        # Fit scaler ONLY on X_train to prevent data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        logger.info("Random Forest Classifier successfully trained.")

    def predict(self, X: pd.DataFrame):
        """
        Predict binary next-day trend (1=UP, 0=DOWN).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame = None) -> dict:
        """
        Evaluate Random Forest model against test dataset and compare with Naive Baseline model.

        Baseline Model: Predicts next-day direction as identical to previous trading day's direction.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        # Calculate metrics for Random Forest
        rf_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred)
        }

        # Naive Baseline Evaluation (Same direction as previous trading day)
        baseline_metrics = {}
        if test_df is not None and "Baseline_Prev_Direction" in test_df.columns:
            y_base = test_df["Baseline_Prev_Direction"]
            baseline_metrics = {
                "Accuracy": accuracy_score(y_test, y_base),
                "Precision": precision_score(y_test, y_base, zero_division=0),
                "Recall": recall_score(y_test, y_base, zero_division=0),
                "F1-Score": f1_score(y_test, y_base, zero_division=0),
                "Confusion_Matrix": confusion_matrix(y_test, y_base)
            }

        return {
            "Random_Forest": rf_metrics,
            "Baseline": baseline_metrics,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get sorted feature importances.
        """
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": self.feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return fi_df

    def save(self, filepath: str = "models/random_forest.pkl"):
        """
        Save model and scaler state using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols
        }
        joblib.dump(state, filepath)
        logger.info(f"Model saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/random_forest.pkl"):
        """
        Load trained model and scaler from joblib pickle.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        state = joblib.load(filepath)
        instance = cls()
        instance.model = state["model"]
        instance.scaler = state["scaler"]
        instance.feature_cols = state["feature_cols"]
        instance.is_fitted = True
        logger.info(f"Model loaded successfully from {filepath}")
        return instance
