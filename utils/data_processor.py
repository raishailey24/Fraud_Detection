"""
Data cleaning and preprocessing utilities.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


class DataProcessor:
    """Handle data cleaning and preprocessing."""
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # Standardize is_fraud to integer
        if "is_fraud" in df.columns:
            df["is_fraud"] = df["is_fraud"].astype(int)
        
        # Clean amount column
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["amount"] = df["amount"].abs()  # Ensure positive values
        
        # Clean text columns
        text_columns = ["merchant", "category", "location"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Remove duplicates based on transaction_id
        if "transaction_id" in df.columns:
            df = df.drop_duplicates(subset=["transaction_id"], keep="first")
        
        # Handle missing values
        df = DataProcessor._handle_missing_values(df)
        
        return df
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Drop rows with missing critical fields
        critical_fields = ["transaction_id", "timestamp", "amount", "is_fraud"]
        existing_critical = [col for col in critical_fields if col in df.columns]
        df = df.dropna(subset=existing_critical)
        
        # Fill missing categorical values
        categorical_cols = ["merchant", "category", "location"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")
        
        return df
    
    @staticmethod
    def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for fraud analysis.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Temporal features
        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["day_of_month"] = df["timestamp"].dt.day
            df["month"] = df["timestamp"].dt.month
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_night"] = df["hour"].between(22, 6).astype(int)
        
        # Amount-based features
        if "amount" in df.columns:
            df["amount_log"] = np.log1p(df["amount"])
            df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
            
            # Amount categories
            df["amount_category"] = pd.cut(
                df["amount"],
                bins=[0, 50, 200, 1000, float("inf")],
                labels=["small", "medium", "large", "very_large"]
            )
        
        # Merchant-based features
        if "merchant" in df.columns:
            merchant_stats = df.groupby("merchant").agg({
                "amount": ["mean", "std", "count"],
                "is_fraud": "mean"
            }).reset_index()
            merchant_stats.columns = ["merchant", "merchant_avg_amount", "merchant_std_amount", 
                                     "merchant_tx_count", "merchant_fraud_rate"]
            df = df.merge(merchant_stats, on="merchant", how="left")
        
        # Category-based features
        if "category" in df.columns:
            category_stats = df.groupby("category").agg({
                "amount": "mean",
                "is_fraud": "mean"
            }).reset_index()
            category_stats.columns = ["category", "category_avg_amount", "category_fraud_rate"]
            df = df.merge(category_stats, on="category", how="left")
        
        # User behavior features (if user_id exists)
        if "user_id" in df.columns:
            user_stats = df.groupby("user_id").agg({
                "amount": ["mean", "std", "sum", "count"],
                "is_fraud": "sum"
            }).reset_index()
            user_stats.columns = ["user_id", "user_avg_amount", "user_std_amount", 
                                 "user_total_amount", "user_tx_count", "user_fraud_count"]
            df = df.merge(user_stats, on="user_id", how="left")
            
            # Transaction velocity
            df = df.sort_values(["user_id", "timestamp"])
            df["time_since_last_tx"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds() / 60
            df["time_since_last_tx"] = df["time_since_last_tx"].fillna(0)
        
        # Risk score calculation
        df = DataProcessor._calculate_risk_score(df)
        
        return df
    
    @staticmethod
    def _calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a composite risk score for each transaction."""
        df = df.copy()
        risk_score = np.zeros(len(df))
        
        # High amount risk
        if "amount_zscore" in df.columns:
            risk_score += np.clip(df["amount_zscore"] / 10, 0, 0.3)
        
        # Merchant fraud rate risk
        if "merchant_fraud_rate" in df.columns:
            risk_score += df["merchant_fraud_rate"].fillna(0) * 0.3
        
        # Category fraud rate risk
        if "category_fraud_rate" in df.columns:
            risk_score += df["category_fraud_rate"].fillna(0) * 0.2
        
        # Night transaction risk
        if "is_night" in df.columns:
            risk_score += df["is_night"] * 0.1
        
        # High velocity risk
        if "time_since_last_tx" in df.columns:
            velocity_risk = (df["time_since_last_tx"] < 5).astype(float) * 0.1
            risk_score += velocity_risk
        
        df["risk_score"] = np.clip(risk_score, 0, 1)
        df["risk_level"] = pd.cut(
            df["risk_score"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=["low", "medium", "high"]
        )
        
        return df
    
    @staticmethod
    def get_aggregated_metrics(df: pd.DataFrame) -> dict:
        """
        Calculate aggregated metrics for AI analysis (no PII).
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of aggregated metrics
        """
        metrics = {}
        
        # Overall statistics
        metrics["total_transactions"] = len(df)
        metrics["total_fraud_cases"] = int(df["is_fraud"].sum())
        metrics["fraud_rate"] = float(df["is_fraud"].mean())
        metrics["total_amount"] = float(df["amount"].sum())
        metrics["total_fraud_amount"] = float(df[df["is_fraud"] == 1]["amount"].sum())
        
        # Amount statistics
        metrics["avg_transaction_amount"] = float(df["amount"].mean())
        metrics["median_transaction_amount"] = float(df["amount"].median())
        metrics["avg_fraud_amount"] = float(df[df["is_fraud"] == 1]["amount"].mean()) if metrics["total_fraud_cases"] > 0 else 0
        metrics["avg_legitimate_amount"] = float(df[df["is_fraud"] == 0]["amount"].mean())
        
        # Temporal patterns
        if "hour" in df.columns:
            fraud_by_hour = df[df["is_fraud"] == 1].groupby("hour").size()
            metrics["peak_fraud_hour"] = int(fraud_by_hour.idxmax()) if len(fraud_by_hour) > 0 else None
        
        if "day_of_week" in df.columns:
            fraud_by_day = df[df["is_fraud"] == 1].groupby("day_of_week").size()
            metrics["peak_fraud_day"] = int(fraud_by_day.idxmax()) if len(fraud_by_day) > 0 else None
        
        # Category analysis
        if "category" in df.columns:
            category_fraud = df.groupby("category")["is_fraud"].agg(["sum", "mean", "count"])
            top_fraud_category = category_fraud.sort_values("sum", ascending=False).head(1)
            if len(top_fraud_category) > 0:
                metrics["highest_fraud_category"] = top_fraud_category.index[0]
                metrics["highest_fraud_category_rate"] = float(top_fraud_category["mean"].values[0])
        
        # Risk distribution
        if "risk_level" in df.columns:
            risk_dist = df["risk_level"].value_counts(normalize=True).to_dict()
            metrics["risk_distribution"] = {str(k): float(v) for k, v in risk_dist.items()}
        
        # Detection metrics
        if "risk_score" in df.columns:
            metrics["avg_risk_score_fraud"] = float(df[df["is_fraud"] == 1]["risk_score"].mean()) if metrics["total_fraud_cases"] > 0 else 0
            metrics["avg_risk_score_legitimate"] = float(df[df["is_fraud"] == 0]["risk_score"].mean())
        
        # Enhanced metrics for better AI analysis
        metrics["data_quality"] = DataProcessor._get_data_quality_metrics(df)
        metrics["temporal_patterns"] = DataProcessor._get_temporal_patterns(df)
        metrics["merchant_insights"] = DataProcessor._get_merchant_insights(df)
        metrics["amount_patterns"] = DataProcessor._get_amount_patterns(df)
        
        return metrics
    
    @staticmethod
    def _get_data_quality_metrics(df: pd.DataFrame) -> dict:
        """Get data quality metrics."""
        return {
            "total_records": len(df),
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "duplicate_records": df.duplicated().sum(),
            "data_completeness": ((len(df) * len(df.columns) - df.isnull().sum().sum()) / (len(df) * len(df.columns))) * 100
        }
    
    @staticmethod
    def _get_temporal_patterns(df: pd.DataFrame) -> dict:
        """Get detailed temporal patterns."""
        patterns = {}
        
        if "timestamp" in df.columns:
            # Hourly fraud distribution
            if "hour" in df.columns:
                hourly_fraud = df[df["is_fraud"] == 1].groupby("hour").size()
                patterns["peak_fraud_hours"] = hourly_fraud.nlargest(3).to_dict()
                patterns["low_fraud_hours"] = hourly_fraud.nsmallest(3).to_dict()
            
            # Daily fraud distribution
            if "day_of_week" in df.columns:
                daily_fraud = df[df["is_fraud"] == 1].groupby("day_of_week").size()
                patterns["fraud_by_day"] = daily_fraud.to_dict()
            
            # Weekend vs weekday patterns
            if "is_weekend" in df.columns:
                weekend_fraud_rate = df[df["is_weekend"] == 1]["is_fraud"].mean()
                weekday_fraud_rate = df[df["is_weekend"] == 0]["is_fraud"].mean()
                patterns["weekend_vs_weekday"] = {
                    "weekend_fraud_rate": weekend_fraud_rate,
                    "weekday_fraud_rate": weekday_fraud_rate,
                    "weekend_risk_multiplier": weekend_fraud_rate / weekday_fraud_rate if weekday_fraud_rate > 0 else 0
                }
        
        return patterns
    
    @staticmethod
    def _get_merchant_insights(df: pd.DataFrame) -> dict:
        """Get merchant-specific insights."""
        insights = {}
        
        if "merchant" in df.columns:
            merchant_stats = df.groupby("merchant").agg({
                "is_fraud": ["count", "sum", "mean"],
                "amount": ["mean", "sum"]
            }).round(4)
            
            merchant_stats.columns = ["total_transactions", "fraud_count", "fraud_rate", "avg_amount", "total_amount"]
            
            # Top risky merchants
            risky_merchants = merchant_stats[merchant_stats["total_transactions"] >= 10].nlargest(5, "fraud_rate")
            insights["top_risky_merchants"] = risky_merchants.to_dict("index")
            
            # High volume merchants
            volume_merchants = merchant_stats.nlargest(5, "total_transactions")
            insights["high_volume_merchants"] = volume_merchants.to_dict("index")
            
            # Merchant statistics
            insights["merchant_summary"] = {
                "total_merchants": len(merchant_stats),
                "merchants_with_fraud": len(merchant_stats[merchant_stats["fraud_count"] > 0]),
                "avg_merchant_fraud_rate": merchant_stats["fraud_rate"].mean()
            }
        
        return insights
    
    @staticmethod
    def _get_amount_patterns(df: pd.DataFrame) -> dict:
        """Get amount-related patterns."""
        patterns = {}
        
        if "amount" in df.columns:
            fraud_amounts = df[df["is_fraud"] == 1]["amount"]
            legit_amounts = df[df["is_fraud"] == 0]["amount"]
            
            patterns["fraud_amount_stats"] = {
                "mean": fraud_amounts.mean(),
                "median": fraud_amounts.median(),
                "std": fraud_amounts.std(),
                "min": fraud_amounts.min(),
                "max": fraud_amounts.max(),
                "q25": fraud_amounts.quantile(0.25),
                "q75": fraud_amounts.quantile(0.75)
            }
            
            patterns["legitimate_amount_stats"] = {
                "mean": legit_amounts.mean(),
                "median": legit_amounts.median(),
                "std": legit_amounts.std(),
                "min": legit_amounts.min(),
                "max": legit_amounts.max(),
                "q25": legit_amounts.quantile(0.25),
                "q75": legit_amounts.quantile(0.75)
            }
            
            # Amount ranges analysis
            if "amount_category" in df.columns:
                amount_fraud_by_category = df.groupby("amount_category")["is_fraud"].agg(["count", "sum", "mean"])
                patterns["fraud_by_amount_category"] = amount_fraud_by_category.to_dict("index")
        
        return patterns
