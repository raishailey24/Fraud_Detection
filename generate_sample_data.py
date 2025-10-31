"""
Generate sample transaction data for testing the fraud analytics dashboard.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(n_transactions: int = 10000, fraud_rate: float = 0.02):
    """
    Generate synthetic transaction data with fraud labels.
    
    Args:
        n_transactions: Number of transactions to generate
        fraud_rate: Proportion of fraudulent transactions
        
    Returns:
        DataFrame with sample transaction data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate timestamps (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    timestamps = [
        start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        for _ in range(n_transactions)
    ]
    
    # Merchants and categories
    merchants = [
        "amazon", "walmart", "target", "bestbuy", "starbucks",
        "mcdonalds", "shell", "exxon", "costco", "home depot",
        "apple store", "netflix", "spotify", "uber", "lyft",
        "airbnb", "booking.com", "expedia", "delta airlines", "marriott",
        "whole foods", "trader joes", "safeway", "cvs", "walgreens"
    ]
    
    categories = [
        "retail", "grocery", "gas", "restaurant", "entertainment",
        "travel", "utilities", "healthcare", "transportation", "online services"
    ]
    
    # Category-merchant mapping
    category_map = {
        "amazon": "retail", "walmart": "retail", "target": "retail",
        "bestbuy": "retail", "starbucks": "restaurant", "mcdonalds": "restaurant",
        "shell": "gas", "exxon": "gas", "costco": "grocery",
        "home depot": "retail", "apple store": "retail", "netflix": "entertainment",
        "spotify": "entertainment", "uber": "transportation", "lyft": "transportation",
        "airbnb": "travel", "booking.com": "travel", "expedia": "travel",
        "delta airlines": "travel", "marriott": "travel", "whole foods": "grocery",
        "trader joes": "grocery", "safeway": "grocery", "cvs": "healthcare",
        "walgreens": "healthcare"
    }
    
    # Generate base data
    data = {
        "transaction_id": [f"TXN{str(i).zfill(8)}" for i in range(n_transactions)],
        "timestamp": timestamps,
        "merchant": [random.choice(merchants) for _ in range(n_transactions)],
        "user_id": [f"USER{random.randint(1, 1000):04d}" for _ in range(n_transactions)],
        "location": [random.choice(["new york", "los angeles", "chicago", "houston", 
                                   "phoenix", "philadelphia", "san antonio", "san diego",
                                   "dallas", "san jose", "austin", "jacksonville",
                                   "online", "international"]) for _ in range(n_transactions)]
    }
    
    # Add categories based on merchants
    data["category"] = [category_map.get(m, random.choice(categories)) for m in data["merchant"]]
    
    # Generate amounts (log-normal distribution)
    base_amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_transactions)
    data["amount"] = np.round(base_amounts, 2)
    
    # Determine fraud cases
    n_fraud = int(n_transactions * fraud_rate)
    fraud_indices = np.random.choice(n_transactions, size=n_fraud, replace=False)
    data["is_fraud"] = np.zeros(n_transactions, dtype=int)
    data["is_fraud"][fraud_indices] = 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Make fraudulent transactions more suspicious
    fraud_mask = df["is_fraud"] == 1
    
    # Fraudulent transactions tend to be:
    # 1. Higher amounts
    df.loc[fraud_mask, "amount"] = df.loc[fraud_mask, "amount"] * np.random.uniform(2, 5, fraud_mask.sum())
    
    # 2. At unusual hours (late night)
    for idx in df[fraud_mask].index:
        hour = random.choice([0, 1, 2, 3, 4, 22, 23])
        df.at[idx, "timestamp"] = df.at[idx, "timestamp"].replace(hour=hour)
    
    # 3. More likely to be international or online
    df.loc[fraud_mask, "location"] = np.random.choice(
        ["international", "online"],
        size=fraud_mask.sum(),
        p=[0.6, 0.4]
    )
    
    # 4. Certain categories have higher fraud rates
    high_risk_categories = ["online services", "travel", "entertainment"]
    for cat in high_risk_categories:
        cat_mask = df["category"] == cat
        if cat_mask.sum() > 0:
            # Increase fraud rate for these categories
            additional_fraud = int(cat_mask.sum() * 0.03)
            if additional_fraud > 0:
                cat_indices = df[cat_mask & ~fraud_mask].sample(min(additional_fraud, cat_mask.sum())).index
                df.loc[cat_indices, "is_fraud"] = 1
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Round amounts
    df["amount"] = df["amount"].round(2)
    
    return df


def main():
    """Generate and save sample data."""
    print("Generating sample transaction data...")
    
    # Generate data
    df = generate_sample_data(n_transactions=10000, fraud_rate=0.02)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/sample_transactions.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Sample data generated successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Data Summary:")
    print(f"  - Total transactions: {len(df):,}")
    print(f"  - Fraudulent transactions: {df['is_fraud'].sum():,}")
    print(f"  - Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Total amount: ${df['amount'].sum():,.2f}")
    print(f"  - Fraud amount: ${df[df['is_fraud']==1]['amount'].sum():,.2f}")
    print(f"\n🔍 Sample records:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
