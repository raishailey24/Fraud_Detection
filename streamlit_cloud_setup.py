"""
Streamlit Cloud Setup Script
Generates sample data when deployed to Streamlit Cloud
"""
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

def ensure_sample_data():
    """Ensure sample data exists for Streamlit Cloud deployment."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_file = data_dir / "sample_transactions.csv"
    
    if not sample_file.exists():
        st.info("🔄 Generating realistic sample data for demo...")
        st.info("📊 Creating 100,000 transaction records with fraud patterns...")
        
        # Generate sample transaction data
        np.random.seed(42)
        n_transactions = 100000
        
        # Generate realistic transaction data
        data = {
            'transaction_id': [f'TXN{str(i).zfill(8)}' for i in range(1, n_transactions + 1)],
            'timestamp': pd.date_range('2023-01-01', periods=n_transactions, freq='1H'),
            'amount': np.random.lognormal(3, 1, n_transactions).round(2),
            'user_id': [f'USER{np.random.randint(1000, 9999)}' for _ in range(n_transactions)],
            'merchant': np.random.choice([
                'Amazon', 'Walmart', 'Target', 'Starbucks', 'McDonalds', 
                'Shell', 'Exxon', 'Best Buy', 'Home Depot', 'CVS'
            ], n_transactions),
            'location': np.random.choice([
                'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
                'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA',
                'Dallas, TX', 'San Jose, CA', 'International', 'Online'
            ], n_transactions),
            'category': np.random.choice([
                'retail', 'restaurant', 'gas', 'grocery', 'entertainment',
                'healthcare', 'transportation', 'financial', 'utilities'
            ], n_transactions),
            'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.98, 0.02])
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(list(range(22, 24)) + list(range(0, 6))).astype(int)
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Merchant statistics
        merchant_stats = df.groupby('merchant').agg({
            'amount': 'mean',
            'transaction_id': 'count',
            'is_fraud': 'mean'
        }).reset_index()
        merchant_stats.columns = ['merchant', 'merchant_avg_amount', 'merchant_tx_count', 'merchant_fraud_rate']
        
        df = df.merge(merchant_stats, on='merchant', how='left')
        
        # Category statistics
        category_stats = df.groupby('category').agg({
            'amount': 'mean',
            'is_fraud': 'mean'
        }).reset_index()
        category_stats.columns = ['category', 'category_avg_amount', 'category_fraud_rate']
        
        df = df.merge(category_stats, on='category', how='left')
        
        # Risk scoring
        df['risk_score'] = (
            df['amount_zscore'] * 0.3 +
            df['merchant_fraud_rate'] * 0.4 +
            df['category_fraud_rate'] * 0.2 +
            df['is_night'] * 0.1
        ).clip(0, 1)
        
        df['risk_level'] = pd.cut(
            df['risk_score'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['low', 'medium', 'high']
        )
        
        # Save to CSV
        df.to_csv(sample_file, index=False)
        st.success(f"✅ Generated {len(df):,} sample transactions for demo")
        
    return sample_file

if __name__ == "__main__":
    ensure_sample_data()
