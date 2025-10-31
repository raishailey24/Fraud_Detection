"""
Create a sample dataset from user's data for quick testing.
Takes first 50,000 transactions to test dashboard functionality.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path


def create_sample_dataset():
    """Create a sample dataset for testing."""
    data_dir = r"C:\Users\shail\OneDrive\Desktop\BizViz Challenge\Financial Transactions Datasets"
    data_path = Path(data_dir)
    
    print("Creating sample dataset from user data...")
    print("=" * 50)
    
    # Load MCC codes
    print("Loading MCC codes...")
    with open(data_path / "mcc_codes.json", 'r') as f:
        mcc_codes = json.load(f)
    print(f"[OK] Loaded {len(mcc_codes)} MCC codes")
    
    # Load sample of transactions (first 50,000)
    print("Loading sample transactions (50,000 rows)...")
    transactions = pd.read_csv(data_path / "transactions_data.csv", nrows=50000)
    print(f"[OK] Loaded {len(transactions):,} transactions")
    
    # Load fraud labels
    print("Loading fraud labels...")
    with open(data_path / "train_fraud_labels.json", 'r') as f:
        fraud_data = json.load(f)
    
    # Convert to DataFrame
    fraud_dict = fraud_data['target']
    fraud_labels = pd.DataFrame([
        {'transaction_id': int(tid), 'is_fraud': 1 if label == 'Yes' else 0}
        for tid, label in fraud_dict.items()
    ])
    print(f"[OK] Loaded fraud labels")
    
    # Process transactions
    print("Processing transactions...")
    
    # Clean amount
    transactions['amount_clean'] = transactions['amount'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Filter valid transactions
    transactions = transactions[transactions['amount_clean'] > 0]
    
    # Create dashboard format
    df = pd.DataFrame({
        'transaction_id': transactions['id'].astype(int),
        'timestamp': pd.to_datetime(transactions['date']),
        'amount': transactions['amount_clean'],
        'user_id': transactions['client_id'].astype(str),
        'merchant': transactions.apply(
            lambda row: f"merchant_{row['merchant_id']}" if pd.isna(row['merchant_city']) or row['merchant_city'] == 'ONLINE'
            else f"{row['merchant_city']}_merchant",
            axis=1
        ),
        'location': transactions.apply(
            lambda row: f"{row['merchant_city']}, {row['merchant_state']}" 
            if pd.notna(row['merchant_city']) and pd.notna(row['merchant_state']) and row['merchant_city'] != 'ONLINE'
            else "online",
            axis=1
        ),
        'mcc': transactions['mcc'].astype(str)
    })
    
    # Map categories
    df['category'] = df['mcc'].map(mcc_codes).fillna('other')
    
    # Simplify categories
    category_mapping = {
        'Eating Places and Restaurants': 'restaurant',
        'Fast Food Restaurants': 'restaurant', 
        'Service Stations': 'gas',
        'Grocery Stores, Supermarkets': 'grocery',
        'Department Stores': 'retail',
        'Discount Stores': 'retail',
        'Utilities - Electric, Gas, Water, Sanitary': 'utilities',
        'Money Transfer': 'financial',
        'Tolls and Bridge Fees': 'transportation',
        'Book Stores': 'retail',
        'Miscellaneous Food Stores': 'grocery',
        'Taxicabs and Limousines': 'transportation',
        'Wholesale Clubs': 'retail',
        'Miscellaneous Home Furnishing Stores': 'retail',
        'Motion Picture Theaters': 'entertainment',
        'Drinking Places (Alcoholic Beverages)': 'restaurant',
        'Amusement Parks, Carnivals, Circuses': 'entertainment',
        'Lumber and Building Materials': 'retail',
        'Computer Network Services': 'technology'
    }
    
    df['category'] = df['category'].replace(category_mapping)
    df = df.drop('mcc', axis=1)
    
    # Add fraud labels
    print("Adding fraud labels...")
    df = df.merge(fraud_labels, on='transaction_id', how='left')
    df['is_fraud'] = df['is_fraud'].fillna(0).astype(int)
    
    # Convert transaction_id back to string for dashboard compatibility
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Add missing columns that dashboard expects
    print("Adding dashboard-required columns...")
    
    # Add temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(22, 6).astype(int)
    
    # Add amount features
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
    # Calculate merchant stats
    merchant_stats = df.groupby('merchant').agg({
        'amount': ['mean', 'count'],
        'is_fraud': 'mean'
    }).reset_index()
    merchant_stats.columns = ['merchant', 'merchant_avg_amount', 'merchant_tx_count', 'merchant_fraud_rate']
    df = df.merge(merchant_stats, on='merchant', how='left')
    
    # Calculate category stats
    category_stats = df.groupby('category').agg({
        'amount': 'mean',
        'is_fraud': 'mean'
    }).reset_index()
    category_stats.columns = ['category', 'category_avg_amount', 'category_fraud_rate']
    df = df.merge(category_stats, on='category', how='left')
    
    # Calculate risk score with improved scoring
    risk_score = np.zeros(len(df))
    
    # High amount risk (more aggressive scoring)
    risk_score += np.clip(df['amount_zscore'] / 3, 0, 0.4)
    
    # Merchant fraud rate risk
    risk_score += df['merchant_fraud_rate'].fillna(0) * 0.5
    
    # Category fraud rate risk  
    risk_score += df['category_fraud_rate'].fillna(0) * 0.3
    
    # Night transaction risk
    risk_score += df['is_night'] * 0.15
    
    # Weekend risk
    risk_score += df['is_weekend'] * 0.05
    
    # High frequency user risk (if user has many transactions)
    user_tx_counts = df.groupby('user_id').size()
    high_freq_users = user_tx_counts[user_tx_counts > user_tx_counts.quantile(0.95)].index
    risk_score += df['user_id'].isin(high_freq_users).astype(float) * 0.1
    
    # Fraud label boost (for known fraud cases)
    risk_score += df['is_fraud'] * 0.3
    
    # Clip risk score and create risk level
    df['risk_score'] = np.clip(risk_score, 0, 1)
    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    
    # Fill any remaining NaN values in risk_level
    df['risk_level'] = df['risk_level'].fillna('low').astype(str)
    
    # Final cleanup
    df = df.dropna(subset=['transaction_id', 'timestamp', 'amount', 'merchant', 'category'])
    
    print(f"[OK] Final sample dataset: {len(df):,} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Categories: {', '.join(sorted(df['category'].unique()))}")
    
    # Save sample
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "user_sample_transactions.csv"
    
    df.to_csv(output_file, index=False)
    
    print(f"\n[SUCCESS] Sample dataset created!")
    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nYou can now test this with your dashboard!")


if __name__ == "__main__":
    create_sample_dataset()
