"""
Create the complete 13M+ transaction dataset from user's data.
Optimized processing with progress tracking and memory management.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
import time
import gc


def load_mcc_codes(mcc_file_path):
    """Load MCC codes mapping from JSON file."""
    print("Loading MCC codes...")
    with open(mcc_file_path, 'r') as f:
        mcc_codes = json.load(f)
    print(f"[OK] Loaded {len(mcc_codes)} MCC codes")
    return mcc_codes


def load_fraud_labels_optimized(fraud_file_path):
    """Load fraud labels from JSON file with optimization."""
    print("Loading fraud labels...")
    start_time = time.time()
    
    with open(fraud_file_path, 'r') as f:
        fraud_data = json.load(f)
    
    print(f"[OK] JSON loaded in {time.time() - start_time:.1f}s")
    print("Converting to DataFrame...")
    
    # More efficient conversion
    fraud_dict = fraud_data['target']
    transaction_ids = [int(tid) for tid in fraud_dict.keys()]
    fraud_values = [1 if label == 'Yes' else 0 for label in fraud_dict.values()]
    
    fraud_labels = pd.DataFrame({
        'transaction_id': transaction_ids,
        'is_fraud': fraud_values
    })
    
    print(f"[OK] Loaded {len(fraud_labels)} fraud labels")
    fraud_rate = fraud_labels['is_fraud'].mean() * 100
    print(f"  Fraud rate: {fraud_rate:.2f}%")
    return fraud_labels


def process_transactions_chunk(chunk, mcc_codes, chunk_num):
    """Process a single chunk of transactions."""
    print(f"  Processing chunk {chunk_num} ({len(chunk):,} rows)...")
    
    # Category mapping for better visualization
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
    
    # Clean amount column
    chunk['amount_clean'] = chunk['amount'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Filter valid transactions
    chunk = chunk[chunk['amount_clean'] > 0]
    
    # Create required columns
    chunk_processed = pd.DataFrame({
        'transaction_id': chunk['id'].astype(int),
        'timestamp': pd.to_datetime(chunk['date']),
        'amount': chunk['amount_clean'],
        'user_id': chunk['client_id'].astype(str),
        'merchant': chunk.apply(
            lambda row: f"merchant_{row['merchant_id']}" if pd.isna(row['merchant_city']) or row['merchant_city'] == 'ONLINE'
            else f"{row['merchant_city']}_merchant",
            axis=1
        ),
        'location': chunk.apply(
            lambda row: f"{row['merchant_city']}, {row['merchant_state']}" 
            if pd.notna(row['merchant_city']) and pd.notna(row['merchant_state']) and row['merchant_city'] != 'ONLINE'
            else "online",
            axis=1
        ),
        'mcc': chunk['mcc'].astype(str)
    })
    
    # Map categories
    chunk_processed['category'] = chunk_processed['mcc'].map(mcc_codes).fillna('other')
    chunk_processed['category'] = chunk_processed['category'].replace(category_mapping)
    
    # Add temporal features
    chunk_processed['hour'] = chunk_processed['timestamp'].dt.hour
    chunk_processed['day_of_week'] = chunk_processed['timestamp'].dt.dayofweek
    chunk_processed['is_weekend'] = chunk_processed['day_of_week'].isin([5, 6]).astype(int)
    chunk_processed['is_night'] = chunk_processed['hour'].between(22, 6).astype(int)
    
    # Add amount features
    chunk_processed['amount_log'] = np.log1p(chunk_processed['amount'])
    
    # Drop MCC column
    chunk_processed = chunk_processed.drop('mcc', axis=1)
    
    print(f"    Processed: {len(chunk_processed):,} valid transactions")
    return chunk_processed


def add_statistical_features(df):
    """Add merchant and category statistics to the dataset."""
    print("Adding statistical features...")
    
    # Calculate merchant stats
    print("  Computing merchant statistics...")
    merchant_stats = df.groupby('merchant').agg({
        'amount': ['mean', 'count'],
        'is_fraud': 'mean'
    }).reset_index()
    merchant_stats.columns = ['merchant', 'merchant_avg_amount', 'merchant_tx_count', 'merchant_fraud_rate']
    df = df.merge(merchant_stats, on='merchant', how='left')
    
    # Calculate category stats
    print("  Computing category statistics...")
    category_stats = df.groupby('category').agg({
        'amount': 'mean',
        'is_fraud': 'mean'
    }).reset_index()
    category_stats.columns = ['category', 'category_avg_amount', 'category_fraud_rate']
    df = df.merge(category_stats, on='category', how='left')
    
    return df


def calculate_risk_scores(df):
    """Calculate risk scores and levels for all transactions."""
    print("Calculating risk scores...")
    
    # Calculate amount z-score
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
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
    
    # High frequency user risk
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
    
    return df


def create_full_dataset():
    """Create the complete 13M+ dataset."""
    data_dir = r"C:\Users\shail\OneDrive\Desktop\BizViz Challenge\Financial Transactions Datasets"
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("CREATING COMPLETE 13M+ TRANSACTION DATASET")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load supporting data
    mcc_codes = load_mcc_codes(data_path / "mcc_codes.json")
    fraud_labels = load_fraud_labels_optimized(data_path / "train_fraud_labels.json")
    
    # Process transactions in chunks
    print("\nProcessing transactions in chunks...")
    chunk_size = 200000  # Process 200k records at a time
    processed_chunks = []
    total_processed = 0
    
    transactions_file = data_path / "transactions_data.csv"
    
    for chunk_num, chunk in enumerate(pd.read_csv(transactions_file, chunksize=chunk_size), 1):
        chunk_processed = process_transactions_chunk(chunk, mcc_codes, chunk_num)
        processed_chunks.append(chunk_processed)
        total_processed += len(chunk_processed)
        
        # Memory management - combine chunks periodically
        if len(processed_chunks) >= 5:
            print("  Combining chunks to manage memory...")
            combined_chunk = pd.concat(processed_chunks, ignore_index=True)
            processed_chunks = [combined_chunk]
            gc.collect()
        
        print(f"  Total processed so far: {total_processed:,} transactions")
    
    # Combine all chunks
    print("Combining all processed chunks...")
    df = pd.concat(processed_chunks, ignore_index=True)
    del processed_chunks
    gc.collect()
    
    print(f"[OK] Combined dataset: {len(df):,} transactions")
    
    # Join with fraud labels
    print("Adding fraud labels...")
    print(f"  Transactions before join: {len(df):,}")
    print(f"  Fraud labels available: {len(fraud_labels):,}")
    
    df = df.merge(fraud_labels, on='transaction_id', how='left')
    df['is_fraud'] = df['is_fraud'].fillna(0).astype(int)
    
    print(f"  Transactions after join: {len(df):,}")
    
    # Add statistical features
    df = add_statistical_features(df)
    
    # Calculate risk scores
    df = calculate_risk_scores(df)
    
    # Convert transaction_id back to string for dashboard compatibility
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Final cleanup
    print("Final data cleanup...")
    initial_count = len(df)
    
    # Remove rows with missing required fields
    df = df.dropna(subset=['transaction_id', 'timestamp', 'amount', 'merchant', 'category'])
    
    final_count = len(df)
    removed_count = initial_count - final_count
    
    print(f"[OK] Removed {removed_count:,} invalid transactions")
    print(f"[OK] Final dataset: {final_count:,} transactions")
    
    # Dataset summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total transactions: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique merchants: {df['merchant'].nunique():,}")
    print(f"Categories: {', '.join(sorted(df['category'].unique()))}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():,.2f}")
    
    # Risk level distribution
    risk_dist = df['risk_level'].value_counts()
    print(f"\nRisk Level Distribution:")
    for level in ['low', 'medium', 'high']:
        count = risk_dist.get(level, 0)
        pct = (count / len(df)) * 100
        print(f"  {level.capitalize()}: {count:,} ({pct:.1f}%)")
    
    # Save dataset
    print("\nSaving complete dataset...")
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "complete_user_transactions.csv"
    
    df.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    
    print(f"\n[SUCCESS] Complete dataset created!")
    print(f"Processing time: {total_time/60:.1f} minutes")
    print(f"Dataset saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return df


if __name__ == "__main__":
    create_full_dataset()
