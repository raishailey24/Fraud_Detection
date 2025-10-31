"""
Optimized data preprocessing script for large datasets.
Processes data in chunks with progress updates.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
import time


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


def process_transactions_optimized(transactions_file, mcc_codes, chunk_size=100000):
    """Process transactions in chunks for memory efficiency."""
    print(f"Processing transactions in chunks of {chunk_size:,}...")
    
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
    
    processed_chunks = []
    total_rows = 0
    
    # Read in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(transactions_file, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_num + 1} ({len(chunk):,} rows)...")
        
        # Clean amount column
        chunk['amount_clean'] = chunk['amount'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Filter valid transactions
        chunk = chunk[chunk['amount_clean'] > 0]
        
        # Create required columns
        chunk_processed = pd.DataFrame({
            'transaction_id': chunk['id'].astype(str),
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
        
        # Drop MCC column
        chunk_processed = chunk_processed.drop('mcc', axis=1)
        
        processed_chunks.append(chunk_processed)
        total_rows += len(chunk_processed)
        
        print(f"  Processed: {len(chunk_processed):,} valid transactions")
        
        # Memory management - limit chunks in memory
        if len(processed_chunks) >= 10:
            print("  Combining chunks...")
            combined_chunk = pd.concat(processed_chunks, ignore_index=True)
            processed_chunks = [combined_chunk]
    
    print("Combining all processed chunks...")
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"[OK] Processed {total_rows:,} transactions")
    return final_df


def combine_datasets_optimized(data_dir):
    """Optimized version for large datasets."""
    data_path = Path(data_dir)
    
    # File paths
    transactions_file = data_path / "transactions_data.csv"
    mcc_file = data_path / "mcc_codes.json"
    fraud_file = data_path / "train_fraud_labels.json"
    
    print("=" * 60)
    print("COMBINING USER DATASETS (OPTIMIZED)")
    print("=" * 60)
    
    # Load supporting data
    mcc_codes = load_mcc_codes(mcc_file)
    fraud_labels = load_fraud_labels_optimized(fraud_file)
    
    # Process transactions in chunks
    print("\nProcessing transactions...")
    df = process_transactions_optimized(transactions_file, mcc_codes)
    
    # Join with fraud labels
    print("Adding fraud labels...")
    print(f"  Transactions before join: {len(df):,}")
    print(f"  Fraud labels available: {len(fraud_labels):,}")
    
    df = df.merge(fraud_labels, on='transaction_id', how='left')
    df['is_fraud'] = df['is_fraud'].fillna(0).astype(int)
    
    print(f"  Transactions after join: {len(df):,}")
    
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
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total transactions: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique merchants: {df['merchant'].nunique():,}")
    print(f"Categories: {', '.join(sorted(df['category'].unique()))}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():,.2f}")
    
    return df


def main():
    """Main function with optimized processing."""
    # Data directory
    data_dir = r"C:\Users\shail\OneDrive\Desktop\BizViz Challenge\Financial Transactions Datasets"
    
    # Output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Combine datasets
        combined_df = combine_datasets_optimized(data_dir)
        
        # Save combined dataset
        print("\nSaving combined dataset...")
        output_file = output_dir / "user_transactions.csv"
        combined_df.to_csv(output_file, index=False)
        
        total_time = time.time() - start_time
        
        print(f"\n[SUCCESS] Data combination completed!")
        print(f"Processing time: {total_time:.1f} seconds")
        print(f"Combined dataset saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. The combined dataset is ready for the dashboard")
        print("2. In the dashboard sidebar, select 'Upload File'")
        print("3. Upload the generated 'user_transactions.csv' file")
        print("4. Or the dashboard can be modified to use this file as default")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
