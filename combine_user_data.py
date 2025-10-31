"""
Data preprocessing script to combine user's 5 data files into dashboard format.
Combines transactions, cards, users, MCC codes, and fraud labels into single dataset.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
import re


def load_mcc_codes(mcc_file_path):
    """Load MCC codes mapping from JSON file."""
    print("Loading MCC codes...")
    with open(mcc_file_path, 'r') as f:
        mcc_codes = json.load(f)
    print(f"[OK] Loaded {len(mcc_codes)} MCC codes")
    return mcc_codes


def load_fraud_labels(fraud_file_path):
    """Load fraud labels from JSON file."""
    print("Loading fraud labels...")
    with open(fraud_file_path, 'r') as f:
        fraud_data = json.load(f)
    
    # Convert to DataFrame for easier processing
    fraud_labels = pd.DataFrame([
        {'transaction_id': int(tid), 'is_fraud_label': label}
        for tid, label in fraud_data['target'].items()
    ])
    
    # Convert Yes/No to 1/0
    fraud_labels['is_fraud'] = fraud_labels['is_fraud_label'].map({'Yes': 1, 'No': 0})
    fraud_labels = fraud_labels.drop('is_fraud_label', axis=1)
    
    print(f"[OK] Loaded {len(fraud_labels)} fraud labels")
    fraud_rate = fraud_labels['is_fraud'].mean() * 100
    print(f"  Fraud rate: {fraud_rate:.2f}%")
    return fraud_labels


def clean_amount(amount_str):
    """Clean amount string and convert to float."""
    if pd.isna(amount_str):
        return 0.0
    
    # Remove $ and convert to float
    amount_clean = str(amount_str).replace('$', '').replace(',', '')
    try:
        return float(amount_clean)
    except:
        return 0.0


def create_merchant_name(merchant_id, merchant_city, merchant_state):
    """Create merchant name from available data."""
    if pd.isna(merchant_city) or merchant_city == 'ONLINE':
        return f"merchant_{merchant_id}"
    elif pd.isna(merchant_state):
        return f"{merchant_city}_merchant"
    else:
        return f"{merchant_city}_{merchant_state}_merchant"


def combine_datasets(data_dir):
    """
    Combine all 5 datasets into dashboard format.
    
    Args:
        data_dir: Path to directory containing the 5 data files
    
    Returns:
        Combined DataFrame in dashboard format
    """
    data_path = Path(data_dir)
    
    # File paths
    transactions_file = data_path / "transactions_data.csv"
    cards_file = data_path / "cards_data.csv"
    users_file = data_path / "users_data.csv"
    mcc_file = data_path / "mcc_codes.json"
    fraud_file = data_path / "train_fraud_labels.json"
    
    print("=" * 60)
    print("COMBINING USER DATASETS FOR FRAUD DASHBOARD")
    print("=" * 60)
    
    # Load supporting data
    mcc_codes = load_mcc_codes(mcc_file)
    fraud_labels = load_fraud_labels(fraud_file)
    
    # Load main datasets
    print("\nLoading main datasets...")
    
    print("Loading transactions data...")
    transactions = pd.read_csv(transactions_file)
    print(f"[OK] Loaded {len(transactions):,} transactions")
    
    print("Loading cards data...")
    cards = pd.read_csv(cards_file)
    print(f"[OK] Loaded {len(cards):,} cards")
    
    print("Loading users data...")
    users = pd.read_csv(users_file)
    print(f"[OK] Loaded {len(users):,} users")
    
    print("\nProcessing and combining data...")
    
    # Start with transactions as base
    df = transactions.copy()
    
    # Clean and transform columns
    print("Transforming transaction data...")
    
    # Required columns for dashboard
    df['transaction_id'] = df['id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].apply(clean_amount)
    df['user_id'] = df['client_id'].astype(str)
    
    # Create merchant names
    df['merchant'] = df.apply(
        lambda row: create_merchant_name(row['merchant_id'], row['merchant_city'], row['merchant_state']),
        axis=1
    )
    
    # Create location from city and state
    df['location'] = df.apply(
        lambda row: f"{row['merchant_city']}, {row['merchant_state']}" 
        if pd.notna(row['merchant_city']) and pd.notna(row['merchant_state']) and row['merchant_city'] != 'ONLINE'
        else "online",
        axis=1
    )
    
    # Map MCC codes to categories
    print("Mapping MCC codes to categories...")
    df['category'] = df['mcc'].astype(str).map(mcc_codes)
    
    # Fill missing categories
    df['category'] = df['category'].fillna('other')
    
    # Simplify category names for better visualization
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
    
    # Join with fraud labels
    print("Adding fraud labels...")
    df = df.merge(fraud_labels, on='transaction_id', how='left')
    
    # Fill missing fraud labels with 0 (assume legitimate if not labeled)
    df['is_fraud'] = df['is_fraud'].fillna(0).astype(int)
    
    # Select final columns for dashboard
    final_columns = [
        'transaction_id',
        'timestamp', 
        'amount',
        'merchant',
        'category',
        'is_fraud',
        'user_id',
        'location'
    ]
    
    df_final = df[final_columns].copy()
    
    # Remove any rows with missing required data
    print("Cleaning final dataset...")
    initial_count = len(df_final)
    
    # Remove rows with missing required fields
    df_final = df_final.dropna(subset=['transaction_id', 'timestamp', 'amount', 'merchant', 'category'])
    
    # Remove zero or negative amounts
    df_final = df_final[df_final['amount'] > 0]
    
    final_count = len(df_final)
    removed_count = initial_count - final_count
    
    print(f"[OK] Removed {removed_count:,} invalid transactions")
    print(f"[OK] Final dataset: {final_count:,} transactions")
    
    # Dataset summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total transactions: {len(df_final):,}")
    print(f"Date range: {df_final['timestamp'].min()} to {df_final['timestamp'].max()}")
    print(f"Unique users: {df_final['user_id'].nunique():,}")
    print(f"Unique merchants: {df_final['merchant'].nunique():,}")
    print(f"Categories: {', '.join(sorted(df_final['category'].unique()))}")
    print(f"Fraud rate: {df_final['is_fraud'].mean()*100:.2f}%")
    print(f"Amount range: ${df_final['amount'].min():.2f} to ${df_final['amount'].max():,.2f}")
    
    return df_final


def main():
    """Main function to combine datasets and save result."""
    # Data directory
    data_dir = r"C:\Users\shail\OneDrive\Desktop\BizViz Challenge\Financial Transactions Datasets"
    
    # Output directory (fraud dashboard data folder)
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Combine datasets
        combined_df = combine_datasets(data_dir)
        
        # Save combined dataset
        output_file = output_dir / "user_transactions.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n[SUCCESS] Data combination completed!")
        print(f"Combined dataset saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. The combined dataset is ready for the dashboard")
        print("2. In the dashboard sidebar, select 'Upload File'")
        print("3. Upload the generated 'user_transactions.csv' file")
        print("4. Or modify the dashboard to use this file as default")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("Please check file paths and data formats")


if __name__ == "__main__":
    main()
