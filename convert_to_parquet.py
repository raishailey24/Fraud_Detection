"""
CSV to Parquet Converter with Chunking for Google Drive
Converts large CSV files to compressed parquet format and splits if >100MB
"""
import pandas as pd
import numpy as np
from pathlib import Path
import math

def convert_csv_to_parquet(csv_path, output_dir=None, max_size_mb=100):
    """
    Convert CSV to parquet format with chunking if file exceeds max_size_mb.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save parquet files (default: same as CSV)
        max_size_mb: Maximum size per parquet file in MB
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f" CSV file not found: {csv_path}")
        return
    
    if output_dir is None:
        output_dir = csv_path.parent / "parquet_files"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Get CSV info
    csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"Input CSV: {csv_path.name} ({csv_size_mb:.1f}MB)")
    
    # Read CSV
    print("Reading CSV file...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Convert timestamp column if exists
    timestamp_cols = ['timestamp', 'date', 'transaction_date', 'created_at']
    for col in timestamp_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f" Converted {col} to datetime")
            except:
                pass
    
    # Optimize data types
    print(" Optimizing data types...")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category for string columns
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
                print(f"   {col} -> category")
        elif df[col].dtype == 'int64':
            # Downcast integers
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Create single parquet file first
    base_name = csv_path.stem
    single_parquet = output_dir / f"{base_name}.parquet"
    
    print(" Creating parquet file...")
    df.to_parquet(
        single_parquet, 
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    # Check file size
    parquet_size_mb = single_parquet.stat().st_size / (1024 * 1024)
    print(f" Parquet created: {parquet_size_mb:.1f}MB (compression: {(1-parquet_size_mb/csv_size_mb)*100:.1f}%)")
    
    if parquet_size_mb <= max_size_mb:
        print(f" File size OK ({parquet_size_mb:.1f}MB ≤ {max_size_mb}MB)")
        print(f" Single file: {single_parquet}")
        return [single_parquet]
    else:
        print(f" File too large ({parquet_size_mb:.1f}MB > {max_size_mb}MB)")
        print(" Splitting into chunks...")
        
        # Remove single file
        single_parquet.unlink()
        
        # Calculate number of chunks needed
        num_chunks = math.ceil(parquet_size_mb / max_size_mb)
        rows_per_chunk = len(df) // num_chunks
        
        print(f" Creating {num_chunks} chunks ({rows_per_chunk:,} rows each)")
        
        chunk_files = []
        for i in range(num_chunks):
            start_idx = i * rows_per_chunk
            if i == num_chunks - 1:  # Last chunk gets remaining rows
                end_idx = len(df)
            else:
                end_idx = (i + 1) * rows_per_chunk
            
            chunk_df = df.iloc[start_idx:end_idx].copy()
            chunk_file = output_dir / f"{base_name}_chunk_{i:02d}.parquet"
            
            chunk_df.to_parquet(
                chunk_file,
                engine='pyarrow', 
                compression='snappy',
                index=False
            )
            
            chunk_size_mb = chunk_file.stat().st_size / (1024 * 1024)
            print(f"   Chunk {i+1}/{num_chunks}: {len(chunk_df):,} rows ({chunk_size_mb:.1f}MB)")
            chunk_files.append(chunk_file)
        
        print(f"\n Successfully created {len(chunk_files)} parquet chunks!")
        print(f" Files saved in: {output_dir}")
        
        # Create metadata file
        metadata = {
            'original_csv': str(csv_path),
            'total_rows': len(df),
            'total_chunks': len(chunk_files),
            'chunk_files': [f.name for f in chunk_files],
            'compression_ratio': f"{(1-parquet_size_mb/csv_size_mb)*100:.1f}%"
        }
        
        metadata_file = output_dir / f"{base_name}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Metadata saved: {metadata_file}")
        return chunk_files

def merge_parquet_chunks(chunk_dir, output_file=None):
    """
    Merge parquet chunks back into a single DataFrame.
    
    Args:
        chunk_dir: Directory containing parquet chunk files
        output_file: Optional output file path
    """
    chunk_dir = Path(chunk_dir)
    
    # Find all parquet chunk files
    chunk_files = sorted(list(chunk_dir.glob("*_chunk_*.parquet")))
    
    if not chunk_files:
        print(" No parquet chunk files found")
        return None
    
    print(f" Merging {len(chunk_files)} parquet chunks...")
    
    # Read and merge all chunks
    dfs = []
    for chunk_file in chunk_files:
        chunk_df = pd.read_parquet(chunk_file)
        dfs.append(chunk_df)
        print(f"   Loaded: {chunk_file.name} ({len(chunk_df):,} rows)")
    
    # Combine all chunks
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f" Merged result: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
    
    if output_file:
        output_file = Path(output_file)
        merged_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f" Saved merged file: {output_file} ({file_size_mb:.1f}MB)")
    
    return merged_df

if __name__ == "__main__":
    print("CSV to Parquet Converter")
    print("=" * 50)
    
    # Use the complete_user_transactions.csv file
    csv_file = "data/complete_user_transactions.csv"
    output_dir = "data/github_chunks"
    max_size = 25  # Smaller chunks for GitHub (25MB limit per file)
    
    print(f"Input file: {csv_file}")
    print(f"Output directory: {output_dir}")
    print(f"Max chunk size: {max_size}MB")
    
    print(f"\nConverting CSV to parquet (max {max_size}MB per file)...")
    chunk_files = convert_csv_to_parquet(csv_file, output_dir=output_dir, max_size_mb=max_size)
    
    if chunk_files:
        print(f"\nUpload these files to Google Drive:")
        for i, file in enumerate(chunk_files, 1):
            print(f"  {i}. {file.name}")
        
        print(f"\nNext steps:")
        print(f"1. Upload all parquet files to Google Drive")
        print(f"2. Set sharing to 'Anyone with the link can view'")
        print(f"3. Get the file IDs from the Google Drive URLs")
        print(f"4. Update the app configuration with the file IDs")
