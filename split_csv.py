"""
CSV File Splitter for Large Datasets
Splits large CSV files into smaller chunks for Streamlit Cloud upload
"""
import pandas as pd
from pathlib import Path
import math

def split_large_csv(input_file_path, chunk_size_mb=200, output_dir=None):
    """
    Split a large CSV file into smaller chunks.
    
    Args:
        input_file_path: Path to the large CSV file
        chunk_size_mb: Size of each chunk in MB (default 200MB for Streamlit Cloud)
        output_dir: Directory to save chunks (default: same as input file)
    """
    input_path = Path(input_file_path)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_file_path}")
        return
    
    if output_dir is None:
        output_dir = input_path.parent / "chunks"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Get file size
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"📊 Input file: {input_path.name} ({file_size_mb:.1f}MB)")
    
    # Calculate approximate number of chunks needed
    estimated_chunks = math.ceil(file_size_mb / chunk_size_mb)
    print(f"📁 Estimated chunks needed: {estimated_chunks}")
    
    # Read CSV in chunks
    print("🔄 Reading and splitting CSV...")
    
    try:
        # First, get the total number of rows
        total_rows = sum(1 for _ in open(input_path)) - 1  # -1 for header
        print(f"📈 Total rows: {total_rows:,}")
        
        # Calculate rows per chunk
        rows_per_chunk = total_rows // estimated_chunks
        if total_rows % estimated_chunks != 0:
            rows_per_chunk += 1
        
        print(f"📋 Rows per chunk: {rows_per_chunk:,}")
        
        # Split the file
        chunk_num = 0
        
        for chunk_df in pd.read_csv(input_path, chunksize=rows_per_chunk):
            chunk_filename = f"{input_path.stem}_chunk_{chunk_num:02d}.csv"
            chunk_path = output_dir / chunk_filename
            
            # Save chunk
            chunk_df.to_csv(chunk_path, index=False)
            
            # Check chunk size
            chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
            print(f"✅ Created: {chunk_filename} ({len(chunk_df):,} rows, {chunk_size_mb:.1f}MB)")
            
            chunk_num += 1
        
        print(f"\n🎉 Successfully split into {chunk_num} chunks!")
        print(f"📁 Chunks saved in: {output_dir}")
        print(f"\n📋 Next steps:")
        print(f"1. Go to your Streamlit app")
        print(f"2. Use 'Multi-File Upload' in the sidebar")
        print(f"3. Upload all {chunk_num} chunk files")
        print(f"4. Click 'Merge Uploaded Files'")
        
    except Exception as e:
        print(f"❌ Error splitting file: {str(e)}")

if __name__ == "__main__":
    # Example usage
    print("🔧 CSV File Splitter for Streamlit Cloud")
    print("=" * 50)
    
    # Get input file path
    input_file = input("Enter path to your large CSV file: ").strip().strip('"')
    
    if not input_file:
        # Use default path if no input
        input_file = "data/complete_user_transactions.csv"
        print(f"Using default: {input_file}")
    
    # Get chunk size
    chunk_size = input("Enter chunk size in MB (default 200): ").strip()
    if not chunk_size:
        chunk_size = 200
    else:
        try:
            chunk_size = int(chunk_size)
        except ValueError:
            chunk_size = 200
    
    print(f"\n🚀 Starting split process...")
    split_large_csv(input_file, chunk_size)
