"""
FraudSight: AI-Driven Financial Intelligence Hub
Optimized for local and cloud deployment with full dataset support.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from config import Config
from components.ai_panel import display_ai_copilot
import requests
import time


# Page configuration
st.set_page_config(
    page_title="FraudSight: AI-Driven Financial Intelligence Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = None
    if "df_processed" not in st.session_state:
        st.session_state["df_processed"] = None


def get_available_datasets():
    """Get list of available datasets."""
    data_dir = Path(__file__).parent / "data"
    
    # Check available datasets
    available_files = []
    file_info = {
        "complete_user_transactions.csv": "Full Dataset (2.97GB)",
        "user_sample_transactions.csv": "Sample Dataset (9.8MB)", 
        "user_transactions.csv": "User Dataset",
        "transactions.csv": "Basic Dataset"
    }
    
    for filename in file_info.keys():
        filepath = data_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024*1024)  # MB
            available_files.append((filename, f"{file_info[filename]} - {file_size:.1f}MB"))
    
    return available_files, data_dir

@st.cache_data(ttl=7200)
def load_dataset_file(file_path: str):
    """Load dataset from specified file path with memory optimization."""
    
    try:
        # Optimized dtypes for memory efficiency
        dtype_dict = {
            'transaction_id': 'int32',
            'amount': 'float32',
            'user_id': 'int32', 
            'is_fraud': 'int8',
            'hour': 'int8',
            'day_of_week': 'int8',
            'is_weekend': 'int8',
            'is_night': 'int8',
            'amount_log': 'float32',
            'amount_zscore': 'float32',
            'merchant_avg_amount': 'float32',
            'merchant_tx_count': 'int32',
            'merchant_fraud_rate': 'float32',
            'category_avg_amount': 'float32',
            'category_fraud_rate': 'float32',
            'risk_score': 'float32'
        }
        
        # Direct loading for maximum speed (your local machine can handle it)
        df = pd.read_csv(file_path, dtype=dtype_dict)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def get_google_drive_datasets():
    """
    Define Google Drive file IDs for parquet datasets.
    Update these IDs after uploading your parquet files to Google Drive.
    """
    return {
        "complete_user_transactions.parquet": {
            "file_id": "YOUR_PARQUET_FILE_ID_HERE",  # Replace with actual Google Drive file ID
            "description": "Complete User Transactions (Parquet Format)",
            "size_mb": "TBD",  # Will be updated after conversion
            "is_chunked": False
        },
        # If file is chunked, add chunk entries like:
        # "complete_user_transactions_chunk_00.parquet": {
        #     "file_id": "CHUNK_0_FILE_ID",
        #     "description": "Complete Transactions - Chunk 1",
        #     "size_mb": 95,
        #     "is_chunked": True,
        #     "chunk_index": 0,
        #     "total_chunks": 3
        # },
    }

def download_from_google_drive(file_id: str, filename: str, description: str = "dataset") -> Path:
    """
    Download parquet dataset from Google Drive with proper large file handling.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / filename
    
    # Check if file already exists and is not empty
    if file_path.exists() and file_path.stat().st_size > 1024:  # At least 1KB
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        st.success(f"✅ {description} already available ({file_size_mb:.1f}MB)")
        return file_path
    
    try:
        # Check if running on Streamlit Cloud
        is_cloud = "streamlit" in str(Path.cwd()).lower() or "app" in str(Path.cwd()).lower()
        
        # Show preparation message
        st.markdown("### 🚀 Downloading Parquet Dataset")
        st.info(f"📥 Downloading {description} from Google Drive...")
        if is_cloud:
            st.info("⚡ **Parquet Format**: Faster downloads and smaller file sizes!")
        
        session = requests.Session()
        
        # Direct download URL for parquet files
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        st.info("🔄 Initiating download...")
        response = session.get(download_url, stream=True, timeout=30)
        
        # Check for large file confirmation
        if 'text/html' in response.headers.get('content-type', ''):
            st.info("🔄 Large file detected, extracting confirmation token...")
            
            import re
            confirm_token = None
            
            # Extract confirmation token
            token_matches = re.findall(r'name="confirm"\s+value="([^"]+)"', response.text)
            if token_matches:
                confirm_token = token_matches[0]
                st.info(f"🔑 Found confirmation token: {confirm_token[:10]}...")
                
                # Download with confirmation token
                url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                response = session.get(url, stream=True, timeout=60)
                response.raise_for_status()
        
        # Final check for HTML response
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            st.error("❌ Received HTML instead of file. Please check Google Drive sharing permissions.")
            st.info("""
            **Fix sharing permissions:**
            1. Go to your Google Drive file
            2. Right-click → Share
            3. Change to "Anyone with the link"
            4. Set permission to "Viewer"
            """)
            return None
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress display
        st.markdown("### 📥 Download Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        chunk_size = 8192
        start_time = time.time()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(min(progress, 1.0))
                        
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        
                        elapsed = time.time() - start_time
                        if elapsed > 1:
                            speed_mbps = downloaded_mb / elapsed
                            eta_seconds = (total_mb - downloaded_mb) / speed_mbps if speed_mbps > 0 else 0
                            eta_minutes = eta_seconds / 60
                            
                            status_text.markdown(f"**📊 Progress:** {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({progress*100:.1f}%) | **⚡ Speed:** {speed_mbps:.2f} MB/s | **⏱️ ETA:** {eta_minutes:.1f} min")
        
        # Verify download
        final_size_mb = downloaded / (1024 * 1024)
        if downloaded < 1024:
            st.error(f"❌ Download failed: File too small ({downloaded} bytes)")
            if file_path.exists():
                file_path.unlink()
            return None
        
        progress_bar.progress(1.0)
        status_text.markdown(f"**✅ Download Complete!** {final_size_mb:.1f}MB")
        
        return file_path
        
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        if file_path.exists():
            file_path.unlink()
        return None

def merge_parquet_chunks(chunk_files: list) -> pd.DataFrame:
    """
    Merge multiple parquet chunks into a single DataFrame.
    """
    if not chunk_files:
        return None
    
    st.info(f"🔗 Merging {len(chunk_files)} parquet chunks...")
    progress_bar = st.progress(0)
    
    dfs = []
    for i, chunk_file in enumerate(chunk_files):
        if chunk_file.exists():
            chunk_df = pd.read_parquet(chunk_file)
            dfs.append(chunk_df)
            progress_bar.progress((i + 1) / len(chunk_files))
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        st.success(f"✅ Merged {len(chunk_files)} chunks: {len(merged_df):,} total records")
        return merged_df
    else:
        st.error("❌ No valid chunk files found")
        return None

def generate_cloud_sample_data():
    """Generate sample data directly for cloud deployment."""
    import numpy as np
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    st.info("🔄 Generating realistic sample data for demo...")
    progress_bar = st.progress(0)
    
    # Generate 50,000 transactions for good demo experience
    np.random.seed(42)
    n_transactions = 50000
    
    # Create realistic transaction patterns
    data = {
        'transaction_id': [f'TXN{str(i).zfill(8)}' for i in range(1, n_transactions + 1)],
        'timestamp': pd.date_range('2023-01-01', periods=n_transactions, freq='30min'),
        'amount': np.random.lognormal(3, 1.5, n_transactions).round(2),
        'user_id': [f'USER{np.random.randint(1000, 9999)}' for _ in range(n_transactions)],
        'merchant': np.random.choice([
            'Amazon', 'Walmart', 'Target', 'Starbucks', 'McDonalds', 
            'Shell', 'Exxon', 'Best Buy', 'Home Depot', 'CVS',
            'Costco', 'Apple Store', 'Netflix', 'Uber', 'Airbnb'
        ], n_transactions),
        'location': np.random.choice([
            'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
            'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA',
            'Dallas, TX', 'San Jose, CA', 'Austin, TX', 'Jacksonville, FL',
            'International', 'Online'
        ], n_transactions),
        'category': np.random.choice([
            'retail', 'restaurant', 'gas', 'grocery', 'entertainment',
            'healthcare', 'transportation', 'financial', 'utilities', 'online'
        ], n_transactions)
    }
    
    progress_bar.progress(0.3)
    
    df = pd.DataFrame(data)
    
    # Add realistic fraud patterns (2% fraud rate)
    fraud_indicators = np.random.random(n_transactions) < 0.02
    
    # Make fraud more realistic - higher amounts, night time, international
    night_hours = df['timestamp'].dt.hour.isin([22, 23, 0, 1, 2, 3])
    high_amounts = df['amount'] > df['amount'].quantile(0.95)
    international = df['location'] == 'International'
    
    # Increase fraud probability for suspicious patterns
    fraud_boost = (night_hours * 0.05) + (high_amounts * 0.03) + (international * 0.08)
    fraud_indicators = (np.random.random(n_transactions) + fraud_boost) > 0.98
    
    df['is_fraud'] = fraud_indicators.astype(int)
    
    progress_bar.progress(0.6)
    
    # Add derived features for analysis
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
    progress_bar.progress(0.8)
    
    # Save the sample data
    sample_file = data_dir / "sample_transactions.csv"
    df.to_csv(sample_file, index=False)
    
    progress_bar.progress(1.0)
    
    file_size_mb = sample_file.stat().st_size / (1024 * 1024)
    st.success(f"✅ Generated {len(df):,} transactions ({file_size_mb:.1f}MB)")
    st.info(f"📊 Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    return sample_file

def load_full_dataset():
    """Load complete user dataset with smart performance handling."""
    st.sidebar.header("📁 Data Source")
    
    # For Streamlit Cloud deployment, ensure sample data exists
    try:
        from streamlit_cloud_setup import ensure_sample_data
        ensure_sample_data()
    except ImportError:
        pass  # Skip if not in cloud environment
    
    # Get available datasets
    available_files, data_dir = get_available_datasets()
    
    # Google Drive Parquet Datasets
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Google Drive Datasets")
    st.sidebar.info(" Parquet Format: Faster loading, smaller files, better compression")
    
    google_datasets = get_google_drive_datasets()
    
    # Check for chunked datasets
    chunked_datasets = {}
    single_datasets = {}
    
    for filename, info in google_datasets.items():
        if info.get('is_chunked', False):
            base_name = filename.split('_chunk_')[0]
            if base_name not in chunked_datasets:
                chunked_datasets[base_name] = []
            chunked_datasets[base_name].append((filename, info))
        else:
            single_datasets[filename] = info
    
    # Display single parquet files
    for filename, info in single_datasets.items():
        file_path = data_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            st.sidebar.success(f" {info['description']} ({size_mb:.1f}MB)")
        else:
            if info['file_id'] != "YOUR_PARQUET_FILE_ID_HERE":
                if st.sidebar.button(f" Download {info['description']}", key=f"gd_{filename}"):
                    result = download_from_google_drive(info["file_id"], filename, info["description"])
                    if result:
                        st.success(" Download completed!")
                        st.rerun()
            else:
                st.sidebar.warning(" Please update Google Drive file ID")
    
    # Display chunked datasets
    for base_name, chunks in chunked_datasets.items():
        with st.sidebar.expander(f"📦 {base_name} ({len(chunks)} chunks)"):
            all_downloaded = True
            chunk_files = []
            
            for filename, info in sorted(chunks, key=lambda x: x[1].get('chunk_index', 0)):
                file_path = data_dir / filename
                chunk_files.append(file_path)
                
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    st.success(f"✅ Chunk {info.get('chunk_index', 0)+1} ({size_mb:.1f}MB)")
                else:
                    all_downloaded = False
                    if info['file_id'] != "CHUNK_FILE_ID":
                        if st.button(f"📥 Download Chunk {info.get('chunk_index', 0)+1}", key=f"chunk_{filename}"):
                            result = download_from_google_drive(info["file_id"], filename, info["description"])
                            if result:
                                st.rerun()
                    else:
                        st.warning(f"⚠️ Update file ID for chunk {info.get('chunk_index', 0)+1}")
            
            if all_downloaded and len([f for f in chunk_files if f.exists()]) == len(chunks):
                if st.button(f"🔗 Load Merged Dataset", key=f"merge_{base_name}"):
                    merged_df = merge_parquet_chunks(chunk_files)
                    if merged_df is not None:
                        return merged_df
    
    # Instructions for setting up Google Drive
    with st.sidebar.expander("📋 Setup Instructions"):
        st.markdown("""
        **Step 1: Convert CSV to Parquet**
        ```bash
        python convert_to_parquet.py
        ```
        
        **Step 2: Upload to Google Drive**
        - Upload all parquet files to Google Drive
        - Set sharing: "Anyone with the link can view"
        
        **Step 3: Update File IDs**
        - Copy file IDs from Google Drive URLs
        - Update the `get_google_drive_datasets()` function
        
        **Benefits:**
        - 🚀 Faster downloads (parquet compression)
        - 📦 Smaller file sizes (60-80% reduction)
        - ⚡ Faster loading in app
        """)
    
    st.sidebar.markdown("---")
    
    # Check for existing local files as fallback
    if available_files:
        # Load local parquet or CSV files
        selected_file = st.sidebar.selectbox(
            "📊 Local Files",
            options=[f[0] for f in available_files],
            format_func=lambda x: next(f[1] for f in available_files if f[0] == x),
            help="Local files found in data directory"
        )
        
        file_path = data_dir / selected_file
        try:
            if selected_file.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = load_dataset_file(str(file_path))
            
            # Auto-rename columns if needed
            from utils.data_loader import DataLoader
            df = DataLoader.auto_rename_columns(df)
            
            file_size_mb = file_path.stat().st_size / (1024*1024)
            st.sidebar.success(f"✅ Loaded: {len(df):,} records ({file_size_mb:.1f}MB)")
            return df
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {selected_file}: {str(e)}")
    
    # No data available - show sample data generation
    st.warning("⚠️ No transaction data available")
    st.info("Convert your CSV to parquet format and upload to Google Drive, or generate sample data.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Generate Sample Data (50K records)", type="primary"):
            try:
                sample_file = generate_cloud_sample_data()
                st.success("✅ Sample data generated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Sample data generation failed: {str(e)}")
    
    with col2:
        if st.button("📊 Generate Large Sample (100K records)"):
            try:
                    # Generate larger sample for better demo
                    import numpy as np
                    np.random.seed(42)
                    
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    
                    st.info("🔄 Generating large sample dataset...")
                    progress_bar = st.progress(0)
                    
                    n_transactions = 100000
                    
                    # Quick generation for large dataset
                    data = {
                        'transaction_id': [f'TXN{str(i).zfill(8)}' for i in range(1, n_transactions + 1)],
                        'timestamp': pd.date_range('2023-01-01', periods=n_transactions, freq='15min'),
                        'amount': np.random.lognormal(3, 1.5, n_transactions).round(2),
                        'user_id': [f'USER{np.random.randint(1000, 9999)}' for _ in range(n_transactions)],
                        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'McDonalds', 'Shell', 'Exxon', 'Best Buy', 'Home Depot', 'CVS'], n_transactions),
                        'location': np.random.choice(['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ', 'International', 'Online'], n_transactions),
                        'category': np.random.choice(['retail', 'restaurant', 'gas', 'grocery', 'entertainment', 'healthcare', 'transportation'], n_transactions),
                        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.98, 0.02])
                    }
                    
                    progress_bar.progress(0.5)
                    
                    df = pd.DataFrame(data)
                    sample_file = data_dir / "sample_transactions.csv"
                    df.to_csv(sample_file, index=False)
                    
                    progress_bar.progress(1.0)
                    
                    file_size_mb = sample_file.stat().st_size / (1024 * 1024)
                    st.success(f"✅ Generated {len(df):,} transactions ({file_size_mb:.1f}MB)")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Large sample generation failed: {str(e)}")
    
    return None


def load_data_section():
    """Wrapper for backward compatibility."""
    return load_full_dataset()


def process_data(df: pd.DataFrame):
    """Process and validate user's pre-processed data."""
    # Basic validation for user's dataset
    required_columns = ['transaction_id', 'timestamp', 'amount', 'merchant', 'category', 'is_fraud']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return None
    
    # User's data is already processed, just ensure correct data types
    df = df.copy()
    
    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Ensure is_fraud is integer
    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].astype(int)
    
    # Ensure amount is numeric
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    
    # Ensure risk_level is string (fix the sorting issue)
    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].fillna('low').astype(str)
    
    st.success(f"✅ Your banking data loaded successfully: {len(df):,} transactions")
    
    return df


def main():
    """Main application logic."""
    initialize_session_state()
    
    # Modern Header with FraudSight branding
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
            🎯 FraudSight: AI-Driven Financial Intelligence Hub 
        </h1>
        <p style="color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 1.1rem; font-weight: 300;">
            Turning reactive detection into proactive intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add data refresh button in header
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh Data", help="Reload data from CSV"):
            # Clear all cached data
            for key in ["data_loaded", "df_raw", "df_processed", "df_filtered", "ai_metrics", "metrics_outdated"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Validate configuration
    config_errors = Config.validate()
    if config_errors:
        st.warning("⚠️ Configuration warnings:")
        for error in config_errors:
            st.warning(f"  • {error}")
        st.info("The dashboard will work, but AI Copilot features will be disabled.")
    
    # Load data only if not already loaded
    if not st.session_state.get("data_loaded", False):
        df = load_data_section()
        
        if df is not None:
            # Process data
            df_processed = process_data(df)
            
            if df_processed is not None:
                # Store in session state
                st.session_state["df_raw"] = df
                st.session_state["df_processed"] = df_processed
                st.session_state["data_loaded"] = True
                st.rerun()  # Rerun once to load the interface
        else:
            return
    
    # Use data from session state
    if st.session_state.get("data_loaded", False):
        df_processed = st.session_state["df_processed"]
        
        # Apply smart filters for performance
        from components.smart_filters import apply_smart_filters, show_filter_summary
        df_filtered = apply_smart_filters(df_processed)
        
        # Update filtered data in session state
        st.session_state["df_filtered"] = df_filtered
        
        # Check if filtered data is empty
        if df_filtered.empty:
            st.warning("⚠️ No data matches the current filters. Please adjust your filter settings.")
            return
        
        # Show filter summary
        show_filter_summary(df_processed, df_filtered)
        
        # Performance warning for large datasets
        if len(df_filtered) > 2000000:  # > 2M records
            st.warning("⚡ **Large Dataset Detected**: Processing 2M+ records may take 30-60 seconds. Consider using 1M records for faster performance.")
        
        # FraudSight Main Tabs
        tab1, tab2, tab3 = st.tabs(["🏦 Overview", "📊 Analytics", "🧠 Ask AI"])
        
        with tab1:
            # FraudSight Overview Tab
            from components.fraudsight_overview import display_fraudsight_overview
            
            # Show loading spinner while generating overview
            with st.spinner("📊 Loading executive dashboard..."):
                display_fraudsight_overview(df_filtered)
        
        with tab2:
            # FraudSight Analytics Tab
            from components.fraudsight_analytics import display_fraudsight_analytics, display_ai_recommendations_panel
            
            # Show loading spinner while generating analytics
            with st.spinner("📈 Loading advanced analytics..."):
                # Main analytics content
                col_main, col_sidebar = st.columns([3, 1])
                
                with col_main:
                    display_fraudsight_analytics(df_filtered)
                
                with col_sidebar:
                    display_ai_recommendations_panel(df_filtered)
        
        with tab3:
            # Enhanced Ask AI Tab
            st.markdown("### 🧠 Ask AI ")
            
            # Show loading spinner while initializing AI
            with st.spinner("🤖 Initializing AI assistant..."):
                display_ai_copilot(df_filtered)
    
    else:
        # Welcome screen for user's dataset
        st.info("""
        👋 **Welcome to Your Fraud Analytics Dashboard!**
        
        **Your Banking Dataset Status:**
        - 📊 **Source**: Your financial transaction data (2010s)
        - 🔍 **Fraud Detection**: Real fraud labels from your training set
        - 📈 **Analytics**: Advanced risk scoring and pattern analysis
        - 🤖 **AI Copilot**: Available with API key configuration
        
        **Dataset Features:**
        - ✅ Transaction records with fraud labels
        - ✅ Merchant and category analysis
        - ✅ Risk scoring and level classification
        - ✅ Temporal and behavioral features
        - ✅ Geographic transaction patterns
        
        Your dataset is automatically loaded and ready for analysis!
        """)
        
        # Show dataset status
        user_data_path = Path(__file__).parent / "data" / "user_sample_transactions.csv"
        if user_data_path.exists():
            st.success("✅ Your banking dataset is ready for analysis!")
        else:
            st.warning("⚠️ Please run the data preparation script to generate your dataset.")


if __name__ == "__main__":
    main()
