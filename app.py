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

# Google Drive functionality removed - using local files and uploads only

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
    
    # Add file upload option
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Upload Your CSV")
    
    # Multi-file upload for large datasets
    with st.sidebar.expander("📁 Multi-File Upload (for 2GB+ datasets)"):
        st.markdown("**Split your large CSV into 200MB chunks and upload them here:**")
        
        uploaded_files = st.file_uploader(
            "Choose CSV chunks",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload multiple CSV files (each <200MB) - they will be merged automatically"
        )
        
        if uploaded_files and len(uploaded_files) > 1:
            if st.button("🔗 Merge Uploaded Files", type="primary"):
                try:
                    st.info(f"📥 Merging {len(uploaded_files)} CSV files...")
                    progress_bar = st.progress(0)
                    
                    # Save and merge files
                    data_dir.mkdir(exist_ok=True)
                    merged_dfs = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Save each chunk
                        chunk_path = data_dir / f"chunk_{i}.csv"
                        with open(chunk_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Read and add to merge list
                        chunk_df = pd.read_csv(chunk_path)
                        merged_dfs.append(chunk_df)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files) * 0.8)
                    
                    # Merge all dataframes
                    st.info("🔗 Combining all chunks...")
                    merged_df = pd.concat(merged_dfs, ignore_index=True)
                    
                    # Save merged file
                    merged_path = data_dir / "merged_transactions.csv"
                    merged_df.to_csv(merged_path, index=False)
                    
                    progress_bar.progress(1.0)
                    
                    # Clean up chunk files
                    for i in range(len(uploaded_files)):
                        chunk_path = data_dir / f"chunk_{i}.csv"
                        if chunk_path.exists():
                            chunk_path.unlink()
                    
                    file_size_mb = merged_path.stat().st_size / (1024 * 1024)
                    st.success(f"✅ Merged {len(uploaded_files)} files: {len(merged_df):,} records ({file_size_mb:.1f}MB)")
                    
                    # Auto-rename columns and return
                    from utils.data_loader import DataLoader
                    merged_df = DataLoader.auto_rename_columns(merged_df)
                    
                    return merged_df
                    
                except Exception as e:
                    st.error(f"Merge failed: {str(e)}")
        
        elif uploaded_files and len(uploaded_files) == 1:
            st.info("💡 Single file detected - use regular upload below")
    
    # Check if running locally or in cloud
    is_local = not ("streamlit" in str(Path.cwd()).lower() or "app" in str(Path.cwd()).lower())
    
    if is_local:
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your large CSV file (no size limit locally)"
        )
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload CSV file (max 200MB for Streamlit Cloud)"
        )
        st.sidebar.warning("⚠️ Cloud deployment has 200MB upload limit. Use sample data or external hosting for larger files.")
    
    if uploaded_file is not None:
        try:
            # Save uploaded file
            data_dir.mkdir(exist_ok=True)
            file_path = data_dir / "uploaded_transactions.csv"
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            st.sidebar.success(f"✅ File uploaded: {file_size_mb:.1f}MB")
            
            # Load and process the uploaded file
            df = pd.read_csv(file_path)
            
            # Auto-rename columns if needed
            from utils.data_loader import DataLoader
            df = DataLoader.auto_rename_columns(df)
            
            st.sidebar.info(f"📊 Records: {len(df):,}")
            return df
            
        except Exception as e:
            st.sidebar.error(f"Upload failed: {str(e)}")
    
    # No data available - show sample data generation
    st.warning("⚠️ No transaction data available")
    st.info("Generate sample data to explore the fraud detection dashboard features.")
    
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
        for i, (filename, _) in enumerate(available_files):
            if "complete" in filename.lower():
                default_index = i
                break
        
        # Let user choose dataset for deployment flexibility
        selected_file = st.sidebar.selectbox(
            "📊 Select Dataset",
            options=[f[0] for f in available_files],
            format_func=lambda x: next(f[1] for f in available_files if f[0] == x),
            index=default_index,
            help="Choose dataset size - defaults to complete dataset for local use"
        )
        user_data_path = data_dir / selected_file
        
        # Load the selected dataset using cached function
        try:
            df = load_dataset_file(str(user_data_path))
            
            # Auto-rename columns if needed
            from utils.data_loader import DataLoader
            df = DataLoader.auto_rename_columns(df)
            
            # Show dataset info
            file_size_mb = user_data_path.stat().st_size / (1024*1024)
            st.sidebar.success(f"✅ Dataset loaded: {len(df):,} records ({file_size_mb:.1f}MB)")
            
            # Show local performance info for large datasets
            if file_size_mb > 1000:  # If > 1GB
                st.sidebar.info("🖥️ **Local Performance Mode**\n- Full dataset loaded in memory\n- All visualizations use complete data\n- Filters apply to entire dataset\n- Cached for 2 hours")
            
            return df
        except Exception as e:
            st.sidebar.error(f"❌ {str(e)}")
            return None
    else:
        st.sidebar.error("❌ No dataset found!")
        st.sidebar.info("Please upload your transaction data to the /data folder")
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
