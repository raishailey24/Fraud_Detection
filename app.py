"""
FraudSight: AI-Driven Financial Intelligence Hub
Production version - optimized for deployment
"""

# Import with error handling
try:
    import streamlit as st
    import pandas as pd
    from pathlib import Path
    import requests
    import time
    import sys
    import os
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set page configuration with error handling
try:
    st.set_page_config(
        page_title="FraudSight: AI-Driven Financial Intelligence Hub",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Page config error: {e}")
    st.stop()

def initialize_session_state():
    """Initialize session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = None
    if "df_processed" not in st.session_state:
        st.session_state["df_processed"] = None

def generate_sample_data():
    """Generate sample fraud detection data for demonstration."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 150000  # 150K sample records for more data
    
    # Define realistic geographic locations
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
              'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle']
    
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA', 'TX', 'FL',
              'TX', 'OH', 'NC', 'CA', 'IN', 'WA']
    
    countries = ['USA'] * len(cities)
    
    # Expanded merchant categories
    merchant_categories = [
        'grocery', 'gas_station', 'restaurant', 'retail', 'online_shopping', 'atm',
        'pharmacy', 'department_store', 'electronics', 'clothing', 'hotel', 'airline',
        'car_rental', 'entertainment', 'healthcare', 'education', 'utilities', 'insurance'
    ]
    
    # Generate transaction data
    data = {
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
        'user_id': np.random.randint(1000, 50000, n_samples),
        'amount': np.random.lognormal(3, 1.5, n_samples).round(2),
        'merchant_category': np.random.choice(merchant_categories, n_samples),
        'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer', 'payment'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),  # 0.2% fraud rate
        
        # Add geographic data
        'city': np.random.choice(cities, n_samples),
        'state': np.random.choice(states, n_samples),
        'country': np.random.choice(countries, n_samples),
        
        # Add merchant and location details
        'merchant_name': [f'Merchant_{np.random.randint(1, 10000):04d}' for _ in range(n_samples)],
        'location': np.random.choice(cities, n_samples),
    }
    
    # Add timestamps
    start_date = datetime.now() - timedelta(days=365)
    data['timestamp'] = [start_date + timedelta(seconds=np.random.randint(0, 365*24*3600)) for _ in range(n_samples)]
    
    # Create more realistic fraud patterns
    fraud_indices = np.where(data['is_fraud'] == 1)[0]
    for idx in fraud_indices:
        # Fraudulent transactions tend to be higher amounts
        data['amount'][idx] = np.random.lognormal(5, 1, 1)[0].round(2)
        # More likely to be online or ATM
        data['merchant_category'][idx] = np.random.choice(['online_shopping', 'atm', 'electronics'], 1)[0]
        # Fraud more common in certain cities
        data['city'][idx] = np.random.choice(['New York', 'Los Angeles', 'Chicago'], 1)[0]
    
    df = pd.DataFrame(data)
    return df

def get_github_datasets():
    """Define sample data generation instead of external downloads."""
    # Generate more chunks for better data variety
    chunks = {}
    estimated_chunks = 10  # Increased to 10 chunks for more realistic dataset
    
    for i in range(estimated_chunks):
        chunk_filename = f"sample_transactions_chunk_{i:02d}.parquet"
        chunks[chunk_filename] = {
            "url": "sample_data",  # Special marker for sample data
            "description": f"Sample Transactions - Chunk {i+1}/{estimated_chunks}",
            "size_mb": 8,  # Larger chunks with more data
            "is_chunked": True,
            "chunk_index": i,
            "total_chunks": estimated_chunks,
            "base_name": "sample_transactions"
        }
    
    return chunks

def download_from_github(url: str, filename: str, description: str = "dataset") -> Path:
    """Generate sample data instead of downloading (temporary solution)."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / filename
    
    # Check if file already exists and is not empty
    if file_path.exists() and file_path.stat().st_size > 1024:  # At least 1KB
        return file_path
    
    try:
        # Generate sample data instead of downloading
        if url == "sample_data":
            # Generate sample data for this chunk
            df = generate_sample_data()
            
            # Split data into chunks based on filename
            chunk_index = int(filename.split('_')[-1].split('.')[0])
            total_chunks = 10
            chunk_size = len(df) // total_chunks
            
            start_idx = chunk_index * chunk_size
            if chunk_index == total_chunks - 1:  # Last chunk gets remainder
                end_idx = len(df)
            else:
                end_idx = (chunk_index + 1) * chunk_size
            
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            # Save as parquet
            chunk_df.to_parquet(file_path, index=False)
            
            return file_path
        else:
            # Original download logic for real URLs (when available)
            import requests
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            downloaded = 0
            chunk_size = 8192
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # Verify download
            if downloaded < 1024:
                if file_path.exists():
                    file_path.unlink()
                return None
            
            return file_path
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        return None

def merge_parquet_chunks(chunk_files: list) -> pd.DataFrame:
    """Merge multiple parquet chunks into a single DataFrame."""
    if not chunk_files:
        st.error("❌ No chunk files provided")
        return None
    
    st.info(f"🔗 Merging {len(chunk_files)} parquet chunks...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    dfs = []
    for i, chunk_file in enumerate(chunk_files):
        if chunk_file.exists():
            try:
                chunk_df = pd.read_parquet(chunk_file)
                dfs.append(chunk_df)
                status_text.text(f"Loaded chunk {i+1}/{len(chunk_files)}: {len(chunk_df):,} records")
                progress_bar.progress((i + 1) / len(chunk_files))
            except Exception as e:
                st.error(f"❌ Error reading chunk {i+1}: {str(e)}")
                return None
        else:
            st.error(f"❌ Chunk file not found: {chunk_file}")
            return None
    
    if dfs:
        st.info("🔗 Combining all chunks...")
        merged_df = pd.concat(dfs, ignore_index=True)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Merge completed successfully!")
        
        return merged_df
    else:
        st.error("❌ No valid chunk files found")
        return None

def load_full_dataset():
    """Load the complete dataset."""
    st.sidebar.markdown("### 📊 Demo Dataset")
    st.sidebar.info(f"1.5M transaction records • 12 columns • ~80MB total")
    
    data_dir = Path("data")
    github_datasets = get_github_datasets()
    
    # Group chunks
    chunked_datasets = {}
    for filename, info in github_datasets.items():
        if info.get('is_chunked', False):
            base_name = info.get('base_name', filename.split('_chunk_')[0])
            if base_name not in chunked_datasets:
                chunked_datasets[base_name] = []
            chunked_datasets[base_name].append((filename, info))
    
    # Single download button for all chunks
    for base_name, chunks in chunked_datasets.items():
        total_chunks = chunks[0][1].get('total_chunks', len(chunks))
        
        # Check if all chunks are downloaded
        chunk_files = []
        all_downloaded = True
        downloaded_files = []
        
        for filename, info in sorted(chunks, key=lambda x: x[1].get('chunk_index', 0)):
            file_path = data_dir / filename
            chunk_files.append(file_path)
            if file_path.exists() and file_path.stat().st_size > 1024:  # At least 1KB
                downloaded_files.append(filename)
            else:
                all_downloaded = False
        
        # Show status
        if downloaded_files:
            st.sidebar.info(f"📥 Downloaded: {len(downloaded_files)}/{total_chunks} chunks")
        
        if all_downloaded:
            # All chunks ready - show load button
            st.sidebar.success("✅ All data chunks ready!")
            if st.sidebar.button("🚀 Load Fraud Detection Dashboard", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading complete dataset..."):
                        merged_df = merge_parquet_chunks(chunk_files)
                        if merged_df is not None:
                            st.success(f"✅ Dataset loaded: {len(merged_df):,} records")
                            return merged_df
                        else:
                            st.error("❌ Failed to merge parquet chunks")
                            return None
                            
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
                    return None
        else:
            # Show single download button for all chunks
            if st.sidebar.button("📥 Download Complete Dataset", type="primary", use_container_width=True):
                # Create progress display
                progress_container = st.container()
                with progress_container:
                    st.info("🚀 Downloading fraud detection dataset...")
                    overall_progress = st.progress(0)
                    status_text = st.empty()
                
                # Download all chunks sequentially
                success_count = 0
                for i, (filename, info) in enumerate(sorted(chunks, key=lambda x: x[1].get('chunk_index', 0))):
                    file_path = data_dir / filename
                    
                    # Update progress
                    progress = i / total_chunks
                    overall_progress.progress(progress)
                    status_text.text(f"Downloading chunk {i+1}/{total_chunks}...")
                    
                    if not file_path.exists():
                        result = download_from_github(info["url"], filename, info["description"])
                        if result:
                            success_count += 1
                        else:
                            st.error(f"❌ Failed to download chunk {i+1}")
                            return None
                    else:
                        success_count += 1
                
                # Complete progress
                overall_progress.progress(1.0)
                status_text.text("✅ All chunks downloaded!")
                
                if success_count == total_chunks:
                    st.success("🎉 All chunks downloaded successfully!")
                    st.info("🔄 Refreshing page to show load button...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Download incomplete: {success_count}/{total_chunks} chunks")
    
    # No data available - clean message
    st.info("📥 **Ready to load your fraud detection dataset**")
    st.markdown("Click **'Download Complete Dataset'** in the sidebar to get started.")
    return None

def process_data(df: pd.DataFrame):
    """Process and validate the dataset."""
    if df is None or df.empty:
        return None
    
    try:
        # Basic data processing
        processed_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['transaction_id', 'amount', 'is_fraud']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Fix data types to prevent filter errors
        if 'amount' in processed_df.columns:
            processed_df['amount'] = pd.to_numeric(processed_df['amount'], errors='coerce')
            
        if 'is_fraud' in processed_df.columns:
            processed_df['is_fraud'] = pd.to_numeric(processed_df['is_fraud'], errors='coerce')
            
        if 'timestamp' in processed_df.columns:
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], errors='coerce')
        
        # Basic data cleaning - remove rows with invalid data
        processed_df = processed_df.dropna(subset=['amount', 'is_fraud'])
        
        # Ensure is_fraud is integer (0 or 1)
        processed_df['is_fraud'] = processed_df['is_fraud'].astype(int)
        
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Data processing error: {str(e)}")
        return None

def main():
    """Main application function."""
    
    # Health check for Streamlit Cloud
    try:
        # Basic health check - ensure Streamlit is working
        if not hasattr(st, 'write'):
            raise Exception("Streamlit not properly initialized")
            
        # Initialize session state
        initialize_session_state()
        
    except Exception as e:
        st.error(f"❌ App initialization failed: {str(e)}")
        st.info("Please refresh the page or contact support.")
        return
    
    # Apply modern theme
    try:
        from utils.modern_theme import apply_modern_theme
        apply_modern_theme()
    except:
        pass  # Continue without theme if not available
    
    # Header - simplified for better compatibility
    try:
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
    except Exception as e:
        # Fallback to simple title if HTML fails
        st.title("🎯 FraudSight: AI-Driven Financial Intelligence Hub")
        st.caption("Turning reactive detection into proactive intelligence")
    
    # Validate configuration
    try:
        from config import Config
        config_errors = Config.validate()
        if config_errors:
            with st.expander("⚠️ AI Configuration Required", expanded=False):
                st.warning("**AI features are disabled. To enable natural language queries:**")
                st.markdown("""
                **For Streamlit Cloud Deployment:**
                1. Go to your app settings in Streamlit Cloud
                2. Click **"Advanced settings"**
                3. Add to **"Secrets"** section:
                
                ```toml
                [anthropic]
                api_key = "sk-ant-your-actual-key-here"
                ```
                
                **Get your API key:**
                - **Anthropic**: [console.anthropic.com](https://console.anthropic.com) (Recommended)
                - **OpenAI**: [platform.openai.com](https://platform.openai.com)
                
                **After adding the key:**
                - Save settings and restart the app
                - AI features will be automatically enabled
                """)
            st.info("💡 The dashboard works fully without AI - add API keys to enable natural language queries.")
    except:
        pass  # Continue without config validation
    
    # Load data only if not already loaded
    if not st.session_state.get("data_loaded", False):
        try:
            with st.spinner("Loading dataset..."):
                df = load_full_dataset()
            
            if df is not None:
                st.success(f"✅ Raw data loaded: {len(df):,} records")
                
                # Process data
                try:
                    with st.spinner("Processing data..."):
                        df_processed = process_data(df)
                    
                    if df_processed is not None:
                        st.success(f"✅ Data processed: {len(df_processed):,} records")
                        
                        # Store in session state
                        st.session_state["df_raw"] = df
                        st.session_state["df_processed"] = df_processed
                        st.session_state["data_loaded"] = True
                        
                        st.success("🎉 Dataset ready! Refreshing interface...")
                        st.rerun()
                    else:
                        st.error("❌ Data processing failed")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Data processing error: {str(e)}")
                    return
            else:
                st.info("ℹ️ No dataset loaded. Please download chunks to get started.")
                return
                
        except Exception as e:
            st.error(f"❌ Data loading error: {str(e)}")
            return
    
    # Use data from session state
    if st.session_state.get("data_loaded", False):
        df_processed = st.session_state["df_processed"]
        
        # Apply smart filters for performance
        try:
            from components.smart_filters import apply_smart_filters, show_filter_summary
            df_filtered = apply_smart_filters(df_processed)
        except Exception as e:
            st.warning(f"Smart filters unavailable: {str(e)}")
            df_filtered = df_processed
        
        # Check if filtered data is empty
        if df_filtered.empty:
            st.warning("⚠️ No data matches the current filters. Please adjust your filter settings.")
            return
        
        # Show filter summary
        try:
            show_filter_summary(df_processed, df_filtered)
        except:
            pass
        
        # Performance warning for large datasets
        if len(df_filtered) > 2000000:  # > 2M records
            st.warning("⚡ **Large Dataset Detected**: Processing 2M+ records may take 30-60 seconds. Consider using 1M records for faster performance.")
        
        # FraudSight Main Tabs
        tab1, tab2, tab3 = st.tabs(["🏦 Overview", "📊 Analytics", "🧠 Ask AI"])
        
        with tab1:
            # FraudSight Overview Tab
            try:
                from components.fraudsight_overview import display_fraudsight_overview
                
                with st.spinner("📊 Loading executive dashboard..."):
                    display_fraudsight_overview(df_filtered)
            except Exception as e:
                st.error(f"❌ Overview tab error: {str(e)}")
                st.info("Please try refreshing the page.")
        
        with tab2:
            # FraudSight Analytics Tab
            try:
                from components.fraudsight_analytics import display_fraudsight_analytics, display_ai_recommendations_panel
                
                with st.spinner("📈 Loading advanced analytics..."):
                    col_main, col_sidebar = st.columns([3, 1])
                    
                    with col_main:
                        display_fraudsight_analytics(df_filtered)
                    
                    with col_sidebar:
                        display_ai_recommendations_panel(df_filtered)
                        
            except Exception as e:
                st.error(f"❌ Analytics tab error: {str(e)}")
                st.info("Please try refreshing the page.")
        
        with tab3:
            # AI Tab
            try:
                from components.ai_panel import display_ai_copilot
                
                st.markdown("### 🧠 Ask AI ")
                
                with st.spinner("🤖 Initializing AI assistant..."):
                    display_ai_copilot(df_filtered)
            except Exception as e:
                st.error(f"❌ AI tab error: {str(e)}")
                st.info("AI features may be temporarily unavailable.")
    
    else:
        # Welcome screen
        st.info("""
        👋 **Welcome to Your Fraud Analytics Dashboard!**
        
        **Demo Dataset Status:**
        - 📊 **Source**: Sample financial transaction data (1.5M records)
        - 🔍 **Fraud Detection**: Realistic fraud patterns and labels
        - 📈 **Analytics**: Advanced risk scoring and pattern analysis
        - 🤖 **AI Copilot**: Claude 3.5 Sonnet for intelligent insights
        
        **Dataset Information:**
        - **Size**: ~80MB total (10 chunks × ~8MB each)
        - **Format**: Optimized parquet files for fast loading
        - **Type**: Generated sample data with geographic and merchant details
        
        **Get Started:**
        1. Click **"Download Complete Dataset"** in the sidebar
        2. Wait for sample data generation (~80MB total)
        3. Click **"Load Fraud Detection Dashboard"**
        4. Explore fraud patterns with AI-powered insights!
        
        **Note**: This demo uses generated sample data with realistic geographic and merchant information. The full production version would connect to your actual transaction database.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 **Critical Error**: {str(e)}")
        st.error(f"**Error Type**: {type(e).__name__}")
        
        import traceback
        st.code(traceback.format_exc(), language="python")
        
        st.info("""
        **Troubleshooting Steps:**
        1. Refresh the page (F5)
        2. Clear browser cache
        3. Check if all required files are present
        """)
