"""
Simplified version of the main app to isolate errors
"""
import streamlit as st
import pandas as pd
from pathlib import Path

# Set page config first
st.set_page_config(
    page_title="FraudSight Analytics",
    page_icon="🔍",
    layout="wide"
)

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
        
        # Show column info
        st.info(f"📊 Merged dataset: {len(merged_df):,} records, {len(merged_df.columns)} columns")
        st.info(f"📋 Columns: {', '.join(merged_df.columns[:10])}{'...' if len(merged_df.columns) > 10 else ''}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Merge completed successfully!")
        
        return merged_df
    else:
        st.error("❌ No valid chunk files found")
        return None

def load_full_dataset():
    """Load the complete dataset from parquet chunks."""
    data_dir = Path("data")
    
    # Look for parquet chunks
    parquet_files = list(data_dir.glob("complete_user_transactions_chunk_*.parquet"))
    
    if len(parquet_files) >= 15:
        st.info(f"Found {len(parquet_files)} parquet chunks")
        return merge_parquet_chunks(sorted(parquet_files))
    else:
        st.warning(f"Only found {len(parquet_files)} chunks, need 15")
        return None

def main():
    st.title("🔍 FraudSight Analytics - Progressive Loading")
    
    # Add progressive loading toggle
    progressive_mode = st.sidebar.checkbox("🚀 Enable Full App", value=False, help="Load full app features progressively")
    
    try:
        st.success("✅ App started successfully!")
        
        if progressive_mode:
            st.info("🔄 Loading full app features...")
            
            # Step 1: Load data functionality
            st.write("### Step 1: Data Loading")
            try:
                df = load_full_dataset()
                if df is not None:
                    st.success(f"✅ Dataset loaded: {len(df):,} records")
                    
                    # Step 2: Load components progressively
                    st.write("### Step 2: Loading Components")
                    
                    try:
                        from components.smart_filters import apply_smart_filters
                        df_filtered = apply_smart_filters(df)
                        st.success("✅ Smart filters loaded")
                        
                        # Step 3: Load overview component
                        try:
                            from components.fraudsight_overview import display_fraudsight_overview
                            st.write("### Step 3: Dashboard Overview")
                            display_fraudsight_overview(df_filtered)
                            st.success("✅ Overview component loaded")
                            
                        except Exception as e:
                            st.error(f"❌ Overview component error: {e}")
                            
                    except Exception as e:
                        st.error(f"❌ Smart filters error: {e}")
                        
                else:
                    st.warning("⚠️ No dataset loaded")
                    
            except Exception as e:
                st.error(f"❌ Data loading error: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Keep the original debug functionality
        st.write("### Testing Core Imports")
        
        try:
            import pandas as pd
            st.success("✅ Pandas OK")
        except Exception as e:
            st.error(f"❌ Pandas: {e}")
            return
            
        try:
            from pathlib import Path
            st.success("✅ Pathlib OK")
        except Exception as e:
            st.error(f"❌ Pathlib: {e}")
            return
            
        # Test data directory
        st.write("### Testing Data Directory")
        data_dir = Path("data")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            st.success(f"✅ Found {len(parquet_files)} parquet files")
            
            # Show first few files
            for i, f in enumerate(parquet_files[:3]):
                size_mb = f.stat().st_size / (1024*1024)
                st.write(f"- {f.name}: {size_mb:.1f}MB")
                
            if len(parquet_files) >= 15:
                st.success("✅ All chunks appear to be downloaded!")
                
                if st.button("🚀 Test Load Data"):
                    try:
                        # Try to load one chunk
                        test_df = pd.read_parquet(parquet_files[0])
                        st.success(f"✅ Successfully loaded test chunk: {len(test_df):,} records")
                        st.write("Columns:", list(test_df.columns))
                        st.write("Sample data:")
                        st.dataframe(test_df.head())
                    except Exception as e:
                        st.error(f"❌ Error loading data: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning(f"⚠️ Only {len(parquet_files)} chunks found, need 15")
        else:
            st.error("❌ Data directory not found")
            
        # Test component imports
        st.write("### Testing Component Imports")
        
        try:
            from config import Config
            st.success("✅ Config OK")
        except Exception as e:
            st.error(f"❌ Config: {e}")
            
        try:
            from components.smart_filters import apply_smart_filters
            st.success("✅ Smart Filters OK")
        except Exception as e:
            st.error(f"❌ Smart Filters: {e}")
            
        try:
            from components.fraudsight_overview import display_fraudsight_overview
            st.success("✅ Overview Component OK")
        except Exception as e:
            st.error(f"❌ Overview Component: {e}")
            
    except Exception as e:
        st.error(f"❌ Error in main function: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Critical Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
