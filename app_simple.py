"""
Simplified version of the main app to isolate errors
"""
import streamlit as st

# Set page config first
st.set_page_config(
    page_title="FraudSight Analytics",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 FraudSight Analytics - Debug Mode")
    
    try:
        st.success("✅ App started successfully!")
        
        # Test basic imports
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
