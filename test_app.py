"""
Minimal test app to isolate the error
"""
import streamlit as st

def main():
    st.title("🔍 Test App - Error Isolation")
    
    try:
        st.success("✅ Basic Streamlit working!")
        
        # Test imports one by one
        st.write("Testing imports...")
        
        try:
            import pandas as pd
            st.success("✅ Pandas imported")
        except Exception as e:
            st.error(f"❌ Pandas error: {e}")
            
        try:
            from pathlib import Path
            st.success("✅ Pathlib imported")
        except Exception as e:
            st.error(f"❌ Pathlib error: {e}")
            
        try:
            from config import Config
            st.success("✅ Config imported")
        except Exception as e:
            st.error(f"❌ Config error: {e}")
            
        try:
            from components.ai_panel import display_ai_copilot
            st.success("✅ AI Panel imported")
        except Exception as e:
            st.error(f"❌ AI Panel error: {e}")
            
        try:
            import requests
            st.success("✅ Requests imported")
        except Exception as e:
            st.error(f"❌ Requests error: {e}")
            
        # Test data directory
        data_dir = Path("data")
        if data_dir.exists():
            files = list(data_dir.glob("*.parquet"))
            st.success(f"✅ Data directory found with {len(files)} parquet files")
        else:
            st.warning("⚠️ Data directory not found")
            
        # Test basic functionality
        if st.button("Test Button"):
            st.success("✅ Button works!")
            
    except Exception as e:
        st.error(f"❌ Error in main: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Critical Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
