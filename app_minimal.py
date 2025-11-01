"""
Minimal FraudSight app for health check testing
"""
import streamlit as st

# Basic page config
st.set_page_config(
    page_title="FraudSight Analytics",
    page_icon="🎯",
    layout="wide"
)

def main():
    st.title("🎯 FraudSight Analytics")
    st.success("✅ App is running successfully!")
    
    st.write("### Health Check")
    st.write("- ✅ Streamlit: Working")
    st.write("- ✅ Page Config: Working") 
    st.write("- ✅ Basic UI: Working")
    
    if st.button("Test Button"):
        st.success("✅ Button interaction working!")
    
    st.info("Minimal version deployed successfully. Ready for full features.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
