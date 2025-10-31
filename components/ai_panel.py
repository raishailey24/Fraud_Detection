"""
AI Copilot panel component.
"""
import streamlit as st
from ai.ai_copilot import AICopilot
from typing import Dict, Any


def display_ai_copilot(df):
    """Display Ask AI interface for natural language queries."""
    
    # Initialize AI Copilot only once
    if "ai_copilot" not in st.session_state:
        st.session_state["ai_copilot"] = AICopilot()
    
    copilot = st.session_state["ai_copilot"]
    
    if not copilot.is_available():
        st.error("""
        🧠 **Ask AI is not configured**
        
        To enable natural language queries:
        1. Copy `.env.example` to `.env`
        2. Add your OpenAI or Anthropic API key
        3. Restart FraudSight
        """)
        return
    
    # Calculate metrics from dataframe only when needed
    if "ai_metrics" not in st.session_state or st.session_state.get("metrics_outdated", True):
        from utils.data_processor import DataProcessor
        metrics = DataProcessor.get_aggregated_metrics(df)
        st.session_state["ai_metrics"] = metrics
        st.session_state["metrics_outdated"] = False
    else:
        metrics = st.session_state["ai_metrics"]
    
    # Display provider info - simplified
    provider_info = copilot.get_provider_name()
    st.success(f"✅ AI Ready: {provider_info}")
    
    # Add refresh button only
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", help="Refresh AI configuration"):
            st.rerun()
    
    # AI Copilot tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Summary",
        "🔮 What-If Analysis",
        "🎯 Detection Rules",
        "💬 Custom Query"
    ])
    
    with tab1:
        st.subheader("Executive Summary")
        st.markdown("Generate an AI-powered executive summary of the fraud analytics data.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_summary = st.button("Generate Summary", key="gen_summary", use_container_width=True)
        with col2:
            if st.button("Clear", key="clear_summary", use_container_width=True):
                if "last_summary" in st.session_state:
                    del st.session_state["last_summary"]
                st.rerun()
        
        if generate_summary:
            with st.spinner("🤖 AI is analyzing your data and generating executive summary..."):
                try:
                    summary = copilot.generate_executive_summary(metrics)
                    st.session_state["last_summary"] = summary
                    st.success("✅ Executive summary generated!")
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
        
        # Display summary if available
        if "last_summary" in st.session_state:
            st.markdown("---")
            st.markdown("**📋 Executive Summary:**")
            st.markdown(st.session_state["last_summary"])
    
    with tab2:
        st.subheader("What-If Analysis")
        st.markdown("Explore hypothetical scenarios and their potential impact on fraud rates.")
        
        scenario = st.text_area(
            "Describe your scenario:",
            placeholder="Example: What if we implement a $500 transaction limit for new users?",
            height=100
        )
        
        if st.button("Analyze Scenario", key="analyze_scenario"):
            if scenario.strip():
                with st.spinner("🔮 AI is running what-if analysis on your scenario..."):
                    analysis = copilot.generate_what_if_analysis(metrics, scenario)
                    st.success("✅ Scenario analysis complete!")
                    st.markdown(analysis)
                    
                    # Save to session state
                    st.session_state["last_analysis"] = analysis
            else:
                st.warning("Please enter a scenario to analyze.")
        
        # Display last analysis
        if "last_analysis" in st.session_state:
            st.markdown("---")
            st.markdown("**Last Analysis:**")
            st.markdown(st.session_state["last_analysis"])
    
    with tab3:
        st.subheader("Fraud Detection Rules")
        st.markdown("Get AI-generated proposals for fraud detection rules based on data patterns.")
        
        if st.button("Generate Detection Rules", key="gen_rules"):
            with st.spinner("🎯 AI is analyzing patterns and generating detection rules..."):
                rules = copilot.generate_detection_rules(metrics)
                st.success("✅ Detection rules generated!")
                st.markdown(rules)
                
                # Save to session state
                st.session_state["last_rules"] = rules
        
        # Display last rules
        if "last_rules" in st.session_state:
            st.markdown("---")
            st.markdown("**Last Generated Rules:**")
            st.markdown(st.session_state["last_rules"])
    
    with tab4:
        st.subheader("Custom Query")
        st.markdown("Ask any question about the fraud analytics data.")
        
        query = st.text_area(
            "Your question:",
            placeholder="Example: What are the main characteristics of fraudulent transactions in this dataset?",
            height=100
        )
        
        if st.button("Ask AI", key="ask_ai"):
            if query.strip():
                with st.spinner("💬 AI is processing your question and analyzing data..."):
                    answer = copilot.answer_custom_query(metrics, query)
                    st.success("✅ AI response ready!")
                    st.markdown(answer)
                    
                    # Save to session state
                    if "query_history" not in st.session_state:
                        st.session_state["query_history"] = []
                    
                    st.session_state["query_history"].append({
                        "query": query,
                        "answer": answer
                    })
            else:
                st.warning("Please enter a question.")
        
        # Display query history
        if "query_history" in st.session_state and st.session_state["query_history"]:
            st.markdown("---")
            st.markdown("**Query History:**")
            
            for i, item in enumerate(reversed(st.session_state["query_history"][-5:])):
                with st.expander(f"Q: {item['query'][:100]}..."):
                    st.markdown(f"**Question:** {item['query']}")
                    st.markdown(f"**Answer:** {item['answer']}")


def display_metrics_summary(metrics: Dict[str, Any]):
    """
    Display a summary of available metrics for AI analysis.
    
    Args:
        metrics: Aggregated metrics dictionary
    """
    with st.expander("📈 View Available Metrics for AI Analysis"):
        st.json(metrics)
