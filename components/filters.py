"""
Filter components for the dashboard.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display filter controls and return filtered DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    st.sidebar.header("🔍 Filters")
    
    # Initialize filtered dataframe
    filtered_df = df.copy()
    
    # Add reset filters button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()
    
    # Date range filter
    if "timestamp" in df.columns:
        st.sidebar.subheader("Date Range")
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df["timestamp"].dt.date >= start_date) &
                (filtered_df["timestamp"].dt.date <= end_date)
            ]
    
    # Amount range filter
    if "amount" in df.columns:
        st.sidebar.subheader("Amount Range")
        min_amount = float(df["amount"].min())
        max_amount = float(df["amount"].max())
        
        amount_range = st.sidebar.slider(
            "Transaction amount ($)",
            min_value=min_amount,
            max_value=max_amount,
            value=(min_amount, max_amount),
            format="$%.2f"
        )
        
        filtered_df = filtered_df[
            (filtered_df["amount"] >= amount_range[0]) &
            (filtered_df["amount"] <= amount_range[1])
        ]
    
    # Fraud status filter
    if "is_fraud" in df.columns:
        st.sidebar.subheader("Transaction Type")
        fraud_filter = st.sidebar.multiselect(
            "Show transactions",
            options=["Legitimate", "Fraud"],
            default=["Legitimate", "Fraud"]
        )
        
        if fraud_filter:
            fraud_values = []
            if "Legitimate" in fraud_filter:
                fraud_values.append(0)
            if "Fraud" in fraud_filter:
                fraud_values.append(1)
            filtered_df = filtered_df[filtered_df["is_fraud"].isin(fraud_values)]
    
    # Category filter
    if "category" in df.columns:
        st.sidebar.subheader("Category")
        categories = sorted(df["category"].unique())
        selected_categories = st.sidebar.multiselect(
            "Select categories",
            options=categories,
            default=categories
        )
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    
    # Risk level filter
    if "risk_level" in df.columns:
        st.sidebar.subheader("Risk Level")
        # Ensure proper ordering of risk levels
        risk_level_order = ['low', 'medium', 'high']
        available_levels = [level for level in risk_level_order if level in df["risk_level"].unique()]
        
        selected_risk = st.sidebar.multiselect(
            "Select risk levels",
            options=available_levels,
            default=available_levels,
            help="Filter transactions by risk assessment level"
        )
        
        if selected_risk:
            filtered_df = filtered_df[filtered_df["risk_level"].isin(selected_risk)]
    
    # Merchant filter (top N)
    if "merchant" in df.columns:
        st.sidebar.subheader("Merchant")
        top_n = st.sidebar.number_input(
            "Show top N merchants",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
        
        top_merchants = df["merchant"].value_counts().head(top_n).index.tolist()
        show_all_merchants = st.sidebar.checkbox("Show all merchants", value=True)
        
        if not show_all_merchants:
            filtered_df = filtered_df[filtered_df["merchant"].isin(top_merchants)]
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.metric(
        "Filtered Records",
        f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df):,}"
    )
    
    return filtered_df
