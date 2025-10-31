"""
KPI card components for the dashboard.
"""
import streamlit as st
import pandas as pd


def display_kpi_cards(df: pd.DataFrame):
    """
    Display key performance indicator cards.
    
    Args:
        df: Processed DataFrame with fraud data
    """
    # Calculate KPIs
    total_transactions = len(df)
    total_fraud = int(df["is_fraud"].sum())
    fraud_rate = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0
    
    total_amount = df["amount"].sum()
    fraud_amount = df[df["is_fraud"] == 1]["amount"].sum()
    fraud_amount_rate = (fraud_amount / total_amount * 100) if total_amount > 0 else 0
    
    avg_transaction = df["amount"].mean()
    avg_fraud_amount = df[df["is_fraud"] == 1]["amount"].mean() if total_fraud > 0 else 0
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{total_transactions:,}",
            delta=None
        )
        st.metric(
            label="Fraud Cases",
            value=f"{total_fraud:,}",
            delta=f"{fraud_rate:.2f}% fraud rate",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Total Amount",
            value=f"${total_amount:,.2f}",
            delta=None
        )
        st.metric(
            label="Fraud Amount",
            value=f"${fraud_amount:,.2f}",
            delta=f"{fraud_amount_rate:.2f}% of total",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Avg Transaction",
            value=f"${avg_transaction:.2f}",
            delta=None
        )
        st.metric(
            label="Avg Fraud Amount",
            value=f"${avg_fraud_amount:.2f}",
            delta=f"{((avg_fraud_amount / avg_transaction - 1) * 100):.1f}% vs avg" if avg_transaction > 0 else "N/A",
            delta_color="inverse"
        )
    
    with col4:
        # Risk distribution
        if "risk_level" in df.columns:
            high_risk = (df["risk_level"] == "high").sum()
            high_risk_pct = (high_risk / total_transactions * 100) if total_transactions > 0 else 0
            
            st.metric(
                label="High Risk Transactions",
                value=f"{high_risk:,}",
                delta=f"{high_risk_pct:.2f}%",
                delta_color="inverse"
            )
        
        # Detection accuracy (if risk_score exists)
        if "risk_score" in df.columns:
            # Simple threshold-based detection
            threshold = 0.5
            predicted_fraud = (df["risk_score"] > threshold).astype(int)
            accuracy = (predicted_fraud == df["is_fraud"]).mean() * 100
            
            st.metric(
                label="Detection Accuracy",
                value=f"{accuracy:.1f}%",
                delta=f"at {threshold} threshold"
            )
