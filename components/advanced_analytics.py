"""
Advanced Analytics Components - Optimized for Large Datasets
Handles data efficiently to prevent MessageSizeError.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


def display_advanced_analytics(df: pd.DataFrame):
    """
    Display advanced analytics with data sampling for performance.
    
    Args:
        df: Input DataFrame (will be sampled if too large)
    """
    st.header("🔬 Advanced Analytics")
    
    # Sample data if too large to prevent MessageSizeError
    max_records = 50000  # Limit for performance
    if len(df) > max_records:
        st.info(f"📊 Analyzing sample of {max_records:,} records (from {len(df):,} total) for performance")
        df_sample = df.sample(n=max_records, random_state=42)
    else:
        df_sample = df
        st.info(f"📊 Analyzing all {len(df_sample):,} records")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Model Performance", 
        "📊 Risk Analysis", 
        "🕐 Temporal Patterns", 
        "🏪 Merchant Insights"
    ])
    
    with tab1:
        display_model_performance(df_sample)
    
    with tab2:
        display_risk_analysis(df_sample)
    
    with tab3:
        display_temporal_patterns(df_sample)
    
    with tab4:
        display_merchant_insights(df_sample)


def display_model_performance(df: pd.DataFrame):
    """Display model performance metrics."""
    st.subheader("🎯 Fraud Detection Performance")
    
    if 'risk_score' not in df.columns:
        st.warning("Risk score not available for performance analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Risk score threshold for fraud classification"
        )
        
        # Create predictions based on threshold
        y_true = df['is_fraud']
        y_pred = (df['risk_score'] >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Legitimate', 'Predicted Fraud'],
            y=['Actual Legitimate', 'Actual Fraud'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix (Threshold: {threshold})",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance Metrics
        st.markdown("#### Performance Metrics")
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Display metrics
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Precision", f"{precision:.3f}", help="True Positives / (True Positives + False Positives)")
            st.metric("Recall", f"{recall:.3f}", help="True Positives / (True Positives + False Negatives)")
        
        with metrics_col2:
            st.metric("F1-Score", f"{f1:.3f}", help="Harmonic mean of Precision and Recall")
            st.metric("Accuracy", f"{accuracy:.3f}", help="(True Positives + True Negatives) / Total")
        
        # ROC-like analysis
        st.markdown("#### Threshold Analysis")
        
        thresholds = np.arange(0.0, 1.01, 0.05)
        precisions, recalls, f1_scores = [], [], []
        
        for thresh in thresholds:
            y_pred_thresh = (df['risk_score'] >= thresh).astype(int)
            cm_thresh = confusion_matrix(y_true, y_pred_thresh)
            tn, fp, fn, tp = cm_thresh.ravel()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_thresh = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1_thresh)
        
        # Plot threshold analysis
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=precisions,
            mode='lines+markers',
            name='Precision',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=recalls,
            mode='lines+markers',
            name='Recall',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=f1_scores,
            mode='lines+markers',
            name='F1-Score',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title="Performance vs Threshold",
            xaxis_title="Threshold",
            yaxis_title="Score",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_risk_analysis(df: pd.DataFrame):
    """Display risk score analysis."""
    st.subheader("📊 Risk Score Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Score Distribution
        st.markdown("#### Risk Score Distribution")
        
        fig = go.Figure()
        
        # Legitimate transactions
        legitimate_scores = df[df['is_fraud'] == 0]['risk_score']
        fig.add_trace(go.Histogram(
            x=legitimate_scores,
            name="Legitimate",
            opacity=0.7,
            marker_color='green',
            nbinsx=50
        ))
        
        # Fraudulent transactions
        fraud_scores = df[df['is_fraud'] == 1]['risk_score']
        if len(fraud_scores) > 0:
            fig.add_trace(go.Histogram(
                x=fraud_scores,
                name="Fraudulent",
                opacity=0.7,
                marker_color='red',
                nbinsx=50
            ))
        
        fig.update_layout(
            title="Risk Score Distribution by Fraud Status",
            xaxis_title="Risk Score",
            yaxis_title="Count",
            barmode="overlay",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk Level Analysis
        st.markdown("#### Risk Level Breakdown")
        
        risk_fraud = df.groupby(['risk_level', 'is_fraud']).size().unstack(fill_value=0)
        risk_fraud_pct = risk_fraud.div(risk_fraud.sum(axis=1), axis=0) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=risk_fraud_pct.index,
            y=risk_fraud_pct[0] if 0 in risk_fraud_pct.columns else [0] * len(risk_fraud_pct.index),
            name='Legitimate',
            marker_color='green'
        ))
        
        if 1 in risk_fraud_pct.columns:
            fig.add_trace(go.Bar(
                x=risk_fraud_pct.index,
                y=risk_fraud_pct[1],
                name='Fraudulent',
                marker_color='red'
            ))
        
        fig.update_layout(
            title="Fraud Rate by Risk Level",
            xaxis_title="Risk Level",
            yaxis_title="Percentage",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level statistics
        st.markdown("#### Risk Level Statistics")
        risk_stats = df.groupby('risk_level').agg({
            'is_fraud': ['count', 'sum', 'mean'],
            'amount': ['mean', 'sum']
        }).round(3)
        
        risk_stats.columns = ['Total', 'Fraud Count', 'Fraud Rate', 'Avg Amount', 'Total Amount']
        st.dataframe(risk_stats, use_container_width=True)


def display_temporal_patterns(df: pd.DataFrame):
    """Display temporal fraud patterns."""
    st.subheader("🕐 Temporal Fraud Patterns")
    
    # Add time-based features if not present
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly patterns
        st.markdown("#### Fraud by Hour of Day")
        
        hourly_fraud = df.groupby('hour').agg({
            'is_fraud': ['sum', 'count']
        }).reset_index()
        hourly_fraud.columns = ['hour', 'fraud_count', 'total_count']
        hourly_fraud['fraud_rate'] = (hourly_fraud['fraud_count'] / hourly_fraud['total_count'] * 100)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=hourly_fraud['hour'], y=hourly_fraud['fraud_count'],
                   name="Fraud Count", marker_color='red'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=hourly_fraud['hour'], y=hourly_fraud['fraud_rate'],
                      name="Fraud Rate %", mode="lines+markers",
                      line=dict(color='orange', width=2)),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="Fraud Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week patterns
        st.markdown("#### Fraud by Day of Week")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_fraud = df.groupby('day_of_week').agg({
            'is_fraud': ['sum', 'count']
        }).reset_index()
        daily_fraud.columns = ['day_of_week', 'fraud_count', 'total_count']
        daily_fraud['fraud_rate'] = (daily_fraud['fraud_count'] / daily_fraud['total_count'] * 100)
        daily_fraud['day_name'] = daily_fraud['day_of_week'].map(lambda x: day_names[x])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=daily_fraud['day_name'],
            y=daily_fraud['fraud_rate'],
            marker_color=daily_fraud['fraud_rate'],
            marker_colorscale='Reds',
            text=[f"{rate:.1f}%" for rate in daily_fraud['fraud_rate']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Fraud Rate by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Fraud Rate (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_merchant_insights(df: pd.DataFrame):
    """Display merchant-level insights."""
    st.subheader("🏪 Merchant Risk Analysis")
    
    # Calculate merchant statistics
    merchant_stats = df.groupby('merchant').agg({
        'is_fraud': ['sum', 'count', 'mean'],
        'amount': ['sum', 'mean']
    }).reset_index()
    
    merchant_stats.columns = ['merchant', 'fraud_count', 'total_count', 'fraud_rate', 'total_amount', 'avg_amount']
    
    # Filter merchants with significant transaction volume
    min_transactions = st.slider(
        "Minimum transactions per merchant",
        min_value=5,
        max_value=100,
        value=20,
        help="Filter merchants with at least this many transactions"
    )
    
    merchant_stats_filtered = merchant_stats[merchant_stats['total_count'] >= min_transactions]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top risky merchants
        st.markdown("#### Highest Risk Merchants")
        
        top_risky = merchant_stats_filtered.nlargest(15, 'fraud_rate')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_risky['merchant'],
            x=top_risky['fraud_rate'] * 100,
            orientation='h',
            marker_color=top_risky['fraud_rate'] * 100,
            marker_colorscale='Reds',
            text=[f"{rate:.1f}%" for rate in top_risky['fraud_rate'] * 100],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Merchants with Highest Fraud Rates",
            xaxis_title="Fraud Rate (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Merchant volume vs fraud rate
        st.markdown("#### Volume vs Risk Scatter")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=merchant_stats_filtered['total_amount'],
            y=merchant_stats_filtered['fraud_rate'] * 100,
            mode='markers',
            marker=dict(
                size=np.log(merchant_stats_filtered['total_count'] + 1) * 3,
                color=merchant_stats_filtered['fraud_rate'] * 100,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Fraud Rate %")
            ),
            text=merchant_stats_filtered['merchant'],
            hovertemplate="<b>%{text}</b><br>Total Amount: $%{x:,.0f}<br>Fraud Rate: %{y:.2f}%<extra></extra>"
        ))
        
        fig.update_layout(
            title="Transaction Volume vs Fraud Rate",
            xaxis_title="Total Transaction Amount ($)",
            yaxis_title="Fraud Rate (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Merchant summary table
    st.markdown("#### Merchant Summary")
    st.dataframe(
        merchant_stats_filtered.sort_values('fraud_rate', ascending=False).head(20),
        use_container_width=True
    )
