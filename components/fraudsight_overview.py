"""
FraudSight Overview Tab - Executive snapshot of fraud landscape.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from components.smart_filters import _format_number, _format_currency


def display_fraudsight_overview(df: pd.DataFrame):
    """Display FraudSight Overview tab with executive KPIs and visualizations."""
    
    # Top Section - KPI Summary Cards
    st.markdown("### 📊 Executive KPI Dashboard")
    
    # Calculate KPIs
    total_transactions = len(df)
    flagged_transactions = df['is_fraud'].sum()
    flagged_percentage = (flagged_transactions / total_transactions * 100) if total_transactions > 0 else 0
    
    # Estimated fraud loss
    fraud_amount = df[df['is_fraud'] == 1]['amount'].sum() if 'amount' in df.columns else 0
    
    # Detection rate trend (mock calculation for demo)
    detection_rate = flagged_percentage
    detection_trend = "+2.3%" if detection_rate > 1 else "+0.8%"
    
    # KPI Cards in 3 columns (removed fraud rate metric)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">🏦 TOTAL TRANSACTIONS</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{}</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">MTD/YTD Volume</div>
        </div>
        """.format(_format_number(total_transactions)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">💰 ESTIMATED FRAUD LOSS</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{}</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">Total exposure amount</div>
        </div>
        """.format(_format_currency(fraud_amount)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">📈 DETECTION RATE TREND</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{}</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">vs last month</div>
        </div>
        """.format(detection_trend), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Middle Section - Transaction Overview
    st.markdown("### 📈 Transaction Volume vs Fraud Detection")
    
    if 'timestamp' in df.columns:
        # Prepare time series data
        df_time = df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time['date'] = df_time['timestamp'].dt.date
        
        daily_stats = df_time.groupby('date').agg({
            'transaction_id': 'count',
            'is_fraud': 'sum',
            'amount': 'sum'
        }).reset_index()
        daily_stats.columns = ['date', 'total_transactions', 'fraud_count', 'total_amount']
        
        # Create dual-axis time series
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=("Daily Transaction Volume vs Fraud Detection",)
        )
        
        # Transaction volume (bars)
        fig.add_trace(
            go.Bar(
                x=daily_stats['date'],
                y=daily_stats['total_transactions'],
                name="Transaction Volume",
                marker_color='rgba(99, 102, 241, 0.6)',
                yaxis="y"
            ),
            secondary_y=False,
        )
        
        # Fraud count (line)
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['fraud_count'],
                mode="lines+markers",
                name="Fraud Cases",
                line=dict(color='#ef4444', width=3),
                marker=dict(size=8, color='#ef4444'),
                yaxis="y2"
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            height=400,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Transaction Volume", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Cases", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom Section - Geographic and Category Analysis
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🗺️ Geographic Risk Distribution")
        
        # Use actual location data from the dataset - optimized for performance
        if 'location' in df.columns:
            # Fast vectorized operations
            geo_data = df.groupby('location', observed=True).agg({
                'is_fraud': ['sum', 'count'],
                'amount': 'sum'
            }).reset_index()
            geo_data.columns = ['location', 'fraud_count', 'total_count', 'total_amount']
            geo_data['fraud_rate'] = (geo_data['fraud_count'] / geo_data['total_count'] * 100)
            
            # Quick filtering and get top 10 for performance
            geo_data = geo_data[geo_data['total_count'] >= 10]  # At least 10 transactions
            geo_data = geo_data.nlargest(10, 'fraud_rate')
            
            # Create horizontal bar chart for top risky locations
            fig = go.Figure(go.Bar(
                x=geo_data['fraud_rate'],
                y=geo_data['location'],
                orientation='h',
                marker_color='rgba(239, 68, 68, 0.7)',
                text=[f"{rate:.1f}%" for rate in geo_data['fraud_rate']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Fraud Rate: %{x:.1f}%<br>Fraud Cases: %{customdata[0]}<br>Total Transactions: %{customdata[1]}<extra></extra>',
                customdata=geo_data[['fraud_count', 'total_count']].values
            ))
            
            fig.update_layout(
                title="Top 15 Locations by Fraud Rate",
                xaxis_title="Fraud Rate (%)",
                yaxis_title="Location",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### 🏷️ Merchant Category Breakdown")
        
        if 'category' in df.columns:
            # Fast category analysis
            category_stats = df.groupby('category', observed=True).agg({
                'is_fraud': ['sum', 'count'],
                'amount': 'sum'
            }).reset_index()
            category_stats.columns = ['category', 'fraud_count', 'total_count', 'total_amount']
            category_stats['fraud_rate'] = (category_stats['fraud_count'] / category_stats['total_count'] * 100)
            category_stats = category_stats.nlargest(6, 'fraud_count')  # Reduced to 6 for performance
            
            # Create treemap
            fig = go.Figure(go.Treemap(
                labels=category_stats['category'],
                values=category_stats['fraud_count'],
                parents=[""] * len(category_stats),
                textinfo="label+value+percent parent",
                marker_colorscale='Reds',
                marker_colorbar_title="Fraud Cases"
            ))
            
            fig.update_layout(
                title="Fraud Volume by Category",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Fraud Alert Ticker
    st.markdown("---")
    st.markdown("### 🚨 Live Fraud Alerts")
    
    # Generate mock alerts based on data
    alerts = generate_fraud_alerts(df)
    
    # Display alerts using Streamlit components instead of raw HTML
    with st.container():
       
        
        # Display each alert as a separate info box
        for i, alert in enumerate(alerts[:3]):
            st.info(f"⚠️ **Alert #{i+1}**: {alert}")
    
    st.markdown("<br>", unsafe_allow_html=True)


def generate_fraud_alerts(df: pd.DataFrame) -> list:
    """Generate contextual fraud alerts based on actual data patterns."""
    alerts = []
    
    try:
        # Category-based insights using actual data
        if 'category' in df.columns:
            category_fraud = df.groupby('category').agg({
                'is_fraud': ['sum', 'count'],
                'amount': 'sum'
            }).reset_index()
            category_fraud.columns = ['category', 'fraud_count', 'total_count', 'total_amount']
            category_fraud['fraud_rate'] = (category_fraud['fraud_count'] / category_fraud['total_count'] * 100)
            
            if len(category_fraud) > 0:
                # Find highest risk category
                top_risky = category_fraud[category_fraud['total_count'] >= 10].nlargest(1, 'fraud_rate')
                if len(top_risky) > 0:
                    top_cat = top_risky.iloc[0]
                    alerts.append(f"Highest risk category: {top_cat['category']} with {top_cat['fraud_rate']:.1f}% fraud rate ({top_cat['fraud_count']} cases)")
        
        # Location-based insights
        if 'location' in df.columns:
            location_fraud = df.groupby('location').agg({
                'is_fraud': ['sum', 'count']
            }).reset_index()
            location_fraud.columns = ['location', 'fraud_count', 'total_count']
            location_fraud['fraud_rate'] = (location_fraud['fraud_count'] / location_fraud['total_count'] * 100)
            
            # Find locations with significant fraud
            risky_locations = location_fraud[(location_fraud['total_count'] >= 5) & (location_fraud['fraud_count'] > 0)]
            if len(risky_locations) > 0:
                top_location = risky_locations.nlargest(1, 'fraud_rate').iloc[0]
                alerts.append(f"Geographic alert: {top_location['location']} shows {top_location['fraud_rate']:.1f}% fraud rate")
        
        # High-value transaction insights
        if 'amount' in df.columns and 'risk_score' in df.columns:
            high_risk_transactions = df[df['risk_score'] > 0.5]
            if len(high_risk_transactions) > 0:
                avg_amount = high_risk_transactions['amount'].mean()
                alerts.append(f"High-risk transactions detected: {len(high_risk_transactions)} cases with avg amount ${avg_amount:.2f}")
        
        # Time-based patterns using actual hour data
        if 'hour' in df.columns:
            night_fraud = df[(df['hour'] >= 22) | (df['hour'] <= 6)]['is_fraud'].sum()
            total_night = len(df[(df['hour'] >= 22) | (df['hour'] <= 6)])
            if night_fraud > 0 and total_night > 0:
                night_rate = (night_fraud / total_night) * 100
                alerts.append(f"Night-time pattern: {night_fraud} fraud cases ({night_rate:.1f}% of night transactions)")
        
    except Exception as e:
        # Fallback with basic stats
        total_fraud = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
        alerts = [f"Monitoring {len(df):,} transactions with {total_fraud:,} fraud cases detected"]
    
    # Ensure we always have at least one alert
    if not alerts:
        total_transactions = len(df)
        total_fraud = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
        fraud_rate = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0
        alerts = [f"System monitoring: {total_transactions:,} transactions, {fraud_rate:.2f}% fraud rate"]
    
    return alerts
