"""
Optimized Fraud Analytics Dashboard - Modern & Fast
Handles large datasets efficiently with modern design.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import optimized components
from utils.optimized_data_loader import OptimizedDataLoader, DataCache
from utils.modern_theme import ModernTheme, apply_modern_plotly_styling
from config import Config

# Page configuration
st.set_page_config(
    page_title="🔍 Modern Fraud Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply modern theme
ModernTheme.apply_custom_css()


@st.cache_data(ttl=3600)
def load_optimized_data():
    """Load data with optimization and caching."""
    # Use complete dataset if available, fallback to sample
    complete_path = Path(__file__).parent / "data" / "complete_user_transactions.csv"
    sample_path = Path(__file__).parent / "data" / "user_sample_transactions.csv"
    
    data_path = complete_path if complete_path.exists() else sample_path
    
    if not data_path.exists():
        return None, None
    
    # Load smaller sample for faster loading (max 25k records)
    df_sample = OptimizedDataLoader.load_sample_data(str(data_path), sample_size=25000)
    
    # Load aggregated stats for KPIs
    stats = OptimizedDataLoader.load_aggregated_data(str(data_path))
    
    return df_sample, stats


def create_modern_kpi_cards(stats):
    """Create modern KPI cards with enhanced styling."""
    if not stats:
        st.error("No statistics available")
        return
    
    # Calculate derived metrics
    fraud_rate = (stats['total_fraud'] / stats['total_transactions']) * 100 if stats['total_transactions'] > 0 else 0
    avg_transaction = stats['total_amount'] / stats['total_transactions'] if stats['total_transactions'] > 0 else 0
    fraud_loss_rate = (stats['fraud_amount'] / stats['total_amount']) * 100 if stats['total_amount'] > 0 else 0
    
    # Create 4 columns for KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💳 Total Transactions",
            value=f"{stats['total_transactions']:,}",
            delta=None,
            help="Total number of transactions in the dataset"
        )
    
    with col2:
        st.metric(
            label="🚨 Fraud Cases",
            value=f"{stats['total_fraud']:,}",
            delta=f"{fraud_rate:.2f}% rate",
            delta_color="inverse",
            help="Number and percentage of fraudulent transactions"
        )
    
    with col3:
        st.metric(
            label="💰 Total Volume",
            value=f"${stats['total_amount']:,.0f}",
            delta=f"${avg_transaction:.2f} avg",
            help="Total transaction volume and average transaction amount"
        )
    
    with col4:
        st.metric(
            label="💸 Fraud Loss",
            value=f"${stats['fraud_amount']:,.0f}",
            delta=f"{fraud_loss_rate:.2f}% of volume",
            delta_color="inverse",
            help="Total fraud losses and percentage of total volume"
        )


def create_modern_filters(df):
    """Create modern filter interface."""
    st.sidebar.markdown("### 🔍 **Smart Filters**")
    
    filters = {}
    
    # Date range filter
    if 'timestamp' in df.columns:
        st.sidebar.markdown("#### 📅 Date Range")
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter transactions by date range"
        )
        
        if len(date_range) == 2:
            filters['date_range'] = date_range
    
    # Amount range filter
    if 'amount' in df.columns:
        st.sidebar.markdown("#### 💰 Amount Range")
        min_amount = float(df['amount'].min())
        max_amount = float(df['amount'].max())
        
        amount_range = st.sidebar.slider(
            "Transaction amount ($)",
            min_value=min_amount,
            max_value=min_amount + min(max_amount - min_amount, 10000),  # Cap for performance
            value=(min_amount, min_amount + min(max_amount - min_amount, 1000)),
            format="$%.2f",
            help="Filter by transaction amount"
        )
        filters['amount_range'] = amount_range
    
    # Category filter
    if 'category' in df.columns:
        st.sidebar.markdown("#### 🏪 Categories")
        categories = sorted(df['category'].unique())
        selected_categories = st.sidebar.multiselect(
            "Select categories",
            options=categories,
            default=categories[:5],  # Limit default selection for faster loading
            help="Filter by transaction categories"
        )
        filters['categories'] = selected_categories
    
    # Risk level filter
    if 'risk_level' in df.columns:
        st.sidebar.markdown("#### ⚠️ Risk Levels")
        risk_levels = ['low', 'medium', 'high']
        available_levels = [level for level in risk_levels if level in df['risk_level'].unique()]
        
        selected_risk = st.sidebar.multiselect(
            "Select risk levels",
            options=available_levels,
            default=available_levels,
            help="Filter by risk assessment"
        )
        filters['risk_levels'] = selected_risk
    
    # Fraud type filter
    st.sidebar.markdown("#### 🔍 Transaction Type")
    fraud_types = st.sidebar.multiselect(
        "Show transactions",
        options=["Legitimate", "Fraud"],
        default=["Legitimate", "Fraud"],
        help="Filter by fraud status"
    )
    filters['fraud_types'] = fraud_types
    
    # Apply filters button
    if st.sidebar.button("🔄 Reset Filters", help="Reset all filters to default"):
        st.rerun()
    
    return filters


def apply_filters_optimized(df, filters):
    """Apply filters efficiently."""
    filtered_df = df.copy()
    
    # Apply each filter
    if filters.get('date_range') and len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) &
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    if filters.get('amount_range'):
        min_amount, max_amount = filters['amount_range']
        filtered_df = filtered_df[
            (filtered_df['amount'] >= min_amount) &
            (filtered_df['amount'] <= max_amount)
        ]
    
    if filters.get('categories'):
        filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
    
    if filters.get('risk_levels'):
        filtered_df = filtered_df[filtered_df['risk_level'].isin(filters['risk_levels'])]
    
    if filters.get('fraud_types'):
        fraud_values = []
        if 'Legitimate' in filters['fraud_types']:
            fraud_values.append(0)
        if 'Fraud' in filters['fraud_types']:
            fraud_values.append(1)
        filtered_df = filtered_df[filtered_df['is_fraud'].isin(fraud_values)]
    
    return filtered_df


def create_modern_time_series(df):
    """Create modern time series visualization."""
    st.markdown("#### 📈 **Fraud Trends Over Time**")
    
    # Aggregate by date
    df_daily = df.groupby(df['timestamp'].dt.date).agg({
        'is_fraud': ['sum', 'count']
    }).reset_index()
    df_daily.columns = ['date', 'fraud_count', 'total_count']
    df_daily['fraud_rate'] = (df_daily['fraud_count'] / df_daily['total_count'] * 100)
    
    # Create subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add fraud count bars
    fig.add_trace(
        go.Bar(
            x=df_daily['date'],
            y=df_daily['fraud_count'],
            name="Fraud Cases",
            marker_color=ModernTheme.COLORS['fraud'],
            hovertemplate="<b>Date:</b> %{x}<br><b>Fraud Cases:</b> %{y}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add fraud rate line
    fig.add_trace(
        go.Scatter(
            x=df_daily['date'],
            y=df_daily['fraud_rate'],
            name="Fraud Rate %",
            mode="lines+markers",
            line=dict(color=ModernTheme.COLORS['warning'], width=3),
            marker=dict(size=6),
            hovertemplate="<b>Date:</b> %{x}<br><b>Fraud Rate:</b> %{y:.2f}%<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Fraud Cases", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
    
    fig = apply_modern_plotly_styling(fig)
    fig.update_layout(height=400, title="Daily Fraud Patterns")
    
    st.plotly_chart(fig, use_container_width=True)


def create_modern_amount_distribution(df):
    """Create modern amount distribution chart."""
    st.markdown("#### 💰 **Amount Distribution Analysis**")
    
    # Sample data for performance if too large
    if len(df) > 10000:
        df_sample = df.sample(n=10000, random_state=42)
    else:
        df_sample = df
    
    fig = go.Figure()
    
    # Legitimate transactions
    legitimate_amounts = df_sample[df_sample['is_fraud'] == 0]['amount']
    fig.add_trace(go.Histogram(
        x=legitimate_amounts,
        name="Legitimate",
        opacity=0.7,
        marker_color=ModernTheme.COLORS['legitimate'],
        nbinsx=50,
        hovertemplate="<b>Amount:</b> $%{x}<br><b>Count:</b> %{y}<br><b>Type:</b> Legitimate<extra></extra>"
    ))
    
    # Fraudulent transactions
    fraud_amounts = df_sample[df_sample['is_fraud'] == 1]['amount']
    if len(fraud_amounts) > 0:
        fig.add_trace(go.Histogram(
            x=fraud_amounts,
            name="Fraudulent",
            opacity=0.7,
            marker_color=ModernTheme.COLORS['fraud'],
            nbinsx=50,
            hovertemplate="<b>Amount:</b> $%{x}<br><b>Count:</b> %{y}<br><b>Type:</b> Fraudulent<extra></extra>"
        ))
    
    fig = apply_modern_plotly_styling(fig)
    fig.update_layout(
        title="Transaction Amount Distribution",
        xaxis_title="Amount ($)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_modern_category_analysis(df):
    """Create modern category analysis."""
    st.markdown("#### 🏪 **Category Risk Analysis**")
    
    # Calculate category statistics
    category_stats = df.groupby('category').agg({
        'is_fraud': ['sum', 'count', 'mean'],
        'amount': 'sum'
    }).reset_index()
    
    category_stats.columns = ['category', 'fraud_count', 'total_count', 'fraud_rate', 'total_amount']
    category_stats = category_stats.sort_values('fraud_rate', ascending=True)
    
    # Take top 15 categories by transaction count
    top_categories = category_stats.nlargest(15, 'total_count')
    
    fig = go.Figure()
    
    # Create horizontal bar chart
    fig.add_trace(go.Bar(
        y=top_categories['category'],
        x=top_categories['fraud_rate'] * 100,
        orientation='h',
        marker_color=top_categories['fraud_rate'] * 100,
        marker_colorscale='RdYlBu_r',
        text=[f"{rate:.1f}%" for rate in top_categories['fraud_rate'] * 100],
        textposition='auto',
        hovertemplate="<b>Category:</b> %{y}<br><b>Fraud Rate:</b> %{x:.2f}%<br><b>Total Transactions:</b> %{customdata}<extra></extra>",
        customdata=top_categories['total_count']
    ))
    
    fig = apply_modern_plotly_styling(fig)
    fig.update_layout(
        title="Fraud Rate by Category",
        xaxis_title="Fraud Rate (%)",
        yaxis_title="Category",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_modern_risk_analysis(df):
    """Create modern risk level analysis."""
    st.markdown("#### ⚠️ **Risk Level Distribution**")
    
    # Risk level distribution
    risk_dist = df['risk_level'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=risk_dist.index,
            values=risk_dist.values,
            hole=0.4,
            marker_colors=[ModernTheme.RISK_COLORS.get(level, '#888888') for level in risk_dist.index],
            textinfo='label+percent',
            textfont_size=12,
            hovertemplate="<b>Risk Level:</b> %{label}<br><b>Count:</b> %{value:,}<br><b>Percentage:</b> %{percent}<extra></extra>"
        )
    ])
    
    fig = apply_modern_plotly_styling(fig)
    fig.update_layout(
        title="Risk Level Distribution",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application logic."""
    # Simple header instead of gradient to fix display issue
    st.title("🔍 Fraud Analytics Dashboard")
    st.markdown("**AI-Powered Financial Crime Detection & Analysis**")
    st.markdown("---")
    
    # Load data with progress indicator
    with st.spinner("🔄 Loading your banking data..."):
        df_sample, stats = load_optimized_data()
    
    if df_sample is None or stats is None:
        st.error("❌ **Dataset not found!**")
        st.info("Please ensure your dataset is available in the data folder.")
        return
    
    # Display data info
    st.sidebar.success(f"✅ **Dataset Loaded**")
    st.sidebar.info(f"📊 **{stats['total_transactions']:,}** total transactions")
    st.sidebar.info(f"🔍 **{len(df_sample):,}** records for analysis")
    
    # Create filters
    filters = create_modern_filters(df_sample)
    
    # Apply filters
    df_filtered = apply_filters_optimized(df_sample, filters)
    
    # Show filter results
    st.sidebar.markdown("---")
    st.sidebar.metric("📋 Filtered Records", f"{len(df_filtered):,}")
    
    if len(df_filtered) == 0:
        st.warning("⚠️ **No data matches your filters.** Please adjust the filter settings.")
        return
    
    # Main content
    st.markdown("## 📊 **Key Performance Indicators**")
    create_modern_kpi_cards(stats)
    
    st.markdown("---")
    
    # Visualizations in tabs
    tab1, tab2, tab3 = st.tabs(["📈 **Trends**", "🔍 **Analysis**", "📋 **Data**"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            create_modern_time_series(df_filtered)
        
        with col2:
            create_modern_amount_distribution(df_filtered)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            create_modern_category_analysis(df_filtered)
        
        with col2:
            create_modern_risk_analysis(df_filtered)
    
    with tab3:
        st.markdown("#### 📋 **Transaction Details**")
        st.info(f"Showing sample of {min(len(df_filtered), 1000):,} transactions (limited for performance)")
        
        # Display limited data for performance
        display_df = df_filtered[[
            'transaction_id', 'timestamp', 'amount', 'merchant',
            'category', 'is_fraud', 'risk_level'
        ]].head(1000)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Data",
            data=csv,
            file_name="fraud_analysis_sample.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
