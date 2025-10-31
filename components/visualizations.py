"""
Modern visualization components using Plotly.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

# Modern color palette
COLORS = {
    'primary': '#1f77b4',
    'fraud': '#d62728', 
    'legitimate': '#2ca02c',
    'warning': '#ff7f0e',
    'info': '#17becf',
    'background': '#f8f9fa'
}

# Modern chart template
CHART_TEMPLATE = {
    'layout': {
        'font': {'family': 'Inter, sans-serif', 'size': 12},
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
        'showlegend': True,
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
    }
}


def plot_fraud_over_time(df: pd.DataFrame):
    """Modern fraud timeline visualization."""
    if "timestamp" not in df.columns:
        st.warning("⚠️ Timeline data not available")
        return
    
    # Prepare time series data
    df_time = df.copy()
    df_time["date"] = pd.to_datetime(df_time["timestamp"]).dt.date
    
    daily_stats = df_time.groupby("date").agg({
        "is_fraud": ["sum", "count"],
        "amount": "sum"
    }).reset_index()
    daily_stats.columns = ["date", "fraud_count", "total_count", "total_amount"]
    daily_stats["fraud_rate"] = (daily_stats["fraud_count"] / daily_stats["total_count"] * 100)
    
    # Create modern dual-axis chart
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=("Daily Fraud Analysis",)
    )
    
    # Fraud cases (bars)
    fig.add_trace(
        go.Bar(
            x=daily_stats["date"],
            y=daily_stats["fraud_count"],
            name="Fraud Cases",
            marker_color=COLORS['fraud'],
            opacity=0.7,
            yaxis="y"
        ),
        secondary_y=False,
    )
    
    # Fraud rate (line)
    fig.add_trace(
        go.Scatter(
            x=daily_stats["date"],
            y=daily_stats["fraud_rate"],
            mode="lines+markers",
            name="Fraud Rate %",
            line=dict(color=COLORS['warning'], width=3),
            marker=dict(size=8, color=COLORS['warning']),
            yaxis="y2"
        ),
        secondary_y=True,
    )
    
    # Modern styling
    fig.update_layout(
        **CHART_TEMPLATE['layout'],
        title=dict(text="📈 Fraud Trends Over Time", x=0.02, font=dict(size=16)),
        height=450,
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis2=dict(showgrid=False)
    )
    
    fig.update_xaxes(title_text="📅 Date")
    fig.update_yaxes(title_text="🚨 Fraud Cases", secondary_y=False)
    fig.update_yaxes(title_text="📊 Fraud Rate (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_amount_distribution(df: pd.DataFrame):
    """Plot transaction amount distribution by fraud status."""
    fig = go.Figure()
    
    # Legitimate transactions
    fig.add_trace(go.Histogram(
        x=df[df["is_fraud"] == 0]["amount"],
        name="Legitimate",
        opacity=0.7,
        marker_color="green",
        nbinsx=50,
        hovertemplate="<b>Amount Range:</b> $%{x}<br>" +
                     "<b>Count:</b> %{y}<br>" +
                     "<b>Type:</b> Legitimate<br>" +
                     "<extra></extra>"
    ))
    
    # Fraudulent transactions
    fig.add_trace(go.Histogram(
        x=df[df["is_fraud"] == 1]["amount"],
        name="Fraud",
        opacity=0.7,
        marker_color="red",
        nbinsx=50,
        hovertemplate="<b>Amount Range:</b> $%{x}<br>" +
                     "<b>Count:</b> %{y}<br>" +
                     "<b>Type:</b> Fraudulent<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title="Transaction Amount Distribution",
        xaxis_title="Amount ($)",
        yaxis_title="Count",
        barmode="overlay",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_fraud_by_category(df: pd.DataFrame):
    """Plot fraud analysis by category."""
    if "category" not in df.columns:
        st.warning("Category column not available.")
        return
    
    category_stats = df.groupby("category").agg({
        "is_fraud": ["sum", "count", "mean"]
    }).reset_index()
    category_stats.columns = ["category", "fraud_count", "total_count", "fraud_rate"]
    category_stats["fraud_rate"] = category_stats["fraud_rate"] * 100
    category_stats = category_stats.sort_values("fraud_count", ascending=False).head(10)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fraud Cases by Category", "Fraud Rate by Category"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=category_stats["category"], y=category_stats["fraud_count"],
               marker_color="red", name="Fraud Cases"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=category_stats["category"], y=category_stats["fraud_rate"],
               marker_color="orange", name="Fraud Rate %"),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_xaxes(title_text="Category", row=1, col=2)
    fig.update_yaxes(title_text="Fraud Cases", row=1, col=1)
    fig.update_yaxes(title_text="Fraud Rate (%)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_hourly_patterns(df: pd.DataFrame):
    """Plot fraud patterns by hour of day."""
    if "hour" not in df.columns:
        st.warning("Hour column not available.")
        return
    
    hourly_stats = df.groupby("hour").agg({
        "is_fraud": ["sum", "count"]
    }).reset_index()
    hourly_stats.columns = ["hour", "fraud_count", "total_count"]
    hourly_stats["fraud_rate"] = (hourly_stats["fraud_count"] / hourly_stats["total_count"] * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_stats["hour"],
        y=hourly_stats["fraud_count"],
        name="Fraud Cases",
        marker_color="red",
        yaxis="y"
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_stats["hour"],
        y=hourly_stats["fraud_rate"],
        name="Fraud Rate %",
        mode="lines+markers",
        line=dict(color="orange", width=2),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Fraud Patterns by Hour of Day",
        xaxis=dict(title="Hour of Day", tickmode="linear"),
        yaxis=dict(title="Fraud Cases", side="left"),
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right"),
        height=400,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_score_distribution(df: pd.DataFrame):
    """Plot risk score distribution."""
    if "risk_score" not in df.columns:
        st.warning("Risk score not available.")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[df["is_fraud"] == 0]["risk_score"],
        name="Legitimate",
        opacity=0.7,
        marker_color="green",
        nbinsx=50
    ))
    
    fig.add_trace(go.Histogram(
        x=df[df["is_fraud"] == 1]["risk_score"],
        name="Fraud",
        opacity=0.7,
        marker_color="red",
        nbinsx=50
    ))
    
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Count",
        barmode="overlay",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_merchant_analysis(df: pd.DataFrame, top_n: int = 10):
    """Plot top merchants by fraud cases."""
    if "merchant" not in df.columns:
        st.warning("Merchant column not available.")
        return
    
    merchant_stats = df.groupby("merchant").agg({
        "is_fraud": ["sum", "count", "mean"],
        "amount": "sum"
    }).reset_index()
    merchant_stats.columns = ["merchant", "fraud_count", "total_count", "fraud_rate", "total_amount"]
    merchant_stats["fraud_rate"] = merchant_stats["fraud_rate"] * 100
    
    # Top by fraud count
    top_merchants = merchant_stats.sort_values("fraud_count", ascending=False).head(top_n)
    
    fig = px.bar(
        top_merchants,
        x="merchant",
        y="fraud_count",
        color="fraud_rate",
        title=f"Top {top_n} Merchants by Fraud Cases",
        labels={"fraud_count": "Fraud Cases", "fraud_rate": "Fraud Rate (%)"},
        color_continuous_scale="Reds",
        height=400
    )
    
    fig.update_xaxes(tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix(df: pd.DataFrame, threshold: float = 0.5):
    """Plot confusion matrix for risk score predictions."""
    if "risk_score" not in df.columns:
        st.warning("Risk score not available for confusion matrix.")
        return
    
    # Create predictions based on threshold
    predictions = (df["risk_score"] > threshold).astype(int)
    actuals = df["is_fraud"]
    
    # Calculate confusion matrix
    tp = ((predictions == 1) & (actuals == 1)).sum()
    tn = ((predictions == 0) & (actuals == 0)).sum()
    fp = ((predictions == 1) & (actuals == 0)).sum()
    fn = ((predictions == 0) & (actuals == 1)).sum()
    
    # Create confusion matrix
    cm = [[tn, fp], [fn, tp]]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Legitimate", "Predicted Fraud"],
        y=["Actual Legitimate", "Actual Fraud"],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale="RdYlGn_r",
        showscale=False
    ))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fig.update_layout(
        title=f"Confusion Matrix (Threshold: {threshold})<br>" +
              f"Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_hourly_patterns(df: pd.DataFrame):
    """Plot fraud patterns by hour of day."""
    st.subheader("🕐 Hourly Fraud Patterns")
    
    if 'timestamp' not in df.columns:
        st.warning("Timestamp data not available for hourly analysis.")
        return
    
    # Extract hour from timestamp
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly['timestamp'].dt.hour
    
    # Aggregate by hour
    hourly_stats = df_hourly.groupby('hour').agg({
        'is_fraud': ['sum', 'count']
    }).reset_index()
    hourly_stats.columns = ['hour', 'fraud_count', 'total_count']
    hourly_stats['fraud_rate'] = (hourly_stats['fraud_count'] / hourly_stats['total_count'] * 100)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_stats['hour'],
        y=hourly_stats['fraud_rate'],
        name="Fraud Rate %",
        marker_color='red',
        hovertemplate="<b>Hour:</b> %{x}:00<br><b>Fraud Rate:</b> %{y:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Fraud Rate by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Fraud Rate (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_merchant_analysis(df: pd.DataFrame, top_n: int = 15):
    """Plot merchant risk analysis."""
    st.subheader("🏪 Top Risk Merchants")
    
    if 'merchant' not in df.columns:
        st.warning("Merchant data not available.")
        return
    
    # Calculate merchant statistics
    merchant_stats = df.groupby('merchant').agg({
        'is_fraud': ['sum', 'count', 'mean'],
        'amount': 'sum'
    }).reset_index()
    
    merchant_stats.columns = ['merchant', 'fraud_count', 'total_count', 'fraud_rate', 'total_amount']
    
    # Filter merchants with significant transactions
    merchant_stats = merchant_stats[merchant_stats['total_count'] >= 10]
    
    # Get top risky merchants
    top_merchants = merchant_stats.nlargest(top_n, 'fraud_rate')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_merchants['merchant'],
        x=top_merchants['fraud_rate'] * 100,
        orientation='h',
        marker_color=top_merchants['fraud_rate'] * 100,
        marker_colorscale='Reds',
        hovertemplate="<b>Merchant:</b> %{y}<br><b>Fraud Rate:</b> %{x:.2f}%<br><b>Total Transactions:</b> %{customdata}<extra></extra>",
        customdata=top_merchants['total_count']
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Merchants by Fraud Rate",
        xaxis_title="Fraud Rate (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_score_distribution(df: pd.DataFrame):
    """Plot risk score distribution."""
    st.subheader("📊 Risk Score Distribution")
    
    if 'risk_score' not in df.columns:
        st.warning("Risk score not available.")
        return
    
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


def display_metrics_summary(metrics: dict):
    """Display metrics summary for AI analysis."""
    st.subheader("📈 Dataset Metrics Summary")
    
    if not metrics:
        st.warning("No metrics available.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", f"{metrics.get('total_transactions', 0):,}")
        st.metric("Fraud Cases", f"{metrics.get('total_fraud_cases', 0):,}")
    
    with col2:
        st.metric("Fraud Rate", f"{metrics.get('fraud_rate', 0)*100:.2f}%")
        st.metric("Total Amount", f"${metrics.get('total_amount', 0):,.0f}")
    
    with col3:
        st.metric("Avg Transaction", f"${metrics.get('avg_transaction_amount', 0):.2f}")
        st.metric("Avg Fraud Amount", f"${metrics.get('avg_fraud_amount', 0):.2f}")
