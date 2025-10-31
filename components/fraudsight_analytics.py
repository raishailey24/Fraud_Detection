"""
FraudSight Analytics Tab - Deep-dive exploration into patterns and correlations.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from components.smart_filters import _format_number, _format_currency


def display_fraudsight_analytics(df: pd.DataFrame):
    """Display FraudSight Analytics tab with pattern exploration and risk analysis."""
    
    # Top Section - Fraud Pattern Explorer
    st.markdown("### 🔍 Fraud Pattern Explorer")
    
    # Create a clear, readable fraud pattern analysis
    if 'category' in df.columns:
        # Get top 6 categories by transaction volume for clarity
        top_categories = df['category'].value_counts().head(6).index.tolist()
        df_filtered = df[df['category'].isin(top_categories)]
        
        # Simple category vs fraud rate analysis
        category_analysis = df_filtered.groupby('category').agg({
            'is_fraud': ['sum', 'count'],
            'amount': 'mean'
        }).reset_index()
        category_analysis.columns = ['category', 'fraud_count', 'total_count', 'avg_amount']
        category_analysis['fraud_rate'] = (category_analysis['fraud_count'] / category_analysis['total_count'] * 100)
        
        # Create a clear bar chart instead of confusing heatmap
        fig = go.Figure()
        
        # Add fraud rate bars
        fig.add_trace(go.Bar(
            name='Fraud Rate (%)',
            x=category_analysis['category'],
            y=category_analysis['fraud_rate'],
            marker_color='rgba(239, 68, 68, 0.8)',
            text=[f"{rate:.2f}%" for rate in category_analysis['fraud_rate']],
            textposition='auto',
            yaxis='y'
        ))
        
        # Add transaction count on secondary axis
        fig.add_trace(go.Scatter(
            name='Total Transactions',
            x=category_analysis['category'],
            y=category_analysis['total_count'],
            mode='markers+lines',
            marker=dict(size=10, color='rgba(59, 130, 246, 0.8)'),
            line=dict(color='rgba(59, 130, 246, 0.8)', width=2),
            yaxis='y2'
        ))
        
        # Update layout for dual axis
        fig.update_layout(
            title="Fraud Analysis by Transaction Category",
            xaxis_title="Transaction Category",
            yaxis=dict(title="Fraud Rate (%)", side="left"),
            yaxis2=dict(title="Total Transactions", side="right", overlaying="y"),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary table for detailed insights
        st.markdown("#### 📊 Category Fraud Summary")
        
        # Format the data for display
        display_data = category_analysis.copy()
        display_data['fraud_rate'] = display_data['fraud_rate'].round(2)
        display_data['avg_amount'] = display_data['avg_amount'].round(2)
        display_data.columns = ['Category', 'Fraud Cases', 'Total Transactions', 'Avg Amount ($)', 'Fraud Rate (%)']
        
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )
    
    # Middle Section - Customer Risk Segmentation & Correlation Matrix
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 👥 Customer Risk Segmentation")
        
        # Create customer segmentation data
        if 'customer_id' not in df.columns:
            df['customer_id'] = np.random.randint(1000, 9999, size=len(df))
        
        customer_stats = df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': 'mean',
            'is_fraud': 'mean'
        }).reset_index()
        customer_stats.columns = ['customer_id', 'transaction_frequency', 'avg_amount', 'fraud_ratio']
        
        # Sample for visualization performance
        if len(customer_stats) > 1000:
            customer_stats = customer_stats.sample(1000, random_state=42)
        
        # Create bubble chart
        fig = go.Figure()
        
        # Separate fraud and non-fraud customers
        fraud_customers = customer_stats[customer_stats['fraud_ratio'] > 0]
        clean_customers = customer_stats[customer_stats['fraud_ratio'] == 0]
        
        # Clean customers
        if len(clean_customers) > 0:
            fig.add_trace(go.Scatter(
                x=clean_customers['transaction_frequency'],
                y=clean_customers['avg_amount'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(34, 197, 94, 0.6)',
                    line=dict(width=1, color='rgba(34, 197, 94, 0.8)')
                ),
                name='Clean Customers',
                hovertemplate='Frequency: %{x}<br>Avg Amount: $%{y:.2f}<br>Status: Clean<extra></extra>'
            ))
        
        # Fraud customers
        if len(fraud_customers) > 0:
            fig.add_trace(go.Scatter(
                x=fraud_customers['transaction_frequency'],
                y=fraud_customers['avg_amount'],
                mode='markers',
                marker=dict(
                    size=fraud_customers['fraud_ratio'] * 50 + 10,  # Size based on fraud ratio
                    color='rgba(239, 68, 68, 0.7)',
                    line=dict(width=2, color='rgba(239, 68, 68, 1)')
                ),
                name='Risky Customers',
                hovertemplate='Frequency: %{x}<br>Avg Amount: $%{y:.2f}<br>Fraud Ratio: %{marker.size:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title="Customer Risk Profile",
            xaxis_title="Transaction Frequency",
            yaxis_title="Average Transaction Amount ($)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### 🔗 Feature Correlation Matrix")
        
        # Prepare numerical features for correlation
        numerical_cols = ['amount', 'is_fraud']
        if 'risk_score' in df.columns:
            numerical_cols.append('risk_score')
        
        # Add derived features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            numerical_cols.extend(['hour', 'day_of_week'])
        
        # Calculate correlation matrix
        corr_data = df[numerical_cols].corr()
        
        # Create correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.index,
            colorscale='RdBu',
            zmid=0,
            text=[[f"{val:.2f}" for val in row] for row in corr_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation with Fraud",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom Section - Fraud Rule Simulation Sandbox
    st.markdown("### 🎯 Fraud Rule Simulation Sandbox")
    
    # Interactive rule controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount_threshold = st.slider(
            "Amount Threshold ($)",
            min_value=0,
            max_value=int(df['amount'].max()) if 'amount' in df.columns else 10000,
            value=5000,
            step=100,
            help="Flag transactions above this amount"
        )
    
    with col2:
        if 'risk_score' in df.columns:
            risk_threshold = st.slider(
                "Risk Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Flag transactions above this risk score"
            )
        else:
            risk_threshold = 0.7
    
    with col3:
        if 'hour' in df.columns:
            night_hours = st.checkbox(
                "Flag Night Transactions",
                value=True,
                help="Flag transactions between 10 PM and 6 AM"
            )
        else:
            night_hours = False
    
    # Apply rules and calculate metrics
    rule_flags = pd.Series(False, index=df.index)
    
    # Amount rule
    if 'amount' in df.columns:
        rule_flags |= (df['amount'] > amount_threshold)
    
    # Risk score rule
    if 'risk_score' in df.columns:
        rule_flags |= (df['risk_score'] > risk_threshold)
    
    # Night hours rule
    if night_hours and 'hour' in df.columns:
        rule_flags |= ((df['hour'] >= 22) | (df['hour'] <= 6))
    
    # Calculate performance metrics
    true_positives = ((rule_flags == True) & (df['is_fraud'] == 1)).sum()
    false_positives = ((rule_flags == True) & (df['is_fraud'] == 0)).sum()
    true_negatives = ((rule_flags == False) & (df['is_fraud'] == 0)).sum()
    false_negatives = ((rule_flags == False) & (df['is_fraud'] == 1)).sum()
    
    total_flagged = rule_flags.sum()
    total_fraud = df['is_fraud'].sum()
    
    catch_rate = (true_positives / total_fraud * 100) if total_fraud > 0 else 0
    false_positive_rate = (false_positives / (false_positives + true_negatives) * 100) if (false_positives + true_negatives) > 0 else 0
    precision = (true_positives / total_flagged * 100) if total_flagged > 0 else 0
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Catch Rate", f"{catch_rate:.1f}%", help="% of fraud cases detected")
    
    with col2:
        st.metric("False Positive Rate", f"{false_positive_rate:.1f}%", help="% of clean transactions flagged")
    
    with col3:
        st.metric("Precision", f"{precision:.1f}%", help="% of flags that are actual fraud")
    
    with col4:
        st.metric("Total Flagged", _format_number(total_flagged), help="Total transactions flagged")
    
    # Performance visualization
    metrics_data = {
        'Metric': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'],
        'Count': [true_positives, false_positives, true_negatives, false_negatives],
        'Color': ['#22c55e', '#ef4444', '#3b82f6', '#f59e0b']
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics_data['Metric'],
            y=metrics_data['Count'],
            marker_color=metrics_data['Color'],
            text=[_format_number(count) for count in metrics_data['Count']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Rule Performance Breakdown",
        xaxis_title="Classification",
        yaxis_title="Count",
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_ai_recommendations_panel(df: pd.DataFrame):
    """Display AI recommendations sidebar panel."""
    
    st.markdown("### 💡 AI Insights")
    
    # Generate contextual recommendations
    recommendations = generate_ai_recommendations(df)
    
    for i, rec in enumerate(recommendations[:3]):
        # Use Streamlit's info component instead of raw HTML
        st.info(f"💡 **Insight #{i+1}**: {rec}")


def generate_ai_recommendations(df: pd.DataFrame) -> list:
    """Generate AI-powered recommendations based on data patterns."""
    recommendations = []
    
    if 'category' in df.columns and 'transaction_mode' in df.columns:
        # Category + mode analysis
        high_risk = df.groupby(['category', 'transaction_mode'])['is_fraud'].mean().reset_index()
        high_risk = high_risk.nlargest(1, 'is_fraud').iloc[0]
        
        recommendations.append(
            f"Transactions in {high_risk['category']} using {high_risk['transaction_mode']} "
            f"show {high_risk['is_fraud']*100:.1f}% fraud risk. Consider enhanced verification."
        )
    
    if 'amount' in df.columns:
        # Amount-based insights
        fraud_amounts = df[df['is_fraud'] == 1]['amount']
        clean_amounts = df[df['is_fraud'] == 0]['amount']
        
        if len(fraud_amounts) > 0 and len(clean_amounts) > 0:
            fraud_median = fraud_amounts.median()
            clean_median = clean_amounts.median()
            
            if fraud_median > clean_median * 1.5:
                recommendations.append(
                    f"Fraudulent transactions have {fraud_median/clean_median:.1f}x higher median amount "
                    f"(${fraud_median:.0f} vs ${clean_median:.0f}). Consider amount-based rules."
                )
    
    if 'timestamp' in df.columns:
        # Time-based insights
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        night_fraud_rate = df[(df['hour'] >= 22) | (df['hour'] <= 6)]['is_fraud'].mean()
        day_fraud_rate = df[(df['hour'] >= 8) & (df['hour'] <= 18)]['is_fraud'].mean()
        
        if night_fraud_rate > day_fraud_rate * 2:
            recommendations.append(
                f"Night-time transactions show {night_fraud_rate*100:.1f}% fraud rate vs "
                f"{day_fraud_rate*100:.1f}% during business hours. Implement time-based scoring."
            )
    
    # Default recommendations if no patterns found
    if not recommendations:
        recommendations = [
            "Consider implementing velocity checks for repeat transactions from same customer.",
            "Geographic anomaly detection could help identify location-based fraud patterns.",
            "Machine learning models could improve detection accuracy beyond rule-based systems."
        ]
    
    return recommendations
