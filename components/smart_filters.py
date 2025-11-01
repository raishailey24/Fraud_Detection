"""
Smart analytical filters for performance optimization.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px


def _format_number(num):
    """Format large numbers with K, M, B abbreviations."""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))


def _format_currency(amount):
    """Format currency amounts with K, M, B abbreviations."""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.1f}K"
    else:
        return f"${amount:.2f}"


def apply_smart_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply intelligent filters for performance and analysis."""
    
    if df is None or df.empty:
        st.sidebar.warning("⚠️ No data available for filtering")
        return df
    
    dataset_size = len(df)
    
    # Remove debug info for cleaner interface
    
    try:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Smart Filters")
        dataset_size = len(df)
        
        # Force convert all numeric columns immediately
        df_clean = df.copy()
        for col in df_clean.columns:
            if col in ['amount', 'is_fraud', 'user_id']:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    pass
            elif col in ['timestamp']:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                except:
                    pass
        
        # Use cleaned dataframe for all operations
        df = df_clean
        
        # Data limit options - exactly 5 options as requested - ensure all are integers
        limit_options = [100000, 500000, 1000000, 5000000, int(dataset_size)]
        limit_labels = [
            "100K records",
            "500K records", 
            "1M records",
            "5M records",
            f"All Data ({_format_number(dataset_size)} records)"
        ]
        
        # Filter options to only show relevant limits
        valid_options = []
        valid_labels = []
        for limit, label in zip(limit_options, limit_labels):
            if limit <= dataset_size:
                valid_options.append(limit)
                valid_labels.append(label)
        
        # Ensure we have at least one option
        if not valid_options:
            valid_options = [dataset_size]
            valid_labels = [f"All {dataset_size:,} records"]
        
        # Default to 1M records if available, otherwise use the largest available option
        default_index = 0  # Start with first option as fallback
        
        # Try to find 1M records option
        for i, limit in enumerate(valid_options):
            if limit == 1000000:  # 1M records
                default_index = i
                break
        
        # If no 1M option, use the largest available (last in list)
        if default_index == 0 and len(valid_options) > 1:
            default_index = len(valid_options) - 1
        
        # Ensure index is within bounds
        if default_index >= len(valid_options):
            default_index = len(valid_options) - 1
        
        selected_limit = st.sidebar.selectbox(
            "📊 Data Limit",
            options=valid_options,
            format_func=lambda x: valid_labels[valid_options.index(x)] if valid_options else "No data",
            index=default_index if valid_options else 0,
            help="Choose how many records to analyze - defaults to 1M records or largest available",
            key="data_limit_selector"
        )
        
        # Ensure selected_limit is always an integer
        selected_limit = int(selected_limit)
        
        filtered_df = df.copy()
        
        # Force data type conversion as backup
        try:
            if 'amount' in filtered_df.columns:
                # More aggressive conversion
                filtered_df['amount'] = filtered_df['amount'].astype(str).str.replace('$', '').str.replace(',', '')
                filtered_df['amount'] = pd.to_numeric(filtered_df['amount'], errors='coerce')
                
            if 'is_fraud' in filtered_df.columns:
                filtered_df['is_fraud'] = pd.to_numeric(filtered_df['is_fraud'], errors='coerce')
            if 'timestamp' in filtered_df.columns:
                filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'], errors='coerce')
            
            # Remove any rows with invalid data
            filtered_df = filtered_df.dropna(subset=[col for col in ['amount', 'is_fraud'] if col in filtered_df.columns])
            
        except Exception as e:
            st.sidebar.error(f"Data conversion error: {e}")
            return df
        
        # Active filters section
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎯 Data Filters**")
        st.sidebar.markdown("*All data selected by default - adjust as needed*")
        
        # Date range filter (active by default)
        if 'timestamp' in df.columns and len(filtered_df) > 0:
            try:
                min_date = filtered_df['timestamp'].min().date()
                max_date = filtered_df['timestamp'].max().date()
                
                st.sidebar.markdown("📅 **Date Range**")
                date_range = st.sidebar.date_input(
                    "Select dates to analyze",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_selector"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df['timestamp'].dt.date >= start_date) & 
                        (filtered_df['timestamp'].dt.date <= end_date)
                    ]
            except Exception as e:
                st.sidebar.warning(f"Date filter skipped: {str(e)}")
                pass
        
        # Amount range filter (active by default)
        if 'amount' in df.columns and len(filtered_df) > 0:
            try:
                amount_min = float(filtered_df['amount'].min())
                amount_max = float(filtered_df['amount'].max())
                
                st.sidebar.markdown("💰 **Amount Range**")
                amount_range = st.sidebar.slider(
                    "Select transaction amounts",
                    min_value=amount_min,
                    max_value=amount_max,
                    value=(amount_min, amount_max),
                    format="$%.2f",
                    key="amount_range_selector"
                )
                
                filtered_df = filtered_df[
                    (filtered_df['amount'] >= amount_range[0]) & 
                    (filtered_df['amount'] <= amount_range[1])
                ]
            except Exception as e:
                st.sidebar.warning(f"Amount filter skipped: {str(e)}")
                pass
        
        # Location filter (active by default) - prioritize 'location' column
        location_columns = ['location', 'city', 'state', 'merchant', 'region']
        available_location_col = None
        for col in location_columns:
            if col in df.columns:
                available_location_col = col
                break
        
        if available_location_col:
            # Convert all location values to strings and remove NaN/None values
            location_values = df[available_location_col].dropna().astype(str).unique()
            locations = sorted([loc for loc in location_values if loc != 'nan' and loc != 'None'])
            
            st.sidebar.markdown("📍 **Location Filter**")
            selected_locations = st.sidebar.multiselect(
                f"Select locations to analyze",
                options=locations,
                default=locations,  # All locations selected by default
                key="location_selector"
            )
            
            if selected_locations:
                filtered_df = filtered_df[filtered_df[available_location_col].isin(selected_locations)]
            else:
                # If nothing selected, show all data
                pass
        
        # Category filter (active by default) - check for merchant_category or category
        category_col = None
        for col in ['merchant_category', 'category', 'transaction_type']:
            if col in df.columns:
                category_col = col
                break
                
        if category_col:
            # Convert all category values to strings and remove NaN/None values
            category_values = df[category_col].dropna().astype(str).unique()
            categories = sorted([cat for cat in category_values if cat != 'nan' and cat != 'None'])
            
            st.sidebar.markdown("🏷️ **Transaction Categories**")
            selected_categories = st.sidebar.multiselect(
                "Select categories to analyze",
                options=categories,
                default=categories,  # All categories selected by default
                key="category_selector"
            )
            
            if selected_categories:
                filtered_df = filtered_df[filtered_df[category_col].isin(selected_categories)]
        
        # Fraud focus filter (active by default)
        st.sidebar.markdown("🚨 **Transaction Focus**")
        fraud_filter = st.sidebar.selectbox(
            "Choose analysis focus",
            options=["All Transactions", "Fraud Only", "Legitimate Only", "High Risk"],
            index=0,
            key="fraud_focus_selector"
        )
        
        try:
            if fraud_filter == "Fraud Only" and 'is_fraud' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['is_fraud'] == 1]
            elif fraud_filter == "Legitimate Only" and 'is_fraud' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['is_fraud'] == 0]
            elif fraud_filter == "High Risk" and 'risk_score' in df.columns:
                filtered_df = filtered_df[filtered_df['risk_score'] > 0.7]
        except Exception as e:
            st.sidebar.warning(f"Fraud filter skipped: {str(e)}")
            pass
        
        # Apply data limit with stratified sampling
        if len(filtered_df) > selected_limit:
            # Stratified sampling to maintain fraud ratio
            if 'is_fraud' in filtered_df.columns:
                fraud_df = filtered_df[filtered_df['is_fraud'] == 1]
                legit_df = filtered_df[filtered_df['is_fraud'] == 0]
                
                fraud_ratio = len(fraud_df) / len(filtered_df)
                fraud_sample_size = int(selected_limit * fraud_ratio)
                legit_sample_size = selected_limit - fraud_sample_size
                
                fraud_sample = fraud_df.sample(n=min(fraud_sample_size, len(fraud_df)), random_state=42)
                legit_sample = legit_df.sample(n=min(legit_sample_size, len(legit_df)), random_state=42)
                
                filtered_df = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)
            else:
                filtered_df = filtered_df.sample(n=selected_limit, random_state=42)
        
        # Show filter results
        st.sidebar.markdown("---")
        st.sidebar.metric("📋 Filtered Records", _format_number(len(filtered_df)))
        
        if len(filtered_df) < len(df):
            reduction = (1 - len(filtered_df) / len(df)) * 100
            st.sidebar.metric("📉 Data Reduction", f"{reduction:.1f}%")
        
        return filtered_df
        
    except Exception as e:
        st.sidebar.error(f"❌ Filter error: {str(e)}")
        st.sidebar.info("Using original dataset without filters")
        
        # Additional debugging
        st.sidebar.write("**Error Debug Info:**")
        st.sidebar.write(f"Error type: {type(e).__name__}")
        
        import traceback
        error_trace = traceback.format_exc()
        st.sidebar.text_area("Full Error:", error_trace, height=100)
        
        return df


def show_filter_summary(original_df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Show a summary of applied filters."""
    
    if len(filtered_df) == len(original_df):
        st.info("🔍 **No filters applied** - Showing complete dataset")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Original Records", 
            _format_number(len(original_df)),
            help="Total records in dataset"
        )
    
    with col2:
        st.metric(
            "Filtered Records", 
            _format_number(len(filtered_df)),
            help="Records after applying filters"
        )
    
    with col3:
        reduction = (1 - len(filtered_df) / len(original_df)) * 100
        st.metric(
            "Reduction", 
            f"{reduction:.1f}%",
            help="Percentage of data filtered out"
        )
