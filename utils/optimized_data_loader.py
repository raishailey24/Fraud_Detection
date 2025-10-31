"""
Optimized data loader with sampling and performance improvements.
Handles large datasets efficiently for Streamlit.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional
import time


class OptimizedDataLoader:
    """Optimized data loader for large datasets."""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_sample_data(file_path: str, sample_size: int = 100000) -> pd.DataFrame:
        """
        Load a representative sample of the dataset for fast visualization.
        
        Args:
            file_path: Path to the CSV file
            sample_size: Number of records to sample
            
        Returns:
            Sampled DataFrame
        """
        try:
            # Get total number of rows first
            total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
            
            if total_rows <= sample_size:
                # If dataset is small enough, load all data
                return pd.read_csv(file_path)
            
            # Calculate skip probability for random sampling
            skip_prob = 1 - (sample_size / total_rows)
            
            # Random sampling with pandas
            df = pd.read_csv(
                file_path,
                skiprows=lambda i: i > 0 and np.random.random() < skip_prob
            )
            
            # Ensure we have the required columns
            required_cols = ['transaction_id', 'timestamp', 'amount', 'merchant', 
                           'category', 'is_fraud', 'risk_level']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            # Optimize data types
            df = OptimizedDataLoader._optimize_dtypes(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_aggregated_data(file_path: str) -> dict:
        """
        Load pre-aggregated statistics for fast KPI display.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary of aggregated statistics
        """
        try:
            # Load data in chunks and aggregate
            chunk_size = 50000
            stats = {
                'total_transactions': 0,
                'total_fraud': 0,
                'total_amount': 0.0,
                'fraud_amount': 0.0,
                'date_range': {'min': None, 'max': None},
                'categories': set(),
                'risk_levels': {'low': 0, 'medium': 0, 'high': 0}
            }
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Basic counts
                stats['total_transactions'] += len(chunk)
                stats['total_fraud'] += chunk['is_fraud'].sum()
                stats['total_amount'] += chunk['amount'].sum()
                stats['fraud_amount'] += chunk[chunk['is_fraud'] == 1]['amount'].sum()
                
                # Date range
                chunk_dates = pd.to_datetime(chunk['timestamp'])
                if stats['date_range']['min'] is None:
                    stats['date_range']['min'] = chunk_dates.min()
                    stats['date_range']['max'] = chunk_dates.max()
                else:
                    stats['date_range']['min'] = min(stats['date_range']['min'], chunk_dates.min())
                    stats['date_range']['max'] = max(stats['date_range']['max'], chunk_dates.max())
                
                # Categories
                stats['categories'].update(chunk['category'].unique())
                
                # Risk levels
                risk_counts = chunk['risk_level'].value_counts()
                for level in ['low', 'medium', 'high']:
                    stats['risk_levels'][level] += risk_counts.get(level, 0)
            
            # Convert sets to lists for JSON serialization
            stats['categories'] = list(stats['categories'])
            
            return stats
            
        except Exception as e:
            st.error(f"Error loading aggregated data: {str(e)}")
            return {}
    
    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Optimize numeric columns
        numeric_cols = ['amount', 'risk_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize categorical columns
        categorical_cols = ['category', 'risk_level', 'merchant']
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        # Optimize boolean/integer columns
        if 'is_fraud' in df.columns:
            df['is_fraud'] = df['is_fraud'].astype('int8')
        
        return df
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_filtered_sample(df: pd.DataFrame, filters: dict, max_size: int = 50000) -> pd.DataFrame:
        """
        Apply filters and return a manageable sample.
        
        Args:
            df: Input DataFrame
            filters: Dictionary of filter conditions
            max_size: Maximum number of records to return
            
        Returns:
            Filtered and sampled DataFrame
        """
        filtered_df = df.copy()
        
        # Apply filters
        if filters.get('date_range'):
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
        
        # Sample if too large
        if len(filtered_df) > max_size:
            filtered_df = filtered_df.sample(n=max_size, random_state=42)
        
        return filtered_df


class DataCache:
    """Simple caching mechanism for expensive operations."""
    
    _cache = {}
    
    @classmethod
    def get(cls, key: str):
        """Get cached value."""
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value, ttl: int = 3600):
        """Set cached value with TTL."""
        cls._cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    @classmethod
    def is_valid(cls, key: str) -> bool:
        """Check if cached value is still valid."""
        if key not in cls._cache:
            return False
        
        item = cls._cache[key]
        return (time.time() - item['timestamp']) < item['ttl']
    
    @classmethod
    def clear(cls):
        """Clear all cached values."""
        cls._cache.clear()
