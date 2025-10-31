"""
Data loading and validation utilities.
"""
import pandas as pd
import streamlit as st
from typing import Optional, Tuple
import os


class DataLoader:
    """Handle data loading from CSV/XLSX files."""
    
    REQUIRED_COLUMNS = [
        "transaction_id",
        "timestamp",
        "amount",
        "merchant",
        "category",
        "is_fraud"
    ]
    
    @staticmethod
    def load_file(file_path: str = None, uploaded_file=None) -> Optional[pd.DataFrame]:
        """
        Load data from file path or uploaded file.
        
        Args:
            file_path: Path to local file
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            DataFrame or None if loading fails
        """
        try:
            if uploaded_file is not None:
                # Handle uploaded file
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                if file_extension == ".csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension in [".xlsx", ".xls"]:
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    return None
            elif file_path:
                # Handle local file
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == ".csv":
                    df = pd.read_csv(file_path)
                elif file_extension in [".xlsx", ".xls"]:
                    df = pd.read_excel(file_path)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    return None
            else:
                st.error("No file provided")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe has required columns for fraud analysis.
        Attempts to map common column name variations.
        
        Args:
            df: Input dataframe
            
        Returns:
            bool: True if valid, raises exception if invalid
        """
        # Column mapping for common variations
        column_mapping = {
            'transaction_id': ['transaction_id', 'txn_id', 'id', 'trans_id', 'transaction_number'],
            'timestamp': ['timestamp', 'date', 'datetime', 'trans_date', 'transaction_date', 'time'],
            'amount': ['amount', 'transaction_amount', 'amt', 'value', 'trans_amount'],
            'merchant': ['merchant', 'merchant_name', 'store', 'vendor', 'business'],
            'category': ['category', 'merchant_category', 'type', 'trans_type', 'category_code'],
            'is_fraud': ['is_fraud', 'fraud', 'fraudulent', 'is_fraudulent', 'label', 'target']
        }
        
        # Try to map columns
        mapped_columns = {}
        available_columns = [col.lower() for col in df.columns]
        
        for required_col, variations in column_mapping.items():
            found = False
            for variation in variations:
                if variation.lower() in available_columns:
                    # Find the actual column name (with original case)
                    actual_col = next(col for col in df.columns if col.lower() == variation.lower())
                    mapped_columns[required_col] = actual_col
                    found = True
                    break
            
            if not found:
                # Show available columns to help user
                raise ValueError(f"❌ Could not find column for '{required_col}'\n"
                               f"Available columns: {list(df.columns)}\n"
                               f"Expected variations: {variations}\n"
                               f"Please rename your columns or update the mapping.")
        
        # Check data types and values
        if "amount" in mapped_columns:
            if not pd.api.types.is_numeric_dtype(df[mapped_columns["amount"]]):
                raise ValueError(f"❌ 'amount' column must be numeric")
            elif (df[mapped_columns["amount"]] < 0).any():
                raise ValueError(f"❌ 'amount' column contains negative values")
        
        if "is_fraud" in mapped_columns:
            unique_vals = df[mapped_columns["is_fraud"]].unique()
            if not all(val in [0, 1, True, False, "0", "1"] for val in unique_vals):
                raise ValueError(f"❌ 'is_fraud' column must contain binary values (0/1 or True/False)")
        
        if "timestamp" in mapped_columns:
            try:
                pd.to_datetime(df[mapped_columns["timestamp"]])
            except Exception:
                raise ValueError(f"❌ 'timestamp' column cannot be parsed as datetime")
        
        return True
    
    @staticmethod
    def auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically rename columns to expected format.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with renamed columns
        """
        column_mapping = {
            'transaction_id': ['transaction_id', 'txn_id', 'id', 'trans_id', 'transaction_number'],
            'timestamp': ['timestamp', 'date', 'datetime', 'trans_date', 'transaction_date', 'time'],
            'amount': ['amount', 'transaction_amount', 'amt', 'value', 'trans_amount'],
            'merchant': ['merchant', 'merchant_name', 'store', 'vendor', 'business'],
            'category': ['category', 'merchant_category', 'type', 'trans_type', 'category_code'],
            'is_fraud': ['is_fraud', 'fraud', 'fraudulent', 'is_fraudulent', 'label', 'target']
        }
        
        df_renamed = df.copy()
        available_columns = [col.lower() for col in df.columns]
        rename_dict = {}
        
        for target_col, variations in column_mapping.items():
            for variation in variations:
                if variation.lower() in available_columns:
                    # Find the actual column name (with original case)
                    actual_col = next(col for col in df.columns if col.lower() == variation.lower())
                    if actual_col != target_col:
                        rename_dict[actual_col] = target_col
                    break
        
        if rename_dict:
            df_renamed = df_renamed.rename(columns=rename_dict)
            st.info(f"🔄 Automatically renamed columns: {rename_dict}")
        
        return df_renamed
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        """
        Get summary statistics about the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            "total_records": len(df),
            "columns": list(df.columns),
            "date_range": (
                df["timestamp"].min() if "timestamp" in df.columns else None,
                df["timestamp"].max() if "timestamp" in df.columns else None
            ),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": df.isnull().sum().to_dict(),
        }
