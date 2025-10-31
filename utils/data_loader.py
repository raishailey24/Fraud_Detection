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
    def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate that DataFrame has required columns and proper data types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if df is None or df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for required columns
        missing_cols = [col for col in DataLoader.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check data types and values
        if "amount" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["amount"]):
                errors.append("'amount' column must be numeric")
            elif (df["amount"] < 0).any():
                errors.append("'amount' column contains negative values")
        
        if "is_fraud" in df.columns:
            unique_vals = df["is_fraud"].unique()
            if not all(val in [0, 1, True, False, "0", "1"] for val in unique_vals):
                errors.append("'is_fraud' column must contain binary values (0/1 or True/False)")
        
        if "timestamp" in df.columns:
            try:
                pd.to_datetime(df["timestamp"])
            except Exception:
                errors.append("'timestamp' column cannot be parsed as datetime")
        
        return len(errors) == 0, errors
    
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
