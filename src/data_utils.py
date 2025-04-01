"""
Utility functions for data processing and manipulation for the S&P 500 forecasting project.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any

def load_sp500_data(file_path: str) -> pd.DataFrame:
    """
    Load S&P 500 data from CSV file and preprocess it
    """
    # Read the data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Sort by date (ascending)
    df = df.sort_values('Date')
    
    # Set Date as index (without forcing a frequency)
    df = df.set_index('Date')
    
    # Convert columns to numeric
    price_columns = ['Close/Last', 'Open', 'High', 'Low']
    for col in price_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
    
    return df

def calculate_returns(df: pd.DataFrame, column: str = 'Close/Last') -> pd.DataFrame:
    """
    Calculate daily returns based on specified price column
    """
    df = df.copy()
    df['Returns'] = df[column].pct_change() * 100
    return df

def train_test_split(df: pd.DataFrame, test_size: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets
    """
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return train_df, test_df

def plot_time_series(df: pd.DataFrame, column: str = 'Close/Last', title: str = 'S&P 500 Index',
                     figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Create a time series plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[column])
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def calculate_evaluation_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for model performance
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

def plot_forecast_comparison(actual: pd.Series, forecast_dict: Dict[str, pd.Series], 
                             title: str = 'Forecast Comparison', figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot comparison of actual values and forecasts from different models
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(actual.index, actual.values, 'o-', label='Actual', linewidth=2)
    
    # Plot forecasts
    for model_name, forecast in forecast_dict.items():
        ax.plot(forecast.index, forecast.values, 'o--', label=f'{model_name} Forecast')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    return fig

def ensure_frequency(series: pd.Series, freq: str = 'B') -> pd.Series:
    """
    Ensure that a time series has frequency information in its DatetimeIndex
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex")
    
    if series.index.freq is None:
        # Try to infer frequency first
        inferred_index = pd.DatetimeIndex(series.index, freq='infer')
        
        # If inference succeeded, use that
        if inferred_index.freq is not None:
            series = series.copy()
            series.index = inferred_index
        # Otherwise use the specified frequency
        else:
            series = series.copy()
            series.index = pd.DatetimeIndex(series.index, freq=freq)
    
    return series 