"""
ARIMA model implementation for S&P 500 forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from typing import Tuple, Dict, Any, Optional, List

def check_stationarity(series: pd.Series) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a time series is stationary using ADF test
    
    Parameters:
    -----------
    series : pd.Series
        Time series to check
        
    Returns:
    --------
    Tuple[bool, Dict[str, Any]]
        Boolean indicating stationarity and test results
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Perform ADF test
    result = adfuller(series.dropna())
    
    # Extract and format test results
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Determine if stationary (p-value < 0.05)
    is_stationary = p_value < 0.05
    
    test_results = {
        'ADF Statistic': adf_stat,
        'p-value': p_value,
        'Critical Values': critical_values,
        'Is Stationary': is_stationary
    }
    
    return is_stationary, test_results

def find_optimal_arima_params(series: pd.Series, seasonal: bool = False, m: int = 1,
                             max_p: int = 5, max_d: int = 2, max_q: int = 5,
                             max_P: int = 2, max_D: int = 1, max_Q: int = 2) -> Dict[str, Any]:
    """
    Find optimal ARIMA parameters using auto_arima
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    seasonal : bool, default=False
        Whether to include seasonal component
    m : int, default=1
        The period for seasonal differencing
    max_p, max_d, max_q : int
        Maximum values for ARIMA(p,d,q) parameters
    max_P, max_D, max_Q : int
        Maximum values for seasonal ARIMA parameters
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with optimal parameters and model summary
    """
    # Find optimal parameters using auto_arima
    model = auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        start_p=0,
        start_q=0,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        start_P=0,
        start_Q=0,
        max_P=max_P,
        max_D=max_D,
        max_Q=max_Q,
        d=None,
        D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    # Get parameters
    params = model.get_params()
    order = model.order
    seasonal_order = model.seasonal_order if seasonal else None
    
    return {
        'order': order,
        'seasonal_order': seasonal_order,
        'aic': model.aic(),
        'bic': model.bic(),
        'model': model
    }

def train_arima_model(series: pd.Series, order: Tuple[int, int, int],
                     seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """
    Train an ARIMA model with specified parameters
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    order : Tuple[int, int, int]
        ARIMA order parameters (p,d,q)
    seasonal_order : Optional[Tuple[int, int, int, int]], default=None
        Seasonal order parameters (P,D,Q,s)
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with model and results
    """
    # Create and fit model
    # Note: We're not trying to force a frequency on the DatetimeIndex anymore
    # statsmodels will work with the data as is, but will generate warnings
    if seasonal_order:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(series, order=order)
    
    # Fit with appropriate options for irregular time series
    results = model.fit()
    
    # Perform diagnostic checks
    residuals = results.resid
    ljung_box_results = acorr_ljungbox(residuals, lags=[10], return_df=True)
    
    return {
        'model': model,
        'results': results,
        'residuals': residuals,
        'ljung_box': ljung_box_results,
        'aic': results.aic,
        'bic': results.bic
    }

def forecast_arima(model_results: Any, steps: int, 
                  return_conf_int: bool = True, alpha: float = 0.05,
                  forecast_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """
    Generate forecasts from an ARIMA model
    
    Parameters:
    -----------
    model_results : Any
        Fitted ARIMA model results
    steps : int
        Number of steps to forecast
    return_conf_int : bool, default=True
        Whether to return confidence intervals
    alpha : float, default=0.05
        Significance level for confidence intervals
    forecast_index : Optional[pd.DatetimeIndex], default=None
        DatetimeIndex to use for the forecast results
        If None, the forecast will have integer indices
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with forecast results
    """
    # Generate forecast
    forecast = model_results.forecast(steps=steps)
    
    # If forecast_index is provided, use it for the forecast
    if forecast_index is not None and len(forecast_index) == len(forecast):
        forecast.index = forecast_index
    
    # Get confidence intervals if requested
    if return_conf_int:
        conf_int = model_results.get_forecast(steps=steps).conf_int(alpha=alpha)
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        
        # Use the same index for confidence intervals if provided
        if forecast_index is not None and len(forecast_index) == len(lower):
            lower.index = forecast_index
            upper.index = forecast_index
    else:
        lower, upper = None, None
    
    return {
        'forecast': forecast,
        'lower_ci': lower,
        'upper_ci': upper
    }

def plot_arima_diagnostics(model_results: Any, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot ARIMA model diagnostics
    
    Parameters:
    -----------
    model_results : Any
        Fitted ARIMA model results
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with diagnostic plots
    """
    fig = model_results.plot_diagnostics(figsize=figsize)
    plt.tight_layout()
    return fig

def plot_arima_forecast(actual: pd.Series, forecast_results: Dict[str, Any], 
                       title: str = 'ARIMA Forecast', figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot ARIMA forecast with confidence intervals
    
    Parameters:
    -----------
    actual : pd.Series
        Actual time series data
    forecast_results : Dict[str, Any]
        Dictionary with forecast results
    title : str, default='ARIMA Forecast'
        Plot title
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with forecast plot
    """
    forecast = forecast_results['forecast']
    lower_ci = forecast_results.get('lower_ci')
    upper_ci = forecast_results.get('upper_ci')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data
    ax.plot(actual.index, actual, 'b-', label='Historical Data')
    
    # Plot forecast
    ax.plot(forecast.index, forecast, 'r--', label='Forecast')
    
    # Plot confidence intervals if available
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(forecast.index, lower_ci, upper_ci, color='pink', alpha=0.3, label='95% Confidence Interval')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig 