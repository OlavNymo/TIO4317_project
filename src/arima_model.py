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
    """
    # Create and fit model

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
    """
    fig = model_results.plot_diagnostics(figsize=figsize)
    plt.tight_layout()
    return fig

def plot_arima_forecast(actual: pd.Series, forecast_results: Dict[str, Any], 
                       title: str = 'ARIMA Forecast', figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot ARIMA forecast with confidence intervals
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
