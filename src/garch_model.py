"""
GARCH model implementation for S&P 500 forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from typing import Tuple, Dict, Any, Optional, List

def find_optimal_garch_params(returns: pd.Series, p_range: List[int] = [1, 2], 
                             q_range: List[int] = [1, 2], power_range: List[float] = [1.0, 2.0],
                             distribution: str = 'normal') -> Dict[str, Any]:
    """
    Find optimal GARCH parameters by fitting models with different parameters
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    p_range : List[int], default=[1, 2]
        Range of p values to try
    q_range : List[int], default=[1, 2] 
        Range of q values to try
    power_range : List[float], default=[1.0, 2.0]
        Power values to try
    distribution : str, default='normal'
        Error distribution
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with optimal parameters and model
    """
    best_aic = np.inf
    best_params = None
    best_model = None
    best_result = None
    
    # Loop through combinations of parameters
    for p in p_range:
        for q in q_range:
            for power in power_range:
                try:
                    # Create model
                    model = arch_model(returns, vol='Garch', p=p, q=q, power=power, dist=distribution)
                    
                    # Fit model
                    result = model.fit(disp='off')
                    
                    # Check if this model has a lower AIC
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_params = (p, q, power)
                        best_model = model
                        best_result = result
                except:
                    continue
    
    return {
        'p': best_params[0],
        'q': best_params[1],
        'power': best_params[2],
        'model': best_model,
        'result': best_result,
        'aic': best_aic,
        'bic': best_result.bic if best_result else None
    }

def train_garch_model(returns: pd.Series, p: int = 1, q: int = 1, 
                     mean: str = 'AR', lags: int = 1, vol: str = 'GARCH',
                     power: float = 2.0, dist: str = 'normal') -> Dict[str, Any]:
    """
    Train a GARCH model with specified parameters
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    p : int, default=1
        GARCH lag order
    q : int, default=1
        ARCH lag order
    mean : str, default='AR'
        Mean model
    lags : int, default=1
        Number of lags in the mean model
    vol : str, default='GARCH'
        Volatility model
    power : float, default=2.0
        Power in the GARCH model
    dist : str, default='normal'
        Error distribution
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with model and results
    """
    # Create model
    model = arch_model(
        returns, 
        mean=mean, 
        lags=lags, 
        vol=vol, 
        p=p, 
        q=q, 
        power=power, 
        dist=dist
    )
    
    # Fit model
    result = model.fit(disp='off')
    
    return {
        'model': model,
        'result': result,
        'aic': result.aic,
        'bic': result.bic,
        'params': result.params
    }

def forecast_garch(model_result: Any, horizon: int = 5, 
                  reindex: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """
    Generate forecasts from a GARCH model
    
    Parameters:
    -----------
    model_result : Any
        Fitted GARCH model result
    horizon : int, default=5
        Forecast horizon
    reindex : Optional[pd.DatetimeIndex], default=None
        DatetimeIndex to reindex the forecast to
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with forecast results
    """
    # Generate forecast
    forecast = model_result.forecast(horizon=horizon)
    
    # Extract mean and variance forecasts
    mean_forecast = forecast.mean.iloc[-1].values
    variance_forecast = forecast.variance.iloc[-1].values
    
    # Calculate volatility forecast
    volatility_forecast = np.sqrt(variance_forecast)
    
    # Create series with forecast
    if reindex is not None:
        mean_forecast_series = pd.Series(mean_forecast, index=reindex)
        volatility_forecast_series = pd.Series(volatility_forecast, index=reindex)
    else:
        mean_forecast_series = pd.Series(mean_forecast)
        volatility_forecast_series = pd.Series(volatility_forecast)
    
    return {
        'forecast': forecast,
        'mean_forecast': mean_forecast_series,
        'volatility_forecast': volatility_forecast_series
    }

def convert_returns_to_price(last_price: float, returns_forecast: pd.Series) -> pd.Series:
    """
    Convert forecasted returns to price levels
    
    Parameters:
    -----------
    last_price : float
        Last observed price
    returns_forecast : pd.Series
        Forecasted returns (in percentage)
        
    Returns:
    --------
    pd.Series
        Forecasted prices
    """
    # Convert percentage returns to decimal
    returns_decimal = returns_forecast / 100
    
    # Initialize price series with last observed price
    prices = [last_price]
    
    # Calculate forecasted prices
    for ret in returns_decimal:
        next_price = prices[-1] * (1 + ret)
        prices.append(next_price)
    
    # Create series with forecasted prices
    price_forecast = pd.Series(prices[1:], index=returns_forecast.index)
    
    return price_forecast

def plot_garch_volatility(returns: pd.Series, model_result: Any, 
                         title: str = 'GARCH Volatility', figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot GARCH model conditional volatility
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    model_result : Any
        Fitted GARCH model result
    title : str, default='GARCH Volatility'
        Plot title
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with volatility plot
    """
    # Get conditional volatility
    conditional_vol = model_result.conditional_volatility
    
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    
    # Plot returns
    ax[0].plot(returns.index, returns)
    ax[0].set_title('Returns')
    ax[0].set_ylabel('Return (%)')
    ax[0].grid(True, alpha=0.3)
    
    # Plot conditional volatility
    ax[1].plot(conditional_vol.index, conditional_vol)
    ax[1].set_title('Conditional Volatility')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Volatility')
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_garch_forecast(returns: pd.Series, forecast_results: Dict[str, Any],
                       title: str = 'GARCH Forecast', figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot GARCH model forecasts
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    forecast_results : Dict[str, Any]
        Dictionary with forecast results
    title : str, default='GARCH Forecast'
        Plot title
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with forecast plot
    """
    # Extract forecasts
    mean_forecast = forecast_results['mean_forecast']
    volatility_forecast = forecast_results['volatility_forecast']
    
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    
    # Plot historical returns and mean forecast
    ax[0].plot(returns.index, returns, 'b-', label='Historical Returns')
    ax[0].plot(mean_forecast.index, mean_forecast, 'r--', label='Mean Forecast')
    ax[0].set_title('Returns Forecast')
    ax[0].set_ylabel('Return (%)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Plot volatility forecast
    ax[1].plot(volatility_forecast.index, volatility_forecast, 'g--', label='Volatility Forecast')
    ax[1].set_title('Volatility Forecast')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Volatility')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig 