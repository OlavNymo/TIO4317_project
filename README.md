# S&P 500 Forecasting Project

This project aims to forecast the S&P 500 closing index price for the period from Monday, February 24th, to Friday, February 28th, 2025, utilizing historical data from 2020-2025. The analysis employs ARIMA and GARCH models to capture price movements and volatility patterns.

## Project Structure

- `data/` - Contains the historical S&P 500 data
- `notebooks/` - Jupyter notebooks for data analysis and modeling
  - `1_data_exploration.ipynb` - Initial data exploration and preprocessing
  - `2_arima_modeling.ipynb` - ARIMA model implementation and analysis
  - `3_garch_modeling.ipynb` - GARCH model implementation and analysis
  - `4_forecast_evaluation.ipynb` - Model comparison and forecasting results
- `src/` - Python modules with reusable code
- `requirements.txt` - Required Python packages
- `README.md` - Project documentation

## Setup Instructions

1. Clone this repository:

```bash
git clone https://github.com/OlavNymo/TIO4317_project.git
cd <repository-directory>
```

2. Create and activate a virtual environment:

```bash
# Using venv
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Navigate to the `notebooks/` directory and open the notebooks in sequential order.

## Methodology

The project utilizes the following methodologies:

- ARIMA (AutoRegressive Integrated Moving Average) for modeling linear trends and autocorrelations
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity) for capturing volatility clustering

Model performance is evaluated using:

- R² (coefficient of determination)
- Root Mean Squared Error (RMSE)
- Residual analysis and diagnostic checks

## Dependencies

All required dependencies are listed in the `requirements.txt` file.

## Contributors

This project was developed by:

- Olav Nikolai Meli Nymo
- Nicolai Harvik
- Olav Berger
- Ole Ekern
