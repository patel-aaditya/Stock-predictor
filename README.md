# Stock Price Predictor (Single File)

A simple, single-file Python app that:

-   Pulls the last \~356 trading days of stock data
-   Calculates basic technical indicators (SMA, EMA, RSI, MACD)
-   Generates buy/sell signals
-   Backtests a simple rule-based strategy
-   Predicts the next 5 days of prices using Linear Regression
-   Plots charts using Matplotlib

------------------------------------------------------------------------

## Requirements

-   Python 3.9+
-   Required packages:
    -   `yfinance`
    -   `pandas`
    -   `numpy`
    -   `scikit-learn`
    -   `matplotlib`

------------------------------------------------------------------------

## Installation

``` powershell
pip install yfinance pandas numpy scikit-learn matplotlib
```

------------------------------------------------------------------------

## Run

``` powershell
python main.py
```

Make sure your script file is named `main.py`, or update the command
accordingly.

------------------------------------------------------------------------

## What It Does

1.  Downloads historical stock data using `yfinance`
2.  Computes technical indicators:
    -   20 & 50 Day Simple Moving Averages (SMA)
    -   12 & 26 Day Exponential Moving Averages (EMA)
    -   MACD (Momentum Indicator)
    -   RSI (Relative Strength Index)
3.  Generates trading signals based on SMA crossover + RSI filter
4.  Backtests strategy performance vs Buy & Hold
5.  Trains a Linear Regression model
6.  Forecasts the next 5 days of prices
7.  Visualizes results with charts

------------------------------------------------------------------------

## Disclaimer

This project is for educational purposes only.

It does NOT: - Include transaction costs - Model slippage - Implement
risk management - Provide financial advice

Use at your own discretion.
