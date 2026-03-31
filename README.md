# Stock Indicator Analyser — IRFC.NS (Single File)

A single-file Python app that downloads historical stock data for **IRFC.NS** (Indian Railway Finance Corporation) and performs multi-indicator technical analysis, signal generation, and rolling accuracy evaluation — all visualised in dark-themed Matplotlib charts.

-   Pulls the last \~356 trading days of **Google (GOOGL)** stock data
-   Calculates basic technical indicators (SMA, EMA, RSI, MACD, OBV)
-   Generates buy/sell signals
-   Backtests a simple rule-based strategy
-   Predicts the next 5 days of prices using Linear Regression
-   Plots charts using Matplotlib

## What It Does

1. **Downloads historical data** for `IRFC.NS` from `2021-01-29` to `2026-02-25` using `yfinance`
2. **Computes five technical indicators** from scratch (no TA libraries):
   - **OBV** — On-Balance Volume
   - **A/D** — Accumulation/Distribution Line
   - **RSI** — Relative Strength Index (14-period)
   - **MA** — 20 & 50-day Simple Moving Averages
   - **LinReg** — 50-period Linear Regression line (via `scipy.stats.linregress`)
3. **Generates buy/sell signals** for each indicator:
   - OBV crossover above/below its 20-day MA
   - A/D line direction change
   - RSI oversold (<30) / overbought (>70) crossover
   - MA 20/50 golden/death cross
   - Price crossover above/below the Linear Regression line
4. **Evaluates rolling signal accuracy** over a 20-day window — how often each indicator's signal correctly predicted the next-day price direction
5. **Produces two multi-panel charts** across three time windows:
   - `5 Years (2021–2026)`
   - `1 Year (last 12m)`
   - `120 Weeks (~2.3 yrs)`

### Output Files

| File | Description |
|------|-------------|
| `irfc_overlay.png` | All indicators normalised to 0–100 and overlaid on price |
| `irfc_accuracy.png` | Rolling 20-day signal accuracy (%) for each indicator |

---

## Requirements

- Python 3.9+
- Required packages:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `scipy`
  - `matplotlib`

---

## Installation

```powershell
pip install yfinance pandas numpy scipy matplotlib
```

---

## Run

```powershell
python main.py
```

Make sure your script file is named `main.py`, or update the command accordingly.

---

## Project Structure

1.  Downloads historical **Google (GOOGL)** stock data using `yfinance`
2.  Computes technical indicators:
    -   20 & 50 Day Simple Moving Averages (SMA)
    -   12 & 26 Day Exponential Moving Averages (EMA)
    -   MACD (Momentum Indicator)
    -   RSI (Relative Strength Index)
    -   OBV (On-Balance Volume)
3.  Generates trading signals based on SMA crossover + RSI filter
4.  Backtests strategy performance vs Buy & Hold
5.  Trains a Linear Regression model
6.  Forecasts the next 5 days of prices
7.  Visualizes results with charts

---

## Disclaimer

This project is for **educational purposes only**.

It does **NOT**:
- Include transaction costs
- Model slippage
- Implement risk management
- Provide financial advice

Use at your own discretion.