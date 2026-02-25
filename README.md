# Stock Indicator Analyser — IRFC.NS (Single File)

A single-file Python app that downloads historical stock data for **IRFC.NS** (Indian Railway Finance Corporation) and performs multi-indicator technical analysis, signal generation, and rolling accuracy evaluation — all visualised in dark-themed Matplotlib charts.

---

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

```
Stock-predictor/
└── main.py   # Single-file app — all logic lives here
```

---

## Disclaimer

This project is for **educational purposes only**.

It does **NOT**:
- Include transaction costs
- Model slippage
- Implement risk management
- Provide financial advice

Use at your own discretion.