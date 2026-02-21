import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with `pip install yfinance`.")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


def fetch_last_356_days(ticker: str) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=540)  # ~356 trading days
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if df.empty:
        raise ValueError("No data returned. Check the ticker symbol.")
    df = df.dropna()
    return df.tail(356).copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    obv_direction = np.sign(delta).fillna(0)
    df["OBV"] = (obv_direction * df["Volume"].fillna(0)).cumsum()

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["Buy"] = (df["SMA_20"] > df["SMA_50"]) & (df["RSI"] < 70)
    df["Sell"] = (df["SMA_20"] < df["SMA_50"]) & (df["RSI"] > 30)
    df["Position"] = 0
    df.loc[df["Buy"], "Position"] = 1
    df.loc[df["Sell"], "Position"] = -1
    df["Position"] = df["Position"].replace(to_replace=0, method="ffill").fillna(0)
    return df


def backtest(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Position"].shift(1) * df["Returns"]
    df = df.dropna()

    df["Equity"] = (1 + df["Strategy"]).cumprod()
    df["BuyHoldEquity"] = (1 + df["Returns"]).cumprod()

    total_return = df["Equity"].iloc[-1] - 1
    buy_hold = df["BuyHoldEquity"].iloc[-1] - 1
    win_rate = (df["Strategy"] > 0).mean()
    return {
        "strategy_return": total_return,
        "buy_hold_return": buy_hold,
        "win_rate": win_rate
    }, df


def predict_next(df: pd.DataFrame, days_ahead: int = 5) -> pd.DataFrame:
    feature_cols = ["Close", "SMA_20", "SMA_50", "EMA_12", "EMA_26", "MACD", "Signal", "RSI"]
    df_model = df.dropna().copy()

    for lag in range(1, 6):
        df_model[f"Close_lag_{lag}"] = df_model["Close"].shift(lag)
    df_model = df_model.dropna()

    X = df_model[feature_cols + [f"Close_lag_{lag}" for lag in range(1, 6)]]
    y = df_model["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = []
    last_row = df_model.iloc[-1].copy()
    for _ in range(days_ahead):
        features = last_row[feature_cols + [f"Close_lag_{lag}" for lag in range(1, 6)]].values.reshape(1, -1)
        pred = model.predict(features)[0]
        preds.append(pred)

        for lag in range(5, 1, -1):
            last_row[f"Close_lag_{lag}"] = last_row[f"Close_lag_{lag-1}"]
        last_row["Close_lag_1"] = pred
        last_row["Close"] = pred

        last_row["SMA_20"] = df["Close"].tail(19).mean() * 0.95 + pred * 0.05
        last_row["SMA_50"] = df["Close"].tail(49).mean() * 0.98 + pred * 0.02
        last_row["EMA_12"] = (pred * (2 / (12 + 1))) + (last_row["EMA_12"] * (1 - (2 / (12 + 1))))
        last_row["EMA_26"] = (pred * (2 / (26 + 1))) + (last_row["EMA_26"] * (1 - (2 / (26 + 1))))
        last_row["MACD"] = last_row["EMA_12"] - last_row["EMA_26"]
        last_row["Signal"] = (last_row["MACD"] * (2 / (9 + 1))) + (last_row["Signal"] * (1 - (2 / (9 + 1))))
        last_row["RSI"] = last_row["RSI"]

    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    return pd.DataFrame({"Date": future_dates, "Predicted_Close": preds})


def plot_charts(df: pd.DataFrame, ticker: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close", linewidth=1.5)
    plt.plot(df.index, df["SMA_20"], label="SMA 20")
    plt.plot(df.index, df["SMA_50"], label="SMA 50")

    buys = df[df["Buy"]]
    sells = df[df["Sell"]]
    plt.scatter(buys.index, buys["Close"], marker="^", color="green", label="Buy", s=60)
    plt.scatter(sells.index, sells["Close"], marker="v", color="red", label="Sell", s=60)

    plt.title(f"{ticker} Price + Signals")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["MACD"], label="MACD")
    plt.plot(df.index, df["Signal"], label="Signal")
    plt.title(f"{ticker} MACD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["RSI"], label="RSI")
    plt.axhline(70, color="red", linestyle="--")
    plt.axhline(30, color="green", linestyle="--")
    plt.title(f"{ticker} RSI")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if "OBV" in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["OBV"], label="OBV", color="purple")
        plt.title(f"{ticker} OBV")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if "Equity" in df.columns and "BuyHoldEquity" in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["Equity"], label="Strategy Equity")
        plt.plot(df.index, df["BuyHoldEquity"], label="Buy & Hold Equity")
        plt.title(f"{ticker} Equity Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    ticker = "GOOGL"

    df = fetch_last_356_days(ticker)
    df = add_indicators(df)
    df = generate_signals(df)
    stats, bt_df = backtest(df)

    print("\n=== Strategy Performance ===")
    print(f"Strategy Return: {stats['strategy_return']:.2%}")
    print(f"Buy & Hold Return: {stats['buy_hold_return']:.2%}")
    print(f"Win Rate: {stats['win_rate']:.2%}")

    last_signal = df[["Buy", "Sell"]].tail(1)
    if last_signal["Buy"].iloc[0]:
        signal_text = "BUY"
    elif last_signal["Sell"].iloc[0]:
        signal_text = "SELL"
    else:
        signal_text = "HOLD"
    print(f"\nLatest Signal: {signal_text}")

    future = predict_next(df, days_ahead=5)
    print("\n=== 5‑Day Price Prediction ===")
    print(future.to_string(index=False))

    plot_charts(bt_df, ticker)


if __name__ == "__main__":
    main()
