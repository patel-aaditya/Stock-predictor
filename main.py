import warnings

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover - environment specific import guard
    raise SystemExit(
        "Missing dependency: yfinance. Install with `pip install yfinance`."
    ) from exc


# DOWNLOAD DATA

def fetch_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    print(f"Downloading {symbol} ...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)

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
    signed_volume = pd.Series(0, index=df.index, dtype="float64")
    positive = delta > 0
    negative = delta < 0
    signed_volume[positive] = df.loc[positive, "Volume"]
    signed_volume[negative] = -df.loc[negative, "Volume"]
    df["OBV"] = signed_volume.cumsum()

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
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df_raw = fetch_data("IRFC.NS", start="2021-01-29", end="2026-02-25", interval="1d")

    # ════════════════════════════════════════════════════════════════════════
    # 5. DEFINE TIME WINDOWS
    # ════════════════════════════════════════════════════════════════════════
    end_date = df_raw["Date"].max()
    windows = {
        "5 Years  (2021–2026)": df_raw[
            df_raw["Date"] >= end_date - pd.DateOffset(years=5)
        ].copy(),
        "1 Year   (last 12m)": df_raw[
            df_raw["Date"] >= end_date - pd.DateOffset(years=1)
        ].copy(),
        "120 Weeks (~2.3 yrs)": df_raw[
            df_raw["Date"] >= end_date - pd.Timedelta(weeks=120)
        ].copy(),
    }

    period_names = list(windows.keys())
    acc_window = 20

    for fig_type in ["overlay", "accuracy"]:
        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor(BG_DARK)

        title_str = (
            "IRFC.NS — Indicators vs Price  (normalised 0–100)"
            if fig_type == "overlay"
            else "IRFC.NS — Rolling Signal Accuracy Timeline  (20-day window)"
        )
        fig.suptitle(
            title_str,
            color=TEXT_COL,
            fontsize=14,
            fontweight="bold",
            y=0.97,
            fontfamily="monospace",
        )

        outer = gridspec.GridSpec(
            1,
            3,
            figure=fig,
            left=0.05,
            right=0.97,
            top=0.91,
            bottom=0.07,
            wspace=0.08,
        )

        for col_idx, pname in enumerate(period_names):
            d = windows[pname].reset_index(drop=True)
            dates = d["Date"].values
            close = d["Close"].values
            high = d["High"].values
            low = d["Low"].values
            volume = d["Volume"].values
            n = len(close)

            obv = calc_obv(close, volume)
            ad = calc_ad(close, high, low, volume)
            rsi = calc_rsi(close, 14)
            ma20 = calc_ma(close, 20)
            ma50 = calc_ma(close, 50)
            linreg = calc_linreg(close, 50)

            ind_arrays = {
                "OBV": obv,
                "A/D": ad,
                "RSI": rsi,
                "MA": ma20,
                "LinReg": linreg,
            }

            sigs = {
                "OBV": gen_signals_obv(close, volume),
                "A/D": gen_signals_ad(close, high, low, volume),
                "RSI": gen_signals_rsi(close),
                "MA": gen_signals_ma(close),
                "LinReg": gen_signals_lr(close),
            }

            inner = gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer[col_idx],
                hspace=0.06,
                height_ratios=[1, 2] if fig_type == "overlay" else [1, 3],
            )

            ax_price = fig.add_subplot(inner[0])
            style_ax(ax_price)
            ax_price.plot(dates, close, color=PRICE_COLOR, linewidth=1.3, label="Close")
            ax_price.set_title(
                pname.strip(), color=TEXT_COL, fontsize=9, fontfamily="monospace", pad=5
            )
            ax_price.set_ylabel("Price ₹", color=TEXT_COL, fontsize=8)
            ax_price.yaxis.set_tick_params(labelsize=7)
            ax_price.plot(
                dates,
                ma50,
                color=IND_COLORS["MA"],
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
                label="MA50",
            )
            ax_price.plot(
                dates,
                linreg,
                color=IND_COLORS["LinReg"],
                linewidth=0.8,
                linestyle=":",
                alpha=0.6,
                label="LR50",
            )
            ax_price.tick_params(axis="x", which="both", labelbottom=False)

            ax_bot = fig.add_subplot(inner[1])
            style_ax(ax_bot)

            if fig_type == "overlay":
                close_n = normalise(close)
                ax_bot.plot(
                    dates, close_n, color=PRICE_COLOR, linewidth=1.6, label="Price", zorder=10, alpha=0.9
                )
                for iname, iarr in ind_arrays.items():
                    norm = normalise(iarr)
                    ax_bot.plot(dates, norm, color=IND_COLORS[iname], linewidth=1.0, label=iname, alpha=0.85)

                obv_n = normalise(obv)
                close_n2 = normalise(close)
                ax_bot.fill_between(
                    dates,
                    close_n2,
                    obv_n,
                    where=(obv_n < close_n2),
                    alpha=0.07,
                    color=IND_COLORS["OBV"],
                    label="_nolegend_",
                )

                ax_bot.set_ylabel("Normalised (0–100)", color=TEXT_COL, fontsize=8)
                ax_bot.set_ylim(-5, 108)

            else:
                acc_data = {iname: rolling_accuracy(sig_arr, close, window=acc_window) for iname, sig_arr in sigs.items()}

                for iname, acc in acc_data.items():
                    ax_bot.plot(dates, acc, color=IND_COLORS[iname], linewidth=1.2, label=iname, alpha=0.9)
                    ax_bot.fill_between(dates, 50, acc, where=(acc >= 50), color=IND_COLORS[iname], alpha=0.06)

                ax_bot.axhline(50, color="#6B7280", linestyle="--", linewidth=0.8, label="50% (random)")
                ax_bot.axhline(60, color="#374151", linestyle=":", linewidth=0.6, alpha=0.7)

                acc_matrix = np.array([acc_data[k] for k in IND_COLORS.keys()])
                ind_names_list = list(IND_COLORS.keys())
                for i in range(acc_window, n - 1):
                    col_vals = acc_matrix[:, i]
                    if np.all(np.isnan(col_vals)):
                        continue
                    best_idx = np.nanargmax(col_vals)
                    ax_bot.axvspan(
                        dates[i], dates[i + 1], color=IND_COLORS[ind_names_list[best_idx]], alpha=0.04, linewidth=0
                    )

                ax_bot.set_ylabel("Rolling Accuracy %", color=TEXT_COL, fontsize=8)
                ax_bot.set_ylim(0, 105)

                for iname, acc in acc_data.items():
                    avg = np.nanmean(acc)
                    if np.isnan(avg):
                        continue
                    ax_bot.annotate(
                        f"{iname}: {avg:.0f}%",
                        xy=(0.02, 0.96 - list(acc_data.keys()).index(iname) * 0.07),
                        xycoords="axes fraction",
                        color=IND_COLORS[iname],
                        fontsize=7.5,
                        fontfamily="monospace",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc=BG_PANEL,
                            ec=IND_COLORS[iname],
                            alpha=0.7,
                            linewidth=0.8,
                        ),
                    )

            fmt_xaxis(ax_bot, dates)

            if col_idx == 0:
                handles = [
                    Line2D([0], [0], color=PRICE_COLOR, linewidth=1.5, label="Price"),
                ] + [Line2D([0], [0], color=c, linewidth=1.5, label=n) for n, c in IND_COLORS.items()]
                if fig_type == "accuracy":
                    handles += [
                        Line2D([0], [0], color="#6B7280", linestyle="--", linewidth=1, label="50% baseline")
                    ]
                ax_bot.legend(
                    handles=handles,
                    loc="lower left",
                    fontsize=8,
                    facecolor=BG_PANEL,
                    labelcolor=TEXT_COL,
                    framealpha=0.9,
                    edgecolor=GRID_COL,
                )

        fname = f"irfc_{'overlay' if fig_type == 'overlay' else 'accuracy'}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        print(f"Saved → {fname}")

    plt.show()
    print("\nDone. Two files saved: irfc_overlay.png  |  irfc_accuracy.png")


if __name__ == "__main__":
    main()
