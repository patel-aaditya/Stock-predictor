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
        raise ValueError(f"No data returned for {symbol}. Check symbol or date range.")

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Robustly produce tz-naive timestamps whether source is tz-aware or not.
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.round(2)
    print(
        f"Downloaded {len(df)} rows  |  {df['Date'].min().date()} → {df['Date'].max().date()}"
    )
    return df


#INDICATOR CALCULATIONS
def calc_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        obv[i] = obv[i - 1] + (
            volume[i]
            if close[i] > close[i - 1]
            else -volume[i] if close[i] < close[i - 1] else 0
        )
    return obv


def calc_ad(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
    denom = np.where((high - low) == 0, 1e-9, high - low)
    clv = ((close - low) - (high - close)) / denom
    return np.cumsum(clv * volume)


def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    rsi = np.full(len(close), np.nan)
    if len(close) <= period:
        return rsi

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    ag = np.zeros(len(close))
    al = np.zeros(len(close))

    ag[period] = gain[1 : period + 1].mean()
    al[period] = loss[1 : period + 1].mean()

    for i in range(period + 1, len(close)):
        ag[i] = (ag[i - 1] * (period - 1) + gain[i]) / period
        al[i] = (al[i - 1] * (period - 1) + loss[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.divide(ag, al, out=np.full_like(ag, np.nan), where=al != 0)

    # If average loss is zero but gain exists, RSI = 100.
    rs[(al == 0) & (ag > 0)] = np.inf
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi


def calc_ma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    for i in range(period - 1, len(arr)):
        out[i] = arr[i - period + 1 : i + 1].mean()
    return out


def calc_linreg(close: np.ndarray, period: int = 50) -> np.ndarray:
    lr = np.full(len(close), np.nan)
    if len(close) < period:
        return lr
    x = np.arange(period)
    for i in range(period - 1, len(close)):
        y = close[i - period + 1 : i + 1]
        s, b, *_ = stats.linregress(x, y)
        lr[i] = b + s * (period - 1)
    return lr


def normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max scale to 0–100 for overlay plotting, ignoring NaNs."""
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn) * 100


# ════════════════════════════════════════════════════════════════════════════
# AI SHIT (Sm1 buy me claude pro)
# ════════════════════════════════════════════════════════════════════════════
def gen_signals_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    obv = calc_obv(close, volume)
    obv_ma = calc_ma(obv, 20)
    sig = np.zeros(len(close))
    for i in range(1, len(close)):
        if np.isnan(obv_ma[i]):
            continue
        if obv[i] > obv_ma[i] and obv[i - 1] <= obv_ma[i - 1]:
            sig[i] = 1
        elif obv[i] < obv_ma[i] and obv[i - 1] >= obv_ma[i - 1]:
            sig[i] = -1
    return sig


def gen_signals_ad(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
    ad = calc_ad(close, high, low, volume)
    sig = np.zeros(len(close))
    for i in range(1, len(close)):
        sig[i] = 1 if ad[i] > ad[i - 1] else -1
    return sig


def gen_signals_rsi(close: np.ndarray) -> np.ndarray:
    rsi = calc_rsi(close, 14)
    sig = np.zeros(len(close))
    active = False
    for i in range(len(close)):
        if np.isnan(rsi[i]):
            continue
        if rsi[i] < 30 and not active:
            sig[i] = 1
            active = True
        elif rsi[i] > 70 and active:
            sig[i] = -1
            active = False
    return sig


def gen_signals_ma(close: np.ndarray) -> np.ndarray:
    ma20 = calc_ma(close, 20)
    ma50 = calc_ma(close, 50)
    sig = np.zeros(len(close))
    for i in range(1, len(close)):
        if np.isnan(ma20[i]) or np.isnan(ma50[i]):
            continue
        if ma20[i] > ma50[i] and ma20[i - 1] <= ma50[i - 1]:
            sig[i] = 1
        elif ma20[i] < ma50[i] and ma20[i - 1] >= ma50[i - 1]:
            sig[i] = -1
    return sig


def gen_signals_lr(close: np.ndarray) -> np.ndarray:
    lr = calc_linreg(close, 50)
    sig = np.zeros(len(close))
    for i in range(1, len(close)):
        if np.isnan(lr[i]):
            continue
        if close[i] > lr[i] and close[i - 1] <= lr[i - 1]:
            sig[i] = 1
        elif close[i] < lr[i] and close[i - 1] >= lr[i - 1]:
            sig[i] = -1
    return sig


def rolling_accuracy(signals: np.ndarray, close: np.ndarray, window: int = 20) -> np.ndarray:
    correct = np.zeros(len(close))
    total = np.zeros(len(close))
    for i in range(len(close) - 1):
        if signals[i] != 0:
            correct[i] = 1 if (signals[i] * (close[i + 1] - close[i]) > 0) else 0
            total[i] = 1

    roll_acc = np.full(len(close), np.nan)
    for i in range(window, len(close)):
        t = total[i - window : i].sum()
        roll_acc[i] = (correct[i - window : i].sum() / t * 100) if t > 0 else np.nan
    return roll_acc


# ════════════════════════════════════════════════════════════════════════════
# 4. STYLE CONFIG
# ════════════════════════════════════════════════════════════════════════════
BG_DARK = "#0A0E17"
BG_PANEL = "#111827"
GRID_COL = "#1F2937"
TEXT_COL = "#E5E7EB"

PRICE_COLOR = "#FFFFFF"
IND_COLORS = {
    "OBV": "#00BCD4",
    "A/D": "#FF9800",
    "RSI": "#F06292",
    "MA": "#69F0AE",
    "LinReg": "#CE93D8",
}


def style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def fmt_xaxis(ax, dates):
    n = len(dates)
    if n > 500:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    elif n > 150:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d%b%y"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


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
