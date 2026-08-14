"""
NSE Sweet-Spot Momentum + Liquidity Screener
=============================================
v17: Targets charts that keep rising across time horizons without chasing
an overbought spike, while requiring genuinely usable trading liquidity.

PRICE RANGE (unchanged):
  - Rs 99 to Rs 1200

DEFAULT "SWEET SPOT" LIQUIDITY:
  - Average 20-day volume >= 75,000 shares
  - Average 20-day traded value >= Rs 3 crore
  - Median 20-day traded value >= Rs 1.5 crore
  - Trading activity must be consistent, not created by one isolated volume day
  - High and Very High liquidity presets are available in the sidebar

MANDATORY MOMENTUM SHAPE:
  - 0 < 1-day return < 1-week return < 1-month return
  - 3-month and 6-month returns must also be positive
  - Default sweet-spot mode allows a small 15% tolerance in the two long-term
    ladder steps, but at least one of 1M<3M or 3M<6M must still be strictly true
  - Optional perfect mode requires 1D<1W<1M<3M<6M exactly
  - At least half of the latest eight weeks must be positive

NOT OVERBOUGHT / NOT CHASING:
  - RSI and MFI remain below their outer overbought levels
  - A blended Overbought Risk score combines RSI, MFI, Stoch RSI, daily spike,
    EMA20/ATR extension and breakout extension, avoiding a brittle single rule
  - The stock must be above rising EMA20/EMA50 and above SMA200, but cannot be
    excessively stretched above EMA20

RANKING:
  - CompositeScore is an anchored 0-100 score blending trend, return staircase,
    market-relative strength, accumulation, liquidity, entry quality and risk
  - Liquidity is measured with both share volume and rupee turnover because
    volume alone does not show how easy a stock is to enter or exit
"""

import io
import datetime as dt
from typing import Iterable, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="NSE Sweet-Spot Momentum & Liquidity Screener", layout="wide")

# ---------------- Price eligibility (kept unchanged) ----------------
PRICE_MIN = 99
PRICE_MAX = 1200
RSI_OVERBOUGHT = 70
MFI_OVERBOUGHT = 80
ROC_LOOKBACK = 10

# ---------------- Sweet-spot trend / momentum settings ----------------
DATA_PERIOD = "1y"
MIN_DATA_DAYS = 40
MIN_TECH_HISTORY_DAYS = 205
BENCHMARK_TICKERS = ("^CRSLDX", "^NSEI")  # NIFTY 500, then NIFTY 50 fallback

TQ_ADX_MIN = 20
TQ_RSI_MIN = 50
TQ_RSI_MAX = 68.5
TQ_MFI_MIN = 45
TQ_MFI_MAX = 78
TQ_STOCH_RSI_HARD_MAX = 92
TQ_MAX_DAILY_RETURN = 4.5
TQ_MAX_DISTANCE_FROM_52W_HIGH = -22.0
TQ_MAX_EXTENSION_ATR = 2.8
TQ_MAX_PRICE_ABOVE_EMA20_PCT = 9.0
TQ_MAX_BREAKOUT_EXTENSION_PCT = 5.0
TQ_MAX_OVERBOUGHT_RISK = 55.0
TQ_MIN_CONFIRMATION_PCT = 57.0
TQ_MIN_POSITIVE_WEEKS_8 = 50.0
TQ_MIN_TREND_CONSISTENCY_20 = 55.0
TQ_MIN_TREND_R2_60 = 0.10

MAX_DAILY_SHARE_OF_WEEK = 80.0
MAX_WEEKLY_SHARE_OF_MONTH = 85.0
LONG_LADDER_TOLERANCE_RATIO = 0.85

# Both share volume and rupee turnover are required.  The default is designed
# to exclude thin stocks without shrinking the result set to only mega-caps.
LIQUIDITY_PRESETS = {
    "Sweet spot": {
        "min_avg_volume": 75_000,
        "min_avg_traded_value": 30_000_000,       # Rs 3 crore/day
        "min_median_traded_value": 15_000_000,    # Rs 1.5 crore/day
        "liquid_days_column": "DaysAbove75LakhPct",
        "min_liquid_days_pct": 70.0,
        "min_active_days_pct": 95.0,
        "min_turnover_stability": 0.30,
        "min_recent_turnover_ratio": 0.55,
        "description": "Good retail liquidity without limiting the scan to only the largest stocks.",
    },
    "High liquidity": {
        "min_avg_volume": 150_000,
        "min_avg_traded_value": 75_000_000,       # Rs 7.5 crore/day
        "min_median_traded_value": 40_000_000,    # Rs 4 crore/day
        "liquid_days_column": "DaysAbove2CrPct",
        "min_liquid_days_pct": 75.0,
        "min_active_days_pct": 95.0,
        "min_turnover_stability": 0.35,
        "min_recent_turnover_ratio": 0.60,
        "description": "Stronger liquidity for larger positions and faster entries/exits.",
    },
    "Very high liquidity": {
        "min_avg_volume": 300_000,
        "min_avg_traded_value": 150_000_000,      # Rs 15 crore/day
        "min_median_traded_value": 80_000_000,    # Rs 8 crore/day
        "liquid_days_column": "DaysAbove4CrPct",
        "min_liquid_days_pct": 80.0,
        "min_active_days_pct": 98.0,
        "min_turnover_stability": 0.40,
        "min_recent_turnover_ratio": 0.65,
        "description": "Only highly active stocks with consistently heavy turnover.",
    },
}
DEFAULT_LIQUIDITY_PRESET = "Sweet spot"

SCORE_WEIGHTS = {
    "Trend": 0.24,
    "Momentum": 0.28,
    "RelativeStrength": 0.13,
    "Volume": 0.10,
    "Trigger": 0.07,
    "Risk": 0.06,
    "Confirmation": 0.03,
    "Liquidity": 0.09,
}

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

FALLBACK_UNIVERSE = {
    "RELIANCE": "Reliance Industries Ltd.", "TCS": "Tata Consultancy Services Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.", "INFY": "Infosys Ltd.", "ICICIBANK": "ICICI Bank Ltd.",
    "HINDUNILVR": "Hindustan Unilever Ltd.", "ITC": "ITC Ltd.", "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel Ltd.", "BAJFINANCE": "Bajaj Finance Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.", "LT": "Larsen & Toubro Ltd.",
    "AXISBANK": "Axis Bank Ltd.", "ASIANPAINT": "Asian Paints Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.", "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
    "TITAN": "Titan Company Ltd.", "ULTRACEMCO": "UltraTech Cement Ltd.",
    "WIPRO": "Wipro Ltd.", "ADANIENT": "Adani Enterprises Ltd.",
}

FALLBACK_SECTORS = {
    "RELIANCE": "Oil Gas & Consumable Fuels", "TCS": "IT - Software", "HDFCBANK": "Banks",
    "INFY": "IT - Software", "ICICIBANK": "Banks", "HINDUNILVR": "FMCG", "ITC": "FMCG",
    "SBIN": "Banks", "BHARTIARTL": "Telecom - Services", "BAJFINANCE": "Finance",
    "KOTAKBANK": "Banks", "LT": "Construction", "AXISBANK": "Banks",
    "ASIANPAINT": "Consumer Durables", "MARUTI": "Automobile", "SUNPHARMA": "Pharmaceuticals",
    "TITAN": "Consumer Durables", "ULTRACEMCO": "Cement", "WIPRO": "IT - Software",
    "ADANIENT": "Metals & Mining",
}


# ---------------- Universe / classification ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_full_nse_universe():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, headers=NSE_HEADERS, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = [str(c).strip() for c in df.columns]

        sym_col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
        name_col = "NAME OF COMPANY" if "NAME OF COMPANY" in df.columns else df.columns[1]
        series_col = "SERIES" if "SERIES" in df.columns else None

        if series_col:
            df = df[df[series_col].astype(str).str.strip() == "EQ"]

        df = df.dropna(subset=[sym_col, name_col])
        mapping = {
            str(row[sym_col]).strip(): str(row[name_col]).strip()
            for _, row in df.iterrows()
        }
        if len(mapping) > 500:
            return mapping
    except Exception:
        pass

    return FALLBACK_UNIVERSE


@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_overlay():
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        response = requests.get(url, headers=NSE_HEADERS, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = [str(c).strip() for c in df.columns]

        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        sector_col = "Industry" if "Industry" in df.columns else None
        if sector_col:
            return {
                str(row[sym_col]).strip(): str(row[sector_col]).strip()
                for _, row in df.iterrows()
            }
    except Exception:
        pass

    return FALLBACK_SECTORS


# ---------------- Data-frame helpers ----------------
def _clean_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, single-ticker OHLCV frame or an empty frame."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    cleaned = frame.copy()
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [str(col[0]) for col in cleaned.columns]
    else:
        cleaned.columns = [str(col) for col in cleaned.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in cleaned.columns for col in required):
        return pd.DataFrame()

    cleaned = cleaned[required].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=["High", "Low", "Close"])
    cleaned["Volume"] = cleaned["Volume"].fillna(0).clip(lower=0)
    if isinstance(cleaned.index, pd.DatetimeIndex):
        if cleaned.index.tz is not None:
            cleaned.index = cleaned.index.tz_convert(None)
        cleaned.index = cleaned.index.normalize()

    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    return cleaned


def _extract_ticker_frame(
    raw: pd.DataFrame,
    ticker: str,
    single_ticker_batch: bool = False,
) -> pd.DataFrame:
    """Handle both possible yfinance MultiIndex layouts and single-ticker output."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    if not isinstance(raw.columns, pd.MultiIndex):
        return _clean_ohlcv_frame(raw) if single_ticker_batch else pd.DataFrame()

    for level in range(raw.columns.nlevels):
        values = raw.columns.get_level_values(level).astype(str)
        if ticker in set(values):
            try:
                frame = raw.xs(ticker, axis=1, level=level, drop_level=True)
                return _clean_ohlcv_frame(frame)
            except (KeyError, ValueError):
                continue

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_close(period: str = DATA_PERIOD) -> Tuple[pd.Series, str]:
    """Use NIFTY 500 when available; fall back to NIFTY 50."""
    for ticker in BENCHMARK_TICKERS:
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            frame = _extract_ticker_frame(raw, ticker, single_ticker_batch=True)
            if not frame.empty and len(frame) >= 126:
                return frame["Close"].dropna().rename(ticker), ticker
        except Exception:
            continue

    return pd.Series(dtype=float), "Unavailable"


# ---------------- Technical indicators ----------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.mask((avg_loss == 0) & (avg_gain > 0), 100)
    value = value.mask((avg_gain == 0) & (avg_loss > 0), 0)
    value = value.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return value


def stochastic_rsi(
    rsi_series: pd.Series,
    period: int = 14,
    smooth: int = 3,
) -> pd.Series:
    """Smoothed Stochastic RSI on a 0-100 scale."""
    rolling_low = rsi_series.rolling(period, min_periods=period).min()
    rolling_high = rsi_series.rolling(period, min_periods=period).max()
    denominator = rolling_high - rolling_low
    raw = (rsi_series - rolling_low) / denominator.replace(0, np.nan) * 100
    raw = raw.mask(denominator == 0, 50).clip(0, 100)
    return raw.rolling(smooth, min_periods=smooth).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def adx_components(
    df: pd.DataFrame,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Wilder ADX plus directional indicators and ATR."""
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = true_range(df)
    atr14 = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_smoothed = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_smoothed = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_smoothed / atr14.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx14 = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx14, plus_di, minus_di, atr14


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    direction = typical_price.diff()

    positive_flow = raw_money_flow.where(direction > 0, 0.0).rolling(
        period, min_periods=period
    ).sum()
    negative_flow = raw_money_flow.where(direction < 0, 0.0).rolling(
        period, min_periods=period
    ).sum()

    ratio = positive_flow / negative_flow.replace(0, np.nan)
    value = 100 - (100 / (1 + ratio))
    value = value.mask((negative_flow == 0) & (positive_flow > 0), 100)
    value = value.mask((positive_flow == 0) & (negative_flow > 0), 0)
    value = value.mask((positive_flow == 0) & (negative_flow == 0), 50)
    return value


def period_return(series: pd.Series, lookback: int) -> float:
    clean = series.dropna()
    if len(clean) <= lookback:
        return np.nan
    start = clean.iloc[-lookback - 1]
    end = clean.iloc[-1]
    if pd.isna(start) or pd.isna(end) or start == 0:
        return np.nan
    return (end / start - 1) * 100


def change_over(series: pd.Series, lookback: int) -> float:
    clean = series.dropna()
    if len(clean) <= lookback:
        return np.nan
    return clean.iloc[-1] - clean.iloc[-lookback - 1]


def max_drawdown_pct(series: pd.Series, lookback: int = 63) -> float:
    clean = series.dropna().tail(lookback + 1)
    if len(clean) < 2:
        return np.nan
    drawdown = clean / clean.cummax() - 1
    return drawdown.min() * 100


def efficiency_ratio(series: pd.Series, lookback: int = 20) -> float:
    clean = series.dropna()
    if len(clean) <= lookback:
        return np.nan
    window = clean.tail(lookback + 1)
    net_change = abs(window.iloc[-1] - window.iloc[0])
    path = window.diff().abs().sum()
    return float(net_change / path) if path > 0 else 0.0


def safe_round(value: float, digits: int = 2) -> float:
    return round(float(value), digits) if pd.notna(value) and np.isfinite(value) else np.nan


def trend_regression_stats(
    series: pd.Series,
    lookback: int = 60,
) -> Tuple[float, float]:
    """Annualized log-price slope and R-squared for trend smoothness."""
    clean = series.dropna().tail(lookback)
    if len(clean) < max(20, lookback // 2) or (clean <= 0).any():
        return np.nan, np.nan

    x = np.arange(len(clean), dtype=float)
    y = np.log(clean.to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual_sum = float(np.square(y - fitted).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    r_squared = 1 - residual_sum / total_sum if total_sum > 0 else 0.0
    annualized_slope = (np.exp(slope * 252) - 1) * 100
    return float(annualized_slope), float(np.clip(r_squared, 0, 1))


def positive_period_ratio(
    series: pd.Series,
    frequency: str,
    periods: int,
) -> float:
    """Percentage of the latest completed/resampled periods with a positive return."""
    clean = series.dropna()
    if not isinstance(clean.index, pd.DatetimeIndex) or clean.empty:
        return np.nan

    if frequency == "weekly":
        sampled = clean.resample("W-FRI").last().dropna()
    elif frequency == "monthly":
        sampled = clean.groupby(clean.index.to_period("M")).last().dropna()
    else:
        raise ValueError("frequency must be 'weekly' or 'monthly'")

    returns = sampled.pct_change().dropna().tail(periods)
    if len(returns) < max(3, periods // 2):
        return np.nan
    return float((returns > 0).mean() * 100)


# ---------------- Dataset construction ----------------
@st.cache_data(ttl=3600, show_spinner=True)
def build_dataset(symbols: Iterable[str], period: str = DATA_PERIOD) -> pd.DataFrame:
    symbols = list(symbols)
    tickers = [f"{symbol}.NS" for symbol in symbols]
    benchmark_close, benchmark_used = get_benchmark_close(period)

    rows = []
    chunk_size = 75

    for start_index in range(0, len(tickers), chunk_size):
        batch = tickers[start_index:start_index + chunk_size]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            continue

        for ticker in batch:
            symbol = ticker.removesuffix(".NS")
            frame = _extract_ticker_frame(
                raw,
                ticker,
                single_ticker_batch=(len(batch) == 1),
            )
            if frame.empty or len(frame) < MIN_DATA_DAYS:
                continue

            close = frame["Close"]
            volume = frame["Volume"]
            daily_returns = close.pct_change()
            daily_turnover = close * volume
            last_close = close.iloc[-1]

            ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
            ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
            sma200 = close.rolling(200, min_periods=200).mean()

            rsi14 = rsi(close, 14)
            stoch_rsi14 = stochastic_rsi(rsi14, 14)
            macd_line, signal_line, macd_hist = macd(close)
            adx14, plus_di, minus_di, atr14 = adx_components(frame, 14)
            mfi14 = mfi(frame, 14)

            # Keep shorter-history stocks in the dataset so the original sector
            # heatmap remains broad. Stocks without enough history for SMA200 simply
            # fail the upgraded technical gate later.

            daily_ret = period_return(close, 1)
            weekly_ret = period_return(close, 5)
            roc10 = period_return(close, ROC_LOOKBACK)
            monthly_ret = period_return(close, 21)
            ret_3m = period_return(close, 63)
            ret_6m = period_return(close, 126)

            return_step_1w_vs_1d = (
                weekly_ret - daily_ret
                if pd.notna(weekly_ret) and pd.notna(daily_ret)
                else np.nan
            )
            return_step_1m_vs_1w = (
                monthly_ret - weekly_ret
                if pd.notna(monthly_ret) and pd.notna(weekly_ret)
                else np.nan
            )
            return_step_3m_vs_1m = (
                ret_3m - monthly_ret
                if pd.notna(ret_3m) and pd.notna(monthly_ret)
                else np.nan
            )
            return_step_6m_vs_3m = (
                ret_6m - ret_3m
                if pd.notna(ret_6m) and pd.notna(ret_3m)
                else np.nan
            )

            daily_share_of_week = (
                daily_ret / weekly_ret * 100
                if pd.notna(daily_ret) and pd.notna(weekly_ret) and weekly_ret > 0
                else np.nan
            )
            weekly_share_of_month = (
                weekly_ret / monthly_ret * 100
                if pd.notna(weekly_ret) and pd.notna(monthly_ret) and monthly_ret > 0
                else np.nan
            )

            core_return_staircase = bool(
                pd.notna(daily_ret)
                and pd.notna(weekly_ret)
                and pd.notna(monthly_ret)
                and 0 < daily_ret < weekly_ret < monthly_ret
            )
            full_return_staircase = bool(
                core_return_staircase
                and pd.notna(ret_3m)
                and pd.notna(ret_6m)
                and monthly_ret < ret_3m < ret_6m
            )

            long_ladder_strict_steps = int(
                pd.notna(ret_3m) and pd.notna(monthly_ret) and ret_3m > monthly_ret
            ) + int(
                pd.notna(ret_6m) and pd.notna(ret_3m) and ret_6m > ret_3m
            )
            long_ladder_with_tolerance = bool(
                pd.notna(ret_3m)
                and pd.notna(ret_6m)
                and pd.notna(monthly_ret)
                and ret_3m > 0
                and ret_6m > 0
                and ret_3m >= monthly_ret * LONG_LADDER_TOLERANCE_RATIO
                and ret_6m >= ret_3m * LONG_LADDER_TOLERANCE_RATIO
                and long_ladder_strict_steps >= 1
            )
            sweet_spot_return_staircase = bool(
                core_return_staircase and long_ladder_with_tolerance
            )

            return_ladder_checks = [
                pd.notna(daily_ret) and daily_ret > 0,
                pd.notna(return_step_1w_vs_1d) and return_step_1w_vs_1d > 0,
                pd.notna(return_step_1m_vs_1w) and return_step_1m_vs_1w > 0,
                pd.notna(return_step_3m_vs_1m) and return_step_3m_vs_1m > 0,
                pd.notna(return_step_6m_vs_3m) and return_step_6m_vs_3m > 0,
            ]
            return_staircase_pct = sum(return_ladder_checks) / len(return_ladder_checks) * 100

            benchmark_aligned = pd.Series(dtype=float)
            if not benchmark_close.empty:
                benchmark_aligned = benchmark_close.reindex(close.index).ffill().dropna()

            benchmark_1m = period_return(benchmark_aligned, 21)
            benchmark_3m = period_return(benchmark_aligned, 63)
            benchmark_6m = period_return(benchmark_aligned, 126)

            rel_strength_1m = (
                monthly_ret - benchmark_1m
                if pd.notna(monthly_ret) and pd.notna(benchmark_1m)
                else np.nan
            )
            rel_strength_3m = (
                ret_3m - benchmark_3m
                if pd.notna(ret_3m) and pd.notna(benchmark_3m)
                else np.nan
            )
            rel_strength_6m = (
                ret_6m - benchmark_6m
                if pd.notna(ret_6m) and pd.notna(benchmark_6m)
                else np.nan
            )

            volume_20 = volume.tail(20)
            turnover_20 = daily_turnover.tail(20)
            avg_volume_20 = volume_20.mean()
            median_volume_20 = volume_20.median()
            avg_volume_5 = volume.tail(5).mean()
            current_rvol = (
                volume.iloc[-1] / avg_volume_20
                if pd.notna(avg_volume_20) and avg_volume_20 > 0
                else np.nan
            )
            volume_trend = (
                avg_volume_5 / avg_volume_20
                if pd.notna(avg_volume_20) and avg_volume_20 > 0
                else np.nan
            )

            avg_traded_value = turnover_20.mean()
            median_traded_value_20 = turnover_20.median()
            avg_traded_value_60 = daily_turnover.tail(60).mean()
            active_days_20_pct = (volume_20 > 0).mean() * 100
            turnover_stability_20 = (
                median_traded_value_20 / avg_traded_value
                if pd.notna(avg_traded_value) and avg_traded_value > 0
                else np.nan
            )
            recent_turnover_ratio = (
                daily_turnover.tail(5).mean() / avg_traded_value
                if pd.notna(avg_traded_value) and avg_traded_value > 0
                else np.nan
            )
            days_above_75_lakh_pct = (turnover_20 >= 7_500_000).mean() * 100
            days_above_2_cr_pct = (turnover_20 >= 20_000_000).mean() * 100
            days_above_4_cr_pct = (turnover_20 >= 40_000_000).mean() * 100

            up_day = close.diff() > 0
            total_volume_20 = volume.tail(20).sum()
            up_volume_20 = volume.where(up_day, 0).tail(20).sum()
            up_volume_ratio_20 = (
                up_volume_20 / total_volume_20 * 100
                if total_volume_20 > 0
                else np.nan
            )

            obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
            obv_impulse_20 = np.nan
            if len(obv) > 20 and avg_volume_20 > 0:
                obv_impulse_20 = (
                    (obv.iloc[-1] - obv.iloc[-21]) / (avg_volume_20 * 20) * 100
                )

            previous_20d_high = close.shift(1).rolling(20, min_periods=20).max().iloc[-1]
            breakout_20_pct = (
                (last_close / previous_20d_high - 1) * 100
                if pd.notna(previous_20d_high) and previous_20d_high > 0
                else np.nan
            )

            high_52w = close.tail(252).max()
            low_52w = close.tail(252).min()
            pct_from_52w_high = (
                (last_close / high_52w - 1) * 100
                if pd.notna(high_52w) and high_52w > 0
                else np.nan
            )
            range_position_52w = (
                (last_close - low_52w) / (high_52w - low_52w) * 100
                if pd.notna(high_52w) and pd.notna(low_52w) and high_52w > low_52w
                else np.nan
            )

            trend_consistency_20 = (
                (close.tail(20) > ema20.reindex(close.index).tail(20)).mean() * 100
            )
            efficiency_ratio_20 = efficiency_ratio(close, 20)
            positive_days_20_pct = (daily_returns.tail(20) > 0).mean() * 100
            positive_weeks_8_pct = positive_period_ratio(close, "weekly", 8)
            positive_months_6_pct = positive_period_ratio(close, "monthly", 6)
            trend_annualized_60, trend_r2_60 = trend_regression_stats(close, 60)
            higher_high_low_20_pct = (
                ((frame["High"].diff() > 0) & (frame["Low"].diff() > 0))
                .tail(20)
                .mean()
                * 100
            )

            atr_value = atr14.iloc[-1]
            extension_atr = (
                (last_close - ema20.iloc[-1]) / atr_value
                if pd.notna(atr_value) and atr_value > 0
                else np.nan
            )
            price_above_ema20_pct = (
                (last_close / ema20.iloc[-1] - 1) * 100
                if pd.notna(ema20.iloc[-1]) and ema20.iloc[-1] > 0
                else np.nan
            )
            atr_pct = (
                atr_value / last_close * 100
                if last_close > 0
                else np.nan
            )

            volatility_20 = daily_returns.tail(20).std(ddof=0) * np.sqrt(252) * 100
            worst_day_20 = daily_returns.tail(20).min() * 100
            max_drawdown_63 = max_drawdown_pct(close, 63)

            macd_hist_pct = macd_hist.iloc[-1] / last_close * 100
            macd_acceleration = (
                (macd_hist.iloc[-1] - macd_hist.iloc[-4]) / last_close * 100
                if len(macd_hist.dropna()) >= 4
                else np.nan
            )

            rsi_change_5 = change_over(rsi14, 5)
            mfi_change_5 = change_over(mfi14, 5)
            momentum_acceleration = (
                weekly_ret - monthly_ret / 4
                if pd.notna(weekly_ret) and pd.notna(monthly_ret)
                else np.nan
            )

            rows.append({
                "Symbol": symbol,
                "HistoryDays": int(len(frame)),
                "Price": safe_round(last_close, 2),
                "Benchmark": benchmark_used,
                "EMA20": safe_round(ema20.iloc[-1], 2),
                "EMA50": safe_round(ema50.iloc[-1], 2),
                "SMA200": safe_round(sma200.iloc[-1], 2),
                "EMA20Slope10": safe_round(period_return(ema20, 10), 2),
                "EMA50Slope20": safe_round(period_return(ema50, 20), 2),
                "SMA200Slope20": safe_round(period_return(sma200, 20), 2),
                "RSI": safe_round(rsi14.iloc[-1], 1),
                "StochRSI": safe_round(stoch_rsi14.iloc[-1], 1),
                "RSIChange5": safe_round(rsi_change_5, 2),
                "MACD_hist_pct": safe_round(macd_hist_pct, 4),
                "MACDAccel": safe_round(macd_acceleration, 4),
                "ADX": safe_round(adx14.iloc[-1], 1),
                "PlusDI": safe_round(plus_di.iloc[-1], 1),
                "MinusDI": safe_round(minus_di.iloc[-1], 1),
                "DISpread": safe_round(plus_di.iloc[-1] - minus_di.iloc[-1], 1),
                "MFI": safe_round(mfi14.iloc[-1], 1),
                "MFIChange5": safe_round(mfi_change_5, 2),
                "RVOL": safe_round(current_rvol, 2),
                "VolumeTrend": safe_round(volume_trend, 2),
                "UpVolumeRatio20": safe_round(up_volume_ratio_20, 1),
                "OBVImpulse20": safe_round(obv_impulse_20, 2),
                "ROC10": safe_round(roc10, 2),
                "DailyRet": safe_round(daily_ret, 2),
                "WeeklyRet": safe_round(weekly_ret, 2),
                "MonthlyRet": safe_round(monthly_ret, 2),
                "Return3M": safe_round(ret_3m, 2),
                "Return6M": safe_round(ret_6m, 2),
                "ReturnStep1Wvs1D": safe_round(return_step_1w_vs_1d, 2),
                "ReturnStep1Mvs1W": safe_round(return_step_1m_vs_1w, 2),
                "ReturnStep3Mvs1M": safe_round(return_step_3m_vs_1m, 2),
                "ReturnStep6Mvs3M": safe_round(return_step_6m_vs_3m, 2),
                "DailyShareOfWeek": safe_round(daily_share_of_week, 1),
                "WeeklyShareOfMonth": safe_round(weekly_share_of_month, 1),
                "CoreReturnStaircase": core_return_staircase,
                "SweetSpotReturnStaircase": sweet_spot_return_staircase,
                "FullReturnStaircase": full_return_staircase,
                "LongLadderStrictSteps": int(long_ladder_strict_steps),
                "ReturnStaircasePct": safe_round(return_staircase_pct, 1),
                "RelStrength1M": safe_round(rel_strength_1m, 2),
                "RelStrength3M": safe_round(rel_strength_3m, 2),
                "RelStrength6M": safe_round(rel_strength_6m, 2),
                "MomentumAcceleration": safe_round(momentum_acceleration, 2),
                "Breakout20Pct": safe_round(breakout_20_pct, 2),
                "PctFrom52WHigh": safe_round(pct_from_52w_high, 2),
                "RangePosition52W": safe_round(range_position_52w, 1),
                "TrendConsistency20": safe_round(trend_consistency_20, 1),
                "EfficiencyRatio20": safe_round(efficiency_ratio_20, 3),
                "PositiveDays20Pct": safe_round(positive_days_20_pct, 1),
                "PositiveWeeks8Pct": safe_round(positive_weeks_8_pct, 1),
                "PositiveMonths6Pct": safe_round(positive_months_6_pct, 1),
                "TrendAnnualized60": safe_round(trend_annualized_60, 2),
                "TrendR2_60": safe_round(trend_r2_60, 3),
                "HigherHighLow20Pct": safe_round(higher_high_low_20_pct, 1),
                "ATR14Pct": safe_round(atr_pct, 2),
                "ExtensionATR": safe_round(extension_atr, 2),
                "PriceAboveEMA20Pct": safe_round(price_above_ema20_pct, 2),
                "Volatility20": safe_round(volatility_20, 2),
                "WorstDay20": safe_round(worst_day_20, 2),
                "MaxDrawdown63": safe_round(max_drawdown_63, 2),
                "AvgVol20": int(avg_volume_20) if pd.notna(avg_volume_20) else 0,
                "MedianVol20": int(median_volume_20) if pd.notna(median_volume_20) else 0,
                "AvgTradedValue": safe_round(avg_traded_value, 0),
                "MedianTradedValue20": safe_round(median_traded_value_20, 0),
                "AvgTradedValue60": safe_round(avg_traded_value_60, 0),
                "AvgTradedValueCr": safe_round(avg_traded_value / 10_000_000, 2),
                "MedianTradedValueCr": safe_round(median_traded_value_20 / 10_000_000, 2),
                "ActiveDays20Pct": safe_round(active_days_20_pct, 1),
                "TurnoverStability20": safe_round(turnover_stability_20, 3),
                "RecentTurnoverRatio": safe_round(recent_turnover_ratio, 2),
                "DaysAbove75LakhPct": safe_round(days_above_75_lakh_pct, 1),
                "DaysAbove2CrPct": safe_round(days_above_2_cr_pct, 1),
                "DaysAbove4CrPct": safe_round(days_above_4_cr_pct, 1),
            })

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        return dataset

    dataset["OverboughtRisk"] = dataset.apply(overbought_risk_score, axis=1)
    dataset["LiquidityGrade"] = dataset.apply(liquidity_grade, axis=1)
    confirmation = dataset.apply(technical_confirmation_summary, axis=1)
    return pd.concat([dataset, confirmation], axis=1)


def _scalar_risk(value: float, safe_level: float, hard_level: float) -> float:
    if pd.isna(value):
        return np.nan
    if hard_level <= safe_level:
        raise ValueError("hard_level must be greater than safe_level")
    return float(np.clip((value - safe_level) / (hard_level - safe_level) * 100, 0, 100))


def overbought_risk_score(row: pd.Series) -> float:
    """Blended 0-100 risk; lower means a fresher, less stretched entry."""
    components = [
        (_scalar_risk(row.get("RSI"), 60, 70), 0.23),
        (_scalar_risk(row.get("MFI"), 65, 80), 0.18),
        (_scalar_risk(row.get("StochRSI"), 70, 95), 0.14),
        (_scalar_risk(row.get("DailyRet"), 1.5, 5.0), 0.12),
        (_scalar_risk(row.get("ExtensionATR"), 1.3, 3.0), 0.15),
        (_scalar_risk(row.get("PriceAboveEMA20Pct"), 4.0, 10.0), 0.12),
        (_scalar_risk(row.get("Breakout20Pct"), 1.0, 6.0), 0.06),
    ]
    if any(pd.isna(value) for value, _ in components):
        return np.nan
    return round(sum(value * weight for value, weight in components), 1)


def liquidity_grade(row: pd.Series) -> str:
    turnover = row.get("AvgTradedValue", np.nan)
    stability = row.get("TurnoverStability20", np.nan)
    if pd.isna(turnover) or pd.isna(stability):
        return "Unavailable"
    if turnover >= 250_000_000 and stability >= 0.45:
        return "Excellent"
    if turnover >= 100_000_000 and stability >= 0.38:
        return "High"
    if turnover >= 30_000_000 and stability >= 0.30:
        return "Good"
    if turnover >= 15_000_000:
        return "Moderate"
    return "Low"


# ---------------- Gates ----------------
def _available_check(value: float, condition: bool) -> Tuple[bool, bool]:
    return pd.notna(value), bool(condition) if pd.notna(value) else False


def technical_confirmation_summary(row: pd.Series) -> pd.Series:
    """Count independent confirmations without failing on a missing benchmark."""
    checks = [
        _available_check(row.get("EMA50"), row.get("EMA50", np.nan) >= row.get("SMA200", np.nan) * 0.98),
        _available_check(row.get("EMA20Slope10"), row.get("EMA20Slope10", np.nan) > 0),
        _available_check(row.get("EMA50Slope20"), row.get("EMA50Slope20", np.nan) > 0),
        _available_check(row.get("SMA200Slope20"), row.get("SMA200Slope20", np.nan) >= -0.5),
        _available_check(row.get("MACDAccel"), row.get("MACDAccel", np.nan) > 0),
        _available_check(row.get("RSIChange5"), row.get("RSIChange5", np.nan) >= -1),
        _available_check(row.get("OverboughtRisk"), row.get("OverboughtRisk", np.nan) <= TQ_MAX_OVERBOUGHT_RISK),
        _available_check(row.get("RelStrength3M"), row.get("RelStrength3M", np.nan) > 0),
        _available_check(row.get("LongLadderStrictSteps"), row.get("LongLadderStrictSteps", 0) >= 1),
        _available_check(row.get("DailyShareOfWeek"), row.get("DailyShareOfWeek", np.nan) <= MAX_DAILY_SHARE_OF_WEEK),
        _available_check(row.get("WeeklyShareOfMonth"), row.get("WeeklyShareOfMonth", np.nan) <= MAX_WEEKLY_SHARE_OF_MONTH),
        _available_check(row.get("TrendConsistency20"), row.get("TrendConsistency20", np.nan) >= 60),
        _available_check(row.get("PositiveWeeks8Pct"), row.get("PositiveWeeks8Pct", np.nan) >= 62.5),
        _available_check(row.get("PositiveMonths6Pct"), row.get("PositiveMonths6Pct", np.nan) >= 50),
        _available_check(row.get("TrendR2_60"), row.get("TrendR2_60", np.nan) >= 0.25),
        _available_check(row.get("UpVolumeRatio20"), row.get("UpVolumeRatio20", np.nan) >= 52),
        _available_check(row.get("VolumeTrend"), row.get("VolumeTrend", np.nan) >= 0.85),
        _available_check(row.get("OBVImpulse20"), row.get("OBVImpulse20", np.nan) > 0),
        _available_check(row.get("TurnoverStability20"), row.get("TurnoverStability20", np.nan) >= 0.35),
        _available_check(row.get("RecentTurnoverRatio"), row.get("RecentTurnoverRatio", np.nan) >= 0.70),
        _available_check(row.get("Breakout20Pct"), row.get("Breakout20Pct", np.nan) >= -5),
        _available_check(row.get("EfficiencyRatio20"), row.get("EfficiencyRatio20", np.nan) >= 0.22),
        _available_check(row.get("Return6M"), row.get("Return6M", np.nan) > 0),
        _available_check(row.get("PriceAboveEMA20Pct"), row.get("PriceAboveEMA20Pct", np.nan) <= 6.5),
        _available_check(row.get("ExtensionATR"), row.get("ExtensionATR", np.nan) <= 2.1),
    ]

    available = sum(1 for is_available, _ in checks if is_available)
    passed = sum(1 for is_available, passed_check in checks if is_available and passed_check)
    pct = passed / available * 100 if available else 0.0

    return pd.Series({
        "TechnicalConfirmations": int(passed),
        "ConfirmationsAvailable": int(available),
        "ConfirmationPct": round(pct, 1),
    })


def passes_basic_gates(row: pd.Series) -> bool:
    """Price and absolute outer overbought limits only."""
    if pd.isna(row.get("Price")) or not (PRICE_MIN < row["Price"] < PRICE_MAX):
        return False
    if pd.isna(row.get("RSI")) or row["RSI"] >= RSI_OVERBOUGHT:
        return False
    if pd.isna(row.get("MFI")) or row["MFI"] >= MFI_OVERBOUGHT:
        return False
    return True


def passes_liquidity_gate(
    row: pd.Series,
    preset_name: str = DEFAULT_LIQUIDITY_PRESET,
) -> bool:
    """Require both share volume and consistent rupee turnover."""
    config = LIQUIDITY_PRESETS.get(preset_name, LIQUIDITY_PRESETS[DEFAULT_LIQUIDITY_PRESET])
    required = [
        "AvgVol20", "AvgTradedValue", "MedianTradedValue20", "ActiveDays20Pct",
        "TurnoverStability20", "RecentTurnoverRatio", config["liquid_days_column"],
    ]
    if any(pd.isna(row.get(column, np.nan)) for column in required):
        return False
    if row["AvgVol20"] < config["min_avg_volume"]:
        return False
    if row["AvgTradedValue"] < config["min_avg_traded_value"]:
        return False
    if row["MedianTradedValue20"] < config["min_median_traded_value"]:
        return False
    if row["ActiveDays20Pct"] < config["min_active_days_pct"]:
        return False
    if row["TurnoverStability20"] < config["min_turnover_stability"]:
        return False
    if row["RecentTurnoverRatio"] < config["min_recent_turnover_ratio"]:
        return False
    if row[config["liquid_days_column"]] < config["min_liquid_days_pct"]:
        return False
    return True


def passes_anti_overbought_gate(row: pd.Series) -> bool:
    """Bullish, but not stretched enough to be a late or spike-driven entry."""
    required = [
        "RSI", "MFI", "StochRSI", "DailyRet", "ExtensionATR",
        "PriceAboveEMA20Pct", "Breakout20Pct", "OverboughtRisk",
    ]
    if any(pd.isna(row.get(column, np.nan)) for column in required):
        return False
    if not (TQ_RSI_MIN <= row["RSI"] < TQ_RSI_MAX):
        return False
    if not (TQ_MFI_MIN <= row["MFI"] < TQ_MFI_MAX):
        return False
    if row["StochRSI"] >= TQ_STOCH_RSI_HARD_MAX:
        return False
    if not (0 < row["DailyRet"] <= TQ_MAX_DAILY_RETURN):
        return False
    if row["ExtensionATR"] > TQ_MAX_EXTENSION_ATR:
        return False
    if row["PriceAboveEMA20Pct"] > TQ_MAX_PRICE_ABOVE_EMA20_PCT:
        return False
    if row["Breakout20Pct"] > TQ_MAX_BREAKOUT_EXTENSION_PCT:
        return False
    if row["OverboughtRisk"] > TQ_MAX_OVERBOUGHT_RISK:
        return False
    return True


def passes_return_staircase(
    row: pd.Series,
    require_perfect_staircase: bool = False,
) -> bool:
    """Core ladder is exact; long horizons use a small tolerance by default."""
    required = [
        "DailyRet", "WeeklyRet", "MonthlyRet", "Return3M", "Return6M",
        "DailyShareOfWeek", "WeeklyShareOfMonth",
    ]
    if any(pd.isna(row.get(column, np.nan)) for column in required):
        return False

    if not (0 < row["DailyRet"] < row["WeeklyRet"] < row["MonthlyRet"]):
        return False
    if row["Return3M"] <= 0 or row["Return6M"] <= 0:
        return False
    if row["DailyShareOfWeek"] > MAX_DAILY_SHARE_OF_WEEK:
        return False
    if row["WeeklyShareOfMonth"] > MAX_WEEKLY_SHARE_OF_MONTH:
        return False

    if require_perfect_staircase:
        return bool(row["MonthlyRet"] < row["Return3M"] < row["Return6M"])

    return bool(row.get("SweetSpotReturnStaircase", False))


def passes_technical_quality(
    row: pd.Series,
    require_perfect_staircase: bool = False,
) -> bool:
    """Smooth, liquid uptrend with positive multi-horizon momentum and a fresh entry."""
    required = [
        "HistoryDays", "Price", "EMA20", "EMA50", "SMA200", "EMA20Slope10",
        "EMA50Slope20", "SMA200Slope20", "ADX", "PlusDI", "MinusDI", "RSI", "MFI",
        "StochRSI", "MACD_hist_pct", "ROC10", "DailyRet", "WeeklyRet", "MonthlyRet",
        "Return3M", "Return6M", "PctFrom52WHigh", "Breakout20Pct", "ExtensionATR",
        "PriceAboveEMA20Pct", "PositiveWeeks8Pct", "TrendConsistency20", "TrendR2_60",
        "TrendAnnualized60", "MaxDrawdown63", "ConfirmationPct", "OverboughtRisk",
    ]
    if any(pd.isna(row.get(column, np.nan)) for column in required):
        return False

    if row["HistoryDays"] < MIN_TECH_HISTORY_DAYS:
        return False
    if not (row["Price"] > row["EMA20"] > row["EMA50"]):
        return False
    if row["Price"] <= row["SMA200"]:
        return False
    # Allow a newly established long-term uptrend, but reject a clearly bearish
    # 50-day/200-day structure.
    if row["EMA50"] < row["SMA200"] * 0.97:
        return False
    if row["EMA20Slope10"] <= 0 or row["EMA50Slope20"] <= 0:
        return False
    if row["SMA200Slope20"] < -0.75:
        return False
    if row["ADX"] < TQ_ADX_MIN or row["PlusDI"] <= row["MinusDI"]:
        return False
    if not passes_anti_overbought_gate(row):
        return False
    if row["MACD_hist_pct"] <= 0 or row["ROC10"] <= 0:
        return False
    if not passes_return_staircase(row, require_perfect_staircase):
        return False
    if row["PositiveWeeks8Pct"] < TQ_MIN_POSITIVE_WEEKS_8:
        return False
    if row["TrendConsistency20"] < TQ_MIN_TREND_CONSISTENCY_20:
        return False
    if row["TrendR2_60"] < TQ_MIN_TREND_R2_60 or row["TrendAnnualized60"] <= 0:
        return False
    if row["MaxDrawdown63"] < -25:
        return False
    if row["PctFrom52WHigh"] < TQ_MAX_DISTANCE_FROM_52W_HIGH:
        return False
    if row["ConfirmationPct"] < TQ_MIN_CONFIRMATION_PCT:
        return False
    return True


# ---------------- Anchored scoring ----------------
def linear_score(
    series: pd.Series,
    low: float,
    high: float,
    neutral: float = 50.0,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if high <= low:
        raise ValueError("high must be greater than low")
    score = (values - low) / (high - low) * 100
    return score.clip(0, 100).fillna(neutral)


def inverse_score(
    series: pd.Series,
    good: float,
    bad: float,
    neutral: float = 50.0,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if bad <= good:
        raise ValueError("bad must be greater than good")
    score = (bad - values) / (bad - good) * 100
    return score.clip(0, 100).fillna(neutral)


def sweet_spot_score(
    series: pd.Series,
    low: float,
    ideal: float,
    high: float,
    neutral: float = 50.0,
) -> pd.Series:
    if not (low < ideal < high):
        raise ValueError("Require low < ideal < high")

    values = pd.to_numeric(series, errors="coerce")
    score = pd.Series(np.nan, index=values.index, dtype=float)

    left = values <= ideal
    right = values > ideal
    score.loc[left] = (values.loc[left] - low) / (ideal - low) * 100
    score.loc[right] = (high - values.loc[right]) / (high - ideal) * 100
    return score.clip(0, 100).fillna(neutral)


def percentile_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(50.0, index=values.index)
    if len(valid) == 1:
        return pd.Series(100.0, index=values.index).where(values.notna(), 50.0)
    score = values.rank(method="average", pct=True) * 100
    return score.fillna(50.0)


def build_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["TrendScore"] = (
        linear_score(d["ADX"], 20, 45) * 0.16
        + linear_score(d["DISpread"], 0, 25) * 0.10
        + linear_score(d["EMA20Slope10"], 0, 5) * 0.13
        + linear_score(d["EMA50Slope20"], 0, 8) * 0.12
        + linear_score(d["SMA200Slope20"], -0.5, 5) * 0.07
        + linear_score(d["TrendConsistency20"], 55, 95) * 0.12
        + linear_score(d["EfficiencyRatio20"], 0.20, 0.65) * 0.09
        + linear_score(d["PositiveWeeks8Pct"], 50, 87.5) * 0.09
        + linear_score(d["TrendR2_60"], 0.10, 0.75) * 0.08
        + linear_score(d["HigherHighLow20Pct"], 25, 65) * 0.04
    )

    d["ReturnStaircaseScore"] = (
        sweet_spot_score(d["DailyRet"], 0.01, 0.80, 3.50) * 0.12
        + sweet_spot_score(d["WeeklyRet"], 0.20, 4.00, 12.00) * 0.16
        + sweet_spot_score(d["MonthlyRet"], 1.00, 10.00, 30.00) * 0.20
        + linear_score(d["ReturnStep1Wvs1D"], 0, 6) * 0.10
        + linear_score(d["ReturnStep1Mvs1W"], 0, 15) * 0.12
        + linear_score(d["ReturnStep3Mvs1M"], -3, 30) * 0.08
        + linear_score(d["ReturnStep6Mvs3M"], -5, 45) * 0.07
        + linear_score(d["LongLadderStrictSteps"], 0, 2) * 0.06
        + sweet_spot_score(d["DailyShareOfWeek"], 5, 30, MAX_DAILY_SHARE_OF_WEEK) * 0.04
        + sweet_spot_score(d["WeeklyShareOfMonth"], 10, 45, MAX_WEEKLY_SHARE_OF_MONTH) * 0.05
    )

    d["MomentumScore"] = (
        sweet_spot_score(d["RSI"], 50, 60, TQ_RSI_MAX) * 0.09
        + sweet_spot_score(d["StochRSI"], 10, 55, TQ_STOCH_RSI_HARD_MAX) * 0.05
        + linear_score(d["ROC10"], 0, 12) * 0.10
        + d["ReturnStaircaseScore"] * 0.40
        + linear_score(d["Return3M"], 0, 45) * 0.12
        + linear_score(d["Return6M"], 0, 80) * 0.10
        + linear_score(d["PositiveMonths6Pct"], 50, 100) * 0.07
        + linear_score(d["TrendAnnualized60"], 0, 100) * 0.04
        + linear_score(d["MomentumAcceleration"], -2, 5) * 0.03
    )

    d["RelativeStrengthScore"] = (
        linear_score(d["RelStrength1M"], -2, 12) * 0.30
        + linear_score(d["RelStrength3M"], 0, 25) * 0.45
        + linear_score(d["RelStrength6M"], -2, 40) * 0.25
    )

    d["VolumeScore"] = (
        sweet_spot_score(d["MFI"], 45, 61, TQ_MFI_MAX) * 0.18
        + linear_score(d["RVOL"], 0.75, 2.50) * 0.18
        + linear_score(d["VolumeTrend"], 0.75, 1.80) * 0.17
        + linear_score(d["UpVolumeRatio20"], 48, 70) * 0.20
        + linear_score(d["OBVImpulse20"], -5, 35) * 0.20
        + linear_score(d["RecentTurnoverRatio"], 0.55, 1.50) * 0.07
    )

    d["TriggerScore"] = (
        linear_score(d["MACD_hist_pct"], 0, 1.50) * 0.28
        + linear_score(d["MACDAccel"], -0.05, 0.40) * 0.22
        + sweet_spot_score(d["Breakout20Pct"], -5, 0.50, TQ_MAX_BREAKOUT_EXTENSION_PCT) * 0.24
        + linear_score(d["PctFrom52WHigh"], -22, 0) * 0.10
        + linear_score(d["RSIChange5"], -2, 8) * 0.08
        + inverse_score(d["StochRSI"], 55, TQ_STOCH_RSI_HARD_MAX) * 0.08
    )

    d["RiskScore"] = (
        inverse_score(d["OverboughtRisk"], 10, TQ_MAX_OVERBOUGHT_RISK) * 0.35
        + sweet_spot_score(d["ExtensionATR"], 0, 1.10, TQ_MAX_EXTENSION_ATR) * 0.18
        + sweet_spot_score(d["PriceAboveEMA20Pct"], 0, 3.5, TQ_MAX_PRICE_ABOVE_EMA20_PCT) * 0.14
        + linear_score(d["MaxDrawdown63"], -25, -3) * 0.18
        + inverse_score(d["Volatility20"], 25, 80) * 0.08
        + linear_score(d["WorstDay20"], -12, -2) * 0.07
    )

    log_avg_turnover = np.log10(d["AvgTradedValue"].clip(lower=1))
    log_median_turnover = np.log10(d["MedianTradedValue20"].clip(lower=1))
    log_avg_volume = np.log10(d["AvgVol20"].clip(lower=1))
    d["LiquidityScore"] = (
        linear_score(log_avg_turnover, np.log10(30_000_000), np.log10(1_000_000_000)) * 0.32
        + linear_score(log_median_turnover, np.log10(15_000_000), np.log10(500_000_000)) * 0.22
        + linear_score(log_avg_volume, np.log10(75_000), np.log10(2_000_000)) * 0.14
        + linear_score(d["ActiveDays20Pct"], 95, 100) * 0.07
        + linear_score(d["TurnoverStability20"], 0.30, 0.85) * 0.10
        + sweet_spot_score(d["RecentTurnoverRatio"], 0.50, 1.05, 2.50) * 0.09
        + linear_score(d["DaysAbove75LakhPct"], 70, 100) * 0.06
    )

    d["ConfirmationScore"] = d["ConfirmationPct"].clip(0, 100).fillna(0)

    d["CompositeScore"] = (
        d["TrendScore"] * SCORE_WEIGHTS["Trend"]
        + d["MomentumScore"] * SCORE_WEIGHTS["Momentum"]
        + d["RelativeStrengthScore"] * SCORE_WEIGHTS["RelativeStrength"]
        + d["VolumeScore"] * SCORE_WEIGHTS["Volume"]
        + d["TriggerScore"] * SCORE_WEIGHTS["Trigger"]
        + d["RiskScore"] * SCORE_WEIGHTS["Risk"]
        + d["ConfirmationScore"] * SCORE_WEIGHTS["Confirmation"]
        + d["LiquidityScore"] * SCORE_WEIGHTS["Liquidity"]
    ).round(1)

    d["TechnicalGrade"] = pd.cut(
        d["CompositeScore"],
        bins=[-np.inf, 55, 65, 75, np.inf],
        labels=["Qualified", "Good", "Strong", "Elite"],
        right=False,
    ).astype(str)

    d["SetupSummary"] = d.apply(build_setup_summary, axis=1)
    return d

def build_setup_summary(row: pd.Series) -> str:
    tags = []
    if bool(row.get("FullReturnStaircase", False)):
        tags.append("Perfect return ladder")
    elif bool(row.get("SweetSpotReturnStaircase", False)):
        tags.append("Sweet-spot return ladder")

    if row.get("OverboughtRisk", 999) <= 30:
        tags.append("Fresh, not overbought")
    elif row.get("OverboughtRisk", 999) <= TQ_MAX_OVERBOUGHT_RISK:
        tags.append("Controlled extension")

    if row.get("LiquidityGrade") in {"Excellent", "High"}:
        tags.append(f"{row.get('LiquidityGrade')} liquidity")
    elif row.get("LiquidityGrade") == "Good":
        tags.append("Good liquidity")

    if row.get("PositiveWeeks8Pct", 0) >= 75:
        tags.append("Consistent weekly rise")
    if row.get("RelStrength3M", -999) >= 10:
        tags.append("Strong vs market")
    elif row.get("RelStrength3M", -999) > 0:
        tags.append("Beating market")
    if row.get("RVOL", 0) >= 1.30 or row.get("VolumeTrend", 0) >= 1.20:
        tags.append("Volume expansion")
    if row.get("MACDAccel", -999) > 0:
        tags.append("MACD accelerating")

    return " | ".join(tags[:4]) if tags else "Qualified smooth uptrend"


# ---------------- UI ----------------
st.title("NSE Sweet-Spot Momentum & Liquidity Screener")

if HAS_AUTOREFRESH:
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")

with st.sidebar:
    st.header("Controls")
    top_n = st.slider("How many stocks to show", 10, 150, 40)
    liquidity_preset = st.selectbox(
        "Liquidity standard",
        options=list(LIQUIDITY_PRESETS.keys()),
        index=list(LIQUIDITY_PRESETS.keys()).index(DEFAULT_LIQUIDITY_PRESET),
        help=(
            "Sweet spot requires both good share volume and at least Rs 3 crore of "
            "average daily traded value. Higher presets are suitable for larger positions."
        ),
    )
    require_perfect_staircase = st.checkbox(
        "Perfect ladder: require 1D < 1W < 1M < 3M < 6M",
        value=False,
        help=(
            "Off by default. Sweet-spot mode keeps 1D<1W<1M exact, requires positive "
            "3M/6M returns, allows only a small long-horizon tolerance, and still requires "
            "at least one strictly rising long-horizon step."
        ),
    )
    refresh_btn = st.button("Refresh data now")

if refresh_btn:
    st.cache_data.clear()

liquidity_config = LIQUIDITY_PRESETS[liquidity_preset]
st.caption(
    f"Scans all NSE main-board EQ stocks from Rs {PRICE_MIN} to Rs {PRICE_MAX}. "
    f"Selected liquidity: avg volume >= {liquidity_config['min_avg_volume']:,}, "
    f"avg turnover >= Rs {liquidity_config['min_avg_traded_value'] / 10_000_000:.1f} crore/day, "
    f"median turnover >= Rs {liquidity_config['min_median_traded_value'] / 10_000_000:.1f} crore/day. "
    "Every result must have 0<1D<1W<1M, positive 3M/6M returns, a rising technical structure, "
    "and controlled overbought risk. Turnover is a liquidity proxy; live bid-ask spread and order-book "
    "depth are not available from the daily Yahoo Finance feed."
)
st.info(f"**{liquidity_preset}:** {liquidity_config['description']}")

universe_map = get_full_nse_universe()
sector_map = get_sector_overlay()
symbols = list(universe_map.keys())
st.write(f"Scanning **{len(symbols)}** NSE-listed EQ-series symbols this refresh.")

data = build_dataset(symbols)

if data.empty:
    st.error("Could not fetch sufficient price data (network/API issue). Try 'Refresh data now'.")
else:
    data["Company Name"] = data["Symbol"].map(lambda symbol: universe_map.get(symbol, symbol))
    data["Sector"] = data["Symbol"].map(lambda symbol: sector_map.get(symbol, "Unclassified"))

    benchmark_used = data["Benchmark"].dropna().iloc[0] if data["Benchmark"].notna().any() else "Unavailable"

    st.subheader(f"Top {top_n} Sweet-Spot Uptrend Stocks")

    basic_ok = data[data.apply(passes_basic_gates, axis=1)].copy()
    liquid_ok = basic_ok[
        basic_ok.apply(lambda row: passes_liquidity_gate(row, liquidity_preset), axis=1)
    ].copy()
    not_overbought_ok = liquid_ok[
        liquid_ok.apply(passes_anti_overbought_gate, axis=1)
    ].copy()
    staircase_ok = not_overbought_ok[
        not_overbought_ok.apply(
            lambda row: passes_return_staircase(row, require_perfect_staircase),
            axis=1,
        )
    ].copy()
    technical_ok = staircase_ok[
        staircase_ok.apply(
            lambda row: passes_technical_quality(row, require_perfect_staircase),
            axis=1,
        )
    ].copy()

    staircase_label = (
        "perfect 1D<1W<1M<3M<6M"
        if require_perfect_staircase
        else "sweet-spot multi-horizon ladder"
    )
    st.write(
        f"{len(data)} scanned -> {len(basic_ok)} passed price/outer limits -> "
        f"{len(liquid_ok)} passed **{liquidity_preset.lower()} liquidity** -> "
        f"{len(not_overbought_ok)} were bullish but not overbought -> "
        f"{len(staircase_ok)} passed the **{staircase_label}** -> "
        f"**{len(technical_ok)} passed the complete smooth-uptrend gate**. "
        f"Relative-strength benchmark: **{benchmark_used}**."
    )

    if technical_ok.empty:
        st.info(
            "No stocks currently clear every liquidity, return-ladder, trend and anti-overbought "
            "condition. Keep the default Sweet spot liquidity and leave Perfect ladder unchecked "
            "for the intended balance between quality and enough results."
        )
    else:
        scored = build_composite_scores(technical_ok)
        result = scored.sort_values(
            ["CompositeScore", "LiquidityScore", "MomentumScore", "TrendScore", "AvgTradedValue"],
            ascending=[False, False, False, False, False],
        ).head(top_n)

        display_df = result[[
            "Company Name", "Price", "CompositeScore", "LiquidityGrade", "AvgTradedValueCr"
        ]].copy()
        display_df = display_df.rename(columns={
            "CompositeScore": "Score",
            "LiquidityGrade": "Liquidity",
            "AvgTradedValueCr": "Avg Turnover (Rs Cr/day)",
        }).reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True)

        with st.expander("Verify liquidity, rising returns and overbought risk"):
            audit_df = result[[
                "Company Name", "Price", "DailyRet", "WeeklyRet", "MonthlyRet",
                "Return3M", "Return6M", "PositiveWeeks8Pct", "AvgVol20",
                "AvgTradedValueCr", "MedianTradedValueCr", "TurnoverStability20",
                "LiquidityGrade", "RSI", "MFI", "StochRSI", "OverboughtRisk",
                "ExtensionATR", "PriceAboveEMA20Pct", "CompositeScore", "SetupSummary",
            ]].copy()
            audit_df = audit_df.rename(columns={
                "DailyRet": "1D %",
                "WeeklyRet": "1W %",
                "MonthlyRet": "1M %",
                "Return3M": "3M %",
                "Return6M": "6M %",
                "PositiveWeeks8Pct": "Positive Weeks (8) %",
                "AvgVol20": "Avg Volume (20D)",
                "AvgTradedValueCr": "Avg Turnover Rs Cr",
                "MedianTradedValueCr": "Median Turnover Rs Cr",
                "TurnoverStability20": "Turnover Stability",
                "LiquidityGrade": "Liquidity",
                "StochRSI": "Stoch RSI",
                "OverboughtRisk": "Overbought Risk (lower better)",
                "ExtensionATR": "EMA20 Extension (ATR)",
                "PriceAboveEMA20Pct": "% Above EMA20",
                "CompositeScore": "Score",
                "SetupSummary": "Why it passed",
            }).reset_index(drop=True)
            audit_df.index = audit_df.index + 1
            st.dataframe(audit_df, use_container_width=True)
            st.caption(
                "Default mode requires an exact 0<1D<1W<1M ladder, positive 3M and 6M returns, "
                "at least one strict long-horizon rise, no more than 15% long-horizon tolerance, "
                "at least four positive weeks out of eight, and strong/consistent rupee turnover."
            )

        download_columns = [
            "Symbol", "Company Name", "HistoryDays", "Price", "CompositeScore", "TechnicalGrade",
            "SetupSummary", "LiquidityGrade", "TrendScore", "MomentumScore", "ReturnStaircaseScore",
            "RelativeStrengthScore", "VolumeScore", "LiquidityScore", "TriggerScore", "RiskScore",
            "TechnicalConfirmations", "ConfirmationsAvailable", "ConfirmationPct", "RSI",
            "StochRSI", "OverboughtRisk", "RSIChange5", "ADX", "PlusDI", "MinusDI", "DISpread",
            "MFI", "MFIChange5", "MACD_hist_pct", "MACDAccel", "ROC10", "DailyRet", "WeeklyRet",
            "MonthlyRet", "Return3M", "Return6M", "ReturnStep1Wvs1D", "ReturnStep1Mvs1W",
            "ReturnStep3Mvs1M", "ReturnStep6Mvs3M", "DailyShareOfWeek", "WeeklyShareOfMonth",
            "CoreReturnStaircase", "SweetSpotReturnStaircase", "FullReturnStaircase",
            "LongLadderStrictSteps", "ReturnStaircasePct", "PositiveDays20Pct", "PositiveWeeks8Pct",
            "PositiveMonths6Pct", "TrendAnnualized60", "TrendR2_60", "HigherHighLow20Pct",
            "RelStrength1M", "RelStrength3M", "RelStrength6M", "EMA20", "EMA50", "SMA200",
            "EMA20Slope10", "EMA50Slope20", "SMA200Slope20", "TrendConsistency20",
            "EfficiencyRatio20", "Breakout20Pct", "PctFrom52WHigh", "RangePosition52W", "RVOL",
            "VolumeTrend", "UpVolumeRatio20", "OBVImpulse20", "ATR14Pct", "ExtensionATR",
            "PriceAboveEMA20Pct", "Volatility20", "WorstDay20", "MaxDrawdown63", "AvgVol20",
            "MedianVol20", "AvgTradedValue", "MedianTradedValue20", "AvgTradedValue60",
            "AvgTradedValueCr", "MedianTradedValueCr", "ActiveDays20Pct", "TurnoverStability20",
            "RecentTurnoverRatio", "DaysAbove75LakhPct", "DaysAbove2CrPct", "DaysAbove4CrPct",
            "Benchmark", "Sector",
        ]

        st.download_button(
            "Download as CSV",
            result[download_columns].to_csv(index=False).encode(),
            file_name="nse_sweet_spot_momentum_liquidity_v17.csv",
        )

    st.markdown("---")

    # Original sector heatmap logic kept unchanged.
    st.subheader("Sector Performance Heatmap (Nifty 500 overlay only)")
    classified = data[data["Sector"] != "Unclassified"]
    grouped = classified.groupby("Sector")[[
        "DailyRet", "WeeklyRet", "MonthlyRet", "Return3M"
    ]].mean()
    grouped = grouped.dropna(subset=["MonthlyRet"])
    grouped["CompositeScore"] = (
        grouped["Return3M"] * 0.40
        + grouped["MonthlyRet"] * 0.30
        + grouped["WeeklyRet"] * 0.20
        + grouped["DailyRet"] * 0.10
    )
    sector_performance = (
        grouped.round(2)
        .sort_values("CompositeScore", ascending=False)
        .reset_index()
    )

    if not sector_performance.empty:
        heat_df = sector_performance.set_index("Sector")[[
            "DailyRet", "WeeklyRet", "MonthlyRet", "Return3M"
        ]]
        heat_df.columns = ["Daily %", "1-Week %", "1-Month %", "3-Month %"]
        fig = px.imshow(
            heat_df,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(color="Return %"),
        )
        fig.update_layout(
            height=max(400, 28 * len(heat_df)),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        top3 = sector_performance.head(3)
        columns = st.columns(3)
        for index, (_, row) in enumerate(top3.iterrows()):
            with columns[index]:
                st.metric(
                    label=f"#{index + 1} {row['Sector']}",
                    value=f"{row['CompositeScore']}",
                    delta=f"3M: {row['Return3M']}% | 1M: {row['MonthlyRet']}%",
                )
    else:
        st.info("Sector data unavailable this refresh.")

    st.caption(
        f"Last data refresh: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        "-- auto-refreshes hourly."
    )

# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   requests
#   streamlit-autorefresh
#   plotly
