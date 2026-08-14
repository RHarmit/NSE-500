"""
NSE Full Universe Momentum Screener + Sector Heatmap
=====================================================
v16 FINAL: Adds a clean positive RETURN STAIRCASE and a stronger
ANTI-OVERBOUGHT entry engine while preserving the original NSE universe,
price range, liquidity floor, compact ranking table, sector heatmap, and
hourly-refresh structure.

UNCHANGED CORE ELIGIBILITY:
  - Price between Rs 99 and Rs 1200
  - Avg volume >= 10,000 shares/day

ANTI-OVERBOUGHT / ANTI-CHASE GATE:
  - 52 <= RSI < 68
  - 50 <= MFI < 75
  - Smoothed Stochastic RSI < 80
  - Latest daily return is positive but no more than 5%
  - Price is no more than 2.5 ATR or 8% above EMA20
  - A 20-day breakout cannot be more than 4% extended

MANDATORY RETURN STAIRCASE:
  - 0 < 1-day return < 1-week return < 1-month return
  - 3-month and 6-month returns must both be positive
  - An optional extra-strict mode also requires 1M < 3M < 6M

This rejects charts where a single day is doing most of the work, such as
1D > 1W > 1M, and favors steady momentum that broadens across time horizons.

TECHNICAL QUALITY:
  - Price > EMA20 > EMA50 and Price > SMA200
  - Wilder ADX >= 20 with +DI > -DI
  - Bullish MACD, positive ROC10, market-relative strength, accumulation,
    smooth trend persistence, and controlled downside/volatility

RANKING:
  CompositeScore is an anchored 0-100 score. The momentum component directly
  rewards a balanced return staircase rather than a one-day price spike.
"""

import io
import datetime as dt
from typing import Iterable, Optional, Tuple

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

st.set_page_config(page_title="NSE Full Universe Momentum Screener", layout="wide")

# ---------------- Original eligibility settings (kept unchanged) ----------------
PRICE_MIN = 99
PRICE_MAX = 1200
RSI_OVERBOUGHT = 70
MFI_OVERBOUGHT = 80
MIN_AVG_VOLUME = 10000
ROC_LOOKBACK = 10

# ---------------- Upgraded technical-quality settings ----------------
DATA_PERIOD = "1y"
MIN_DATA_DAYS = 40
MIN_TECH_HISTORY_DAYS = 205
BENCHMARK_TICKERS = ("^CRSLDX", "^NSEI")  # NIFTY 500, then NIFTY 50 fallback

TQ_ADX_MIN = 20
TQ_RSI_MIN = 52
TQ_RSI_MAX = 68
TQ_MFI_MIN = 50
TQ_MFI_MAX = 75
TQ_STOCH_RSI_MAX = 80
TQ_MAX_DAILY_RETURN = 5.0
TQ_MAX_DISTANCE_FROM_52W_HIGH = -20.0
TQ_MAX_EXTENSION_ATR = 2.5
TQ_MAX_PRICE_ABOVE_EMA20_PCT = 8.0
TQ_MAX_BREAKOUT_EXTENSION_PCT = 4.0
TQ_MIN_CONFIRMATION_PCT = 60.0

SCORE_WEIGHTS = {
    "Trend": 0.27,
    "Momentum": 0.27,
    "RelativeStrength": 0.16,
    "Volume": 0.12,
    "Trigger": 0.08,
    "Risk": 0.05,
    "Confirmation": 0.03,
    "Liquidity": 0.02,
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

            avg_volume_20 = volume.tail(20).mean()
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
            avg_traded_value = (
                avg_volume_20 * last_close
                if pd.notna(avg_volume_20)
                else np.nan
            )

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
                "FullReturnStaircase": full_return_staircase,
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
                "ATR14Pct": safe_round(atr_pct, 2),
                "ExtensionATR": safe_round(extension_atr, 2),
                "PriceAboveEMA20Pct": safe_round(price_above_ema20_pct, 2),
                "Volatility20": safe_round(volatility_20, 2),
                "WorstDay20": safe_round(worst_day_20, 2),
                "MaxDrawdown63": safe_round(max_drawdown_63, 2),
                "AvgVol20": int(avg_volume_20) if pd.notna(avg_volume_20) else 0,
                "AvgTradedValue": safe_round(avg_traded_value, 0),
            })

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        return dataset

    confirmation = dataset.apply(technical_confirmation_summary, axis=1)
    return pd.concat([dataset, confirmation], axis=1)


# ---------------- Gates ----------------
def _available_check(value: float, condition: bool) -> Tuple[bool, bool]:
    return pd.notna(value), bool(condition) if pd.notna(value) else False


def technical_confirmation_summary(row: pd.Series) -> pd.Series:
    """Count independent confirmations without failing on a missing benchmark."""
    checks = [
        _available_check(row.get("EMA50"), row.get("EMA50", np.nan) > row.get("SMA200", np.nan)),
        _available_check(row.get("EMA20Slope10"), row.get("EMA20Slope10", np.nan) > 0),
        _available_check(row.get("EMA50Slope20"), row.get("EMA50Slope20", np.nan) > 0),
        _available_check(row.get("SMA200Slope20"), row.get("SMA200Slope20", np.nan) > 0),
        _available_check(row.get("MACDAccel"), row.get("MACDAccel", np.nan) > 0),
        _available_check(row.get("RSIChange5"), row.get("RSIChange5", np.nan) >= 0),
        _available_check(row.get("StochRSI"), row.get("StochRSI", np.nan) < TQ_STOCH_RSI_MAX),
        _available_check(row.get("RelStrength3M"), row.get("RelStrength3M", np.nan) > 0),
        _available_check(
            row.get("ReturnStep3Mvs1M"),
            row.get("ReturnStep3Mvs1M", np.nan) > 0,
        ),
        _available_check(
            row.get("ReturnStep6Mvs3M"),
            row.get("ReturnStep6Mvs3M", np.nan) > 0,
        ),
        _available_check(
            row.get("DailyShareOfWeek"),
            row.get("DailyShareOfWeek", np.nan) <= 70,
        ),
        _available_check(
            row.get("WeeklyShareOfMonth"),
            row.get("WeeklyShareOfMonth", np.nan) <= 80,
        ),
        _available_check(
            row.get("TrendConsistency20"),
            row.get("TrendConsistency20", np.nan) >= 65,
        ),
        _available_check(
            row.get("UpVolumeRatio20"),
            row.get("UpVolumeRatio20", np.nan) >= 52,
        ),
        _available_check(row.get("VolumeTrend"), row.get("VolumeTrend", np.nan) >= 0.90),
        _available_check(row.get("Breakout20Pct"), row.get("Breakout20Pct", np.nan) >= -5),
        _available_check(
            row.get("EfficiencyRatio20"),
            row.get("EfficiencyRatio20", np.nan) >= 0.25,
        ),
        _available_check(row.get("Return6M"), row.get("Return6M", np.nan) > 0),
        _available_check(
            row.get("DailyRet"),
            0 < row.get("DailyRet", np.nan) <= TQ_MAX_DAILY_RETURN,
        ),
        _available_check(
            row.get("PriceAboveEMA20Pct"),
            row.get("PriceAboveEMA20Pct", np.nan) <= 6.0,
        ),
        _available_check(row.get("ExtensionATR"), row.get("ExtensionATR", np.nan) <= 2.0),
        _available_check(row.get("OBVImpulse20"), row.get("OBVImpulse20", np.nan) > 0),
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
    """Original price, outer overbought limits, and liquidity eligibility."""
    if pd.isna(row["Price"]) or not (PRICE_MIN < row["Price"] < PRICE_MAX):
        return False
    if pd.isna(row["RSI"]) or row["RSI"] >= RSI_OVERBOUGHT:
        return False
    if pd.isna(row["MFI"]) or row["MFI"] >= MFI_OVERBOUGHT:
        return False
    if row["AvgVol20"] < MIN_AVG_VOLUME:
        return False
    return True


def passes_anti_overbought_gate(row: pd.Series) -> bool:
    """Bullish, but not stretched enough to be a late/chasing entry."""
    required = [
        "RSI", "MFI", "StochRSI", "DailyRet", "ExtensionATR",
        "PriceAboveEMA20Pct", "Breakout20Pct",
    ]
    if any(pd.isna(row.get(column, np.nan)) for column in required):
        return False
    if not (TQ_RSI_MIN <= row["RSI"] < TQ_RSI_MAX):
        return False
    if not (TQ_MFI_MIN <= row["MFI"] < TQ_MFI_MAX):
        return False
    if row["StochRSI"] >= TQ_STOCH_RSI_MAX:
        return False
    if not (0 < row["DailyRet"] <= TQ_MAX_DAILY_RETURN):
        return False
    if row["ExtensionATR"] > TQ_MAX_EXTENSION_ATR:
        return False
    if row["PriceAboveEMA20Pct"] > TQ_MAX_PRICE_ABOVE_EMA20_PCT:
        return False
    if row["Breakout20Pct"] > TQ_MAX_BREAKOUT_EXTENSION_PCT:
        return False
    return True


def passes_return_staircase(
    row: pd.Series,
    require_full_staircase: bool = True,
) -> bool:
    """Require positive returns that increase cleanly with each horizon."""
    required = ["DailyRet", "WeeklyRet", "MonthlyRet", "Return3M", "Return6M"]
    if any(pd.isna(row.get(column, np.nan)) for column in required):
        return False

    if not (0 < row["DailyRet"] < row["WeeklyRet"] < row["MonthlyRet"]):
        return False
    if row["Return3M"] <= 0 or row["Return6M"] <= 0:
        return False
    if require_full_staircase and not (
        row["MonthlyRet"] < row["Return3M"] < row["Return6M"]
    ):
        return False
    return True


def passes_technical_quality(
    row: pd.Series,
    require_full_staircase: bool = True,
) -> bool:
    """High-conviction bullish trend, smooth momentum, and controlled entry gate."""
    required = [
        "HistoryDays", "Price", "EMA20", "EMA50", "SMA200", "ADX", "PlusDI", "MinusDI",
        "RSI", "MFI", "StochRSI", "MACD_hist_pct", "ROC10", "DailyRet", "WeeklyRet",
        "MonthlyRet", "Return3M", "Return6M", "PctFrom52WHigh", "Breakout20Pct",
        "ExtensionATR", "PriceAboveEMA20Pct", "ConfirmationPct",
    ]
    if any(pd.isna(row.get(column, np.nan)) for column in required):
        return False

    if row["HistoryDays"] < MIN_TECH_HISTORY_DAYS:
        return False
    if not (row["Price"] > row["EMA20"] > row["EMA50"]):
        return False
    if row["Price"] <= row["SMA200"]:
        return False
    if row["ADX"] < TQ_ADX_MIN or row["PlusDI"] <= row["MinusDI"]:
        return False
    if not passes_anti_overbought_gate(row):
        return False
    if row["MACD_hist_pct"] <= 0 or row["ROC10"] <= 0:
        return False
    if not passes_return_staircase(row, require_full_staircase):
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
        linear_score(d["ADX"], 20, 45) * 0.20
        + linear_score(d["DISpread"], 0, 25) * 0.12
        + linear_score(d["EMA20Slope10"], 0, 5) * 0.18
        + linear_score(d["EMA50Slope20"], 0, 8) * 0.15
        + linear_score(d["SMA200Slope20"], 0, 5) * 0.10
        + linear_score(d["TrendConsistency20"], 55, 95) * 0.15
        + linear_score(d["EfficiencyRatio20"], 0.20, 0.65) * 0.10
    )

    d["ReturnStaircaseScore"] = (
        sweet_spot_score(d["DailyRet"], 0.01, 0.80, 3.50) * 0.14
        + sweet_spot_score(d["WeeklyRet"], 0.20, 4.00, 12.00) * 0.18
        + sweet_spot_score(d["MonthlyRet"], 1.00, 10.00, 30.00) * 0.22
        + linear_score(d["ReturnStep1Wvs1D"], 0, 6) * 0.10
        + linear_score(d["ReturnStep1Mvs1W"], 0, 15) * 0.14
        + linear_score(d["ReturnStep3Mvs1M"], 0, 30) * 0.08
        + linear_score(d["ReturnStep6Mvs3M"], 0, 45) * 0.06
        + sweet_spot_score(d["DailyShareOfWeek"], 5, 30, 75) * 0.04
        + sweet_spot_score(d["WeeklyShareOfMonth"], 10, 45, 85) * 0.04
    )

    d["MomentumScore"] = (
        sweet_spot_score(d["RSI"], 52, 61, 68) * 0.10
        + sweet_spot_score(d["StochRSI"], 10, 55, TQ_STOCH_RSI_MAX) * 0.07
        + linear_score(d["ROC10"], 0, 12) * 0.12
        + d["ReturnStaircaseScore"] * 0.36
        + linear_score(d["Return3M"], 0, 45) * 0.14
        + linear_score(d["Return6M"], 0, 80) * 0.13
        + linear_score(d["MomentumAcceleration"], -2, 5) * 0.08
    )

    d["RelativeStrengthScore"] = (
        linear_score(d["RelStrength1M"], -2, 12) * 0.30
        + linear_score(d["RelStrength3M"], 0, 25) * 0.45
        + linear_score(d["RelStrength6M"], -2, 40) * 0.25
    )

    d["VolumeScore"] = (
        sweet_spot_score(d["MFI"], 50, 63, 75) * 0.20
        + linear_score(d["RVOL"], 0.80, 2.50) * 0.20
        + linear_score(d["VolumeTrend"], 0.80, 1.80) * 0.18
        + linear_score(d["UpVolumeRatio20"], 48, 70) * 0.20
        + linear_score(d["OBVImpulse20"], -5, 35) * 0.22
    )

    d["TriggerScore"] = (
        linear_score(d["MACD_hist_pct"], 0, 1.50) * 0.28
        + linear_score(d["MACDAccel"], -0.05, 0.40) * 0.23
        + sweet_spot_score(d["Breakout20Pct"], -5, 0.50, 4.00) * 0.24
        + linear_score(d["PctFrom52WHigh"], -20, 0) * 0.10
        + linear_score(d["RSIChange5"], -2, 8) * 0.08
        + inverse_score(d["StochRSI"], 55, TQ_STOCH_RSI_MAX) * 0.07
    )

    d["RiskScore"] = (
        sweet_spot_score(d["ExtensionATR"], 0, 1.10, TQ_MAX_EXTENSION_ATR) * 0.30
        + sweet_spot_score(
            d["PriceAboveEMA20Pct"], 0, 3.5, TQ_MAX_PRICE_ABOVE_EMA20_PCT
        ) * 0.20
        + linear_score(d["MaxDrawdown63"], -25, -3) * 0.25
        + inverse_score(d["Volatility20"], 25, 80) * 0.12
        + linear_score(d["WorstDay20"], -12, -2) * 0.13
    )

    d["ConfirmationScore"] = d["ConfirmationPct"].clip(0, 100).fillna(0)
    d["LiquidityScore"] = percentile_score(np.log1p(d["AvgTradedValue"].clip(lower=0)))

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
        tags.append("1D<1W<1M<3M<6M")
    elif bool(row.get("CoreReturnStaircase", False)):
        tags.append("1D<1W<1M")

    if (
        row.get("RSI", 999) < 65
        and row.get("MFI", 999) < 70
        and row.get("StochRSI", 999) < 70
        and row.get("PriceAboveEMA20Pct", 999) <= 6
    ):
        tags.append("Not overbought")

    if row.get("Breakout20Pct", -999) >= 0:
        tags.append("20D breakout")
    elif row.get("Breakout20Pct", -999) >= -2:
        tags.append("Near 20D breakout")

    if row.get("RelStrength3M", -999) >= 10:
        tags.append("Strong vs market")
    elif row.get("RelStrength3M", -999) > 0:
        tags.append("Beating market")

    if row.get("RVOL", 0) >= 1.30 or row.get("VolumeTrend", 0) >= 1.20:
        tags.append("Volume expansion")
    if row.get("MACDAccel", -999) > 0:
        tags.append("MACD accelerating")
    if row.get("PctFrom52WHigh", -999) >= -5:
        tags.append("Near 52W high")

    return " | ".join(tags[:3]) if tags else "Qualified bullish trend"


# ---------------- UI ----------------
st.title("NSE Full Universe Momentum Screener")
st.caption(
    f"Scans ALL NSE main-board (EQ series) stocks. Price remains Rs {PRICE_MIN}-{PRICE_MAX} "
    f"with avg volume >= {MIN_AVG_VOLUME:,}. Every displayed stock must be bullish but not "
    f"overbought (RSI {TQ_RSI_MIN}-{TQ_RSI_MAX}, MFI {TQ_MFI_MIN}-{TQ_MFI_MAX}, "
    f"smoothed StochRSI<{TQ_STOCH_RSI_MAX}, daily return<={TQ_MAX_DAILY_RETURN:.0f}%, "
    f"extension<={TQ_MAX_EXTENSION_ATR} ATR and <={TQ_MAX_PRICE_ABOVE_EMA20_PCT:.0f}% above EMA20). "
    "The mandatory pattern is 0 < 1D < 1W < 1M with positive 3M and 6M returns."
)

if HAS_AUTOREFRESH:
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")

with st.sidebar:
    st.header("Controls")
    top_n = st.slider("How many stocks to show", 10, 150, 40)
    require_full_staircase = st.checkbox(
        "Extra strict: require 1D < 1W < 1M < 3M < 6M",
        value=False,
        help=(
            "Off by default to preserve enough valid candidates. The requested "
            "0 < 1D < 1W < 1M pattern always remains mandatory, and 3M/6M must "
            "still both be positive."
        ),
    )
    refresh_btn = st.button("Refresh data now")

if refresh_btn:
    st.cache_data.clear()

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

    st.subheader(f"Top {top_n} Most Relevant Stocks (Technically Sound + Strong Momentum)")

    basic_ok = data[data.apply(passes_basic_gates, axis=1)].copy()
    not_overbought_ok = basic_ok[
        basic_ok.apply(passes_anti_overbought_gate, axis=1)
    ].copy()
    staircase_ok = not_overbought_ok[
        not_overbought_ok.apply(
            lambda row: passes_return_staircase(row, require_full_staircase),
            axis=1,
        )
    ].copy()
    technical_ok = staircase_ok[
        staircase_ok.apply(
            lambda row: passes_technical_quality(row, require_full_staircase),
            axis=1,
        )
    ].copy()

    staircase_label = "full 1D<1W<1M<3M<6M" if require_full_staircase else "core 1D<1W<1M"
    st.write(
        f"{len(data)} scanned -> {len(basic_ok)} passed price/liquidity/outer limits -> "
        f"{len(not_overbought_ok)} were bullish but not overbought -> "
        f"{len(staircase_ok)} passed the **{staircase_label} positive-return staircase** -> "
        f"**{len(technical_ok)} passed the complete Technical Quality gate**. "
        f"Relative-strength benchmark: **{benchmark_used}**."
    )

    if technical_ok.empty:
        st.info(
            "No stocks currently clear the anti-overbought, positive-return staircase, "
            "and technical-quality gates. This intentionally avoids one-day spikes. "
            "You can temporarily uncheck the full 3M/6M staircase while the mandatory "
            "1D < 1W < 1M rule remains active."
        )
    else:
        scored = build_composite_scores(technical_ok)
        result = scored.sort_values(
            ["CompositeScore", "MomentumScore", "TrendScore", "AvgTradedValue"],
            ascending=[False, False, False, False],
        ).head(top_n)

        # Preserve the original compact ranking table.
        display_df = result[["Company Name", "Price", "CompositeScore"]].reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True)

        # Make every gate auditable without cluttering the main ranking table.
        with st.expander("Show momentum-staircase and anti-overbought checks"):
            audit_df = result[[
                "Company Name", "Price", "DailyRet", "WeeklyRet", "MonthlyRet",
                "Return3M", "Return6M", "RSI", "MFI", "StochRSI",
                "ExtensionATR", "PriceAboveEMA20Pct", "CompositeScore",
            ]].copy()
            audit_df = audit_df.rename(columns={
                "DailyRet": "1D %",
                "WeeklyRet": "1W %",
                "MonthlyRet": "1M %",
                "Return3M": "3M %",
                "Return6M": "6M %",
                "StochRSI": "Stoch RSI",
                "ExtensionATR": "EMA20 Extension (ATR)",
                "PriceAboveEMA20Pct": "% Above EMA20",
                "CompositeScore": "Score",
            }).reset_index(drop=True)
            audit_df.index = audit_df.index + 1
            st.dataframe(audit_df, use_container_width=True)
            st.caption(
                "Every row has 0 < 1D < 1W < 1M, positive 3M/6M returns, and passes "
                "the anti-overbought and anti-chasing limits."
            )

        download_columns = [
            "Symbol", "Company Name", "HistoryDays", "Price", "CompositeScore", "TechnicalGrade",
            "SetupSummary", "TrendScore", "MomentumScore", "ReturnStaircaseScore",
            "RelativeStrengthScore", "VolumeScore", "TriggerScore", "RiskScore",
            "TechnicalConfirmations", "ConfirmationsAvailable", "ConfirmationPct", "RSI",
            "StochRSI", "RSIChange5", "ADX", "PlusDI", "MinusDI", "DISpread", "MFI",
            "MFIChange5", "MACD_hist_pct", "MACDAccel", "ROC10", "DailyRet", "WeeklyRet",
            "MonthlyRet", "Return3M", "Return6M", "ReturnStep1Wvs1D", "ReturnStep1Mvs1W",
            "ReturnStep3Mvs1M", "ReturnStep6Mvs3M", "DailyShareOfWeek",
            "WeeklyShareOfMonth", "CoreReturnStaircase", "FullReturnStaircase",
            "ReturnStaircasePct",
            "RelStrength1M", "RelStrength3M", "RelStrength6M", "EMA20", "EMA50",
            "SMA200", "EMA20Slope10", "EMA50Slope20", "SMA200Slope20",
            "TrendConsistency20", "EfficiencyRatio20", "Breakout20Pct",
            "PctFrom52WHigh", "RangePosition52W", "RVOL", "VolumeTrend",
            "UpVolumeRatio20", "OBVImpulse20", "ATR14Pct", "ExtensionATR",
            "PriceAboveEMA20Pct", "Volatility20", "WorstDay20", "MaxDrawdown63", "AvgVol20",
            "AvgTradedValue", "Benchmark", "Sector",
        ]

        st.download_button(
            "Download as CSV",
            result[download_columns].to_csv(index=False).encode(),
            file_name="nse_technical_quality_momentum_v16_final.csv",
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
