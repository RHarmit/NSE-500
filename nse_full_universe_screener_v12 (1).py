"""
NSE FULL Universe Momentum Screener (Soft Composite Score) + Sector Heatmap
==============================================================================
v12: Switched from Nifty 500 only to the FULL NSE listed-equities universe
(EQUITY_L.csv from NSE archives), filtered to Series == 'EQ' (main-board,
normal settlement stocks -- excludes BE/trade-to-trade and SME-listed
names, which carry extra settlement restrictions unsuitable for swing
trading). This covers ~2,000+ stocks instead of 500.

Sector labels: NSE's full listing file has no Industry column, so sectors
are cross-referenced from the Nifty 500 index file where a stock overlaps;
everything else is marked "Unclassified" (mostly smaller-cap names).

Same logic as before otherwise:
  Hard gates: Price < Rs 1000, NOT overbought (RSI<70, MFI<80)
  Soft composite score: ADX, MFI, RVOL, ROC10, MACD histogram (weighted)
"""

import time
import io
import datetime as dt
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

PRICE_CAP = 1000
RSI_OVERBOUGHT = 70
MFI_OVERBOUGHT = 80
ROC_LOOKBACK = 10
WEIGHTS = {"ADX": 0.25, "MFI": 0.20, "RVOL": 0.20, "ROC10": 0.20, "MACD_hist": 0.15}

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

# --------------------------------------------------------------------------
# 1. FULL NSE universe (all listed equities, EQ series only)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_full_nse_universe():
    """Returns dict {symbol: company_name} for ALL NSE-listed EQ-series stocks."""
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        r = requests.get(url, headers=NSE_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        sym_col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
        name_col = "NAME OF COMPANY" if "NAME OF COMPANY" in df.columns else df.columns[1]
        series_col = " SERIES" if " SERIES" in df.columns else ("SERIES" if "SERIES" in df.columns else None)
        if series_col and series_col in df.columns:
            df = df[df[series_col].astype(str).str.strip() == "EQ"]
        df = df.dropna(subset=[sym_col, name_col])
        mapping = {str(row[sym_col]).strip(): str(row[name_col]).strip() for _, row in df.iterrows()}
        if len(mapping) > 500:
            return mapping
    except Exception:
        pass
    return FALLBACK_UNIVERSE

@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_overlay():
    """Cross-reference sectors from the Nifty 500 index file where available."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r = requests.get(url, headers=NSE_HEADERS, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        sector_col = "Industry" if "Industry" in df.columns else None
        if sector_col:
            return {row[sym_col].strip(): str(row[sector_col]).strip() for _, row in df.iterrows()}
    except Exception:
        pass
    return FALLBACK_SECTORS

# --------------------------------------------------------------------------
# 2. Indicators
# --------------------------------------------------------------------------
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()

def mfi(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_flow = tp * df["Volume"]
    direction = tp.diff()
    pos_flow = raw_flow.where(direction > 0, 0).rolling(period).sum()
    neg_flow = raw_flow.where(direction < 0, 0).rolling(period).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))

# --------------------------------------------------------------------------
# 3. Build dataset (batched, tolerant of the much larger universe)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def build_dataset(symbols, period="6mo"):
    tickers = [f"{s}.NS" for s in symbols]
    rows = []
    chunk = 75
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        try:
            data = yf.download(batch, period=period, interval="1d",
                                group_by="ticker", threads=True,
                                progress=False, auto_adjust=True)
        except Exception:
            continue
        for t in batch:
            sym = t.replace(".NS", "")
            try:
                df = data[t].dropna()
            except Exception:
                continue
            if len(df) < 40:
                continue

            close = df["Close"]
            vol = df["Volume"]
            last_close = close.iloc[-1]

            rsi14 = rsi(close, 14)
            macd_line, signal_line = macd(close)
            macd_hist = macd_line - signal_line
            adx14 = adx(df, 14)
            mfi14 = mfi(df, 14)
            avgvol20 = vol.rolling(20).mean()
            roc10 = close.pct_change(ROC_LOOKBACK).iloc[-1] * 100

            daily_ret = close.pct_change(1).iloc[-1] * 100
            weekly_ret = close.pct_change(5).iloc[-1] * 100
            monthly_ret = close.pct_change(21).iloc[-1] * 100 if len(df) > 21 else np.nan
            ret_3m = close.pct_change(63).iloc[-1] * 100 if len(df) > 63 else np.nan

            rvol = vol.iloc[-1] / avgvol20.iloc[-1] if avgvol20.iloc[-1] and avgvol20.iloc[-1] > 0 else np.nan
            macd_hist_pct = (macd_hist.iloc[-1] / last_close) * 100 if last_close > 0 else np.nan

            rows.append({
                "Symbol": sym,
                "Price": round(last_close, 2),
                "RSI": round(rsi14.iloc[-1], 1) if not pd.isna(rsi14.iloc[-1]) else np.nan,
                "MACD_hist_pct": round(macd_hist_pct, 3) if not pd.isna(macd_hist_pct) else np.nan,
                "ADX": round(adx14.iloc[-1], 1) if not pd.isna(adx14.iloc[-1]) else np.nan,
                "MFI": round(mfi14.iloc[-1], 1) if not pd.isna(mfi14.iloc[-1]) else np.nan,
                "RVOL": round(rvol, 2) if not pd.isna(rvol) else np.nan,
                "ROC10": round(roc10, 2) if not pd.isna(roc10) else np.nan,
                "DailyRet": round(daily_ret, 2) if not pd.isna(daily_ret) else np.nan,
                "WeeklyRet": round(weekly_ret, 2) if not pd.isna(weekly_ret) else np.nan,
                "MonthlyRet": round(monthly_ret, 2) if not pd.isna(monthly_ret) else np.nan,
                "Return3M": round(ret_3m, 2) if not pd.isna(ret_3m) else np.nan,
            })
    return pd.DataFrame(rows)

def passes_hard_gates(row):
    if pd.isna(row["Price"]) or row["Price"] >= PRICE_CAP:
        return False
    if pd.isna(row["RSI"]) or row["RSI"] >= RSI_OVERBOUGHT:
        return False
    if pd.isna(row["MFI"]) or row["MFI"] >= MFI_OVERBOUGHT:
        return False
    return True

def normalize(series):
    s = series.copy()
    valid = s.dropna()
    if valid.empty or valid.max() == valid.min():
        return s.fillna(0) * 0
    return ((s - valid.min()) / (valid.max() - valid.min()) * 100).fillna(0)

def build_composite_scores(df):
    d = df.copy()
    d["N_ADX"] = normalize(d["ADX"])
    d["N_MFI"] = normalize(d["MFI"])
    d["N_RVOL"] = normalize(d["RVOL"].clip(upper=5))
    d["N_ROC10"] = normalize(d["ROC10"])
    d["N_MACD"] = normalize(d["MACD_hist_pct"])
    d["CompositeScore"] = round(
        d["N_ADX"] * WEIGHTS["ADX"] + d["N_MFI"] * WEIGHTS["MFI"] +
        d["N_RVOL"] * WEIGHTS["RVOL"] + d["N_ROC10"] * WEIGHTS["ROC10"] +
        d["N_MACD"] * WEIGHTS["MACD_hist"], 1
    )
    return d

# ---------------- UI ----------------
st.title("NSE Full Universe Momentum Screener")
st.caption(
    f"Scans ALL NSE main-board (EQ series) stocks -- the full listed universe, not just "
    f"the Nifty 500. Hard rules: price < Rs {PRICE_CAP}, NOT overbought (RSI < "
    f"{RSI_OVERBOUGHT}, MFI < {MFI_OVERBOUGHT}). ADX, MFI strength, volume, momentum and "
    "MACD are blended into one Composite Score."
)
st.warning(
    "Note: this universe includes many smaller/less liquid stocks beyond the Nifty 500. "
    "Sector labels are only available for names that overlap with the Nifty 500 index; "
    "everything else shows as 'Unclassified'. Refreshes take longer due to the larger scan."
)

if HAS_AUTOREFRESH:
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")

with st.sidebar:
    st.header("Controls")
    top_n = st.slider("How many stocks to show", 10, 150, 40)
    refresh_btn = st.button("Refresh data now")

if refresh_btn:
    st.cache_data.clear()

universe_map = get_full_nse_universe()
sector_map = get_sector_overlay()
symbols = list(universe_map.keys())
st.write(f"Scanning **{len(symbols)}** NSE-listed EQ-series symbols this refresh.")

data = build_dataset(symbols)

if data.empty:
    st.error("Could not fetch price data (network/API issue). Try 'Refresh data now'.")
else:
    st.write(f"Successfully pulled data for **{len(data)}** of {len(symbols)} symbols.")

    data["Company Name"] = data["Symbol"].map(lambda s: universe_map.get(s, s))
    data["Sector"] = data["Symbol"].map(lambda s: sector_map.get(s, "Unclassified"))

    st.subheader(f"Top {top_n} Momentum Stocks (Price < Rs {PRICE_CAP}, Not Overbought)")
    candidates = data[data.apply(passes_hard_gates, axis=1)].copy()

    if candidates.empty:
        st.info("No stocks currently meet the hard gates. Try the next hourly refresh.")
    else:
        scored = build_composite_scores(candidates)
        result = scored.sort_values("CompositeScore", ascending=False).head(top_n)

        st.write(f"{len(candidates)} stocks passed the hard gates out of {len(data)} scanned; "
                 f"showing top {len(result)} by Composite Score.")

        display_df = result[["Company Name", "Price", "CompositeScore"]].reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True)

        st.download_button("Download as CSV",
                            result[["Company Name", "Price", "CompositeScore", "RSI", "ADX", "MFI",
                                    "RVOL", "ROC10", "MACD_hist_pct", "Sector"]].to_csv(index=False).encode(),
                            file_name="nse_full_momentum_composite.csv")

    st.markdown("---")

    st.subheader("Sector Performance Heatmap (Nifty 500 overlay only)")
    classified = data[data["Sector"] != "Unclassified"]
    grp = classified.groupby("Sector")[["DailyRet", "WeeklyRet", "MonthlyRet", "Return3M"]].mean()
    grp = grp.dropna(subset=["MonthlyRet"])
    grp["CompositeScore"] = (
        grp["Return3M"] * 0.40 + grp["MonthlyRet"] * 0.30 +
        grp["WeeklyRet"] * 0.20 + grp["DailyRet"] * 0.10
    )
    sector_perf = grp.round(2).sort_values("CompositeScore", ascending=False).reset_index()

    if not sector_perf.empty:
        heat_df = sector_perf.set_index("Sector")[["DailyRet", "WeeklyRet", "MonthlyRet", "Return3M"]]
        heat_df.columns = ["Daily %", "1-Week %", "1-Month %", "3-Month %"]
        fig = px.imshow(heat_df, text_auto=".1f", color_continuous_scale="RdYlGn",
                         color_continuous_midpoint=0, aspect="auto", labels=dict(color="Return %"))
        fig.update_layout(height=max(400, 28 * len(heat_df)), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        top3 = sector_perf.head(3)
        cols = st.columns(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                st.metric(label=f"#{i+1} {row['Sector']}", value=f"{row['CompositeScore']}",
                          delta=f"3M: {row['Return3M']}% | 1M: {row['MonthlyRet']}%")
    else:
        st.info("Sector data unavailable this refresh.")

    st.caption(f"Last data refresh: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- auto-refreshes hourly.")

# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   requests
#   streamlit-autorefresh
#   plotly
