"""
NSE 500 Swing Trading Screener — Web App (Streamlit)
=====================================================
Combines momentum + technical analysis + volume + MFI + FII/institutional-buying
signals to shortlist NSE-500 stocks worth watching for swing trades.

RUN LOCALLY:
    pip install -r requirements.txt   (see bottom of this file for the list)
    streamlit run app.py

DEPLOY AS A REAL WEB APP (free):
    1. Push this file + requirements.txt to a GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> pick the repo/branch/app.py.
    3. It deploys with a public URL and keeps running 24/7, refreshing hourly.

DATA REFRESH:
    - Cached with TTL=3600s (1 hour). A background auto-refresh component
      re-runs the script every hour so the browser tab updates itself too.
    - You can also hit the "Refresh now" button any time.

NOTES ON DATA SOURCES:
    - NSE 500 list: pulled live from NSE's own index CSV, with an offline
      fallback list bundled in FALLBACK_500 (top-liquidity subset) in case
      NSE blocks the request (it frequently rate-limits/geoblocks bots).
    - Prices/Volume: yfinance, using "<SYMBOL>.NS" tickers.
    - FII/DII market-wide net flow: NSE FII/DII daily report (aggregate only —
      NSE does NOT publish per-stock FII data).
    - Stock-level "institutional buying" proxy: NSE bulk/block deal reports.
"""

import time
import io
import datetime as dt
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="NSE 500 Swing Screener", layout="wide")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# A small offline fallback (liquid large/mid caps) used only if the live
# NSE 500 CSV fetch fails. Extend this list as you like.
FALLBACK_500 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","BAJFINANCE","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
    "SUNPHARMA","TITAN","ULTRACEMCO","WIPRO","ADANIENT","ADANIPORTS","ONGC",
    "NTPC","POWERGRID","TATAMOTORS","TATASTEEL","JSWSTEEL","HCLTECH","M&M",
    "BAJAJFINSV","DIVISLAB","GRASIM","DRREDDY","CIPLA","EICHERMOT","HEROMOTOCO",
    "TECHM","INDUSINDBK","COALINDIA","BPCL","IOC","SBILIFE","HDFCLIFE",
    "NESTLEIND","BRITANNIA","DABUR","GODREJCP","PIDILITIND","HAVELLS","DLF",
    "SIEMENS","ABB","PFC","RECLTD","IRCTC","ZOMATO","TRENT","PAYTM","POLYCAB",
]

# --------------------------------------------------------------------------
# 1. Universe: fetch NSE 500 constituents
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_nse500_symbols():
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r = requests.get(url, headers=NSE_HEADERS, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        symbols = df["Symbol"].dropna().unique().tolist()
        if len(symbols) > 50:
            return symbols
    except Exception:
        pass
    return FALLBACK_500

# --------------------------------------------------------------------------
# 2. Market-wide FII/DII net flow (aggregate, not per-stock)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_fii_dii_flow():
    """Returns latest FII/DII net cash-market flow (in INR Cr) if reachable."""
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        r = s.get(url, headers=NSE_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

# --------------------------------------------------------------------------
# 3. Bulk / Block deals -> proxy for large institutional buying in a stock
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_bulk_deal_buyers(lookback_days=5):
    """Best-effort scrape of NSE bulk deal buy-side symbols in last N sessions."""
    buys = set()
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        today = dt.date.today()
        for i in range(lookback_days):
            d = today - dt.timedelta(days=i)
            ds = d.strftime("%d-%m-%Y")
            url = f"https://www.nseindia.com/api/historical/bulk-deals?from={ds}&to={ds}"
            r = s.get(url, headers=NSE_HEADERS, timeout=8)
            if r.status_code == 200:
                js = r.json()
                for row in js.get("data", []):
                    if str(row.get("BD_BUY_SELL", "")).upper().startswith("BUY"):
                        buys.add(row.get("BD_SYMBOL"))
    except Exception:
        pass
    return buys

# --------------------------------------------------------------------------
# 4. Technical indicator helpers (no external TA library needed)
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
# 5. Bulk data fetch + indicator computation for the whole universe
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def build_screener_dataset(symbols, period="9mo"):
    tickers = [f"{s}.NS" for s in symbols]
    rows = []
    chunk = 40
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
            if len(df) < 60:
                continue
            close = df["Close"]
            vol = df["Volume"]
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean() if len(df) >= 200 else pd.Series([np.nan] * len(df))
            rsi14 = rsi(close, 14)
            macd_line, signal_line = macd(close)
            adx14 = adx(df, 14)
            mfi14 = mfi(df, 14)
            avgvol20 = vol.rolling(20).mean()
            roc10 = close.pct_change(10) * 100

            last = -1
            row = {
                "Symbol": sym,
                "Price": round(close.iloc[last], 2),
                "RSI": round(rsi14.iloc[last], 1) if not pd.isna(rsi14.iloc[last]) else np.nan,
                "MACD_bullish": bool(macd_line.iloc[last] > signal_line.iloc[last]),
                "ADX": round(adx14.iloc[last], 1) if not pd.isna(adx14.iloc[last]) else np.nan,
                "MFI": round(mfi14.iloc[last], 1) if not pd.isna(mfi14.iloc[last]) else np.nan,
                "RVOL": round(vol.iloc[last] / avgvol20.iloc[last], 2) if avgvol20.iloc[last] > 0 else np.nan,
                "ROC10": round(roc10.iloc[last], 2) if not pd.isna(roc10.iloc[last]) else np.nan,
                "Above_SMA50": bool(close.iloc[last] > sma50.iloc[last]) if not pd.isna(sma50.iloc[last]) else False,
                "Above_SMA200": bool(close.iloc[last] > sma200.iloc[last]) if not pd.isna(sma200.iloc[last]) else False,
                "SMA50_gt_SMA200": bool(sma50.iloc[last] > sma200.iloc[last]) if not pd.isna(sma200.iloc[last]) else False,
                "AvgVol20": int(avgvol20.iloc[last]) if not pd.isna(avgvol20.iloc[last]) else 0,
            }
            rows.append(row)
        time.sleep(0.5)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------
# 6. Scoring + human-readable rationale
# --------------------------------------------------------------------------
def score_and_explain(row, bulk_buyers):
    reasons = []
    score = 0
    if not pd.isna(row["RSI"]) and 40 <= row["RSI"] <= 65:
        reasons.append(f"RSI {row['RSI']} (bullish zone, not overbought)")
        score += 1
    if row["MACD_bullish"]:
        reasons.append("MACD bullish crossover")
        score += 1
    if not pd.isna(row["ADX"]) and row["ADX"] >= 20:
        reasons.append(f"ADX {row['ADX']} (trending)")
        score += 1
    if row["Above_SMA50"]:
        reasons.append("Price above 50-DMA")
        score += 1
    if row["SMA50_gt_SMA200"]:
        reasons.append("50-DMA above 200-DMA (uptrend)")
        score += 1
    if not pd.isna(row["MFI"]) and 40 <= row["MFI"] <= 80:
        reasons.append(f"MFI {row['MFI']} (money flowing in)")
        score += 1
    if not pd.isna(row["RVOL"]) and row["RVOL"] >= 1.5:
        tag = "strong" if row["RVOL"] >= 2 else "elevated"
        reasons.append(f"Volume {row['RVOL']}x avg ({tag})")
        score += 2 if row["RVOL"] >= 2 else 1
    if not pd.isna(row["ROC10"]) and row["ROC10"] > 0:
        reasons.append(f"+{row['ROC10']}% momentum (10d)")
        score += 1
    if row["Symbol"] in bulk_buyers:
        reasons.append("Recent NSE bulk-deal BUY (institutional accumulation)")
        score += 2
    return score, "; ".join(reasons) if reasons else "No strong signals"

# --------------------------------------------------------------------------
# 7. Streamlit UI
# --------------------------------------------------------------------------
st.title("NSE 500 Swing Trading Screener")
st.caption("Momentum + technical + volume + MFI + institutional-buying screener, refreshed hourly.")

if HAS_AUTOREFRESH:
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")

with st.sidebar:
    st.header("Filters")
    price_min, price_max = st.slider("Price range (₹)", 10, 10000, (50, 5000), step=10)
    rsi_lo, rsi_hi = st.slider("RSI range", 0, 100, (40, 65))
    mfi_lo, mfi_hi = st.slider("MFI range", 0, 100, (40, 80))
    min_rvol = st.slider("Min relative volume (RVOL)", 1.0, 5.0, 1.5, step=0.1)
    min_adx = st.slider("Min ADX (trend strength)", 0, 50, 20)
    require_macd = st.checkbox("Require MACD bullish crossover", value=True)
    require_uptrend = st.checkbox("Require price > 50-DMA", value=True)
    require_roc_pos = st.checkbox("Require positive 10-day momentum", value=True)
    min_avg_vol = st.number_input("Min avg daily volume (shares)", value=200000, step=50000)
    max_results = st.slider("Max stocks to show", 5, 100, 30)
    refresh_btn = st.button("Refresh data now")

if refresh_btn:
    st.cache_data.clear()

symbols = get_nse500_symbols()
st.write(f"Universe size: **{len(symbols)}** symbols")

fii_dii_df = get_fii_dii_flow()
bulk_buyers = get_recent_bulk_deal_buyers()

data = build_screener_dataset(symbols)

if data.empty:
    st.error("Could not fetch price data (network/API issue). Try 'Refresh data now'.")
else:
    scored = data.copy()
    scored["Score"], scored["Why"] = zip(*scored.apply(lambda r: score_and_explain(r, bulk_buyers), axis=1))

    mask = (
        (scored["Price"].between(price_min, price_max)) &
        (scored["RSI"].between(rsi_lo, rsi_hi)) &
        (scored["MFI"].between(mfi_lo, mfi_hi)) &
        (scored["RVOL"] >= min_rvol) &
        (scored["ADX"] >= min_adx) &
        (scored["AvgVol20"] >= min_avg_vol)
    )
    if require_macd:
        mask &= scored["MACD_bullish"]
    if require_uptrend:
        mask &= scored["Above_SMA50"]
    if require_roc_pos:
        mask &= scored["ROC10"] > 0

    result = scored[mask].sort_values("Score", ascending=False).head(max_results)

    st.subheader(f"Shortlisted stocks ({len(result)})")
    display_cols = ["Symbol", "Price", "RSI", "MFI", "ADX", "RVOL", "ROC10",
                     "MACD_bullish", "Above_SMA50", "SMA50_gt_SMA200", "Score", "Why"]
    st.dataframe(result[display_cols], use_container_width=True, hide_index=True)

    st.download_button("Download shortlist as CSV",
                        result[display_cols].to_csv(index=False).encode(),
                        file_name="nse500_swing_shortlist.csv")

    st.subheader("Market-wide FII/DII flow (context, not per-stock)")
    if not fii_dii_df.empty:
        st.dataframe(fii_dii_df, use_container_width=True, hide_index=True)
    else:
        st.info("FII/DII live feed unreachable right now (NSE often blocks server-side requests). "
                "Check https://www.nseindia.com/reports/fii-dii manually if needed.")

    if bulk_buyers:
        st.caption(f"Stocks with recent NSE bulk-deal buying (last 5 sessions): {', '.join(sorted(bulk_buyers))}")

    st.caption(f"Last data refresh: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — auto-refreshes hourly.")

# --------------------------------------------------------------------------
# requirements.txt (create this as a separate file alongside app.py):
#   streamlit
#   yfinance
#   pandas
#   numpy
#   requests
#   streamlit-autorefresh
# --------------------------------------------------------------------------
