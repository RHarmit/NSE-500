
"""
NSE 500 Momentum Screener + Sector Heatmap — Web App (Streamlit)
==================================================================
v8: Stripped down to exactly two things, per request:
  1. Top 10 Momentum Stocks — strong MFI & ADX improvement, but explicitly
     NOT overbought (RSI and MFI capped below classic overbought levels),
     with real volume, and price strictly below Rs 1500.
  2. Sector Performance Heatmap (daily / 1-week / 1-month / 3-month) with
     top 3 sectors by composite score.
No other sections (no FII/DII panel, no bulk deals, no analyst ratings,
no main shortlist) — kept intentionally minimal.

RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py
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

st.set_page_config(page_title="NSE 500 Momentum Screener", layout="wide")

PRICE_CAP = 1500          # HARD RULE

# --- Momentum, but explicitly NOT overbought ---
RSI_MIN, RSI_MAX = 55, 68        # strong momentum, capped below classic overbought (70)
MFI_MIN, MFI_MAX = 50, 75        # money flowing in strongly, capped below overbought (80)
ADX_MIN = 25                     # strong trend
ADX_MIN_ACCEL = 5                # ADX must have risen at least this much in ROC_LOOKBACK sessions
MFI_MIN_ACCEL = 5                # MFI must have risen at least this much in ROC_LOOKBACK sessions
RVOL_MIN = 1.2                   # real volume participation
ROC_LOOKBACK = 10                # ~2 weeks, for "improved strongly" measurement

SECTOR_WEIGHTS = {"Return3M": 0.40, "MonthlyRet": 0.30, "WeeklyRet": 0.20, "DailyRet": 0.10}

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

FALLBACK_UNIVERSE = {
    "RELIANCE": ("Reliance Industries Ltd.", "Oil Gas & Consumable Fuels"),
    "TCS": ("Tata Consultancy Services Ltd.", "IT - Software"),
    "HDFCBANK": ("HDFC Bank Ltd.", "Banks"),
    "INFY": ("Infosys Ltd.", "IT - Software"),
    "ICICIBANK": ("ICICI Bank Ltd.", "Banks"),
    "HINDUNILVR": ("Hindustan Unilever Ltd.", "FMCG"),
    "ITC": ("ITC Ltd.", "FMCG"),
    "SBIN": ("State Bank of India", "Banks"),
    "BHARTIARTL": ("Bharti Airtel Ltd.", "Telecom - Services"),
    "BAJFINANCE": ("Bajaj Finance Ltd.", "Finance"),
    "KOTAKBANK": ("Kotak Mahindra Bank Ltd.", "Banks"),
    "LT": ("Larsen & Toubro Ltd.", "Construction"),
    "AXISBANK": ("Axis Bank Ltd.", "Banks"),
    "ASIANPAINT": ("Asian Paints Ltd.", "Consumer Durables"),
    "MARUTI": ("Maruti Suzuki India Ltd.", "Automobile"),
    "SUNPHARMA": ("Sun Pharmaceutical Industries Ltd.", "Pharmaceuticals"),
    "TITAN": ("Titan Company Ltd.", "Consumer Durables"),
    "ULTRACEMCO": ("UltraTech Cement Ltd.", "Cement"),
    "WIPRO": ("Wipro Ltd.", "IT - Software"),
    "ADANIENT": ("Adani Enterprises Ltd.", "Metals & Mining"),
    "ADANIPORTS": ("Adani Ports and Special Economic Zone Ltd.", "Infrastructure"),
    "ONGC": ("Oil and Natural Gas Corporation Ltd.", "Oil Gas & Consumable Fuels"),
    "NTPC": ("NTPC Ltd.", "Power"),
    "POWERGRID": ("Power Grid Corporation of India Ltd.", "Power"),
    "TATAMOTORS": ("Tata Motors Ltd.", "Automobile"),
    "TATASTEEL": ("Tata Steel Ltd.", "Metals & Mining"),
    "JSWSTEEL": ("JSW Steel Ltd.", "Metals & Mining"),
    "HCLTECH": ("HCL Technologies Ltd.", "IT - Software"),
    "M&M": ("Mahindra & Mahindra Ltd.", "Automobile"),
    "BAJAJFINSV": ("Bajaj Finserv Ltd.", "Finance"),
    "DIVISLAB": ("Divi's Laboratories Ltd.", "Pharmaceuticals"),
    "GRASIM": ("Grasim Industries Ltd.", "Cement"),
    "DRREDDY": ("Dr. Reddy's Laboratories Ltd.", "Pharmaceuticals"),
    "CIPLA": ("Cipla Ltd.", "Pharmaceuticals"),
    "EICHERMOT": ("Eicher Motors Ltd.", "Automobile"),
    "HEROMOTOCO": ("Hero MotoCorp Ltd.", "Automobile"),
    "TECHM": ("Tech Mahindra Ltd.", "IT - Software"),
    "INDUSINDBK": ("IndusInd Bank Ltd.", "Banks"),
    "COALINDIA": ("Coal India Ltd.", "Metals & Mining"),
    "BPCL": ("Bharat Petroleum Corporation Ltd.", "Oil Gas & Consumable Fuels"),
    "IOC": ("Indian Oil Corporation Ltd.", "Oil Gas & Consumable Fuels"),
    "SBILIFE": ("SBI Life Insurance Company Ltd.", "Insurance"),
    "HDFCLIFE": ("HDFC Life Insurance Company Ltd.", "Insurance"),
    "NESTLEIND": ("Nestle India Ltd.", "FMCG"),
    "BRITANNIA": ("Britannia Industries Ltd.", "FMCG"),
    "DABUR": ("Dabur India Ltd.", "FMCG"),
    "GODREJCP": ("Godrej Consumer Products Ltd.", "FMCG"),
    "PIDILITIND": ("Pidilite Industries Ltd.", "Chemicals"),
    "HAVELLS": ("Havells India Ltd.", "Consumer Durables"),
    "DLF": ("DLF Ltd.", "Realty"),
    "SIEMENS": ("Siemens Ltd.", "Capital Goods"),
    "ABB": ("ABB India Ltd.", "Capital Goods"),
    "PFC": ("Power Finance Corporation Ltd.", "Finance"),
    "RECLTD": ("REC Ltd.", "Finance"),
    "IRCTC": ("Indian Railway Catering and Tourism Corporation Ltd.", "Services"),
    "ZOMATO": ("Eternal Ltd. (Zomato)", "Retailing"),
    "TRENT": ("Trent Ltd.", "Retailing"),
    "PAYTM": ("One97 Communications Ltd. (Paytm)", "Financial Technology (Fintech)"),
    "POLYCAB": ("Polycab India Ltd.", "Capital Goods"),
}

# --------------------------------------------------------------------------
# 1. Universe
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_nse500_universe():
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r = requests.get(url, headers=NSE_HEADERS, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        name_col = "Company Name" if "Company Name" in df.columns else df.columns[0]
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        sector_col = "Industry" if "Industry" in df.columns else None
        df = df.dropna(subset=[sym_col, name_col])
        if sector_col:
            mapping = {
                row[sym_col].strip(): (row[name_col].strip(), str(row[sector_col]).strip())
                for _, row in df.iterrows()
            }
        else:
            mapping = {row[sym_col].strip(): (row[name_col].strip(), "Unclassified") for _, row in df.iterrows()}
        if len(mapping) > 50:
            return mapping
    except Exception:
        pass
    return FALLBACK_UNIVERSE

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
# 3. Build dataset
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def build_dataset(symbols, period="9mo"):
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
            last_close = close.iloc[-1]

            rsi14 = rsi(close, 14)
            adx14 = adx(df, 14)
            mfi14 = mfi(df, 14)
            avgvol20 = vol.rolling(20).mean()

            daily_ret = close.pct_change(1).iloc[-1] * 100
            weekly_ret = close.pct_change(5).iloc[-1] * 100
            monthly_ret = close.pct_change(21).iloc[-1] * 100 if len(df) > 21 else np.nan
            ret_3m = close.pct_change(63).iloc[-1] * 100 if len(df) > 63 else np.nan

            rvol = vol.iloc[-1] / avgvol20.iloc[-1] if avgvol20.iloc[-1] and avgvol20.iloc[-1] > 0 else np.nan

            mfi_change_10d = mfi14.iloc[-1] - mfi14.iloc[-(ROC_LOOKBACK + 1)] if len(mfi14) > ROC_LOOKBACK else np.nan
            adx_change_10d = adx14.iloc[-1] - adx14.iloc[-(ROC_LOOKBACK + 1)] if len(adx14) > ROC_LOOKBACK else np.nan

            row = {
                "Symbol": sym,
                "Price": round(last_close, 2),
                "RSI": round(rsi14.iloc[-1], 1) if not pd.isna(rsi14.iloc[-1]) else np.nan,
                "ADX": round(adx14.iloc[-1], 1) if not pd.isna(adx14.iloc[-1]) else np.nan,
                "ADX_Change_10d": round(adx_change_10d, 1) if not pd.isna(adx_change_10d) else np.nan,
                "MFI": round(mfi14.iloc[-1], 1) if not pd.isna(mfi14.iloc[-1]) else np.nan,
                "MFI_Change_10d": round(mfi_change_10d, 1) if not pd.isna(mfi_change_10d) else np.nan,
                "RVOL": round(rvol, 2) if not pd.isna(rvol) else np.nan,
                "DailyRet": round(daily_ret, 2) if not pd.isna(daily_ret) else np.nan,
                "WeeklyRet": round(weekly_ret, 2) if not pd.isna(weekly_ret) else np.nan,
                "MonthlyRet": round(monthly_ret, 2) if not pd.isna(monthly_ret) else np.nan,
                "Return3M": round(ret_3m, 2) if not pd.isna(ret_3m) else np.nan,
            }
            rows.append(row)
        time.sleep(0.5)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------
# 4. Momentum filter: strong MFI/ADX improvement, NOT overbought, price cap
# --------------------------------------------------------------------------
def passes_momentum_rules(row):
    if pd.isna(row["Price"]) or row["Price"] >= PRICE_CAP:
        return False
    if pd.isna(row["RSI"]) or not (RSI_MIN <= row["RSI"] <= RSI_MAX):
        return False
    if pd.isna(row["MFI"]) or not (MFI_MIN <= row["MFI"] <= MFI_MAX):
        return False
    if pd.isna(row["ADX"]) or row["ADX"] < ADX_MIN:
        return False
    if pd.isna(row["ADX_Change_10d"]) or row["ADX_Change_10d"] < ADX_MIN_ACCEL:
        return False
    if pd.isna(row["MFI_Change_10d"]) or row["MFI_Change_10d"] < MFI_MIN_ACCEL:
        return False
    if pd.isna(row["RVOL"]) or row["RVOL"] < RVOL_MIN:
        return False
    return True

def momentum_score(row):
    return round(
        row["ADX_Change_10d"] * 0.30 +
        row["MFI_Change_10d"] * 0.30 +
        (row["ADX"] - ADX_MIN) * 0.15 +
        min(row["RVOL"], 5) * 4 * 0.15 +
        (row["RSI"] - RSI_MIN) * 0.10,
        2
    )

def explain(row):
    return (
        f"RSI {row['RSI']} and MFI {row['MFI']} — strong momentum, both safely below "
        f"overbought levels (RSI<{RSI_MAX}, MFI<{MFI_MAX}); ADX {row['ADX']} confirms a real "
        f"trend, up {row['ADX_Change_10d']} pts in {ROC_LOOKBACK} sessions; MFI up "
        f"{row['MFI_Change_10d']} pts in the same window (money flowing in fast); volume "
        f"{row['RVOL']}x average (genuine participation); price Rs {row['Price']} (under Rs {PRICE_CAP})."
    )

# --------------------------------------------------------------------------
# 5. Sector performance aggregation
# --------------------------------------------------------------------------
def build_sector_performance(data_with_sector):
    grp = data_with_sector.groupby("Sector")[["DailyRet", "WeeklyRet", "MonthlyRet", "Return3M"]].mean()
    grp["StockCount"] = data_with_sector.groupby("Sector")["Symbol"].count()
    grp = grp.dropna(subset=["MonthlyRet"])
    grp["CompositeScore"] = (
        grp["Return3M"] * SECTOR_WEIGHTS["Return3M"] +
        grp["MonthlyRet"] * SECTOR_WEIGHTS["MonthlyRet"] +
        grp["WeeklyRet"] * SECTOR_WEIGHTS["WeeklyRet"] +
        grp["DailyRet"] * SECTOR_WEIGHTS["DailyRet"]
    )
    grp = grp.round(2).sort_values("CompositeScore", ascending=False)
    return grp.reset_index()

# --------------------------------------------------------------------------
# 6. UI
# --------------------------------------------------------------------------
st.title("NSE 500 Momentum Screener")
st.caption(
    f"Top 10 stocks under Rs {PRICE_CAP} with strong momentum: MFI and ADX both "
    f"improving fast, but NOT overbought (RSI {RSI_MIN}-{RSI_MAX}, MFI {MFI_MIN}-{MFI_MAX}), "
    "with real volume behind the move."
)

if HAS_AUTOREFRESH:
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")

with st.sidebar:
    st.header("Controls")
    refresh_btn = st.button("Refresh data now")

if refresh_btn:
    st.cache_data.clear()

universe_map = get_nse500_universe()
symbols = list(universe_map.keys())
st.write(f"Universe size: **{len(symbols)}** symbols")

data = build_dataset(symbols)

if data.empty:
    st.error("Could not fetch price data (network/API issue). Try 'Refresh data now'.")
else:
    data["Company Name"] = data["Symbol"].map(lambda s: universe_map.get(s, (s, "Unclassified"))[0])
    data["Sector"] = data["Symbol"].map(lambda s: universe_map.get(s, (s, "Unclassified"))[1])

    # ---------------- Top 10 Momentum Stocks ----------------
    st.subheader(f"Top 10 Momentum Stocks (Price < Rs {PRICE_CAP}, Not Overbought)")
    candidates = data[data.apply(passes_momentum_rules, axis=1)].copy()
    assert (candidates["Price"] < PRICE_CAP).all(), "Price cap violated -- should not happen"

    if candidates.empty:
        st.info("No stocks currently meet these criteria. Try the next hourly refresh.")
    else:
        candidates["MomentumScore"] = candidates.apply(momentum_score, axis=1)
        candidates["Why"] = candidates.apply(explain, axis=1)
        top10 = candidates.sort_values("MomentumScore", ascending=False).head(10)

        display_cols = ["Symbol", "Company Name", "Sector", "Price", "RSI", "MFI", "MFI_Change_10d",
                         "ADX", "ADX_Change_10d", "RVOL", "MomentumScore", "Why"]
        st.dataframe(top10[display_cols], use_container_width=True, hide_index=True)

        st.download_button("Download as CSV",
                            top10[display_cols].to_csv(index=False).encode(),
                            file_name="nse500_top10_momentum.csv")

    st.markdown("---")

    # ---------------- Sector Heatmap ----------------
    st.subheader("Sector Performance Heatmap")
    sector_perf = build_sector_performance(data)

    if not sector_perf.empty:
        heat_df = sector_perf.set_index("Sector")[["DailyRet", "WeeklyRet", "MonthlyRet", "Return3M"]]
        heat_df.columns = ["Daily %", "1-Week %", "1-Month %", "3-Month %"]
        fig = px.imshow(
            heat_df,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(color="Return %"),
        )
        fig.update_layout(height=max(400, 28 * len(heat_df)), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        top3 = sector_perf.head(3)
        cols = st.columns(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                st.metric(
                    label=f"#{i+1} {row['Sector']}",
                    value=f"{row['CompositeScore']}",
                    delta=f"3M: {row['Return3M']}% | 1M: {row['MonthlyRet']}%",
                )
        st.caption(
            "Composite score = 40% x 3-month return + 30% x 1-month + 20% x 1-week + "
            "10% x daily, averaged across all NSE 500 stocks in each sector."
        )
    else:
        st.info("Sector data unavailable this refresh.")

    st.caption(f"Last data refresh: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — auto-refreshes hourly.")

# --------------------------------------------------------------------------
# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   requests
#   streamlit-autorefresh
#   plotly
# --------------------------------------------------------------------------
