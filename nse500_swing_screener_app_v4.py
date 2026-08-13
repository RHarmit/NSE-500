
"""
NSE 500 Swing Trading Screener — Web App (Streamlit)
=====================================================
v4 change: adds full company names (not just ticker symbols) throughout
the app, sourced from the same NSE 500 index CSV used for the universe
list (no extra API calls needed).
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

PRICE_CAP = 1500
RSI_MIN = 55
ADX_MIN = 20
RVOL_MIN = 1.2
MFI_MIN, MFI_MAX = 45, 90
LOW_LOOKBACK = 180
LOW_RECENCY_MAX_DAYS = 150
PCT_ABOVE_LOW_MIN = 5
PCT_ABOVE_LOW_MAX = 80
MFI_TREND_LOOKBACK = 10

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Fallback universe WITH full company names (used only if the live NSE CSV
# fetch fails, e.g. NSE blocking the request).
FALLBACK_UNIVERSE = {
    "RELIANCE": "Reliance Industries Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "INFY": "Infosys Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "HINDUNILVR": "Hindustan Unilever Ltd.",
    "ITC": "ITC Ltd.",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel Ltd.",
    "BAJFINANCE": "Bajaj Finance Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
    "LT": "Larsen & Toubro Ltd.",
    "AXISBANK": "Axis Bank Ltd.",
    "ASIANPAINT": "Asian Paints Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.",
    "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
    "TITAN": "Titan Company Ltd.",
    "ULTRACEMCO": "UltraTech Cement Ltd.",
    "WIPRO": "Wipro Ltd.",
    "ADANIENT": "Adani Enterprises Ltd.",
    "ADANIPORTS": "Adani Ports and Special Economic Zone Ltd.",
    "ONGC": "Oil and Natural Gas Corporation Ltd.",
    "NTPC": "NTPC Ltd.",
    "POWERGRID": "Power Grid Corporation of India Ltd.",
    "TATAMOTORS": "Tata Motors Ltd.",
    "TATASTEEL": "Tata Steel Ltd.",
    "JSWSTEEL": "JSW Steel Ltd.",
    "HCLTECH": "HCL Technologies Ltd.",
    "M&M": "Mahindra & Mahindra Ltd.",
    "BAJAJFINSV": "Bajaj Finserv Ltd.",
    "DIVISLAB": "Divi's Laboratories Ltd.",
    "GRASIM": "Grasim Industries Ltd.",
    "DRREDDY": "Dr. Reddy's Laboratories Ltd.",
    "CIPLA": "Cipla Ltd.",
    "EICHERMOT": "Eicher Motors Ltd.",
    "HEROMOTOCO": "Hero MotoCorp Ltd.",
    "TECHM": "Tech Mahindra Ltd.",
    "INDUSINDBK": "IndusInd Bank Ltd.",
    "COALINDIA": "Coal India Ltd.",
    "BPCL": "Bharat Petroleum Corporation Ltd.",
    "IOC": "Indian Oil Corporation Ltd.",
    "SBILIFE": "SBI Life Insurance Company Ltd.",
    "HDFCLIFE": "HDFC Life Insurance Company Ltd.",
    "NESTLEIND": "Nestle India Ltd.",
    "BRITANNIA": "Britannia Industries Ltd.",
    "DABUR": "Dabur India Ltd.",
    "GODREJCP": "Godrej Consumer Products Ltd.",
    "PIDILITIND": "Pidilite Industries Ltd.",
    "HAVELLS": "Havells India Ltd.",
    "DLF": "DLF Ltd.",
    "SIEMENS": "Siemens Ltd.",
    "ABB": "ABB India Ltd.",
    "PFC": "Power Finance Corporation Ltd.",
    "RECLTD": "REC Ltd.",
    "IRCTC": "Indian Railway Catering and Tourism Corporation Ltd.",
    "ZOMATO": "Eternal Ltd. (Zomato)",
    "TRENT": "Trent Ltd.",
    "PAYTM": "One97 Communications Ltd. (Paytm)",
    "POLYCAB": "Polycab India Ltd.",
}

# --------------------------------------------------------------------------
# 1. Universe: symbols + full company names
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_nse500_universe():
    """Returns a dict {symbol: full_company_name}."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r = requests.get(url, headers=NSE_HEADERS, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        name_col = "Company Name" if "Company Name" in df.columns else df.columns[0]
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        df = df.dropna(subset=[sym_col, name_col])
        mapping = dict(zip(df[sym_col].str.strip(), df[name_col].str.strip()))
        if len(mapping) > 50:
            return mapping
    except Exception:
        pass
    return FALLBACK_UNIVERSE

# --------------------------------------------------------------------------
# 2. Market-wide FII/DII flow (aggregate only)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_fii_dii_flow():
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        r = s.get(url, headers=NSE_HEADERS, timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()

# --------------------------------------------------------------------------
# 3. Bulk / block deals -> institutional buying proxy
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_bulk_deal_buyers(lookback_days=5):
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
# 3b. Best-effort per-stock FII holding change (quarterly data only)
# --------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_fii_holding_change(symbol):
    link = f"https://www.screener.in/company/{symbol}/"
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=8)
        url = f"https://www.nseindia.com/api/corp-info?symbol={symbol}&corpType=shp&market=equities"
        r = s.get(url, headers=NSE_HEADERS, timeout=8)
        if r.status_code == 200:
            return None, link
    except Exception:
        pass
    return None, link

# --------------------------------------------------------------------------
# 3c. Analyst consensus (Buy Score out of 5) via yfinance
# --------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_analyst_rating(symbol):
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        rec_mean = info.get("recommendationMean")
        n_analysts = info.get("numberOfAnalystOpinions")
        target_mean = info.get("targetMeanPrice")
        if rec_mean is None:
            return np.nan, None, None
        buy_score = round(6 - rec_mean, 1)
        return buy_score, n_analysts, target_mean
    except Exception:
        return np.nan, None, None

# --------------------------------------------------------------------------
# 4. Indicators
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
# 5. Build dataset
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
            last_close = close.iloc[-1]

            rsi14 = rsi(close, 14)
            macd_line, signal_line = macd(close)
            adx14 = adx(df, 14)
            mfi14 = mfi(df, 14)
            avgvol20 = vol.rolling(20).mean()

            daily_ret = close.pct_change(1).iloc[-1] * 100
            weekly_ret = close.pct_change(5).iloc[-1] * 100
            monthly_ret = close.pct_change(21).iloc[-1] * 100 if len(df) > 21 else np.nan

            window = df.tail(LOW_LOOKBACK)
            low_val = window["Low"].min()
            low_idx_pos = window["Low"].values.argmin()
            days_since_low = len(window) - 1 - low_idx_pos
            pct_above_low = (last_close - low_val) / low_val * 100 if low_val > 0 else np.nan

            rvol = vol.iloc[-1] / avgvol20.iloc[-1] if avgvol20.iloc[-1] and avgvol20.iloc[-1] > 0 else np.nan

            mfi_today = mfi14.iloc[-1]
            mfi_prior_avg = mfi14.iloc[-(MFI_TREND_LOOKBACK + 1):-1].mean() if len(mfi14) > MFI_TREND_LOOKBACK + 1 else np.nan
            mfi_rising = bool(mfi_today > mfi_prior_avg) if not pd.isna(mfi_prior_avg) else False

            row = {
                "Symbol": sym,
                "Price": round(last_close, 2),
                "RSI": round(rsi14.iloc[-1], 1) if not pd.isna(rsi14.iloc[-1]) else np.nan,
                "MACD_bullish": bool(macd_line.iloc[-1] > signal_line.iloc[-1]),
                "ADX": round(adx14.iloc[-1], 1) if not pd.isna(adx14.iloc[-1]) else np.nan,
                "MFI": round(mfi_today, 1) if not pd.isna(mfi_today) else np.nan,
                "MFI_2wAvg": round(mfi_prior_avg, 1) if not pd.isna(mfi_prior_avg) else np.nan,
                "MFI_Rising": mfi_rising,
                "RVOL": round(rvol, 2) if not pd.isna(rvol) else np.nan,
                "DailyRet": round(daily_ret, 2) if not pd.isna(daily_ret) else np.nan,
                "WeeklyRet": round(weekly_ret, 2) if not pd.isna(weekly_ret) else np.nan,
                "MonthlyRet": round(monthly_ret, 2) if not pd.isna(monthly_ret) else np.nan,
                "DaysSinceLow": int(days_since_low),
                "PctAboveLow": round(pct_above_low, 1) if not pd.isna(pct_above_low) else np.nan,
                "AvgVol20": int(avgvol20.iloc[-1]) if not pd.isna(avgvol20.iloc[-1]) else 0,
            }
            rows.append(row)
        time.sleep(0.5)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------
# 6. Filter logic + rationale text
# --------------------------------------------------------------------------
def passes_all_rules(row):
    if pd.isna(row["Price"]) or row["Price"] >= PRICE_CAP:
        return False
    if pd.isna(row["RSI"]) or row["RSI"] < RSI_MIN:
        return False
    if not row["MACD_bullish"]:
        return False
    if pd.isna(row["ADX"]) or row["ADX"] < ADX_MIN:
        return False
    if pd.isna(row["RVOL"]) or row["RVOL"] < RVOL_MIN:
        return False
    if pd.isna(row["MFI"]) or not (MFI_MIN <= row["MFI"] <= MFI_MAX):
        return False
    if not row["MFI_Rising"]:
        return False
    if row["DaysSinceLow"] > LOW_RECENCY_MAX_DAYS:
        return False
    if pd.isna(row["PctAboveLow"]) or not (PCT_ABOVE_LOW_MIN <= row["PctAboveLow"] <= PCT_ABOVE_LOW_MAX):
        return False
    if pd.isna(row["MonthlyRet"]) or pd.isna(row["WeeklyRet"]) or pd.isna(row["DailyRet"]):
        return False
    if not (row["MonthlyRet"] > row["WeeklyRet"] > row["DailyRet"] > 0):
        return False
    return True

def explain(row, bulk_buyers, buy_score, n_analysts, target_mean, fii_link):
    reasons = [
        f"RSI {row['RSI']} (strong momentum)",
        "MACD bullish crossover",
        f"ADX {row['ADX']} (trending)",
        f"Volume {row['RVOL']}x avg",
        f"MFI {row['MFI']} rising vs 2wk avg {row['MFI_2wAvg']}",
        f"Bounced {row['PctAboveLow']}% off a low made {row['DaysSinceLow']} sessions ago",
        f"Rising on all timeframes: {row['MonthlyRet']}% (mo) > {row['WeeklyRet']}% (wk) > {row['DailyRet']}% (day), all positive",
    ]
    if row["Symbol"] in bulk_buyers:
        reasons.append("Recent NSE bulk-deal BUY (institutional accumulation)")
    if not pd.isna(buy_score):
        reasons.append(f"Analyst Buy Score {buy_score}/5 ({n_analysts or 0} analysts, target Rs {target_mean})")
    else:
        reasons.append("No analyst coverage found")
    reasons.append(f"Check FII holding trend: {fii_link}")
    return "; ".join(reasons)

def score(row):
    s = 0
    s += (row["RSI"] - RSI_MIN) / 10
    s += (row["ADX"] - ADX_MIN) / 10
    s += min(row["RVOL"], 4)
    s += (row["MonthlyRet"] - row["WeeklyRet"]) / 5
    s += (row["MFI"] - row["MFI_2wAvg"]) / 10
    return round(s, 2)

# --------------------------------------------------------------------------
# 7. UI
# --------------------------------------------------------------------------
st.title("NSE 500 Swing Trading Screener")
st.caption(
    f"Rules: price < Rs {PRICE_CAP}, momentum + trend confirmed, MFI rising vs its "
    "2-week average, recovering off a recent low, and genuinely RISING on all three "
    "timeframes (monthly > weekly > daily, all positive)."
)

if HAS_AUTOREFRESH:
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")

with st.sidebar:
    st.header("Controls")
    max_results = st.slider("Max stocks to show", 5, 100, 30)
    refresh_btn = st.button("Refresh data now")
    st.markdown("---")
    st.caption("Screening thresholds are fixed in code. Company names sourced from the "
               "NSE 500 index CSV. Analyst ratings via yfinance consensus. FII per-stock "
               "change is best-effort — use the screener.in link if blank.")

if refresh_btn:
    st.cache_data.clear()

universe_map = get_nse500_universe()      # {symbol: company name}
symbols = list(universe_map.keys())
st.write(f"Universe size: **{len(symbols)}** symbols")

fii_dii_df = get_fii_dii_flow()
bulk_buyers = get_recent_bulk_deal_buyers()

data = build_screener_dataset(symbols)

if data.empty:
    st.error("Could not fetch price data (network/API issue). Try 'Refresh data now'.")
else:
    data["Company Name"] = data["Symbol"].map(universe_map).fillna(data["Symbol"])

    filtered = data[data.apply(passes_all_rules, axis=1)].copy()
    if filtered.empty:
        st.warning("No stocks currently satisfy all conditions. Try the next hourly refresh.")
    else:
        filtered["Score"] = filtered.apply(score, axis=1)
        result = filtered.sort_values("Score", ascending=False).head(max_results)

        buy_scores, n_analysts_list, targets, fii_links, whys = [], [], [], [], []
        for _, r in result.iterrows():
            bscore, nan_, tgt = get_analyst_rating(r["Symbol"])
            _, link = get_fii_holding_change(r["Symbol"])
            buy_scores.append(bscore)
            n_analysts_list.append(nan_)
            targets.append(tgt)
            fii_links.append(link)
            whys.append(explain(r, bulk_buyers, bscore, nan_, tgt, link))

        result["BuyScore_5"] = buy_scores
        result["NumAnalysts"] = n_analysts_list
        result["AnalystTarget"] = targets
        result["FII_Check_Link"] = fii_links
        result["Why"] = whys

        st.subheader(f"Shortlisted stocks ({len(result)})")
        display_cols = ["Symbol", "Company Name", "Price", "RSI", "ADX", "MFI", "MFI_2wAvg", "RVOL",
                         "DailyRet", "WeeklyRet", "MonthlyRet", "PctAboveLow",
                         "BuyScore_5", "NumAnalysts", "AnalystTarget", "Score", "Why"]
        st.dataframe(result[display_cols], use_container_width=True, hide_index=True)

        st.download_button("Download shortlist as CSV",
                            result[display_cols].to_csv(index=False).encode(),
                            file_name="nse500_swing_shortlist.csv")

    st.subheader("Market-wide FII/DII flow (context, not per-stock)")
    if not fii_dii_df.empty:
        st.dataframe(fii_dii_df, use_container_width=True, hide_index=True)
    else:
        st.info("FII/DII live feed unreachable right now. Check "
                "https://www.nseindia.com/reports/fii-dii manually if needed.")

    if bulk_buyers:
        bulk_names = [f"{s} ({universe_map.get(s, s)})" for s in sorted(bulk_buyers)]
        st.caption(f"Stocks with recent NSE bulk-deal buying (last 5 sessions): {', '.join(bulk_names)}")

    st.caption(f"Last data refresh: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — auto-refreshes hourly.")

# --------------------------------------------------------------------------
# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   requests
#   streamlit-autorefresh
# --------------------------------------------------------------------------
