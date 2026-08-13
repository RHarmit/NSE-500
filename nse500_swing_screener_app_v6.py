
"""
NSE 500 Swing Trading Screener — Web App (Streamlit)
=====================================================
v6 change: adds a "Top 10 Recovery Leaders" section — stocks that fell
significantly from a recent high (worst drop), have since bounced back
meaningfully, and where MFI and ADX are both rising RAPIDLY alongside
strong (above-average) volume.
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
ROC_LOOKBACK = 10          # window for "rapid" MFI/ADX change

SECTOR_WEIGHTS = {"Return3M": 0.40, "MonthlyRet": 0.30, "WeeklyRet": 0.20, "DailyRet": 0.10}

# Recovery-leaders minimum bar (kept loose on purpose so we always get a top 10
# when candidates exist; ranking does the heavy lifting)
RECOVERY_MIN_DRAWDOWN = 12     # stock must have fallen at least 12% from its high
RECOVERY_MIN_BOUNCE = 3        # must have bounced at least 3% off the low
RECOVERY_MIN_RVOL = 1.2

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
# 1. Universe: symbols + full company names + sector/industry
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
# 5. Build dataset (now includes drawdown + MFI/ADX rate-of-change)
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
            ret_3m = close.pct_change(63).iloc[-1] * 100 if len(df) > 63 else np.nan

            window = df.tail(LOW_LOOKBACK)
            low_val = window["Low"].min()
            low_idx_pos = window["Low"].values.argmin()
            days_since_low = len(window) - 1 - low_idx_pos
            pct_above_low = (last_close - low_val) / low_val * 100 if low_val > 0 else np.nan

            # Worst drop: highest High in the window BEFORE the low vs the low itself
            pre_low_window = window.iloc[:low_idx_pos + 1] if low_idx_pos > 0 else window.iloc[:1]
            peak_before_low = pre_low_window["High"].max()
            max_drawdown_pct = (peak_before_low - low_val) / peak_before_low * 100 if peak_before_low > 0 else np.nan

            rvol = vol.iloc[-1] / avgvol20.iloc[-1] if avgvol20.iloc[-1] and avgvol20.iloc[-1] > 0 else np.nan

            mfi_today = mfi14.iloc[-1]
            mfi_prior_avg = mfi14.iloc[-(MFI_TREND_LOOKBACK + 1):-1].mean() if len(mfi14) > MFI_TREND_LOOKBACK + 1 else np.nan
            mfi_rising = bool(mfi_today > mfi_prior_avg) if not pd.isna(mfi_prior_avg) else False

            # Rate-of-change (rapid increase) for MFI and ADX over the last ROC_LOOKBACK sessions
            mfi_change_10d = mfi14.iloc[-1] - mfi14.iloc[-(ROC_LOOKBACK + 1)] if len(mfi14) > ROC_LOOKBACK else np.nan
            adx_change_10d = adx14.iloc[-1] - adx14.iloc[-(ROC_LOOKBACK + 1)] if len(adx14) > ROC_LOOKBACK else np.nan

            row = {
                "Symbol": sym,
                "Price": round(last_close, 2),
                "RSI": round(rsi14.iloc[-1], 1) if not pd.isna(rsi14.iloc[-1]) else np.nan,
                "MACD_bullish": bool(macd_line.iloc[-1] > signal_line.iloc[-1]),
                "ADX": round(adx14.iloc[-1], 1) if not pd.isna(adx14.iloc[-1]) else np.nan,
                "ADX_Change_10d": round(adx_change_10d, 1) if not pd.isna(adx_change_10d) else np.nan,
                "MFI": round(mfi_today, 1) if not pd.isna(mfi_today) else np.nan,
                "MFI_2wAvg": round(mfi_prior_avg, 1) if not pd.isna(mfi_prior_avg) else np.nan,
                "MFI_Change_10d": round(mfi_change_10d, 1) if not pd.isna(mfi_change_10d) else np.nan,
                "MFI_Rising": mfi_rising,
                "RVOL": round(rvol, 2) if not pd.isna(rvol) else np.nan,
                "DailyRet": round(daily_ret, 2) if not pd.isna(daily_ret) else np.nan,
                "WeeklyRet": round(weekly_ret, 2) if not pd.isna(weekly_ret) else np.nan,
                "MonthlyRet": round(monthly_ret, 2) if not pd.isna(monthly_ret) else np.nan,
                "Return3M": round(ret_3m, 2) if not pd.isna(ret_3m) else np.nan,
                "DaysSinceLow": int(days_since_low),
                "PctAboveLow": round(pct_above_low, 1) if not pd.isna(pct_above_low) else np.nan,
                "MaxDrawdownPct": round(max_drawdown_pct, 1) if not pd.isna(max_drawdown_pct) else np.nan,
                "AvgVol20": int(avgvol20.iloc[-1]) if not pd.isna(avgvol20.iloc[-1]) else 0,
            }
            rows.append(row)
        time.sleep(0.5)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------
# 6. Filter logic + rationale text (main strict shortlist)
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
# 6b. Sector performance aggregation
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
# 6c. Top 10 Recovery Leaders: big drop -> strong bounce -> rapid MFI/ADX rise -> good volume
# --------------------------------------------------------------------------
def build_recovery_leaders(data_with_meta, top_n=10):
    d = data_with_meta.copy()
    mask = (
        (d["MaxDrawdownPct"] >= RECOVERY_MIN_DRAWDOWN) &
        (d["PctAboveLow"] >= RECOVERY_MIN_BOUNCE) &
        (d["MFI_Change_10d"] > 0) &
        (d["ADX_Change_10d"] > 0) &
        (d["RVOL"] >= RECOVERY_MIN_RVOL)
    )
    cand = d[mask].copy()
    if cand.empty:
        return cand

    # Composite recovery score: rewards deep prior drop + strong bounce +
    # fast MFI/ADX acceleration + volume conviction.
    cand["RecoveryScore"] = (
        cand["MaxDrawdownPct"] * 0.20 +
        cand["PctAboveLow"] * 0.30 +
        cand["MFI_Change_10d"] * 0.20 +
        cand["ADX_Change_10d"] * 0.20 +
        cand["RVOL"].clip(upper=5) * 4 * 0.10
    )
    cand["Why"] = cand.apply(
        lambda r: (
            f"Fell {r['MaxDrawdownPct']}% from its high, bounced {r['PctAboveLow']}% off the low "
            f"({r['DaysSinceLow']} sessions ago); MFI up {r['MFI_Change_10d']} pts and ADX up "
            f"{r['ADX_Change_10d']} pts in {ROC_LOOKBACK} sessions (accelerating momentum/trend); "
            f"volume {r['RVOL']}x average (real conviction, not a fluke)."
        ),
        axis=1,
    )
    return cand.sort_values("RecoveryScore", ascending=False).head(top_n)

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
    max_results = st.slider("Max stocks to show (main shortlist)", 5, 100, 30)
    refresh_btn = st.button("Refresh data now")
    st.markdown("---")
    st.caption("Screening thresholds are fixed in code. Company names & sectors sourced "
               "from the NSE 500 index CSV. Analyst ratings via yfinance consensus. "
               "FII per-stock change is best-effort — use the screener.in link if blank.")

if refresh_btn:
    st.cache_data.clear()

universe_map = get_nse500_universe()
symbols = list(universe_map.keys())
st.write(f"Universe size: **{len(symbols)}** symbols")

fii_dii_df = get_fii_dii_flow()
bulk_buyers = get_recent_bulk_deal_buyers()

data = build_screener_dataset(symbols)

if data.empty:
    st.error("Could not fetch price data (network/API issue). Try 'Refresh data now'.")
else:
    data["Company Name"] = data["Symbol"].map(lambda s: universe_map.get(s, (s, "Unclassified"))[0])
    data["Sector"] = data["Symbol"].map(lambda s: universe_map.get(s, (s, "Unclassified"))[1])

    # ---------------- Top 10 Recovery Leaders ----------------
    st.subheader("Top 10 Recovery Leaders (Biggest Drop -> Strongest Bounce)")
    st.caption(
        "Stocks that fell sharply from a recent high, have already bounced back "
        "meaningfully, AND where MFI and ADX are both rising rapidly with strong volume "
        "— i.e. money and trend strength are accelerating INTO the recovery, not just "
        "a random bounce."
    )
    recovery = build_recovery_leaders(data, top_n=10)
    if recovery.empty:
        st.info("No stocks currently meet the recovery-leader criteria. Try the next hourly refresh.")
    else:
        rec_cols = ["Symbol", "Company Name", "Sector", "Price", "MaxDrawdownPct", "PctAboveLow",
                    "DaysSinceLow", "MFI", "MFI_Change_10d", "ADX", "ADX_Change_10d", "RVOL",
                    "RecoveryScore", "Why"]
        st.dataframe(recovery[rec_cols], use_container_width=True, hide_index=True)
        st.download_button("Download recovery leaders as CSV",
                            recovery[rec_cols].to_csv(index=False).encode(),
                            file_name="nse500_recovery_leaders.csv")

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

    st.markdown("---")

    # ---------------- Main Strict Shortlist ----------------
    filtered = data[data.apply(passes_all_rules, axis=1)].copy()
    if filtered.empty:
        st.warning("No stocks currently satisfy all main shortlist conditions. Try the next hourly refresh.")
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

        st.subheader(f"Main Shortlist ({len(result)})")
        display_cols = ["Symbol", "Company Name", "Sector", "Price", "RSI", "ADX", "MFI", "MFI_2wAvg", "RVOL",
                         "DailyRet", "WeeklyRet", "MonthlyRet", "Return3M", "PctAboveLow",
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
        bulk_names = [f"{s} ({universe_map.get(s, (s, ''))[0]})" for s in sorted(bulk_buyers)]
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
#   plotly
# --------------------------------------------------------------------------
