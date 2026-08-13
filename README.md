# NSE 500 Momentum Screener

A Streamlit-based research tool that scans the **full NSE 500 universe**, filters out overbought names, ranks candidates using a multi-factor momentum score, and summarizes sector performance across multiple time horizons.

## Why this project exists

Most stock screeners rely on rigid pass/fail rules. That can eliminate potentially attractive setups simply because one indicator is only moderate.

This project uses a **soft composite scoring model** instead. Only a small number of hard rules remain, while trend strength, money flow, relative volume, momentum, and MACD are blended into a weighted score.

## Current screening logic

### Hard gates

- Price below ₹1,000
- RSI below 70
- MFI below 80

### Composite score

| Factor | Weight | Interpretation |
|---|---:|---|
| ADX | 25% | Strength of the prevailing trend |
| MFI | 20% | Price-and-volume based buying pressure |
| Relative Volume | 20% | Whether current volume is elevated versus its recent average |
| 10-Day ROC | 20% | Short-term price momentum |
| MACD Histogram | 15% | Direction and strength of momentum acceleration |

Each factor is normalized across the current candidate universe before the weighted score is calculated.

## Sector analysis

The application also calculates average sector returns for:

- 1 day
- 1 week
- 1 month
- 3 months

A separate sector composite emphasizes the longer timeframes while retaining short-term confirmation.

## Data flow

```text
NSE constituent list
        ↓
NSE 500 symbols
        ↓
yfinance OHLCV history
        ↓
RSI / ADX / MFI / RVOL / ROC / MACD
        ↓
Hard-gate validation
        ↓
Cross-sectional normalization
        ↓
Composite momentum ranking
        ↓
Stock table + sector heatmap + CSV export
```

## Features

- Scans the NSE 500 universe rather than a manually selected watchlist
- Pulls the current NSE 500 constituent list from NSE archives
- Uses cached market data with hourly refresh behavior
- Computes RSI, ADX, MFI, relative volume, ROC, and MACD
- Ranks candidates using a transparent weighted score
- Displays sector performance as a heatmap
- Highlights the strongest sectors
- Exports detailed screening results to CSV

## Tech stack

- Python
- Streamlit
- pandas
- NumPy
- yfinance
- Plotly
- Requests

## Run locally

```bash
git clone https://github.com/RHarmit/NSE-500.git
cd NSE-500
pip install -r requirements.txt
streamlit run nse500_momentum_composite_v11.py
```

## Important data note

The application attempts to pull the current constituent universe from NSE. The code also contains a fallback universe so that the interface can remain usable if the NSE constituent request temporarily fails.

Market-price history is retrieved through `yfinance`, so temporary upstream data gaps or symbol coverage issues can affect individual observations.

## Interpretation

The score is intended as a **research ranking**, not a guarantee of future return. A high score means a stock currently compares favorably with other eligible NSE 500 names across the selected momentum and volume factors.

For an actual trade decision, the output should still be combined with price structure, liquidity, upcoming events, position sizing, and risk management.

## Possible next improvements

- Walk-forward testing of factor weights
- Out-of-sample hit-rate and drawdown analysis
- ATR-based entry, target, and stop framework
- Earnings and corporate-event flags
- Stronger data-source validation
- Fundamental quality overlay
- Persistent historical rankings for signal-performance analysis
