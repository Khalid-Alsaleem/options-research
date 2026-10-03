"""
Download daily price history for the engine-verification sample.
Sample = 50 liquid large caps + SPY + VIX, from 2015 (warm-up for long indicators).
Note: this sample is for building/verifying the engine only (protocol section 2),
not for discovering indicators, so the fixed ticker list is acceptable here.
"""
import time
from pathlib import Path

import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "BAC", "XOM",
    "CVX", "JNJ", "PFE", "MRK", "ABBV", "UNH", "HD", "WMT", "COST", "PG",
    "KO", "PEP", "DIS", "NFLX", "INTC", "AMD", "CSCO", "ORCL", "CRM", "ADBE",
    "QCOM", "TXN", "AVGO", "MU", "IBM", "GS", "MS", "C", "WFC", "V",
    "MA", "PYPL", "BA", "CAT", "GE", "F", "GM", "T", "VZ", "CMCSA",
]
EXTRA = ["SPY", "^VIX"]
START = "2015-01-01"

out = Path("data/prices")
out.mkdir(parents=True, exist_ok=True)

failed = []
for t in TICKERS + EXTRA:
    for attempt in range(3):
        try:
            df = yf.Ticker(t).history(start=START, auto_adjust=False)
            if df.empty:
                raise ValueError("empty result")
            df.index = df.index.tz_localize(None).date
            df.index.name = "Date"
            df.to_csv(out / f"{t.replace('^', '')}.csv")
            print(f"{t}: {len(df)} rows")
            break
        except Exception as e:
            print(f"{t}: attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    else:
        failed.append(t)
    time.sleep(1)

print("FAILED:", failed)
if len(failed) > 5:
    raise SystemExit("Too many failures - stopping")
