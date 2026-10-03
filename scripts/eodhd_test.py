"""
Data coverage test before the full download.
Prices: EODHD (paid EOD plan). Earnings dates: two FREE sources (Yahoo via yfinance,
and Alpha Vantage). Sample = stocks that LEFT the S&P 500 + AAPL as a control.
Keys come from GitHub secrets and are never printed.
"""
import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

EOD_TOKEN = os.environ["EODHD_API_TOKEN"]
AV_KEY = os.environ["ALPHAVANTAGE_API_KEY"]
TEST = {
    "AAPL": "active (control)",
    "ATVI": "acquired 2023",
    "CELG": "acquired 2019",
    "TWTR": "taken private 2022",
    "XLNX": "acquired 2022",
    "CTXS": "taken private 2022",
    "FRC": "bank failure 2023",
    "SIVB": "bank failure 2023",
}

def span(dates):
    dates = sorted(d for d in dates if d and str(d) >= "2015-01-01")
    return len(dates), (dates[0] if dates else None), (dates[-1] if dates else None)

out = Path("data/eodhd_test")
out.mkdir(parents=True, exist_ok=True)
rows = []
for t, note in TEST.items():
    row = {"ticker": t, "note": note}
    # 1) prices from EODHD
    try:
        r = requests.get(f"https://eodhd.com/api/eod/{t}.US",
                         params={"api_token": EOD_TOKEN, "fmt": "json", "from": "2015-01-01"}, timeout=60)
        px = r.json() if r.status_code == 200 else []
        row.update(price_status=r.status_code, price_rows=len(px),
                   price_first=px[0]["date"] if px else None, price_last=px[-1]["date"] if px else None)
    except Exception as e:
        row.update(price_status=f"error: {type(e).__name__}")
    # 2) earnings dates from Yahoo (free)
    try:
        ed = yf.Ticker(t).get_earnings_dates(limit=60)
        dates = [] if ed is None else [str(d.date()) for d in ed.index]
        n, a, b = span(dates)
        row.update(yahoo_earn_count=n, yahoo_earn_first=a, yahoo_earn_last=b)
    except Exception as e:
        row.update(yahoo_earn_count=f"error: {type(e).__name__}")
    # 3) earnings dates from Alpha Vantage (free key)
    try:
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function": "EARNINGS", "symbol": t, "apikey": AV_KEY}, timeout=60)
        q = r.json().get("quarterlyEarnings", [])
        n, a, b = span([x.get("reportedDate") for x in q])
        row.update(av_earn_count=n, av_earn_first=a, av_earn_last=b)
    except Exception as e:
        row.update(av_earn_count=f"error: {type(e).__name__}")
    rows.append(row)
    print({k: v for k, v in row.items()})

pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
