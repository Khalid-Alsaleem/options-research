"""Pull 1-hour bars for a fixed list of liquid stocks (SMC multi-timeframe test).
Source 1: EODHD intraday API (needs an intraday-enabled plan). Source 2 (fallback): Yahoo via yfinance (last ~730 days only).
Writes data/intraday/<T>_1h.csv.gz and data/reports/intraday_coverage.csv. Secret: EODHD_API_TOKEN (never printed)."""
import os, time, datetime as dt
from pathlib import Path
import pandas as pd, requests
TOK = os.environ.get("EODHD_API_TOKEN", "")
TICK = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "ORCL", "CRM", "INTC", "JPM", "BAC", "GS",
        "V", "MA", "XOM", "CVX", "UNH", "LLY", "JNJ", "PFE", "WMT", "COST", "HD", "KO", "DIS", "BA", "CAT", "SPY"]
out = Path("data/intraday"); out.mkdir(parents=True, exist_ok=True); cov = []
def eodhd(t):
    frames = []
    for y in range(2016, dt.date.today().year + 1):
        a = int(dt.datetime(y, 1, 1).timestamp()); b = int(dt.datetime(y, 12, 31, 23).timestamp())
        r = requests.get(f"https://eodhd.com/api/intraday/{t}.US", params={"api_token": TOK, "interval": "1h", "from": a, "to": b, "fmt": "json"}, timeout=120)
        if r.status_code != 200: return None, f"http {r.status_code}"
        j = r.json()
        if isinstance(j, list) and j: frames.append(pd.DataFrame(j))
        time.sleep(0.3)
    if not frames: return None, "empty"
    d = pd.concat(frames); d["dt"] = pd.to_datetime(d["datetime"], utc=True)
    return d[["dt", "open", "high", "low", "close", "volume"]].drop_duplicates("dt").sort_values("dt"), "eodhd"
def yahoo(t):
    import yfinance as yf
    d = yf.Ticker(t).history(period="730d", interval="1h", auto_adjust=False)
    if d.empty: return None, "yahoo empty"
    d = d.reset_index().rename(columns={"Datetime": "dt", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    d["dt"] = pd.to_datetime(d["dt"], utc=True); return d[["dt", "open", "high", "low", "close", "volume"]], "yahoo"
for t in TICK:
    d, src = (eodhd(t) if TOK else (None, "no token"))
    note = src
    if d is None or len(d) < 1000:
        d, src = yahoo(t); note = f"eodhd failed ({note}); {src}"
    if d is not None:
        d.to_csv(out / f"{t}_1h.csv.gz", index=False)
        cov.append(dict(ticker=t, source=note, rows=len(d), first=str(d.dt.iloc[0]), last=str(d.dt.iloc[-1])))
    else: cov.append(dict(ticker=t, source=note, rows=0))
    print(cov[-1], flush=True)
Path("data/reports").mkdir(parents=True, exist_ok=True); pd.DataFrame(cov).to_csv("data/reports/intraday_coverage.csv", index=False)
