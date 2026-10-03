"""
Full data pull for the research (protocol section 2).
1) Universe: point-in-time S&P 500 membership since 2016 (fja05680/sp500 on GitHub).
2) Prices: EODHD end-of-day (raw OHLCV + adjusted_close) and splits, incl. delisted names.
3) Earnings dates: SEC EDGAR 8-K filings with Item 2.02 (official, free, keeps delisted firms).
Every step writes a coverage report so gaps are visible, never silent.
Secrets: EODHD_API_TOKEN, SEC_USER_AGENT (never printed).
"""
import io
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

EOD = os.environ["EODHD_API_TOKEN"]
UA = {"User-Agent": os.environ["SEC_USER_AGENT"], "Accept-Encoding": "gzip, deflate"}
START = "2015-01-01"          # one year of warm-up before the 2016 learning period
UNIVERSE_FROM = "2016-01-01"
ROOT = Path("data")
for sub in ["universe", "eod", "splits", "earnings", "reports"]:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)


def sec_get(url):
    time.sleep(0.15)          # SEC fair-access limit: max 10 requests/second
    return requests.get(url, headers=UA, timeout=60)


# ---------- 1) Universe ----------
z = requests.get("https://codeload.github.com/fja05680/sp500/zip/refs/heads/master", timeout=120)
zf = zipfile.ZipFile(io.BytesIO(z.content))
name = [n for n in zf.namelist() if "Historical Components & Changes (Updated)" in n][0]
h = pd.read_csv(zf.open(name), parse_dates=["date"]).sort_values("date")
h = h[h.date >= pd.Timestamp(UNIVERSE_FROM) - pd.Timedelta(days=10)]

intervals, open_since, prev = [], {}, set()
for _, r in h.iterrows():
    cur = set(r.tickers.split(","))
    for t in cur - prev:
        open_since[t] = r.date
    for t in prev - cur:
        intervals.append((t, open_since.pop(t), r.date))
    prev = cur
for t, s in open_since.items():
    intervals.append((t, s, pd.NaT))
U = pd.DataFrame(intervals, columns=["ticker_raw", "start", "end"])
U["ticker"] = U.ticker_raw.str.replace(r"-\d{6}$", "", regex=True)
U["reused_ticker"] = U.ticker_raw.str.contains(r"-\d{6}$")
U.to_csv(ROOT / "universe/membership.csv", index=False)
tickers = sorted(U.ticker_raw.unique())
print("universe tickers:", len(tickers))

# ---------- 2) Prices + splits from EODHD ----------
cov = []
for raw in tickers:
    base = re.sub(r"-\d{6}$", "", raw)
    sym = base.replace(".", "-") + ".US"
    row = {"ticker_raw": raw, "eod_symbol": sym}
    try:
        r = requests.get(f"https://eodhd.com/api/eod/{sym}",
                         params={"api_token": EOD, "fmt": "json", "from": START}, timeout=60)
        px = pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
        row.update(status=r.status_code, rows=len(px))
        if len(px):
            px.to_csv(ROOT / f"eod/{raw}.csv.gz", index=False)
            row.update(first=px.date.iloc[0], last=px.date.iloc[-1])
        s = requests.get(f"https://eodhd.com/api/splits/{sym}",
                         params={"api_token": EOD, "fmt": "json", "from": START}, timeout=60)
        sp = pd.DataFrame(s.json()) if s.status_code == 200 else pd.DataFrame()
        if len(sp):
            sp.to_csv(ROOT / f"splits/{raw}.csv", index=False)
        row["splits"] = len(sp)
    except Exception as e:
        row["status"] = f"error: {type(e).__name__}"
    cov.append(row)
C = pd.DataFrame(cov).merge(U.groupby("ticker_raw").agg(member_start=("start", "min"),
                                                         member_end=("end", "max")).reset_index())
C.to_csv(ROOT / "reports/price_coverage.csv", index=False)
print("prices ok:", int((C.rows > 0).sum()), "/", len(C))

# ---------- 3) Earnings dates from SEC EDGAR ----------
cur = sec_get("https://www.sec.gov/files/company_tickers.json").json()
cik_map = {v["ticker"].upper(): int(v["cik_str"]) for v in cur.values()}
ciks = []
for raw in tickers:
    base = re.sub(r"-\d{6}$", "", raw)
    cik, how = None, None
    if raw != base:
        # ticker later reused by another company: a lookup by ticker would return the wrong firm
        ciks.append({"ticker_raw": raw, "cik": None, "how": "reused_ticker_manual"})
        continue
    if base.replace(".", "-") in cik_map:
        cik, how = cik_map[base.replace(".", "-")], "current_list"
    if cik is None:
        t = sec_get("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                    f"&CIK={base}&type=8-K&dateb=&owner=include&count=10&output=atom").text
        m = re.search(r"<cik>(\d+)</cik>", t)
        if m:
            cik, how = int(m.group(1)), "ticker_lookup"
    ciks.append({"ticker_raw": raw, "cik": cik, "how": how})
K = pd.DataFrame(ciks)
K.to_csv(ROOT / "universe/cik_map.csv", index=False)

earn, ecov = [], []
for _, k in K.dropna(subset=["cik"]).iterrows():
    cik = int(k.cik)
    try:
        j = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        frames = [pd.DataFrame(j["filings"]["recent"])]
        for f in j["filings"].get("files", []):
            frames.append(pd.DataFrame(sec_get("https://data.sec.gov/submissions/" + f["name"]).json()))
        df = pd.concat(frames, ignore_index=True)
        df = df[(df.form == "8-K") & df["items"].fillna("").str.contains("2.02") & (df.filingDate >= START)]
        for _, f in df.iterrows():
            earn.append({"ticker_raw": k.ticker_raw, "cik": cik, "filing_date": f.filingDate,
                         "accepted": f.get("acceptanceDateTime")})
        ecov.append({"ticker_raw": k.ticker_raw, "cik": cik, "sec_name": j.get("name"), "count": len(df)})
    except Exception as e:
        ecov.append({"ticker_raw": k.ticker_raw, "cik": cik, "error": type(e).__name__})
pd.DataFrame(earn).to_csv(ROOT / "earnings/edgar_8k_item202.csv", index=False)
E = K.merge(pd.DataFrame(ecov), on=["ticker_raw", "cik"], how="left")
E.to_csv(ROOT / "reports/earnings_coverage.csv", index=False)
print("cik found:", int(K.cik.notna().sum()), "/", len(K),
      "| with earnings dates:", int((E["count"].fillna(0) > 0).sum()))
