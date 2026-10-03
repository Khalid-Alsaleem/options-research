"""
Free earnings-date source test: SEC EDGAR 8-K filings with Item 2.02
("Results of Operations and Financial Condition") = the official earnings release filing.
EDGAR keeps filings of delisted companies too. Compared against Yahoo for AAPL.
No API key needed; SEC only asks for a User-Agent with contact info.
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "options-research Khalid-Alsaleem@users.noreply.github.com"}
TEST = ["AAPL", "ATVI", "CELG", "TWTR", "XLNX", "CTXS", "FRC", "SIVB"]

def get(url):
    time.sleep(0.25)                      # SEC fair-access limit: max 10 requests/second
    return requests.get(url, headers=UA, timeout=60)

def cik_for(ticker):
    r = get("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={ticker}&type=8-K&dateb=&owner=include&count=10&output=atom")
    m = re.search(r"<cik>(\d+)</cik>", r.text)
    return int(m.group(1)) if m else None

def earnings_8k_dates(cik):
    base = "https://data.sec.gov/submissions/"
    j = get(f"{base}CIK{cik:010d}.json").json()
    frames = [pd.DataFrame(j["filings"]["recent"])]
    for f in j["filings"].get("files", []):
        frames.append(pd.DataFrame(get(base + f["name"]).json()))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.form == "8-K") & df["items"].fillna("").str.contains("2.02")]
    return sorted(d for d in df.filingDate if d >= "2015-01-01"), j.get("name")

out = Path("data/edgar_test")
out.mkdir(parents=True, exist_ok=True)
rows, all_dates = [], []
for t in TEST:
    row = {"ticker": t}
    try:
        cik = cik_for(t)
        row["cik"] = cik
        if cik:
            dates, name = earnings_8k_dates(cik)
            row.update(name=name, count=len(dates), first=dates[0] if dates else None,
                       last=dates[-1] if dates else None)
            all_dates += [{"ticker": t, "date": d} for d in dates]
    except Exception as e:
        row["error"] = type(e).__name__
    rows.append(row)
    print(row)

pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
pd.DataFrame(all_dates).to_csv(out / "dates.csv", index=False)
