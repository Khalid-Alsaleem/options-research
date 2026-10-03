"""
Free earnings-date source test (v2): SEC EDGAR 8-K filings with Item 2.02
("Results of Operations and Financial Condition").
v3: User-Agent from secret (v2 got HTTP 403). v2: known CIK numbers for the test set (the ticker lookup returned nothing in v1),
and full diagnostics (HTTP status + start of response) so any block is visible.
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

# SEC requires a declared "Name email" User-Agent; kept in a GitHub secret so the email is not public
UA = {"User-Agent": os.environ["SEC_USER_AGENT"], "Accept-Encoding": "gzip, deflate"}
TEST = {"AAPL": 320193, "ATVI": 718877, "CELG": 816284, "TWTR": 1418091,
        "XLNX": 743988, "CTXS": 877890, "FRC": 1132979, "SIVB": 719739}
BASE = "https://data.sec.gov/submissions/"

def get(url):
    time.sleep(0.25)                      # SEC fair-access limit: max 10 requests/second
    return requests.get(url, headers=UA, timeout=60)

out = Path("data/edgar_test")
out.mkdir(parents=True, exist_ok=True)
rows, all_dates = [], []
for t, cik in TEST.items():
    row = {"ticker": t, "cik": cik}
    r = get(f"{BASE}CIK{cik:010d}.json")
    row["status"] = r.status_code
    if r.status_code != 200:
        row["response_start"] = r.text[:150].replace("\n", " ")
        rows.append(row); print(row); continue
    j = r.json()
    frames = [pd.DataFrame(j["filings"]["recent"])]
    for f in j["filings"].get("files", []):
        frames.append(pd.DataFrame(get(BASE + f["name"]).json()))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.form == "8-K") & df["items"].fillna("").str.contains("2.02")]
    dates = sorted(d for d in df.filingDate if d >= "2015-01-01")
    row.update(name=j.get("name"), count=len(dates),
               first=dates[0] if dates else None, last=dates[-1] if dates else None)
    all_dates += [{"ticker": t, "date": d} for d in dates]
    rows.append(row); print(row)

pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
pd.DataFrame(all_dates, columns=["ticker", "date"]).to_csv(out / "dates.csv", index=False)
