"""
Second review pass: hidden ticker reuse and companies that moved to a new holding-company CIK.
Found by name consistency checks and earnings-density checks (not visible by dates alone).
  PRICES  : ticker -> correct EODHD symbol
  EARNINGS: ticker -> list of SEC CIKs (old + new holding company) or name tokens for lookup
"""
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

EOD = os.environ["EODHD_API_TOKEN"]
UA = {"User-Agent": os.environ["SEC_USER_AGENT"], "Accept-Encoding": "gzip, deflate"}
START = "2015-01-01"
ROOT = Path("data")
(ROOT / "eod_override").mkdir(parents=True, exist_ok=True)

PRICES = {           # EODHD symbol now points to a different company
    "BBBY": "BBBY_old",      # EODHD BBBY = former Overstock
    "BBT": "TFC",            # EODHD BBT = Beacon Financial; BB&T became Truist (TFC)
    "PARA": "PARA_old1",     # EODHD PARA = Banzai International
    "XL": "XL_old",          # EODHD XL = XL Fleet
}
EARNINGS = {         # CIKs: old company + new holding company where it reorganized
    "XOM": [34088],
    "BLK": [1364742, 2012383],
    "XRX": [108772, 1770450],
    "APA": [6769, 1841666],
    "WRK": [1636023, 1732845],
    "DIS": [1001039, 1744489],
    "CI": [701221, 1739940],
    "PARA": [813828],
    "BBT": [92230],
    "XL": ["XL GROUP"],
}


def sec_get(url):
    time.sleep(0.15)
    return requests.get(url, headers=UA, timeout=120)


def norm(s):
    s = re.sub(r"/[^/]*/", " ", str(s).upper())
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", s.replace("&", " "))).strip()


def filings_202(cik):
    j = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    frames = [pd.DataFrame(j["filings"]["recent"])]
    for f in j["filings"].get("files", []):
        frames.append(pd.DataFrame(sec_get("https://data.sec.gov/submissions/" + f["name"]).json()))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.form == "8-K") & df["items"].fillna("").str.contains("2.02") & (df.filingDate >= START)]
    return j.get("name"), df


report = []
for t, code in PRICES.items():
    r = requests.get(f"https://eodhd.com/api/eod/{code}.US",
                     params={"api_token": EOD, "fmt": "json", "from": START}, timeout=60)
    px = pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
    if len(px):
        px.to_csv(ROOT / f"eod_override/{t}.csv.gz", index=False)
    s = requests.get(f"https://eodhd.com/api/splits/{code}.US",
                     params={"api_token": EOD, "fmt": "json", "from": START}, timeout=60)
    sp = pd.DataFrame(s.json()) if s.status_code == 200 else pd.DataFrame()
    if len(sp):
        sp.to_csv(ROOT / f"eod_override/{t}__splits.csv", index=False)
    report.append({"ticker": t, "kind": "price", "source": code, "rows": len(px),
                   "first": px.date.iloc[0] if len(px) else None, "last": px.date.iloc[-1] if len(px) else None,
                   "first_close": px.close.iloc[0] if len(px) else None})

lookup = None
rows = []
for t, ciks in EARNINGS.items():
    resolved = []
    for c in ciks:
        if isinstance(c, int):
            resolved.append(c)
            continue
        if lookup is None:
            txt = sec_get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt").content.decode("latin-1")
            lookup = [(norm(m.group(1)), int(m.group(2)))
                      for m in (re.match(r"^(.*):(\d{10}):$", l) for l in txt.splitlines()) if m]
        toks = norm(c).split()
        cands = sorted({k for n, k in lookup if all(re.search(rf"\b{x}\b", n) for x in toks)})[:10]
        best = max(cands, key=lambda k: len(filings_202(k)[1]), default=None)
        if best:
            resolved.append(best)
    for cik in resolved:
        name, df = filings_202(cik)
        for _, f in df.iterrows():
            rows.append({"ticker_raw": t, "cik": cik, "filing_date": f.filingDate,
                         "accepted": f.acceptanceDateTime})
        report.append({"ticker": t, "kind": "earnings", "source": cik, "sec_name": name, "rows": len(df),
                       "first": df.filingDate.min() if len(df) else None,
                       "last": df.filingDate.max() if len(df) else None})

pd.DataFrame(rows).to_csv(ROOT / "earnings/edgar_8k_item202_override.csv", index=False)
pd.DataFrame(report).to_csv(ROOT / "reports/override_report.csv", index=False)
print(pd.DataFrame(report).to_string())
