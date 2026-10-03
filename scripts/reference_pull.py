"""
Reference lists to fix ticker renames / reused tickers found in the first data pull
(111 of 745 tickers, almost all removed from the index). One-off; outputs are raw
reference data that Claude matches by company name before a targeted re-pull.
"""
import gzip
import io
import os
import re
from pathlib import Path

import pandas as pd
import requests

EOD = os.environ["EODHD_API_TOKEN"]
UA = {"User-Agent": os.environ["SEC_USER_AGENT"], "Accept-Encoding": "gzip, deflate"}
OUT = Path("data/reference")
OUT.mkdir(parents=True, exist_ok=True)

# 1) EODHD US symbol lists (active and delisted), with names and ISINs
for flag, fname in [(0, "eodhd_us_active.csv.gz"), (1, "eodhd_us_delisted.csv.gz")]:
    r = requests.get("https://eodhd.com/api/exchange-symbol-list/US",
                     params={"api_token": EOD, "fmt": "json", "delisted": flag}, timeout=300)
    df = pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
    df.to_csv(OUT / fname, index=False)
    print(fname, r.status_code, len(df))

# 2) SEC name -> CIK lookup, filtered to the companies we need
KEYWORDS = """ALTABA ALLIANCE ADT AETNA ALLERGAN ALEXION ANDEAVOR ANTHEM ELEVANCE ANADARKO AIRGAS ARCONIC
BARD BAKER BALL BROADCOM BAXALTA CA CAMERON CBS COCA-COLA CERIDIAN DAYFORCE CELGENE CERNER CABOT COTERRA
ROCKWELL COLUMBIA CSRA CENTURYLINK CABLEVISION CONCHO DISCOVERY DUN DIAMOND DOW DOWDUPONT EMC ENDO EXPRESS
ENSCO E*TRADE ENVISION FACEBOOK META FISERV FLIR FOX TWENTY-FIRST FIRST FRONTIER AGL GGP KEURIG HARMAN HCP
HEALTHPEAK HOLLYFRONTIER STARWOOD HARRIS IHS INGERSOLL JACOBS MICHAEL KANSAS L BRANDS L3 LINEAR LEGG LEVEL
MEAD MALLINCKRODT MONSANTO MAXIM MYLAN NOBLE NEWFIELD PEOPLES PLUM PRECISION PEPCO PRAXAIR REYNOLDS EVEREST
RED RAYTHEON SIGNATURE SCANA SPECTRA SANDISK SCRIPPS STAPLES SUNTRUST ST SYMANTEC TECO TIFFANY TORCHMARK
TOTAL TIME UNITED VARIAN VIACOM VIACOMCBS WELLCARE WHOLE WILLIS WYNDHAM CIMAREX XILINX""".split()
r = requests.get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", headers=UA, timeout=600)
text = r.content.decode("latin-1")
keys = tuple(k + " " for k in KEYWORDS)
rows = []
for line in text.splitlines():
    m = re.match(r"^(.*):(\d{10}):$", line)
    if m and m.group(1).upper().startswith(keys):
        rows.append((m.group(1), int(m.group(2))))
pd.DataFrame(rows, columns=["sec_name", "cik"]).to_csv(OUT / "sec_cik_candidates.csv.gz", index=False)
print("sec lookup", r.status_code, "candidates", len(rows))
