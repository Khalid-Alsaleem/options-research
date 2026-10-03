"""
Targeted re-pull for the 111 tickers with renamed / reused symbols (found in the first data pull).
Each index-membership segment gets: candidate EODHD symbols (old, renamed, successor) and company-name
token sets for SEC. The script DOWNLOADS every candidate and PICKS BY EVIDENCE:
  prices   -> the candidate whose history best covers the membership window
  earnings -> the SEC entity with the most 8-K Item 2.02 filings inside the window
Output: data/universe/segments.csv = final map (one row per ticker segment) used by the engine.
"""
import io
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
(ROOT / "eod_fixed").mkdir(parents=True, exist_ok=True)

# ticker_raw | segment from | segment to | EODHD candidates | SEC name token sets (alternatives separated by ;)
# "=" exact name, "#" fixed CIK. v2: ANDV, DNB, VIAB pinned to CIK after review; PBCT name fixed.
MAP = """
AABA||||AABA|ALTABA
ADS||||ADS,BFH|ALLIANCE DATA SYSTEMS;BREAD FINANCIAL
ADT||||ADT_old|=ADT CORP
AET||||AET|AETNA INC
AGN||||AGN,AGN_old|ALLERGAN PLC;ACTAVIS PLC
ALXN||||ALXN|ALEXION PHARMACEUTICALS
ANDV||||ANDV|#50104
ANTM||||ELV,ANTM|ANTHEM INC;ELEVANCE HEALTH
APC||||APC_old|ANADARKO PETROLEUM
ARG||||ARG|AIRGAS INC
ARNC||||HWM,ARNC_old,ARNC|ARCONIC INC;HOWMET AEROSPACE;ALCOA INC
BCR||||BCR|BARD C R;C R BARD
BHGE||2017-07-05||BHI|BAKER HUGHES INC
BHGE|2017-07-05|||BKR,BHGE|BAKER HUGHES A GE;BAKER HUGHES CO;BAKER HUGHES HOLDINGS
BLL||||BALL,BLL|=BALL CORP
BRCM||||BRCM|BROADCOM CORP
BXLT||||BXLT|BAXALTA
CA||||CA_old|=CA INC
CAM||||CAM_old|CAMERON INTERNATIONAL
CBS||||CBS,PARA|CBS CORP;VIACOMCBS;PARAMOUNT GLOBAL
CCE||||CCE|COCA COLA ENTERPRISES
CCEP||||CCEP|COCA COLA EUROPEAN PARTNERS;COCA COLA EUROPACIFIC
CDAY||||DAY|CERIDIAN HCM;DAYFORCE
CELG||||CELG|CELGENE CORP
CERN||||CERN|CERNER CORP
COG||||CTRA,COG|CABOT OIL;COTERRA ENERGY
COL||||COL|ROCKWELL COLLINS
CPGX||||CPGX|COLUMBIA PIPELINE GROUP
CSRA||||CSRA_old|CSRA INC
CTL||||LUMN,CTL|CENTURYLINK;LUMEN TECHNOLOGIES
CVC||||CVC|CABLEVISION SYSTEMS
CXO||||CXO|CONCHO RESOURCES
DAY||||DAY|DAYFORCE;CERIDIAN HCM
DISCA||||DISCA|DISCOVERY INC;DISCOVERY COMMUNICATIONS;WARNER BROS DISCOVERY
DISCK||||DISCK|DISCOVERY INC;DISCOVERY COMMUNICATIONS;WARNER BROS DISCOVERY
DNB||||DNB_old|#1115222
DO||||DO_old|DIAMOND OFFSHORE
DOW||2018-01-01||DOW_old|DOW CHEMICAL
DOW|2018-01-01|||DOW|=DOW INC
DWDP||||DD,DWDP|DOWDUPONT;DUPONT DE NEMOURS
EMC||||EMC_old|=EMC CORP
ENDP||||ENDP|ENDO INTERNATIONAL
ESRX||||ESRX|EXPRESS SCRIPTS HOLDING
ESV||||ESV|ENSCO PLC;VALARIS
ETFC||||ETFC|E TRADE FINANCIAL
EVHC||||EVHC|ENVISION HEALTHCARE
FB||||META,FB_old|FACEBOOK INC;META PLATFORMS
FI||||FI,FISV|FISERV INC
FLIR||||FLIR|FLIR SYSTEMS
FOX||2019-03-19||TFCF|TWENTY FIRST CENTURY FOX
FOX|2019-03-19|||FOX|=FOX CORP
FOXA||2019-03-19||TFCFA|TWENTY FIRST CENTURY FOX
FOXA|2019-03-19|||FOXA|=FOX CORP
FRC||||FRC,FRCB|FIRST REPUBLIC BANK
FTR||||FTR|FRONTIER COMMUNICATIONS
GAS||||GAS|AGL RESOURCES
GGP||||GGP|=GGP INC;GENERAL GROWTH PROPERTIES
GMCR||||GMCR|KEURIG GREEN MOUNTAIN;GREEN MOUNTAIN COFFEE
HAR||||HAR|HARMAN INTERNATIONAL
HCP||||DOC,HCP_old,PEAK|=HCP INC;HEALTHPEAK PROPERTIES
HFC||||HFC,DINO|HOLLYFRONTIER;HF SINCLAIR
HOT||||HOT|STARWOOD HOTELS
HRS||||LHX,HRS|HARRIS CORP;L3HARRIS TECHNOLOGIES
INFO||||INFO_old1,INFO_old|IHS MARKIT
IR||2020-03-02||TT|INGERSOLL RAND PLC;TRANE TECHNOLOGIES
IR|2020-03-02|||IR|INGERSOLL RAND INC;GARDNER DENVER
JEC||||J,JEC|JACOBS ENGINEERING;JACOBS SOLUTIONS
KORS||||CPRI,KORS|MICHAEL KORS;CAPRI HOLDINGS
KSU||||KSU|KANSAS CITY SOUTHERN
LB||||BBWI,LB_old|L BRANDS;BATH BODY WORKS
LLL||||LLL_old|L3 TECHNOLOGIES;L 3 COMMUNICATIONS
LLTC||||LLTC|LINEAR TECHNOLOGY
LM||||LM|LEGG MASON
LVLT||||LVLT|LEVEL 3 COMMUNICATIONS;LEVEL 3 PARENT
MJN||||MJN|MEAD JOHNSON
MNK||||MNKKQ,MNK_old,MNK|MALLINCKRODT PLC
MON||||MON_old|=MONSANTO CO
MXIM||||MXIM|MAXIM INTEGRATED
MYL||||MYL,VTRS|MYLAN N V;MYLAN NV;MYLAN INC
NBL||||NBL|NOBLE ENERGY
NFX||||NFX_old,NFX|NEWFIELD EXPLORATION
PBCT||||PBCT|PEOPLE S UNITED FINANCIAL;PEOPLES UNITED FINANCIAL
PCL||||PCL_old|PLUM CREEK TIMBER
PCP||||PCP|PRECISION CASTPARTS
PEAK||||DOC,PEAK|HEALTHPEAK PROPERTIES
POM||||POM_old|PEPCO HOLDINGS
PX||||PX_old|PRAXAIR INC
RAI||||RAI|REYNOLDS AMERICAN
RE||||EG|EVEREST RE GROUP;EVEREST GROUP
RHT||||RHT|RED HAT INC
RTN||||RTN|=RAYTHEON CO
SBNY||||SBNY|SIGNATURE BANK
SCG||||SCG|SCANA CORP
SE||||SE1,SE_old|SPECTRA ENERGY CORP
SNDK||2017-01-01||SNDK_old|SANDISK CORP
SNDK|2017-01-01|||SNDK|SANDISK CORP
SNI||||SNI|SCRIPPS NETWORKS
SPLS||||SPLS_old|=STAPLES INC
STI||||STI_old|SUNTRUST BANKS
STJ||||STJ|ST JUDE MEDICAL
SYMC||||GEN,SYMC|SYMANTEC CORP;NORTONLIFELOCK;GEN DIGITAL
TE||||TE_old1,TE_old|TECO ENERGY
TIF||||TIF|TIFFANY
TMK||||GL,TMK|TORCHMARK;GLOBE LIFE
TSS||||TSS|TOTAL SYSTEM SERVICES
TWC||||TWC|TIME WARNER CABLE
TWX||||TWX|=TIME WARNER INC
UTX||||RTX,UTX|UNITED TECHNOLOGIES;RAYTHEON TECHNOLOGIES;RTX CORP
VAR||||VAR|VARIAN MEDICAL
VIAB||||VIAB|#1339947
VIAC||||PARA,VIAC|VIACOMCBS;PARAMOUNT GLOBAL
WCG||||WCG|WELLCARE HEALTH
WFM||||WFM|WHOLE FOODS
WLTW||||WTW,WLTW|WILLIS TOWERS WATSON
WYND||||WYN,TNL|WYNDHAM WORLDWIDE;WYNDHAM DESTINATIONS;TRAVEL LEISURE CO
XEC||||XEC|CIMAREX
XLNX||||XLNX|XILINX
"""


def norm(s):
    s = re.sub(r"/[^/]*/", " ", str(s).upper())          # drop SEC state tags like /DE/
    s = re.sub(r"[^A-Z0-9 ]", " ", s.replace("&", " "))
    return re.sub(r"\s+", " ", s).strip()


def sec_get(url):
    time.sleep(0.15)
    return requests.get(url, headers=UA, timeout=120)


def eod(code):
    r = requests.get(f"https://eodhd.com/api/eod/{code}.US",
                     params={"api_token": EOD, "fmt": "json", "from": START}, timeout=60)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()


U = pd.read_csv(ROOT / "universe/membership.csv", parse_dates=["start", "end"])
# Row format: TICKER|segment_start|segment_end||candidates|names  (empty = whole membership)
segs = []
for l in MAP.strip().splitlines():
    p = l.split("|")
    raw, cands, names = p[0], p[-2], p[-1]
    seg_start = p[1] or None
    seg_end = p[2] or None
    segs.append([raw, seg_start, seg_end, cands, names])

# SEC name -> CIK master list (all current and former names)
lookup = sec_get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt").content.decode("latin-1")
sec_names = []
for line in lookup.splitlines():
    m = re.match(r"^(.*):(\d{10}):$", line)
    if m:
        sec_names.append((norm(m.group(1)), int(m.group(2))))
print("sec names:", len(sec_names))

sub_cache = {}


def filings_202(cik):
    if cik in sub_cache:
        return sub_cache[cik]
    try:
        j = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        frames = [pd.DataFrame(j["filings"]["recent"])]
        for f in j["filings"].get("files", []):
            frames.append(pd.DataFrame(sec_get("https://data.sec.gov/submissions/" + f["name"]).json()))
        df = pd.concat(frames, ignore_index=True)
        df = df[(df.form == "8-K") & df["items"].fillna("").str.contains("2.02") & (df.filingDate >= START)]
        res = (j.get("name"), df[["filingDate", "acceptanceDateTime"]].reset_index(drop=True))
    except Exception:
        res = (None, pd.DataFrame(columns=["filingDate", "acceptanceDateTime"]))
    sub_cache[cik] = res
    return res


report, earn_rows = [], []
for i, (raw, s_start, s_end, cands, names) in enumerate(segs):
    mem = U[U.ticker_raw == raw]
    lo = max(pd.Timestamp("2016-01-01"), pd.Timestamp(s_start) if s_start else pd.Timestamp("1900-01-01"))
    hi = pd.Timestamp(s_end) if s_end else pd.Timestamp.today().normalize()
    days = pd.DatetimeIndex([])
    for _, iv in mem.iterrows():
        a = max(iv.start, lo); b = min(iv.end if pd.notna(iv.end) else pd.Timestamp.today().normalize(), hi)
        if a <= b:
            days = days.union(pd.bdate_range(a, b))
    if len(days) == 0:
        continue
    m_start, m_end = days.min(), days.max()
    seg_id = f"{raw}__{i}"
    row = {"seg_id": seg_id, "ticker_raw": raw, "seg_start": m_start.date(), "seg_end": m_end.date()}

    # prices: pick the candidate that best covers the segment
    best = None
    window = days
    for code in cands.split(","):
        px = eod(code)
        if px.empty:
            continue
        d = pd.to_datetime(px.date)
        cover = len(window.intersection(pd.DatetimeIndex(d))) / max(len(window), 1)
        score = (round(cover, 3), d.min() <= m_start - pd.Timedelta(days=200))
        if best is None or score > best[0]:
            best = (score, code, px)
    if best:
        (cover, warm), code, px = best
        px.to_csv(ROOT / f"eod_fixed/{seg_id}.csv.gz", index=False)
        sp = requests.get(f"https://eodhd.com/api/splits/{code}.US",
                          params={"api_token": EOD, "fmt": "json", "from": START}, timeout=60)
        sp = pd.DataFrame(sp.json()) if sp.status_code == 200 else pd.DataFrame()
        if len(sp):
            sp.to_csv(ROOT / f"eod_fixed/{seg_id}__splits.csv", index=False)
        row.update(price_symbol=code, price_cover=cover, price_warmup=warm,
                   price_first=px.date.iloc[0], price_last=px.date.iloc[-1])

    # earnings: SEC entity with most 8-K 2.02 filings inside the segment
    cand_ciks = []
    for alt in names.split(";"):
        if alt.startswith("#"):                 # CIK fixed by hand after review (verified by filings count)
            cand_ciks.append((0, int(alt[1:])))
            continue
        exact = alt.startswith("=")
        toks = norm(alt.lstrip("=")).split()
        for nm, cik in sec_names:
            if (nm == " ".join(toks)) if exact else all(re.search(rf"\b{re.escape(t)}\b", nm) for t in toks):
                cand_ciks.append((len(nm), cik))
    seen, ordered = set(), []
    for _, c in sorted(cand_ciks):
        if c not in seen:
            seen.add(c); ordered.append(c)
    best_e = None
    for cik in ordered[:10]:
        name, f = filings_202(cik)
        fd = pd.to_datetime(f.filingDate)
        n_in = int(((fd >= m_start - pd.Timedelta(days=60)) & (fd <= m_end + pd.Timedelta(days=30))).sum())
        if best_e is None or n_in > best_e[0]:
            best_e = (n_in, cik, name, f)
    if best_e and best_e[0] > 0:
        n_in, cik, name, f = best_e
        row.update(cik=cik, sec_name=name, earnings_in_window=n_in)
        for _, r in f.iterrows():
            earn_rows.append({"seg_id": seg_id, "ticker_raw": raw, "cik": cik,
                              "filing_date": r.filingDate, "accepted": r.acceptanceDateTime})
    else:
        row.update(cik=None, sec_name=None, earnings_in_window=0, sec_candidates=len(ordered))
    report.append(row)
    print(row)

R = pd.DataFrame(report)
R.to_csv(ROOT / "reports/fix_report.csv", index=False)
pd.DataFrame(earn_rows).to_csv(ROOT / "earnings/edgar_8k_item202_fixed.csv", index=False)
print("segments:", len(R), "| price cover>=0.95:", int((R.price_cover >= 0.95).sum()),
      "| earnings found:", int((R.earnings_in_window > 0).sum()))
