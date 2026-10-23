"""Bullish ONE-direction trade (bull put credit spread: sell put + buy lower put) entered on each school's UP signal.
No calls+puts together. Earnings: no entry within 10 trading days before an earnings filing. SPY>MA200 filter as variant."""
import sys, numpy as np, pandas as pd, pickle
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1]); import schools as S, single as SG
TICK = sys.argv[2].split(",")
E = pd.read_csv("/home/claude/options-research/data/earnings/edgar_8k_item202.csv", parse_dates=["filing_date"])
spy = pd.read_csv("/home/claude/options-research/data/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
CFG = {"std": dict(delta=0.25, width=0.08, dte=35, target=0.15, stop=1.0, hold=20),
       "safe": dict(delta=0.20, width=0.05, dte=35, target=0.10, stop=1.0, hold=20)}
def run(t):
    try: px = S.load(t)
    except Exception: return []
    q = pd.DataFrame({"close": px.c, "iv": px.iv, "rs": px.rs}, index=px.index)
    blk = np.zeros(len(q), bool)
    for d in E[E.ticker_raw == t].filing_date:
        i = q.index.searchsorted(d); blk[max(0, i - 10):i + 1] = True
    s = spy.reindex(q.index).ffill(); spyu = (s > s.rolling(200).mean()).values
    c = px.c; up_trend = ((c > c.rolling(50).mean()) & (c > c.rolling(200).mean())).values
    sigs = {f"{f}{p}": S.setups(px, f, p)[1][0] for f, p in S.FAMS}
    sigs["trend_state"] = up_trend
    out = []
    for name, ent in sigs.items():
        for flt in ("none", "spy"):
            m = ent & ~blk & (spyu if flt == "spy" else True)
            for cn, cfg in CFG.items():
                busy = -1
                for i in np.flatnonzero(m):
                    if i <= busy or i < 252: continue
                    r = SG.trade(q, i, "put_credit", cfg)
                    if r is None: continue
                    out.append((t, name, flt, cn, q.index[i], r[0], r[1], r[2])); busy = i + r[1]
    return out
if __name__ == "__main__":
    with Pool(2) as pool: rows = sum(pool.map(run, TICK), [])
    T = pd.DataFrame(rows, columns=["ticker", "school", "flt", "cfg", "date", "ret", "days", "why"]).dropna(subset=["ret"])
    T.to_pickle("credit6.pkl")
    T["per"] = np.where(T.date < "2023-01-01", "2016-22", "2023-26")
    g = T.groupby(["school", "flt", "cfg", "per"]).ret
    R = pd.DataFrame({"n": g.size(), "avg": g.mean(), "win": g.apply(lambda x: (x > 0).mean())}).unstack("per")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(R.round(3).sort_values(("avg", "2016-22"), ascending=False).to_string())
