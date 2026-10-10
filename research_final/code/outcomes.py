"""Outcome table: for every eligible stock-day (learning period AND holdout kept separate),
the result of buying the standard contract (delta 0.8, ~90 DTE) as a call and as a put,
held under protocol rules (target 30%, 10 days, stops 20/30%). Indicators then become lookups."""
import sys, numpy as np, pandas as pd, warnings, pickle
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import eng as E
root = sys.argv[1]
P = E.build(root); vrp, spy_r, vix_r = E.market(root)

def work(sid):
    seg = P[sid]; px = E.prepare(seg, vrp)
    elig = np.where(px.in_index.values & ~px.earn_block.values & (px.index >= "2016-01-01"))[0]
    neg = np.zeros(len(px), bool); rows = []
    for i in elig:
        for call in (True, False):
            r = E.simulate(px, i, call, neg, spy_r, vix_r)
            if r: rows.append((sid, px.index[i], call, r["s20_fixed"], r["s30_fixed"], r["s20_hit"], r["s30_hit"], r["s30_shockfail"]))
    return rows

if __name__ == "__main__":
    sids = [s for s in P if s not in E.EXCLUDE]
    out = []
    with Pool(2) as pool:
        for k, rows in enumerate(pool.imap_unordered(work, sids, chunksize=4)):
            out += rows
            if k % 100 == 0: print(k, len(out), flush=True)
    df = pd.DataFrame(out, columns=["seg", "date", "call", "s20", "s30", "hit20", "hit30", "shockfail"])
    df.to_pickle(sys.argv[2] if len(sys.argv) > 2 else "outcomes.pkl")
    print("done", len(df))
