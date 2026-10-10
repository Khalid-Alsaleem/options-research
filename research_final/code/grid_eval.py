"""All 645 single-indicator configs (same grid as step 4) on the NEW shape + calibrated IV, via outcome lookup.
Learning period only. Also records each config's entry list so confluence (pairs) can be built later."""
import sys, numpy as np, pandas as pd, warnings, pickle, time
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import eng as E
ROOT, OUTC = sys.argv[1], sys.argv[2]
G = None; P = None; LK = None
def init():
    global G, P, LK
    G = E.grid(); P = {k: v for k, v in E.build(ROOT).items() if k not in E.EXCLUDE}
    O = pd.read_pickle(OUTC); LK = {}
    for (sid, call), g in O.groupby(["seg", "call"]):
        idx = P[sid]["px"].index
        arr = np.full(len(idx), np.nan); pos = idx.get_indexer(g.date)
        arr[pos[pos >= 0]] = g.s30.values[pos >= 0]
        LK[(sid, call)] = arr
def nonover(mask):
    out = np.zeros_like(mask); last = -999
    for i in np.flatnonzero(mask):
        if i > last + 10: out[i] = True; last = i
    return out
def one(k):
    fam, params, fn = G[k]; vals = []; ents = []
    for sid, seg in P.items():
        px = seg["px"]; le, se, _, _ = fn(px)
        learn = (px.index >= "2016-01-01") & (px.index <= "2022-11-30")
        for call, trig in ((True, le), (False, se)):
            a = LK.get((sid, call))
            if a is None: continue
            m = np.asarray(trig, bool) & learn & ~np.isnan(a)
            m = nonover(m)
            for i in np.flatnonzero(m):
                vals.append((call, px.index[i], a[i])); ents.append((sid, i, call))
    V = pd.DataFrame(vals, columns=["call", "date", "r"])
    row = dict(trial=f"{fam}{params}", family=fam, n=len(V))
    for side, g in (("call", V[V.call]), ("put", V[~V.call])):
        row[f"n_{side}"] = len(g); row[f"r_{side}"] = g.r.mean() if len(g) else np.nan
        if len(g) > 30:
            wk = g.groupby(g.date.dt.to_period("W")).r.mean()
            row[f"t_{side}"] = wk.mean() / (wk.std() / np.sqrt(len(wk)))
            d = g.date.dt.year <= 2019
            row[f"disc_{side}"] = g.r[d].mean(); row[f"val_{side}"] = g.r[~d].mean()
            row[f"yrs_pos_{side}"] = int((g.groupby(g.date.dt.year).r.mean() > 0).sum())
    return row, (k, ents)
if __name__ == "__main__":
    init(); n = len(G); t0 = time.time(); rows = []; ent = {}
    with Pool(2, initializer=init) as pool:
        for j, (row, (k, e)) in enumerate(pool.imap_unordered(one, range(n), chunksize=4)):
            rows.append(row); ent[k] = e
            if j % 50 == 0:
                print(j, round(time.time() - t0), flush=True); pd.DataFrame(rows).to_csv("grid_iv2.csv", index=False)
    pd.DataFrame(rows).to_csv("grid_iv2.csv", index=False); pickle.dump(ent, open("grid_entries.pkl", "wb"))
    print("done", round(time.time() - t0))
