"""Confluence: two indicators from DIFFERENT families both signal the same side within 3 trading days.
Top 25 configs per side by learning-period mean (one per family-param set). Learning period only."""
import sys, pickle, numpy as np, pandas as pd, itertools, warnings
warnings.filterwarnings("ignore")
import eng as E
G = E.grid(); R = pd.read_csv("grid_iv2.csv")
P = {k: v for k, v in E.build(sys.argv[1]).items() if k not in E.EXCLUDE}
O = pd.read_pickle(sys.argv[2]); LK = {}
for (sid, call), g in O.groupby(["seg", "call"]):
    idx = P[sid]["px"].index; arr = np.full(len(idx), np.nan); pos = idx.get_indexer(g.date)
    arr[pos[pos >= 0]] = g.s30.values[pos >= 0]; LK[(sid, call)] = arr
key = {f"{g[0]}{g[1]}": k for k, g in enumerate(G)}
rows = []
for side, call in (("call", True), ("put", False)):
    top = R[R[f"n_{side}"] >= 300].sort_values(f"r_{side}", ascending=False)
    top = top.groupby("family").head(3).head(25)
    sets = {}
    for tr in top.trial:
        fn = G[key[tr]][2]; s = {}
        for sid, seg in P.items():
            px = seg["px"]; le, se, _, _ = fn(px); trig = np.asarray(le if call else se, bool)
            learn = (px.index >= "2016-01-01") & (px.index <= "2022-11-30")
            idx = np.flatnonzero(trig & learn)
            if len(idx): s[sid] = list(idx)
        sets[tr] = s
    for a, b in itertools.combinations(top.trial, 2):
        if R.loc[R.trial == a, "family"].iloc[0] == R.loc[R.trial == b, "family"].iloc[0]: continue
        vals = []
        for sid, ia in sets[a].items():
            ib = sets[b].get(sid)
            if not ib: continue
            ib = np.array(ib); last = -999
            for i in sorted(ia):
                if np.any((ib >= i - 3) & (ib <= i)) and i > last + 10:
                    a_ = LK.get((sid, call)); v = a_[i] if a_ is not None else np.nan
                    if not np.isnan(v): vals.append((P[sid]["px"].index[i], v)); last = i
        if len(vals) < 100: continue
        V = pd.DataFrame(vals, columns=["date", "r"]); wk = V.groupby(V.date.dt.to_period("W")).r.mean()
        d = V.date.dt.year <= 2019
        rows.append(dict(side=side, a=a, b=b, n=len(V), r=V.r.mean(), t=wk.mean() / (wk.std() / np.sqrt(len(wk))),
                         disc=V.r[d].mean(), val=V.r[~d].mean(), yrs_pos=int((V.groupby(V.date.dt.year).r.mean() > 0).sum()),
                         hit=(V.r >= 0.299).mean()))
C = pd.DataFrame(rows); C.to_csv("confluence_iv2.csv", index=False)
print("pairs tested:", len(C)); pd.set_option("display.width", 250)
print(C.sort_values("r", ascending=False).head(20).round(3).to_string())
print("r>0 & t>2:", ((C.r > 0) & (C.t > 2)).sum(), "| r>0 both halves:", ((C.disc > 0) & (C.val > 0)).sum())
