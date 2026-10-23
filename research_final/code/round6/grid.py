import pickle, numpy as np, pandas as pd, itertools, sys
sys.path.insert(0, sys.argv[1]); from longopt import CONTRACTS
R = pickle.load(open(sys.argv[1] + "/long_paths.pkl", "rb"))
meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "side", "spy_up")} for r in R])
PATHS = np.stack([r["paths"] for r in R])  # N x C x H
meta["learn"] = meta.date < "2023-01-01"; LM = meta.learn.values; meta["week"] = meta.date.dt.to_period("W")
def exit_ret(p, Hh, tp, sl, trail):
    p = p[:, :Hh]; N, L = p.shape; valid = ~np.isnan(p)
    last = np.where(valid.any(1), L - 1 - np.argmax(valid[:, ::-1], 1), -1)
    hit = np.zeros_like(valid)
    if tp is not None: hit |= valid & (p >= tp)
    if sl is not None: hit |= valid & (p <= sl)
    if trail:
        pk = np.fmax.accumulate(np.nan_to_num(p, nan=-9), axis=1)
        hit |= valid & (pk >= 0.3) & (p <= pk * 0.5)
    first = np.where(hit.any(1), np.argmax(hit, 1), last)
    out = p[np.arange(N), np.clip(first, 0, L - 1)]; out[last < 0] = np.nan; return out
rows = []
EX = list(itertools.product((5, 10, 20, 40), (None, 0.3, 0.5, 1.0, 2.0), (None, -0.25, -0.5), (False, True)))
for ci, ct in enumerate(CONTRACTS):
    P = PATHS[:, ci, :]
    for (Hh, tp, sl, tr) in EX:
        if tr and tp is not None and tp <= 0.3: continue
        r = exit_ret(P, Hh, tp, sl, tr)
        for sig in list(meta.sig.unique()) + ["ALL"]:
            ms = (meta.sig == sig).values if sig != "ALL" else np.ones(len(meta), bool)
            for sides in ("call", "put", "both"):
                msd = ms & ((meta.side == 1).values if sides == "call" else (meta.side == -1).values if sides == "put" else True)
                for flt in ("none", "aligned"):
                    m = msd & np.isfinite(r)
                    if flt == "aligned": m &= ((meta.side == 1) & meta.spy_up | (meta.side == -1) & ~meta.spy_up).values
                    row = dict(delta=ct[0], dte=ct[1], H=Hh, tp=tp, sl=sl, trail=tr, sig=sig, sides=sides, flt=flt)
                    for per, pm in (("L", LM), ("T", ~LM)):
                        x = r[m & pm]
                        row[per + "_n"] = len(x)
                        if len(x) < 30: continue
                        wn = x > 0
                        row.update({per + "_avg": x.mean(), per + "_win": wn.mean(), per + "_avgwin": x[wn].mean() if wn.any() else np.nan,
                                    per + "_avgloss": x[~wn].mean() if (~wn).any() else np.nan})
                    rows.append(row)
G = pd.DataFrame(rows); G.to_pickle(sys.argv[1] + "/grid.pkl"); print(len(G))
