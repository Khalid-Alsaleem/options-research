import pickle, numpy as np, pandas as pd, itertools, sys
from exits import exit_ret
from stageB import CON
R = pickle.load(open(sys.argv[1], "rb")); out = sys.argv[2]
meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "spy_up")} for r in R]); PA = np.stack([r["paths"] for r in R]); del R
LM = (meta.date < "2023-01-01").values; MO = meta.date.dt.to_period("M").values; AL = meta.spy_up.values
rows = []
for ci, ct in enumerate(CON):
    for Hh, tp, sl in itertools.product((20, 40, 60, 120), (0.5, 1.0, 2.0, None), (None, -0.5)):
        r = exit_ret(PA[:, ci, :], Hh, tp, sl, False)
        for sig in meta.sig.unique():
            for flt in ("none", "aligned"):
                m = (meta.sig == sig).values & np.isfinite(r) & (AL if flt == "aligned" else True)
                row = dict(sig=sig, flt=flt, dte=ct[0], mny=ct[1], H=Hh, tp=tp, sl=sl)
                for per, pm in (("L", LM), ("T", ~LM)):
                    x = r[m & pm]; w = x > 0
                    row.update({per + "_n": len(x), per + "_avg": x.mean() if len(x) else np.nan, per + "_win": w.mean() if len(x) else np.nan,
                                per + "_aw": x[w].mean() if w.any() else np.nan, per + "_al": x[~w].mean() if (~w).any() else np.nan})
                rows.append(row)
G = pd.DataFrame(rows); G.to_pickle(out); print(len(G))
