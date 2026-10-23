import pickle, numpy as np, pandas as pd, itertools, sys
sys.path.insert(0, sys.argv[1]); from exits import exit_ret
CON = [(d, t) for d in (0.5, 0.7) for t in (180, 365)]
R = pickle.load(open(sys.argv[1] + "/long_paths_ext.pkl", "rb"))
meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "side", "spy_up")} for r in R]); PATHS = np.stack([r["paths"] for r in R])
keep = (meta.side == 1).values; meta = meta[keep].reset_index(drop=True); PATHS = PATHS[keep]
LM = (meta.date < "2023-01-01").values; AL = meta.spy_up.values; WK = meta.date.dt.to_period("W").values
def tstat(x, w):
    s = pd.Series(x).groupby(w).mean(); return s.mean() / (s.std() / np.sqrt(len(s)))
rows = []
for ci, ct in enumerate(CON):
    for Hh, tp, sl, tr in itertools.product((20, 40, 60, 80, 120), (None, 0.5, 1.0, 2.0, 3.0), (None, -0.5), (False, True)):
        r = exit_ret(PATHS[:, ci, :], Hh, tp, sl, tr)
        for sig in meta.sig.unique():
            for flt in ("none", "aligned"):
                m = (meta.sig == sig).values & np.isfinite(r) & (AL if flt == "aligned" else True)
                row = dict(sig=sig, flt=flt, delta=ct[0], dte=ct[1], H=Hh, tp=tp, sl=sl, trail=tr)
                for per, pm in (("L", LM), ("T", ~LM)):
                    x = r[m & pm]; wn = x > 0
                    row.update({per + "_n": len(x), per + "_avg": x.mean(), per + "_win": wn.mean(), per + "_avgwin": x[wn].mean(), per + "_avgloss": x[~wn].mean()})
                rows.append(row)
G = pd.DataFrame(rows); G.to_pickle(sys.argv[1] + "/grid_ext.pkl")
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
c = ["sig", "flt", "delta", "dte", "H", "tp", "sl", "trail", "L_n", "L_avg", "L_win", "L_avgwin", "L_avgloss", "T_n", "T_avg", "T_win", "T_avgwin", "T_avgloss"]
print(G.sort_values("L_avg", ascending=False)[c].head(30).round(3).to_string(index=False))
print(G[(G.L_avg > 0.03) & (G.T_avg > 0.03)].groupby(["sig", "flt"]).size())
