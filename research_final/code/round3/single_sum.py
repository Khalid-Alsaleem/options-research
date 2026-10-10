import sys, pickle, numpy as np, pandas as pd
T = pd.read_pickle(sys.argv[1]); G = pickle.load(open(sys.argv[3] if len(sys.argv) > 3 else "single_grid.pkl", "rb"))
rows = []
for (N, gi), g in T.groupby(["N", "gi"]):
    name, p = G[gi]; wk = g.groupby(g.date.dt.to_period("W")).ret.mean(); by = g.groupby(g.date.dt.year).ret.mean()
    yrs = g.date.dt.year.nunique()
    rows.append(dict(strategy=name, N=N, **{k: v for k, v in p.items() if k in ("delta", "width", "target", "dte", "stop", "hold")}, n=len(g),
                     per_year=round(len(g) / max((g.date.max() - g.date.min()).days / 365, 0.5)), avg=g.ret.mean(),
                     win=(g.ret > 0).mean(), hit_target=(g.why == "target").mean(), avg_win=g.ret[g.ret > 0].mean(),
                     avg_loss=g.ret[g.ret <= 0].mean(), t=wk.mean() / (wk.std() / np.sqrt(len(wk))),
                     yrs_pos=int((by > 0).sum()), yrs=len(by), days=g.days.mean()))
R = pd.DataFrame(rows); R.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(R.round(3).sort_values("avg", ascending=False).to_string())
