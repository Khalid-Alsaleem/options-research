import sys, pickle, numpy as np, pandas as pd
T = pd.read_pickle(sys.argv[1]).dropna(subset=["ret"]); G = {g[0]: g for g in pickle.load(open(sys.argv[2], "rb"))}
YRS = 6.92 if "learn" in sys.argv[1] else 3.69
rows = []
for (N, gi), g in T.groupby(["N", "gi"]):
    _, sig, kind, p = G[gi]
    wk = g.groupby(g.date.dt.to_period("W")).ret.mean(); by = g.groupby(g.date.dt.year).ret.mean()
    g = g.assign(exit=g.date + pd.to_timedelta(np.ceil(g.days * 1.45), "D")).sort_values("exit")
    eq = np.cumprod(1 + 0.01 * g.ret.values); dd = (eq / np.maximum.accumulate(eq) - 1).min()
    ann = eq[-1] ** (1 / YRS) - 1
    rows.append(dict(id=gi, N=N, signal=sig, kind=kind, delta=p["delta"], width=p["width"], target=p["target"], dte=p["dte"],
        n=len(g), per_year=round(len(g) / YRS), avg=g.ret.mean(), win=(g.ret > 0).mean(), hit=(g.why == "target").mean(),
        avg_win=g.ret[g.ret > 0].mean(), avg_loss=g.ret[g.ret <= 0].mean(), t=wk.mean() / (wk.std() / np.sqrt(len(wk))),
        yrs_pos=f"{int((by > 0).sum())}/{len(by)}", worst_year=by.min(), port_ann_1pct=ann, port_maxdd_1pct=dd, days=g.days.mean()))
R = pd.DataFrame(rows); R.to_csv(sys.argv[3], index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
print(R.round(3).to_string())
