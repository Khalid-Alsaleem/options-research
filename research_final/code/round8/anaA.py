import pandas as pd, numpy as np
T = pd.read_pickle("stageA.pkl"); T["per"] = np.where(T.date < "2023-01-01", "L", "T")
T["al"] = np.where(T.side == 1, T.spy_up, ~T.spy_up); T["wk"] = T.date.dt.to_period("W")
HZ = [c for c in T.columns if c.startswith("f")]
rows = []
for flt in ("none", "aligned"):
    X = T if flt == "none" else T[T.al]
    base = X[X.sig == "baseline"].groupby(["side", "per"])[HZ].mean()
    for (sig, side), g in X[X.sig != "baseline"].groupby(["sig", "side"]):
        for hz in HZ:
            r = dict(sig=sig, side=side, flt=flt, hz=hz)
            for per, gp in g.groupby("per"):
                x = gp[hz].dropna(); b = base.loc[(side, per), hz]
                if len(x) < 30: continue
                wk = (x - b).groupby(gp.loc[x.index, "wk"]).mean()
                r.update({f"{per}_n": len(x), f"{per}_mean": x.mean(), f"{per}_ex": x.mean() - b, f"{per}_t": wk.mean() / (wk.std() / np.sqrt(len(wk))), f"{per}_win": (x > 0).mean()})
            rows.append(r)
R = pd.DataFrame(rows); R.to_pickle("anaA.pkl")
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
c = ["sig", "side", "flt", "hz", "L_n", "L_mean", "L_ex", "L_t", "T_n", "T_mean", "T_ex", "T_t"]
good = R[(R.L_ex > 0) & (R.T_ex > 0) & (R.L_t > 2) & (R.T_t > 1.5)]
print("robust excess (both periods, t>2/1.5):", len(good), "of", len(R))
print(good.sort_values("L_t", ascending=False)[c].round(4).head(60).to_string(index=False))
print("--- puts with robust excess:"); print(good[good.side == -1][c].round(4).to_string(index=False))
