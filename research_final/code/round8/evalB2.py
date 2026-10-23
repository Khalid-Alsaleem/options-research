import pickle, numpy as np, pandas as pd, itertools
from exits import exit_ret
R = pickle.load(open("stageB2.pkl", "rb")); G = pd.read_pickle("regimes.pkl")
meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "spy_up", "raw_price")} for r in R]); PA = np.stack([r["paths"] for r in R])
INFO = np.array([[x[0] for x in r["info"]] for r in R]); MNY = np.array([[x[1] for x in r["info"]] for r in R])
prev = G.shift(1).reindex(meta.date.values); F = {"none": np.ones(len(meta), bool), "spy50>200": (prev.spy50 > prev.spy200).values}
rows = []
for ci, dte in enumerate((60, 180, 365)):
    for Hh, tp in itertools.product((40, 60, 120), (0.3, 0.5, 1.0, 2.0, None)):
        r = exit_ret(PA[:, ci, :], Hh, tp, None, False)
        for sig in meta.sig.unique():
            for fn, fm in F.items():
                m = (meta.sig == sig).values & np.isfinite(r) & fm
                x = pd.Series(r[m], index=meta.date[m]); L, T = x[x.index < "2023"], x[x.index >= "2023"]; y = x.groupby(x.index.year).mean()
                if len(L) < 50 or len(T) < 30: continue
                rows.append(dict(sig=sig, flt=fn, dte=dte, H=Hh, tp=tp, n=len(x), per_year=round(len(x) / 10.6), L=L.mean(), T=T.mean(), win=(x > 0).mean(),
                                 avg_win=x[x > 0].mean(), avg_loss=x[x <= 0].mean(), neg_years=int((y < 0).sum()), usd=np.nanmedian(INFO[m, ci]), otm=np.nanmedian(MNY[m, ci]), price=np.median(meta.raw_price[m])))
O = pd.DataFrame(rows); O.to_pickle("gridB2.pkl"); O["mn"] = O[["L", "T"]].min(axis=1)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 100)
print(O.sort_values("mn", ascending=False).groupby("sig").head(3).round(3).to_string(index=False))
print("--- highest win rates with both periods positive")
print(O[(O.L > 0) & (O.T > 0)].sort_values("win", ascending=False).head(12).round(3).to_string(index=False))
