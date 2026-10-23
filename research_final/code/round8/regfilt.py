import pickle, numpy as np, pandas as pd, sys
from exits import exit_ret
from stageB import CON
R = pickle.load(open("stageB_base.pkl", "rb")); G = pd.read_pickle("regimes.pkl")
meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "spy_up")} for r in R]); PA = np.stack([r["paths"] for r in R]); del R
prev = G.shift(1).reindex(meta.date.values)   # regime known at the signal day's close (entry is next open)
F = {"none": np.ones(len(meta), bool), "spy200": meta.spy_up.values,
     "vix<20": (prev.vix < 20).values, "vix<vixma50": (prev.vix < prev.vix_ma50).values,
     "breadth>0.6": (prev.breadth > 0.6).values, "breadth_rising": (prev.breadth > prev.breadth_ma20).values,
     "spy50>200": (prev.spy50 > prev.spy200).values, "breadth>0.5&vix<vixma50": ((prev.breadth > 0.5) & (prev.vix < prev.vix_ma50)).values}
rows = []
for sig, ct, tp in (("strong_dip", (365, 0.1), 2.0), ("rsi2_10", (365, 0.1), 2.0), ("trend_day", (365, 0.1), 2.0), ("high52", (365, 0.1), 2.0),
                    ("gap3", (180, -0.1), None), ("gap2_vol", (180, -0.1), None), ("gap2", (180, -0.1), None), ("strong_dip", (365, -0.1), None), ("gap3", (365, -0.1), None)):
    r = exit_ret(PA[:, CON.index(ct), :], 120, tp, None, False)
    for fn, fm in F.items():
        m = (meta.sig == sig).values & np.isfinite(r) & fm
        x = pd.Series(r[m], index=meta.date[m]); y = x.groupby(x.index.year).mean()
        L, T = x[x.index < "2023"], x[x.index >= "2023"]
        rows.append(dict(sig=sig, con=ct, tp=tp, flt=fn, n=len(x), L=L.mean(), T=T.mean(), win=(x > 0).mean(), neg_years=int((y < 0).sum()), worst=y.min(), y2021=y.get(2021), y2022=y.get(2022), y2024=y.get(2024)))
O = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100); print(O.round(3).to_string(index=False))
