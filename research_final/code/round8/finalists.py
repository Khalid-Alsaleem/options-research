import pickle, numpy as np, pandas as pd
from exits import exit_ret
from stageB import CON
G = pd.read_pickle("regimes.pkl")
FIN = {"F1 strong_dip": ("strong_dip", "spy50>200", (365, 0.1), 2.0), "F2 gap3": ("gap3", "spy50>200", (365, -0.1), None),
       "F3 rsi2_10": ("rsi2_10", "spy50>200", (365, 0.1), 2.0), "F4 high52": ("high52", "spy50>200", (365, 0.1), 2.0),
       "F5 trend_day": ("trend_day", "spy50>200", (365, 0.1), 2.0)}
def load(f):
    R = pickle.load(open(f, "rb")); meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "spy_up")} for r in R])
    return meta, np.stack([r["paths"] for r in R]), np.stack([r["cost_pct"] for r in R])
out = {}
for tag, f in (("base", "stageB_base.pkl"), ("x2", "stageB_x2.pkl")):
    meta, PA, CP = load(f); prev = G.shift(1).reindex(meta.date.values); flt = {"spy50>200": (prev.spy50 > prev.spy200).values}
    for k, (sig, fl, ct, tp) in FIN.items():
        ci = CON.index(ct); r = exit_ret(PA[:, ci, :], 120, tp, None, False)
        m = (meta.sig == sig).values & np.isfinite(r) & flt[fl]
        x = pd.Series(r[m], index=meta.date[m]); mo = x.groupby(x.index.to_period("M")).mean()
        L, T = x[x.index < "2023"], x[x.index >= "2023"]; y = x.groupby(x.index.year).mean()
        out[(k, tag)] = dict(n=len(x), per_year=round(len(x) / 10.6), L=L.mean(), T=T.mean(), win=(x > 0).mean(), avg_win=x[x > 0].mean(), avg_loss=x[x <= 0].mean(),
                            t=mo.mean() / (mo.std() / np.sqrt(len(mo))), neg_years=int((y < 0).sum()), cost_pct=float(np.median(CP[m, ci])),
                            years=" ".join(f"{k2}:{v:+.0%}" for k2, v in y.items()))
        if tag == "base": x.to_frame("ret").assign(seg=meta.seg[m].values).to_pickle(f"fin_{k.split()[0]}.pkl")
O = pd.DataFrame(out).T; pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 200); print(O.to_string())
