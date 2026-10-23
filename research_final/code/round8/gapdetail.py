import pickle, numpy as np, pandas as pd, sys
from exits import exit_ret
from stageB import CON
R = pickle.load(open(sys.argv[1], "rb"))
meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "spy_up")} for r in R]); PA = np.stack([r["paths"] for r in R]); del R
def show(sig, flt, dte, mny, H, tp):
    ci = CON.index((dte, mny)); r = exit_ret(PA[:, ci, :], H, tp, None, False)
    m = (meta.sig == sig).values & np.isfinite(r) & (meta.spy_up.values if flt == "aligned" else True)
    x = pd.Series(r[m], index=meta.date[m]); mo = x.groupby(x.index.to_period("M")).mean()
    t = mo.mean() / (mo.std() / np.sqrt(len(mo)))
    yr = {k: f"{v:+.0%}({c})" for (k, v), c in zip(x.groupby(x.index.year).mean().items(), x.groupby(x.index.year).size())}
    print(f"{sig:10s} {flt:7s} dte{dte} m{mny:+.2f} H{H} tp{tp}: n {len(x)} avg {x.mean():+.3f} win {(x>0).mean():.2f} med {x.median():+.2f} t_month {t:.2f} | {yr}")
for sig in ("gap5", "gap3", "gap2_vol", "gap2", "gap2_trend"):
    for flt in ("none", "aligned"):
        show(sig, flt, 180, -0.1, 120, None)
print("--- neighbours of gap3 none")
for dte in (90, 180, 365):
    for mny in (-0.1, 0.0, 0.1):
        for H in (60, 120):
            show("gap3", "none", dte, mny, H, None)
