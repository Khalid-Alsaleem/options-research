import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0, "."); from exits import exit_ret
def load(f, ci):
    R = pickle.load(open(f, "rb")); m = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "side", "spy_up")} for r in R])
    P = np.stack([r["paths"][ci] for r in R]); return m, P
mb, Pb = load("long_paths_base.pkl", 0)
me, Pe = load("long_paths_ext.pkl", 1)   # (0.5, 365)
for H, tp in ((120, 3.0), (120, 2.0), (80, 2.0), (60, None)):
    print("== H", H, "tp", tp)
    for nm, m, P in (("signals", me, Pe), ("baseline", mb, Pb)):
        r = exit_ret(P, H, tp, None, False)
        for sig in m.sig.unique():
            k = (m.sig == sig).values & (m.side == 1).values & m.spy_up.values & np.isfinite(r)
            x = pd.Series(r[k], index=m.date[k])
            L, T = x[x.index < "2023-01-01"], x[x.index >= "2023-01-01"]
            yr = {k: round(float(v), 2) for k, v in x.groupby(x.index.year).mean().items()}
            print(f"{sig:14s} L {len(L):5d} avg {L.mean():+.3f} win {(L>0).mean():.2f} | T {len(T):5d} avg {T.mean():+.3f} win {(T>0).mean():.2f} | {yr}")
