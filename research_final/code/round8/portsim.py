import pickle, numpy as np, pandas as pd
from stageB import CON
G = pd.read_pickle("regimes.pkl")
def trades(fpkl, sig, ci, H, tp, flt, cost_key=None):
    R = pickle.load(open(fpkl, "rb")); meta = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig")} for r in R]); PA = np.stack([r["paths"][ci] for r in R])
    prev = G.shift(1).reindex(meta.date.values); m = (meta.sig == sig).values & ((prev.spy50 > prev.spy200).values if flt else True)
    out = []
    for k in np.flatnonzero(m):
        p = PA[k, :H]; v = np.flatnonzero(np.isfinite(p))
        if not len(v): continue
        h = np.flatnonzero(p[v] >= tp) if tp else []
        j = v[h[0]] if len(h) else v[-1]
        out.append((meta.date[k], meta.seg[k], float(p[j]), int(j + 1)))
    T = pd.DataFrame(out, columns=["date", "seg", "ret", "hold"]).sort_values("date"); T["exit"] = T.date + pd.to_timedelta((T.hold * 1.45).round(), "D"); return T
def sim(T, w=0.02, mx=30, one_per_stock=True):
    cash = 1.0; opn = []; curve = {}
    for _, t in T.iterrows():
        keep = []
        for x in opn:
            if x[0] <= t.date: cash += x[1] * (1 + x[2])
            else: keep.append(x)
        opn = keep; eq = cash + sum(x[1] for x in opn); curve[t.date] = eq
        if len(opn) < mx and not (one_per_stock and any(x[3] == t.seg for x in opn)):
            a = w * eq
            if cash >= a: cash -= a; opn.append((t.exit, a, t.ret, t.seg))
    for x in opn: cash += x[1] * (1 + x[2])
    C = pd.Series(curve); yrs = (T.date.max() - T.date.min()).days / 365
    yr = C.groupby(C.index.year).last(); yr = yr / yr.shift(1).fillna(1.0) - 1
    return cash, cash ** (1 / yrs) - 1, (C / C.cummax() - 1).min(), yr
for name, args in (("F4 high52 1y OTM10 tp200", ("stageB_base.pkl", "high52", CON.index((365, 0.1)), 120, 2.0, True)),
                   ("F1 strong_dip 1y OTM10 tp200", ("stageB_base.pkl", "strong_dip", CON.index((365, 0.1)), 120, 2.0, True)),
                   ("F2 gap3 1y ITM10 hold", ("stageB_base.pkl", "gap3", CON.index((365, -0.1)), 120, None, True)),
                   ("F4 high52 $2-5 1y hold", ("stageB2.pkl", "high52", 2, 120, None, True))):
    T = trades(*args)
    for w, mx in ((0.02, 30), (0.04, 15)):
        fin, cagr, dd, yr = sim(T, w, mx)
        print(f"{name:32s} w={w} max={mx}: x{fin:.2f} CAGR {cagr:+.1%} maxDD {dd:.1%} | " + " ".join(f"{k}:{v:+.0%}" for k, v in yr.items()))
