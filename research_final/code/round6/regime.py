import pickle, numpy as np, pandas as pd
from exits import exit_ret
spy = pd.read_csv("/home/claude/options-research/data/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
below = (spy < spy.rolling(200).mean()); cal = spy.index
def load(f, ci):
    R = pickle.load(open(f, "rb")); m = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "side", "spy_up")} for r in R])
    return m, np.stack([r["paths"][ci] for r in R]), np.stack([r["stock"] for r in R])
def apply(P, S, m, H, tp, regime, sstop):
    P = P[:, :H].copy(); N = len(P)
    cut = np.full(N, H)
    if regime:
        pos = cal.searchsorted(m.date.values); bv = below.values
        for k in range(N):
            w = bv[pos[k]:pos[k] + H]; f = np.argmax(w) if w.any() else H
            cut[k] = min(cut[k], f + 1)
    if sstop is not None:
        hit = S[:, :H] <= sstop; f = np.where(hit.any(1), np.argmax(hit, 1) + 1, H); cut = np.minimum(cut, f)
    for k in range(N):
        if cut[k] < H: P[k, cut[k]:] = np.nan
    return exit_ret(P, H, tp, None, False)
out = []
for f, ci in (("long_paths_ext.pkl", 1), ("long_paths_base.pkl", 0)):
    m, P, S = load(f, ci)
    for sig in m.sig.unique():
        k = ((m.sig == sig) & (m.side == 1) & m.spy_up).values
        mm, PP, SS = m[k].reset_index(drop=True), P[k], S[k]
        for H, tp in ((120, 3.0), (120, 2.0)):
            for regime in (False, True):
                for sstop in (None, -0.10, -0.15):
                    r = apply(PP, SS, mm, H, tp, regime, sstop); ok = np.isfinite(r); x = pd.Series(r[ok], index=mm.date[ok])
                    L, T = x[x.index < "2023"], x[x.index >= "2023"]
                    out.append(dict(sig=sig, H=H, tp=tp, regime=regime, sstop=sstop, L_n=len(L), L_avg=L.mean(), L_win=(L > 0).mean(), L_aw=L[L > 0].mean(),
                                    T_n=len(T), T_avg=T.mean(), T_win=(T > 0).mean(), T_aw=T[T > 0].mean(), y2022=x[x.index.year == 2022].mean(), y2018=x[x.index.year == 2018].mean()))
O = pd.DataFrame(out); O.to_pickle("regime.pkl"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(O[O.sig.isin(["strong_dip", "trend_only", "baseline", "breakout55", "high52_volume"])].round(3).to_string(index=False))
