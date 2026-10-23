import pickle, numpy as np, pandas as pd
R = pickle.load(open("long_paths_ext.pkl", "rb"))
m = pd.DataFrame([{k: r[k] for k in ("seg", "date", "sig", "side", "spy_up")} for r in R]); P = np.stack([r["paths"][1] for r in R])
k = ((m.sig == "strong_dip") & (m.side == 1) & m.spy_up).values; m = m[k].reset_index(drop=True); P = P[k]
H, TP = 120, 2.0
ret, hold = [], []
for p in P:
    p = p[:H]; v = np.flatnonzero(np.isfinite(p))
    if len(v) == 0: ret.append(np.nan); hold.append(0); continue
    h = np.flatnonzero(p[v] >= TP); j = v[h[0]] if len(h) else v[-1]
    ret.append(p[j]); hold.append(j + 1)
m["ret"] = ret; m["hold"] = hold; m = m.dropna(subset=["ret"]).sort_values("date")
m["exit"] = m.date + pd.to_timedelta((m.hold * 1.45).round(), "D")
print("trades", len(m), "per year", round(len(m) / 10.6), "avg hold days", m.hold.mean().round(1))
for w, mx in ((0.02, 30), (0.03, 25), (0.05, 15)):
    eq = 1.0; cash = 1.0; open_ = []; curve = []
    for _, t in m.iterrows():
        still = []
        for (ex, amt, r) in open_:
            if ex <= t.date: cash += amt * (1 + r)
            else: still.append((ex, amt, r))
        open_ = still
        eq = cash + sum(a for _, a, _ in open_)  # mark open at cost (conservative proxy)
        curve.append((t.date, cash + sum(a * (1 + r) * 0 + a for _, a, r in open_)))
        if len(open_) < mx:
            amt = w * eq
            if cash >= amt: cash -= amt; open_.append((t.exit, amt, t.ret))
    for (ex, amt, r) in open_: cash += amt * (1 + r)
    C = pd.Series(dict(curve)); yrs = (m.date.max() - m.date.min()).days / 365
    yr = C.groupby(C.index.year).last(); yr = yr.pct_change().fillna(yr.iloc[0] - 1)
    print(f"w={w} max={mx}: final x{cash:.2f} CAGR {cash ** (1 / yrs) - 1:+.1%} maxDD {(C / C.cummax() - 1).min():.1%}", {k: f"{v:+.0%}" for k, v in yr.items()})
