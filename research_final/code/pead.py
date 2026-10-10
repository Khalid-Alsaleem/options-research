"""Post-earnings drift on options: enter AFTER the earnings reaction (IV already crushed).
Reaction = close(t+1)/close(t-1)-1 around the 8-K 2.02 filing day t. Learning period only."""
import sys, numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings("ignore")
import eng as E
root, outc = sys.argv[1], sys.argv[2]
P = {k: v for k, v in E.build(root).items() if k not in E.EXCLUDE}
O = pd.read_pickle(outc); LK = {}
for (sid, call), g in O.groupby(["seg", "call"]):
    idx = P[sid]["px"].index; arr = np.full(len(idx), np.nan); pos = idx.get_indexer(g.date)
    arr[pos[pos >= 0]] = g.s30.values[pos >= 0]; LK[(sid, call)] = arr
ev = []
for sid, seg in P.items():
    px = seg["px"]; idx = px.index; c = px.close.values; v = px.volume.values
    vavg = pd.Series(v).rolling(50).mean().shift(1).values
    for d in seg["earn"]:
        t = idx.searchsorted(d)
        if t < 51 or t + 6 >= len(idx): continue
        if not px.in_index.iat[t]: continue
        reac = c[t + 1] / c[t - 1] - 1
        vr = max(v[t], v[t + 1]) / vavg[t] if vavg[t] > 0 else np.nan
        pre = c[t - 1] / c[t - 21] - 1
        for lag in (1, 2, 5):
            i = t + lag
            ev.append((sid, idx[i], lag, reac, vr, pre, LK.get((sid, True), np.full(len(idx), np.nan))[i],
                       LK.get((sid, False), np.full(len(idx), np.nan))[i]))
EV = pd.DataFrame(ev, columns=["seg", "date", "lag", "reac", "vr", "pre", "call", "put"])
EV.to_pickle("pead_events.pkl")
learn = (EV.date >= "2016-01-01") & (EV.date <= "2022-11-30")
rows = []
def st(x, d):
    x = x.dropna(); d = d[x.index]
    if len(x) < 30: return None
    wk = x.groupby(d.dt.to_period("W")).mean()
    return dict(n=len(x), r=x.mean(), t=wk.mean() / (wk.std() / np.sqrt(len(wk))), hit=(x >= .299).mean(),
                disc=x[d.dt.year <= 2019].mean(), val=x[d.dt.year > 2019].mean(),
                yrs=int((x.groupby(d.dt.year).mean() > 0).sum()))
for lag, thr, vol in itertools.product((1, 2, 5), (0.03, 0.05, 0.07, 0.10), (None, 2.0)):
    b = EV[learn & (EV.lag == lag)]
    if vol: b = b[b.vr >= vol]
    for side, m, col in (("call", b.reac >= thr, "call"), ("put", b.reac <= -thr, "put")):
        s = st(b.loc[m, col], b.loc[m, "date"])
        if s: s.update(lag=lag, thr=thr, vol=vol, side=side); rows.append(s)
R = pd.DataFrame(rows); R.to_csv("pead_learn.csv", index=False)
pd.set_option("display.width", 220); print(R.round(3).sort_values("r", ascending=False).to_string())
