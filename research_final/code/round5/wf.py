"""Walk-forward per stock: each test year Y, pick params on Y-4..Y-1 (score = sum of R, needs >=6 trades,
with plateau smoothing over neighbouring L), trade year Y. Compare with ONE shared param set chosen on pooled training."""
import sys, numpy as np, pandas as pd, itertools
sys.path.insert(0, sys.argv[1]); import msc as M
TICK = sys.argv[2].split(","); YEARS = range(2020, 2027)
combos = list(itertools.product(M.GRID["L"], M.GRID["R"], M.GRID["trend"]))
PX = {t: M.load(t) for t in TICK}
def score(t, cb, a, b):
    T = M.backtest(PX[t], *cb, start=a, end=b); return (T.R.sum() if len(T) >= 6 else -99), T
rows = []; chosen = []
for Y in YEARS:
    a, b = f"{Y-4}-01-01", f"{Y-1}-12-31"
    S = {t: {cb: score(t, cb, a, b)[0] for cb in combos} for t in TICK}
    # plateau: average score with neighbouring L values (same R, trend)
    Ls = M.GRID["L"]
    def smooth(s, cb):
        k = Ls.index(cb[0]); nb = [Ls[x] for x in (k - 1, k, k + 1) if 0 <= x < len(Ls)]
        return np.mean([s[(x, cb[1], cb[2])] for x in nb])
    shared = max(combos, key=lambda cb: sum(smooth(S[t], cb) for t in TICK))
    for t in TICK:
        own = max(combos, key=lambda cb: smooth(S[t], cb))
        for mode, cb in (("own", own), ("shared", shared)):
            T = M.backtest(PX[t], *cb, start=f"{Y}-01-01", end=f"{Y}-12-31"); T["ticker"] = t; T["mode"] = mode; T["year"] = Y
            rows.append(T); chosen.append(dict(year=Y, ticker=t, mode=mode, L=cb[0], R=cb[1], trend=cb[2]))
T = pd.concat(rows); C = pd.DataFrame(chosen)
T.to_pickle("wf_trades.pkl"); C.to_csv("wf_params.csv", index=False)
yrs = 2026.75 - 2020
for (t, mode), g in T.groupby(["ticker", "mode"]):
    print(t, mode, {k: round(v, 3) for k, v in M.stats(g, yrs).items()})
for t in TICK:
    px = PX[t]; bh = px.c.loc["2020-01-01":].iloc[-1] / px.c.loc["2020-01-01":].iloc[0]
    print(t, "buy&hold 2020-2026 cagr", round(bh ** (1 / yrs) - 1, 3))
print(C.pivot_table(index=["ticker", "mode"], columns="year", values="L", aggfunc="first"))
print(C.pivot_table(index=["ticker", "mode"], columns="year", values="R", aggfunc="first"))
