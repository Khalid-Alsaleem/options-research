import pandas as pd, numpy as np
P = ["Le", "Lh", "age", "kmin"]
CUT = {"h1": "2025-06-01", "h4": "2025-06-01", "daily": "2023-01-01"}
def stats(x, yrs):
    if len(x) == 0: return {}
    eq = np.cumprod(1 + 0.02 * x.sort_index().values); dd = (eq / np.maximum.accumulate(eq) - 1).min(); y = x.groupby(x.index.year).mean()
    return dict(n=len(x), per_yr=round(len(x) / yrs, 1), win=(x > 0).mean(), avg=x.mean(), avg_win=x[x > 0].mean(), avg_loss=x[x <= 0].mean(),
                worst_year=f"{y.idxmin()}: {y.min():+.1%}", losing_years=f"{int((y < 0).sum())}/{len(y)}", maxdd_2pct=dd)
out = []
for m in ("h1", "h4", "daily"):
    T = pd.read_pickle(f"trades_{m}.pkl"); T["t"] = pd.to_datetime(T.t); T = T.set_index("t").sort_index(); cut = pd.Timestamp(CUT[m])
    span_is = (cut - T.index.min()).days / 365; span_oos = (T.index.max() - cut).days / 365
    base = T[T.cost == 1]
    g = base[base.index < cut].groupby(P).net.agg(["mean", "size"]); g = g[g["size"] >= 15]
    best = g["mean"].idxmax(); print(m, "IS-chosen params", dict(zip(P, best)), "IS mean", round(g["mean"].max(), 3), "| combos with IS mean>0:", int((g["mean"] > 0).sum()), "of", len(g))
    for cost in (0, 1, 2):
        X = T[(T.cost == cost)]; sel = X[(X[P] == pd.Series(best, index=P)).all(axis=1)]
        for per, part, yrs in (("IS", sel[sel.index < cut], span_is), ("OOS", sel[sel.index >= cut], span_oos)):
            out.append(dict(mode=m, version="IS-best params", cost=cost, period=per, **stats(part.net if cost else part.gross, yrs)))
        allp = X.groupby(P)
        for per, msk, yrs in (("IS", X.index < cut, span_is), ("OOS", X.index >= cut, span_oos)):
            v = X[msk]; v = v.net if cost else v.gross
            out.append(dict(mode=m, version="avg of all 24 param sets", cost=cost, period=per, n=round(len(v) / 24), avg=v.mean(), win=(v > 0).mean()))
    # walk-forward
    wf = []
    if m == "daily": steps = [(pd.Timestamp(f"{y-4}-01-01"), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01")) for y in range(2020, 2027)]
    else: steps = [(s - pd.DateOffset(months=12), s, s + pd.DateOffset(months=3)) for s in pd.date_range("2024-12-01", "2026-09-01", freq="3MS")]
    for a, b, c in steps:
        tr = base[(base.index >= a) & (base.index < b)].groupby(P).net.agg(["mean", "size"]); tr = tr[tr["size"] >= 8]
        if not len(tr): continue
        bp = tr["mean"].idxmax(); te = base[(base.index >= b) & (base.index < c)]; te = te[(te[P] == pd.Series(bp, index=P)).all(axis=1)]; wf.append(te)
    W = pd.concat(wf) if wf else pd.DataFrame()
    if len(W): out.append(dict(mode=m, version="walk-forward", cost=1, period="rolling OOS", **stats(W.net, (W.index.max() - W.index.min()).days / 365 + 0.25)))
    for sd, nm in ((1, "calls"), (-1, "puts")):
        v = base[base.side == sd].net
        out.append(dict(mode=m, version=f"all params, {nm} only", cost=1, period="all", n=round(len(v) / 24), avg=v.mean(), win=(v > 0).mean()))
O = pd.DataFrame(out); O.to_csv("smc_results.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200); print(O.round(3).to_string(index=False))
