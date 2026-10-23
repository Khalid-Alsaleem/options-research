import pandas as pd, numpy as np, sys
def wf(T, top=1, years=range(2020, 2027)):
    T = T.copy(); T["key"] = T.school + "|" + T.flt + "|" + T.cfg; out = []
    for Y in years:
        tr = T[(T.date >= f"{Y-4}-01-01") & (T.date <= f"{Y-1}-12-31")]; te = T[(T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31")]
        for t, g in tr.groupby("ticker"):
            s = g.groupby("key").ret.agg(["mean", "size"]); s = s[(s["size"] >= 5) & (s["mean"] > 0)]
            for k in s["mean"].nlargest(top).index:
                out.append(te[(te.ticker == t) & (te.key == k)].assign(pick=k))
    return pd.concat(out)
def live(T, top=1):
    T = T.copy(); T["key"] = T.school + "|" + T.flt + "|" + T.cfg
    tr = T[T.date >= "2022-10-01"]; rows = []
    for t, g in tr.groupby("ticker"):
        s = g.groupby("key").ret.agg(["mean", "size"]); s = s[(s["size"] >= 5) & (s["mean"] > 0)]
        for r, k in enumerate(s["mean"].nlargest(top).index):
            rows.append(dict(ticker=t, rank=r + 1, pick=k, train_n=int(s.loc[k, "size"]), train_avg=s.loc[k, "mean"]))
    return pd.DataFrame(rows)
def summ(x, yrs=6.75):
    eq = np.cumprod(1 + 0.01 * x.sort_values("date").ret.values); dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(n=len(x), per_year=round(len(x) / yrs), win=round((x.ret > 0).mean(), 3), avg=round(x.ret.mean(), 4),
                avg_win=round(x.ret[x.ret > 0].mean(), 3), avg_loss=round(x.ret[x.ret <= 0].mean(), 3),
                port_ann_1pct=round(eq[-1] ** (1 / yrs) - 1, 3), maxdd_1pct=round(dd, 3),
                by_year=x.groupby(x.date.dt.year).ret.mean().round(3).to_dict())
if __name__ == "__main__":
    B = pd.read_pickle("credit6_base.pkl"); X = pd.read_pickle("credit6_x2.pkl")
    for top in (1, 2, 3):
        o = wf(B, top); print("TOP", top, summ(o))
        if top <= 2:
            ox = wf(X, top); print("  x2 costs", {k: v for k, v in summ(ox).items() if k != "by_year"})
        o.to_pickle(f"final_oos_top{top}.pkl")
    o = pd.read_pickle("final_oos_top2.pkl")
    ps = o.groupby("ticker").apply(lambda g: pd.Series(dict(n=len(g), per_year=len(g) / 6.75, win=(g.ret > 0).mean(), avg=g.ret.mean(),
          avg_win=g.ret[g.ret > 0].mean(), avg_loss=g.ret[g.ret <= 0].mean())))
    L = live(B, 2); L.to_csv("final_live_top2.csv", index=False); ps.to_csv("final_perstock_oos_top2.csv")
    print(ps.round(3).sort_values("avg", ascending=False).to_string())
    print(L.to_string())
