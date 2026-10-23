"""Per-stock walk-forward over 6 schools. For each test year Y (2020..2026) and each stock: choose the strategy
(school+settings, direction mode, profit-taking rule) with the highest SUM of option returns on trades entered in Y-4..Y-1
(>= 6 trades, and score > 0, else stay out that year); trade it during Y. Also a 'shared' baseline: one strategy for all stocks."""
import sys, numpy as np, pandas as pd, pickle, itertools
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1]); import schools as S
TICK = sys.argv[2].split(",")
MODES = {"long": (1,), "short": (-1,), "both": (1, -1)}; COLS = {"none": "opt", "tp50": "opt_tp50", "tp100": "opt_tp100"}
def build(t):
    try: px = S.load(t)
    except Exception as e: return t, None
    out = []
    for fam, p in S.FAMS:
        for side in (1, -1):
            T = S.trades(px, fam, p, side)
            if len(T): T["fam"] = fam; T["p"] = str(p); out.append(T)
    return t, pd.concat(out) if out else None
if __name__ == "__main__":
    with Pool(2) as pool: TR = dict(pool.map(build, TICK))
    TR = {t: v for t, v in TR.items() if v is not None}; pickle.dump(TR, open("tr6.pkl", "wb"))
    STRATS = [(f, str(p), m, x) for (f, p) in S.FAMS for m in MODES for x in COLS]
    def pick(T, a, b, strats):
        w = T[(T.entry >= a) & (T.entry <= b)]; best, bs = None, 0
        G = {k: g for k, g in w.groupby(["fam", "p", "side"])}
        for f, p, m, x in strats:
            parts = [G[(f, p, s)][COLS[x]] for s in MODES[m] if (f, p, s) in G]
            if not parts: continue
            r = pd.concat(parts)
            if len(r) >= 6 and r.sum() > bs: best, bs = (f, p, m, x), r.sum()
        return best
    def take(T, st, a, b):
        f, p, m, x = st; w = T[(T.entry >= a) & (T.entry <= b) & (T.fam == f) & (T.p == p) & T.side.isin(MODES[m])].copy()
        w["ret"] = w[COLS[x]]; w["strategy"] = f"{f}{p}|{m}|{x}"; return w
    rows = []; picks = []
    for Y in range(2020, 2027):
        a, b = pd.Timestamp(f"{Y-4}-01-01"), pd.Timestamp(f"{Y-1}-12-31"); ya, yb = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y}-12-31")
        # shared: strategy with best total over all stocks
        tot = {}
        for t, T in TR.items():
            w = T[(T.entry >= a) & (T.entry <= b)]
            for (f, p, s), g in w.groupby(["fam", "p", "side"]):
                for m, sides in MODES.items():
                    if s in sides:
                        for x, col in COLS.items(): tot[(f, p, m, x)] = tot.get((f, p, m, x), 0) + g[col].sum()
        shared = max(tot, key=tot.get)
        for t, T in TR.items():
            own = pick(T, a, b, STRATS)
            picks.append(dict(year=Y, ticker=t, own="|".join(own) if own else "OUT", shared="|".join(shared)))
            if own: w = take(T, own, ya, yb); w["ticker"] = t; w["mode_"] = "own"; w["year"] = Y; rows.append(w)
            w = take(T, shared, ya, yb); w["ticker"] = t; w["mode_"] = "shared"; w["year"] = Y; rows.append(w)
    O = pd.concat(rows); O.to_pickle("wf6_oos.pkl"); pd.DataFrame(picks).to_csv("wf6_picks.csv", index=False)
    # live pick: train on the most recent 4 years
    a, b = pd.Timestamp("2022-10-01"), pd.Timestamp("2026-10-02"); live = []
    for t, T in TR.items():
        st = pick(T, a, b, STRATS); w = take(T, st, a, b) if st else None
        live.append(dict(ticker=t, live="|".join(st) if st else "OUT", train_n=len(w) if st else 0, train_avg=w.ret.mean() if st else np.nan))
    pd.DataFrame(live).to_csv("wf6_live.csv", index=False)
    yrs = 2026.75 - 2020
    for m, g in O.groupby("mode_"):
        print(m, "n", len(g), "per yr", round(len(g) / yrs), "win", round((g.ret > 0).mean(), 3), "avg", round(g.ret.mean(), 3),
              "avg_win", round(g.ret[g.ret > 0].mean(), 3), "avg_loss", round(g.ret[g.ret <= 0].mean(), 3))
        print("  by year", g.groupby("year").ret.mean().round(3).to_dict())
