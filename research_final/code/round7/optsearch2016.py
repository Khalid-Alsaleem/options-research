"""Automated 'change one element at a time' search, all combinations at once.
Elements: direction filter (which moving averages), school (MSC/Elliott/breakout/MA cross/MACD/RSI2 pullback),
where the school reads (option chart vs stock chart), contract (days to expiry at entry, strike moneyness),
exit (take-profit level, exit on school signal or hold to 7 days before expiry). Calls when filter says up, puts when down."""
import sys, numpy as np, pandas as pd, pickle, itertools
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"
sys.path.insert(0, A + "/auto"); sys.path.insert(0, A + "/msc")
import single as SG, eng as E, schools as S
B = np.load(A + "/auto/spread_coef.npy")
TICK = ["AAPL", "GOOGL", "MSFT", "BRK.B", "XOM", "AMZN", "META", "JNJ", "GE", "WFC", "JPM", "T", "PG", "WMT", "VZ", "CVX", "PFE", "KO", "ORCL", "DIS"]
SCH = [("msc", (2,)), ("msc", (3,)), ("msc", (5,)), ("elliott", (3,)), ("elliott", (5,)), ("donch", (20, 10)), ("ma", (10, 30)), ("macd", ()), ("pullback", (10, 90))]
FILTERS = ["ma20", "ma50", "ma100", "ma200", "ma50_100_200", "ma20_50"]
DTES = (30, 60, 90); MNY = (-0.05, 0.0, 0.05)
def bs(S_, K, T, iv, call): return SG.bs(S_, K, T, SG.ivk(iv, S_, K, call), call)
def run(t):
    d = pd.read_csv(f"/home/claude/options-research/data/eod/{t}.csv.gz", parse_dates=["date"]).set_index("date")
    d = d[~d.index.duplicated()].sort_index(); f = (d.adjusted_close / d.close).values
    lr = np.log(d.adjusted_close).diff(); iv = (1.2 * (0.5 * lr.rolling(60).std() + 0.5 * lr.rolling(252, min_periods=120).std()) * np.sqrt(252)).clip(0.1, 1.5).values
    dv = (d.close * d.volume).rolling(20).mean(); rs = np.exp(B[0] + B[1] * np.log(dv) + B[2] * np.log(d.close)).clip(0.005, 0.25).values
    ac = d.adjusted_close; M = {n: ac.rolling(n).mean() for n in (20, 50, 100, 200)}
    FL = {"ma20": (ac > M[20], ac < M[20]), "ma50": (ac > M[50], ac < M[50]), "ma100": (ac > M[100], ac < M[100]), "ma200": (ac > M[200], ac < M[200]),
          "ma50_100_200": ((ac > M[50]) & (ac > M[100]) & (ac > M[200]), (ac < M[50]) & (ac < M[100]) & (ac < M[200])),
          "ma20_50": ((ac > M[20]) & (M[20] > M[50]), (ac < M[20]) & (M[20] < M[50]))}
    FL = {k: (u.values, v.values) for k, (u, v) in FL.items()}
    adj = pd.DataFrame({"o": d.open * f, "h": d.high * f, "l": d.low * f, "c": d.adjusted_close}, index=d.index)
    ssig = {f"{a}{p}": S.setups(adj, a, p) for a, p in SCH}
    idx = d.index; ro, rh, rl, rc = d.open.values, d.high.values, d.low.values, d.close.values; split = np.r_[False, np.abs(np.diff(f) / f[:-1]) > 0.02]
    out = []
    for ex in E.EXP:
        if ex < pd.Timestamp("2016-03-01") or ex > pd.Timestamp("2026-09-30"): continue
        for dte in DTES:
            a = idx.searchsorted(ex - pd.Timedelta(days=dte)); b = idx.searchsorted(ex - pd.Timedelta(days=7))
            if a < 260 or b >= len(idx) or b - a < 10 or split[a:b + 1].any(): continue
            js = np.arange(a, b + 1); T = (ex - idx[js]).days.values / 365; v = iv[js - 1]; n = len(js)
            for side in (1, -1):
                call = side == 1
                for mny in MNY:
                    st = E.step(ro[a]); K = np.round(ro[a] * (1 + side * mny) / st) * st
                    hi_s, lo_s = (rh[js], rl[js]) if call else (rl[js], rh[js])
                    oc = pd.DataFrame({"o": bs(ro[js], K, T, v, call), "h": bs(hi_s, K, T, v, call), "l": bs(lo_s, K, T, v, call), "c": bs(rc[js], K, T, v, call)}, index=idx[js])
                    sp = lambda x: np.maximum(rs[a] * x, 0.05)
                    hb = oc.h.values - sp(oc.h.values) / 2; ob = oc.o.values - sp(oc.o.values) / 2; cb = oc.c.values - sp(oc.c.values) / 2
                    for (fa, p) in SCH:
                        nm = f"{fa}{p}"; osg = S.setups(oc, fa, p)[1]; ss = ssig[nm][side]
                        for src, ent, ext in (("option", osg[0], osg[1]), ("stock", ss[0][js], ss[1][js])):
                            ent = np.asarray(ent, bool); ext = np.asarray(ext, bool)
                            for flt in FILTERS:
                                reg = FL[flt][0 if call else 1][js]; i = 0
                                while i < n - 2:
                                    if not (ent[i] and reg[i]): i += 1; continue
                                    e = i + 1; cost = oc.o.iat[e] + sp(oc.o.iat[e]) / 2
                                    if cost < 0.3: i += 1; continue
                                    w = np.flatnonzero(ext[e:n - 1]); ks = e + w[0] if len(w) else n - 1
                                    out.append(dict(t=t, date=idx[js[e]], side=side, dte=dte, mny=mny, sch=nm, src=src, flt=flt,
                                                    hb=(hb[e:] / cost - 1).astype(np.float32), cb=(cb[e:] / cost - 1).astype(np.float32),
                                                    ob=(np.r_[ob[e + 1:], cb[-1]] / cost - 1).astype(np.float32), ks=ks - e))
                                    i = ks + 1
    return out
if __name__ == "__main__":
    with Pool(2) as pool: rows = sum(pool.map(run, TICK), [])
    pickle.dump(rows, open(A + "/cheap/optsearch_raw_2016.pkl", "wb")); print(len(rows))
