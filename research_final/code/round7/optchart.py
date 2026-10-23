"""Khalid's idea: direction from the stock (above MA50/100/200 -> calls, below all three -> puts),
timing from a chart-reading school applied to the OPTION CONTRACT's own daily chart (MSC / Elliott / breakout / MA cross).
Compared against the same school read on the STOCK chart, same contracts.
Contracts: each monthly expiry, strike at-the-money when it has ~60 days left; tracked until 7 days before expiry.
Option bars are synthetic (Black-Scholes on the stock's daily open/high/low/close with the calibrated IV model)."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"
sys.path.insert(0, A + "/auto"); sys.path.insert(0, A + "/msc")
import single as SG, eng as E, schools as S
B = np.load(A + "/auto/spread_coef.npy")
TICK = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK.B", "LLY", "JPM", "WMT", "V", "ORCL", "MA", "NFLX", "XOM", "COST", "JNJ", "HD"]
SCH = [("msc", (2,)), ("msc", (3,)), ("elliott", (3,)), ("donch", (20, 10)), ("ma", (10, 30))]
def bs(S_, K, T, iv, call): return SG.bs(S_, K, T, SG.ivk(iv, S_, K, call), call)
def run(t):
    d = pd.read_csv(f"/home/claude/options-research/data/eod/{t}.csv.gz", parse_dates=["date"]).set_index("date")
    d = d[~d.index.duplicated()].sort_index(); f = (d.adjusted_close / d.close).values
    lr = np.log(d.adjusted_close).diff(); iv = (1.2 * (0.5 * lr.rolling(60).std() + 0.5 * lr.rolling(252, min_periods=120).std()) * np.sqrt(252)).clip(0.1, 1.5).values
    dv = (d.close * d.volume).rolling(20).mean(); rs = np.exp(B[0] + B[1] * np.log(dv) + B[2] * np.log(d.close)).clip(0.005, 0.25).values
    ac = d.adjusted_close; up = ((ac > ac.rolling(50).mean()) & (ac > ac.rolling(100).mean()) & (ac > ac.rolling(200).mean())).values
    dn = ((ac < ac.rolling(50).mean()) & (ac < ac.rolling(100).mean()) & (ac < ac.rolling(200).mean())).values
    adj = pd.DataFrame({"o": d.open * f, "h": d.high * f, "l": d.low * f, "c": d.adjusted_close}, index=d.index)
    stock_sig = {f"{a}{p}": S.setups(adj, a, p) for a, p in SCH}
    idx = d.index; ro, rh, rl, rc = d.open.values, d.high.values, d.low.values, d.close.values; split = np.r_[False, np.abs(np.diff(f) / f[:-1]) > 0.02]
    out = []
    for ex in E.EXP:
        if ex < pd.Timestamp("2016-03-01") or ex > pd.Timestamp("2026-09-30"): continue
        a = idx.searchsorted(ex - pd.Timedelta(days=60)); b = idx.searchsorted(ex - pd.Timedelta(days=7))
        if a < 260 or b >= len(idx) or b - a < 20 or split[a:b + 1].any(): continue
        for side in (1, -1):
            call = side == 1; K = np.round(ro[a] / E.step(ro[a])) * E.step(ro[a]); js = np.arange(a, b + 1)
            T = (ex - idx[js]).days.values / 365; v = iv[js - 1]
            hi_s, lo_s = (rh[js], rl[js]) if call else (rl[js], rh[js])
            oc = pd.DataFrame({"o": bs(ro[js], K, T, v, call), "h": bs(hi_s, K, T, v, call), "l": bs(lo_s, K, T, v, call), "c": bs(rc[js], K, T, v, call)}, index=idx[js])
            if (oc.c < 0.5).all(): continue
            regime = (up if call else dn)[js]
            for nm, (fa, p) in zip([f"{a_}{p_}" for a_, p_ in SCH], SCH):
                # option-chart signals: an 'up' signal on the option chart = option price breaking up
                L = p[0] if fa in ("msc", "elliott") else None
                osg = S.setups(oc, fa, p)[1]
                ssg = stock_sig[nm][side]; s_ent = ssg[0][js]; s_ext = ssg[1][js]
                for src, ent, ext in (("option_chart", osg[0], osg[1]), ("stock_chart", s_ent, s_ext)):
                    i = 0; n = len(js)
                    while i < n - 2:
                        if not (ent[i] and regime[i]): i += 1; continue
                        e = i + 1; sp = lambda x: np.maximum(rs[js[i]] * x, 0.05)
                        cost = oc.o.iat[e] + sp(oc.o.iat[e]) / 2
                        if cost < 0.3: i += 1; continue
                        k = e; why = "expiry7d"
                        while k < n - 1:
                            if (oc.h.iat[k] - sp(oc.h.iat[k]) / 2) >= cost * 1.5: why = "tp50"; break
                            if ext[k]: why = "signal"; break
                            k += 1
                        val = cost * 1.5 if why == "tp50" else (oc.o.iat[k + 1] if why == "signal" and k + 1 < n else oc.c.iat[k])
                        if why != "tp50": val = val - sp(val) / 2
                        out.append((t, idx[js[e]], side, nm, src, val / cost - 1, why, k - e + 1, round(cost * 100)))
                        i = k + 1
    return out
if __name__ == "__main__":
    with Pool(2) as pool: rows = sum(pool.map(run, TICK), [])
    T = pd.DataFrame(rows, columns=["ticker", "date", "side", "school", "src", "ret", "why", "days", "cost_usd"])
    T.to_pickle(A + "/cheap/optchart.pkl")
    T["per"] = np.where(T.date < "2023-01-01", "2016-22", "2023-26")
    g = T.groupby(["src", "school", "side", "per"]).ret.agg(n="size", win=lambda r: (r > 0).mean(), avg="mean", avg_win=lambda r: r[r > 0].mean(), avg_loss=lambda r: r[r <= 0].mean())
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); print(g.round(3).to_string())
