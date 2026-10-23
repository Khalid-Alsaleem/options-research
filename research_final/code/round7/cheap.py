"""Cheap-contract search for the 20 largest US companies.
Buy ONE call (up signal) or ONE put (down signal) priced $2-5 per share ($200-500 per contract).
Take profit +30% with a resting limit order (filled if the option's bid reaches +30% at the day's high/low).
Signals read on split-adjusted charts; option prices use the real (unadjusted) stock price of that day.
Pricing: Black-Scholes, IV = 1.2*(0.5*HV60+0.5*HV252) + put skew; spread = max(model spread, $0.05)."""
import sys, numpy as np, pandas as pd, pickle
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"
sys.path.insert(0, A + "/auto"); sys.path.insert(0, A + "/msc")
import single as SG, eng as E, schools as S
TICK = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK.B", "LLY", "JPM", "WMT", "V", "ORCL", "MA", "NFLX", "XOM", "COST", "JNJ", "HD"]
B = np.load(A + "/auto/spread_coef.npy")
FRI = pd.date_range("2015-01-02", "2027-12-31", freq="W-FRI")
DTES = (7, 14, 30, 60); HOLDS = (3, 5, 10, 20, 999); TP = 0.30
EARN = pd.read_csv("/home/claude/options-research/data/earnings/edgar_8k_item202.csv", parse_dates=["filing_date"])
spy = pd.read_csv("/home/claude/options-research/data/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
def bs(S_, K, T, iv, call): return SG.bs(S_, K, T, SG.ivk(iv, S_, K, call), call)
def run(t):
    d = pd.read_csv(f"/home/claude/options-research/data/eod/{t}.csv.gz", parse_dates=["date"]).set_index("date")
    d = d[~d.index.duplicated()].sort_index(); f = (d.adjusted_close / d.close).values
    adj = pd.DataFrame({"o": d.open * f, "h": d.high * f, "l": d.low * f, "c": d.adjusted_close}, index=d.index)
    lr = np.log(adj.c).diff(); iv = (1.2 * (0.5 * lr.rolling(60).std() + 0.5 * lr.rolling(252, min_periods=120).std()) * np.sqrt(252)).clip(0.1, 1.5).values
    dv = (d.close * d.volume).rolling(20).mean(); rs = np.exp(B[0] + B[1] * np.log(dv) + B[2] * np.log(d.close)).clip(0.005, 0.25).values
    ro, rh, rl, rc = d.open.values, d.high.values, d.low.values, d.close.values; idx = d.index; n = len(d)
    split = np.r_[False, np.abs(np.diff(f) / f[:-1]) > 0.02]
    eb = np.zeros(n, bool)
    for e in EARN[EARN.ticker_raw == t].filing_date:
        k = idx.searchsorted(e); eb[max(0, k - 10):k + 1] = True
    s = spy.reindex(idx).ffill(); spyu = (s > s.rolling(200).mean()).values
    c = adj.c; ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    sig = {f"{fa}{p}": S.setups(adj, fa, p) for fa, p in S.FAMS}
    sig = {k: (v[1][0], v[-1][0]) for k, v in sig.items()}
    rs63 = c.pct_change(63) - s.pct_change(63); r2 = S.rsi(c, 2).values
    sig["strong_dip"] = (((c > ma50) & (ma50 > ma200) & (rs63 > 0)).values & (r2 < 10), ((c < ma50) & (ma50 < ma200) & (rs63 < 0)).values & (r2 > 90))
    sig["trend_day"] = (((c > ma50) & (ma50 > ma200)).values, ((c < ma50) & (ma50 < ma200)).values)
    out = []
    for name, (up, dn) in sig.items():
        for side, m in ((1, up), (-1, dn)):
            last = -99
            for i in np.flatnonzero(np.asarray(m, bool)):
                if i < 252 or i + 2 >= n or i - last < 5 or eb[i] or idx[i] < pd.Timestamp("2016-01-01") or not np.isfinite(iv[i]): continue
                last = i; e = i + 1; S0 = ro[e]; call = side == 1; st = E.step(S0)
                for dte in DTES:
                    ex = FRI[FRI.searchsorted(idx[e] + pd.Timedelta(days=dte))]
                    T0 = (ex - idx[e]).days / 365
                    base = np.round(S0 / st) * st
                    ks = base + side * st * np.arange(0, 60)
                    pr = bs(S0, ks, T0, iv[i], call)
                    ok = (pr >= 2) & (pr <= 5)
                    if not ok.any(): continue
                    K = ks[ok][np.argmin(np.abs(pr[ok] - 3.5))]
                    sprd = lambda p, S_: np.maximum(rs[i] * p, 0.05)
                    p0 = bs(S0, K, T0, iv[i], call); cost = p0 + sprd(p0, S0) / 2
                    js = np.arange(e, n); js = js[idx[js] < ex]
                    js = js[:40]
                    if len(js) == 0 or split[js].any(): continue
                    Tj = (ex - idx[js]).days.values / 365
                    best = rh[js] if call else rl[js]
                    vb = bs(best, K, Tj, iv[js - 1], call); vb = vb - sprd(vb, best) / 2
                    vc = bs(rc[js], K, Tj, iv[js - 1], call); vc = vc - sprd(vc, rc[js]) / 2
                    hit = np.flatnonzero(vb >= cost * (1 + TP))
                    for H in HOLDS:
                        L = min(H, len(js))
                        if hit.size and hit[0] < L: r, why, k = TP, "target", hit[0]
                        else: r, why, k = vc[L - 1] / cost - 1, "time", L - 1
                        out.append((t, idx[e], name, side, dte, H, round(cost * 100), r, why, k + 1, bool(spyu[i])))
    return out
if __name__ == "__main__":
    with Pool(2) as pool: rows = sum(pool.map(run, TICK), [])
    T = pd.DataFrame(rows, columns=["ticker", "date", "sig", "side", "dte", "H", "cost_usd", "ret", "why", "days", "spy_up"])
    T.to_pickle(A + "/cheap/cheap_trades.pkl"); print(len(T)); print(T.groupby(["dte"]).cost_usd.describe())
