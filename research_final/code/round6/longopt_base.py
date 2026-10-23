"""Long-only option search: BUY a call on up signals, BUY a put on down signals (no spreads, no call+put together).
Universe: point-in-time S&P 500 members, top 100 by 20-day dollar volume each day (no survivorship bias).
Entry next day's open; option value tracked on every close for up to 40 trading days (forced exit 7 days before expiry).
Pricing: Black-Scholes, IV = 1.2*(0.5*HV60+0.5*HV252) + put skew, per-stock half-spread on entry and exit."""
import sys, numpy as np, pandas as pd, pickle, time
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad/auto"; sys.path.insert(0, A)
import single as SG
from panel import rsi
ROOT = "/home/claude/options-research/data"; H = 120
CONTRACTS = [(0.5, 365)]
P = spy = None
def init():
    global P, spy
    import os; os.chdir(A); P, spy = SG.load(ROOT)
def cross_up(a, b): return ((a > b) & (a.shift(1) <= b.shift(1))).values
def signals(px, s, earn):
    o, h, l, c, v = px.open, px.high, px.low, px.close, px.volume
    ma20, ma50, ma200 = (c.rolling(n).mean() for n in (20, 50, 200)); sd20 = c.rolling(20).std()
    r2 = rsi(c, 2).values; S = {}
    S["ma_cross"] = (cross_up(ma20, ma50), cross_up(ma50, ma20))
    hi55, lo55 = h.rolling(55).max().shift(), l.rolling(55).min().shift()
    S["breakout55"] = (((c > hi55) & (c.shift() <= hi55.shift())).values, ((c < lo55) & (c.shift() >= lo55.shift())).values)
    bbw = 4 * sd20 / ma20; sq = (bbw <= bbw.rolling(252, min_periods=120).quantile(0.1)).rolling(5).max() > 0
    S["squeeze"] = ((sq & (c > ma20 + 2 * sd20) & (c.shift() <= (ma20 + 2 * sd20).shift())).values,
                    (sq & (c < ma20 - 2 * sd20) & (c.shift() >= (ma20 - 2 * sd20).shift())).values)
    up, dn = np.zeros(len(c), bool), np.zeros(len(c), bool); cv = c.values; idx = px.index
    for d in earn:
        k = idx.searchsorted(d)
        if 1 <= k < len(c) - 1:
            r = cv[k + 1] / cv[k - 1] - 1
            if r > 0.04: up[k + 1] = True
            if r < -0.04: dn[k + 1] = True
    S["earnings_gap"] = (up, dn)
    rs63 = c.pct_change(63) - s.pct_change(63)
    S["strong_dip"] = (((c > ma50) & (ma50 > ma200) & (rs63 > 0)).values & (r2 < 10), ((c < ma50) & (ma50 < ma200) & (rs63 < 0)).values & (r2 > 90))
    hi252, lo252 = c.rolling(252).max().shift(), c.rolling(252).min().shift(); vr = (v / v.rolling(50).mean()).values
    S["high52_volume"] = ((c > hi252).values & (vr > 1.5), (c < lo252).values & (vr > 1.5))
    S["baseline"] = (np.ones(len(c), bool), np.zeros(len(c), bool))
    S["trend_only"] = (((c > ma50) & (ma50 > ma200) & (rs63 > 0)).values, np.zeros(len(c), bool))
    return S
def work(sid):
    seg = P[sid]; px = seg["prep"]; out = []
    if len(px) < 300: return out
    s = spy.reindex(px.index).ffill(); spyu = (s > s.rolling(200).mean()).values
    S = signals(px, s, seg["earn"])
    ok = px.in_index.values & (px.dvrank.values <= 100) & (px.index >= "2016-01-01") & (px.index <= "2026-08-20")
    eb = px.earn_block.values; o, c, iv, rs = px.open.values, px.close.values, px.iv.values, px.rs.values; idx = px.index; n = len(c)
    for name, (up, dn) in S.items():
        if name not in ("baseline", "trend_only"): continue
        for side, sig in ((1, up), (-1, dn)):
            last = -99
            for i in np.flatnonzero(ok & np.asarray(sig, bool)):
                if i - last < 10 or i + 2 >= n or not np.isfinite(iv[i]) or not np.isfinite(rs[i]): continue
                if eb[i] and name != "earnings_gap": continue
                last = i; e = i + 1; S0 = o[e]; call = side == 1
                js = np.arange(e, min(e + H, n)); stk = (c[js] / S0 - 1) * side
                paths = np.full((len(CONTRACTS), H), np.nan, np.float32)
                for ci, (dl, dte) in enumerate(CONTRACTS):
                    ex = SG.pick_exp(idx[e], int(dte * 0.75), int(dte * 1.6), dte)
                    if ex is None: continue
                    T0 = (ex - idx[e]).days / 365; K = SG.strike(S0, T0, iv[i], call, dl)
                    cost = SG.bs(S0, K, T0, SG.ivk(iv[i], S0, K, call), call) * (1 + rs[i] / 2)
                    if not cost > 0: continue
                    days_left = (ex - idx[js]).days.values; m = days_left > 7
                    jj = js[m]
                    if len(jj) == 0: continue
                    val = SG.bs(c[jj], K, days_left[m] / 365, SG.ivk(iv[jj], c[jj], K, call), call) * (1 - rs[i] / 2)
                    paths[ci, :len(jj)] = val / cost - 1
                out.append(dict(seg=sid, date=idx[e], sig=name, side=side, spy_up=bool(spyu[i]), stock=np.pad(stk, (0, H - len(stk)), constant_values=np.nan).astype(np.float32), paths=paths))
    return out
if __name__ == "__main__":
    init(); t0 = time.time(); rows = []
    with Pool(2, initializer=init) as pool:
        for k, o in enumerate(pool.imap_unordered(work, list(P), chunksize=8)):
            rows += o
            if k % 150 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
    pickle.dump(rows, open(f"{sys.argv[1]}/long_paths_base.pkl", "wb")); print("done", len(rows), round(time.time() - t0))
