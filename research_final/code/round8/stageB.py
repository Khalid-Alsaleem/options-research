"""Stage B: buy-only option grid on the stage-A survivors. PIT top-100 universe, calls only (puts failed stage A in absolute terms).
Paths of option value (bid) on closes for up to 120 trading days; forced exit 7 days before expiry."""
import sys, numpy as np, pandas as pd, pickle, time
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"; sys.path.insert(0, A + "/auto"); sys.path.insert(0, A + "/deep")
import single as SG, sigs, eng as E
H = 120; CON = [(dte, m) for dte in (60, 90, 180, 365) for m in (-0.10, 0.0, 0.10)]   # m<0 = in the money
P = spy = None
def init():
    global P, spy
    import os; os.chdir(A + "/auto"); P, spy = SG.load("/home/claude/options-research/data")
def extra(px):
    o, c, v = px.open, px.close, px.volume; gap = (o / c.shift() - 1).values; vr = (v / v.rolling(50).mean()).values; co = (c > o).values
    up200 = (c > c.rolling(200).mean()).values
    return {"gap1": (gap > 0.01) & co, "gap3": (gap > 0.03) & co, "gap5": (gap > 0.05) & co, "gap2_vol": (gap > 0.02) & co & (vr > 1.5),
            "gap2_trend": (gap > 0.02) & co & up200, "trend_day": ((c > c.rolling(50).mean()) & (c.rolling(50).mean() > c.rolling(200).mean())).values}
KEEP = ["gap2", "strong_dip", "rsi2_10", "rsi2_20", "high52", "donch55", "pre_earnings", "elliott5", "msc5", "ma20_50"]
def work(sid):
    seg = P[sid]; px = seg["prep"]
    if len(px) < 300: return []
    s = spy.reindex(px.index).ffill(); spyu = (s > s.rolling(200).mean()).values
    L = sigs.library(px, spy, seg["earn"]); S = {k: L[k][0] for k in KEEP}; S.update(extra(px))
    ok = px.in_index.values & (px.dvrank.values <= 100) & (px.index >= "2016-01-01") & (px.index <= "2026-08-20")
    eb = px.earn_block.values; o, c, iv, rs = px.open.values, px.close.values, px.iv.values, px.rs.values; idx = px.index; n = len(c); out = []
    for name, sg in S.items():
        last = -99; cd = 20 if name == "trend_day" else 5
        for i in np.flatnonzero(ok & np.asarray(sg, bool)):
            if i - last < cd or i + 2 >= n or not np.isfinite(iv[i]) or not np.isfinite(rs[i]): continue
            if eb[i] and name != "pre_earnings": continue
            last = i; e = i + 1; S0 = o[e]; js = np.arange(e, min(e + H, n))
            paths = np.full((len(CON), H), np.nan, np.float32); costs = np.zeros(len(CON), np.float32)
            for ci, (dte, m) in enumerate(CON):
                ex = SG.pick_exp(idx[e], int(dte * 0.8), int(dte * 1.5), dte)
                if ex is None: continue
                T0 = (ex - idx[e]).days / 365; st = E.step(S0); K = np.round(S0 * (1 + m) / st) * st
                cost = SG.bs(S0, K, T0, iv[i], True) * (1 + rs[i] / 2)
                if not cost > 0: continue
                dl = (ex - idx[js]).days.values; mm = dl > 7; jj = js[mm]
                if len(jj) == 0: continue
                val = SG.bs(c[jj], K, dl[mm] / 365, iv[jj], True) * (1 - rs[i] / 2)
                paths[ci, :len(jj)] = val / cost - 1; costs[ci] = cost / S0
            out.append(dict(seg=sid, date=idx[e], sig=name, spy_up=bool(spyu[i]), paths=paths, cost_pct=costs))
    return out
if __name__ == "__main__":
    init(); t0 = time.time(); rows = []
    with Pool(2, initializer=init) as pool:
        for k, o in enumerate(pool.imap_unordered(work, list(P), chunksize=4)):
            rows += o
            if k % 100 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
    pickle.dump(rows, open(A + "/deep/stageB_" + sys.argv[1] + ".pkl", "wb")); print("done", len(rows), round(time.time() - t0))
