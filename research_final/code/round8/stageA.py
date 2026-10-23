import sys, numpy as np, pandas as pd, pickle, time
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"; sys.path.insert(0, A + "/auto"); sys.path.insert(0, A + "/deep")
import single as SG, sigs
HZ = (5, 10, 20, 40, 60, 120)
P = spy = None
def init():
    global P, spy
    import os; os.chdir(A + "/auto"); P, spy = SG.load("/home/claude/options-research/data")
def work(sid):
    seg = P[sid]; px = seg["prep"]
    if len(px) < 300: return []
    s = spy.reindex(px.index).ffill(); spyu = (s > s.rolling(200).mean()).values
    L = sigs.library(px, spy, seg["earn"])
    ok = px.in_index.values & (px.dvrank.values <= 100) & (px.index >= "2016-01-01") & (px.index <= "2026-04-01")
    o, c = px.open.values, px.close.values; n = len(c); out = []
    fwd = np.full((n, len(HZ)), np.nan)
    for k, hz in enumerate(HZ):
        e = np.arange(n) + 1; x = np.arange(n) + hz
        m = x < n; fwd[m, k] = c[x[m]] / o[e[m]] - 1
    for name, (up, dn) in L.items():
        for side, sg in ((1, up), (-1, dn)):
            last = -99; cd = 20 if name == "baseline" else 5
            for i in np.flatnonzero(ok & np.asarray(sg, bool)):
                if i - last < cd or i + 1 >= n: continue
                last = i; out.append((name, side, px.index[i], bool(spyu[i]), *(side * fwd[i]).astype(np.float32)))
    return out
if __name__ == "__main__":
    init(); t0 = time.time(); rows = []
    with Pool(2, initializer=init) as pool:
        for k, o in enumerate(pool.imap_unordered(work, list(P), chunksize=4)):
            rows += o
            if k % 100 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
    T = pd.DataFrame(rows, columns=["sig", "side", "date", "spy_up"] + [f"f{h}" for h in HZ]); T.to_pickle(A + "/deep/stageA.pkl"); print("done", len(T), round(time.time() - t0))
