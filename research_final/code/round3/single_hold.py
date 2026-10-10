import sys, numpy as np, pandas as pd, warnings, itertools, pickle, time
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import single as SG, eng as E
from panel import rsi
ROOT, PERIOD = sys.argv[1], sys.argv[2]
NS = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (50, 100)
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
A_, B_ = ("2016-01-01", "2022-11-30") if PERIOD == "learn" else ("2023-01-01", "2026-09-10")
P = spy = None
def init():
    global P, spy
    P, spy = SG.load(ROOT)
def sigs(px, spyu, spyd):
    c = px.close; ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    r2 = rsi(c, 2).values; hi20 = px.high.rolling(20).max().shift(); vr = px.volume / px.volume.rolling(50).mean()
    up = (c > ma50).values
    return {"pullback_call": ("long_call", spyu & up & (r2 < 10)),
            "breakout_call": ("long_call", spyu & (c > hi20).values & (vr > 1.5).values),
            "trend_call": ("long_call", spyu & up & (ma20 > ma50).values & (c > ma200).values),
            "bear_put": ("long_put", spyd & (c < ma50).values & (r2 > 90)),
            "credit_uptrend": ("put_credit", spyu & up & (c > ma200).values),
            "credit_pullback": ("put_credit", spyu & up & (r2 < 20)),
            "debit_pullback": ("call_debit", spyu & up & (r2 < 10))}
# PRE-COMMITTED before opening 2023-2026 (2026-10-11): A/B/C below; run at N=30 and N=50
GRID = [("credit_uptrend", dict(delta=0.20, width=0.05, dte=35, target=0.10, stop=1.0, hold=20)),   # A
        ("credit_pullback", dict(delta=0.20, width=0.05, dte=35, target=0.10, stop=1.0, hold=20)),  # B
        ("credit_uptrend", dict(delta=0.25, width=0.08, dte=35, target=0.15, stop=1.0, hold=20))]   # C
def work(sid):
    seg = P[sid]; px = seg["prep"]; out = []
    s = spy.reindex(px.index).ffill(); m = s.rolling(200).mean(); spyu = (s > m).values; spyd = (s < m).values
    S = sigs(px, spyu, spyd)
    base = px.in_index.values & ~px.earn_block.values & (px.index >= A_) & (px.index <= B_)
    for N in NS:
        univ = base & (px.dvrank.values <= N)
        for gi, (name, p) in enumerate(GRID):
            kind, sig = S[name]; busy = -1
            for i in np.flatnonzero(univ & np.asarray(sig, bool)):
                if i <= busy: continue
                r = SG.trade(px, i, kind, p)
                if r is None: continue
                out.append((N, gi, sid, px.index[i], r[0], r[1], r[2])); busy = i + r[1]
    return out
if __name__ == "__main__":
    init(); t0 = time.time(); rows = []
    with Pool(2, initializer=init) as pool:
        for k, o in enumerate(pool.imap_unordered(work, list(P), chunksize=8)):
            rows += o
            if k % 150 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
    T = pd.DataFrame(rows, columns=["N", "gi", "seg", "date", "ret", "days", "why"])
    T.to_pickle(f"single_trades_{PERIOD}{TAG}.pkl"); pickle.dump(GRID, open(f"single_grid{TAG}.pkl", "wb"))
    print("done", len(T), round(time.time() - t0))
