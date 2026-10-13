"""20-strategy sweep (2026-10-13). Configs fixed BEFORE any run; same configs on learn and holdout."""
import sys, numpy as np, pandas as pd, warnings, pickle, time
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import single as SG
from panel import rsi
ROOT, PERIOD = sys.argv[1], sys.argv[2]
NS = tuple(int(x) for x in sys.argv[3].split(","))
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
A_, B_ = ("2016-01-01", "2022-11-30") if PERIOD == "learn" else ("2023-01-01", "2026-09-10")
P = spy = None
def init():
    global P, spy
    P, spy = SG.load(ROOT)
D = dict(delta=0.20, width=0.05, dte=35, target=0.10, stop=1.0, hold=20)
def q(**k): d = dict(D); d.update(k); return d
GRID = [  # (id, signal, kind, params)
 (1, "uptrend", "put_credit", q()),
 (2, "pullback_rsi2_20", "put_credit", q()),
 (3, "uptrend", "put_credit", q(delta=0.25, width=0.08, target=0.15)),
 (4, "deep_pullback_rsi2_5", "put_credit", q(delta=0.25, width=0.08, target=0.15)),
 (5, "three_down_days", "put_credit", q()),
 (6, "bollinger_low_in_uptrend", "put_credit", q()),
 (7, "near_52w_high", "put_credit", q()),
 (8, "relative_strength", "put_credit", q()),
 (9, "post_earnings_gap_up", "put_credit", q()),
 (10, "high_vol_uptrend", "put_credit", q()),
 (11, "low_vol_uptrend", "put_credit", q()),
 (12, "uptrend", "put_credit", q(dte=21, hold=10)),
 (13, "pullback_rsi2_20", "put_credit", q(delta=0.30, width=0.10, target=0.20)),
 (14, "ma20_bounce", "put_credit", q()),
 (15, "macd_turn_up", "put_credit", q()),
 (16, "downtrend", "call_credit", q()),
 (17, "bounce_in_downtrend", "call_credit", q()),
 (18, "weak_stock_strong_market", "call_credit", q()),
 (19, "range_compression", "iron_condor", q(delta=0.15)),
 (20, "pullback_rsi2_20", "put_credit", q(target=0.25, hold=30, dte=45)),
]
def sigs(px, spyc, spyu, spyd, earn):
    c = px.close; ma20, ma50, ma200 = (c.rolling(n).mean() for n in (20, 50, 200))
    r2 = rsi(c, 2).values; sd20 = c.rolling(20).std()
    lr = np.log(c).diff(); hv60 = lr.rolling(60).std() * np.sqrt(252); hv252 = lr.rolling(252, min_periods=120).std() * np.sqrt(252)
    hvrank = hv60.rolling(252, min_periods=120).rank(pct=True)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean(); macd = ema12 - ema26; h = macd - macd.ewm(span=9).mean()
    up = (c > ma50).values; up2 = up & (c > ma200).values
    v = lambda x: np.asarray(x, bool)
    pe = np.zeros(len(c), bool); idx = px.index; cv = c.values
    for d in earn:
        k = idx.searchsorted(d)
        if 1 <= k < len(c) - 1 and cv[k + 1] / cv[k - 1] - 1 > 0.03: pe[k + 1] = True
    s63 = pd.Series(spyc, index=idx).pct_change(63).values
    return {
     "uptrend": spyu & up2,
     "pullback_rsi2_20": spyu & up & (r2 < 20),
     "deep_pullback_rsi2_5": spyu & up2 & (r2 < 5),
     "three_down_days": spyu & up & v((c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3))),
     "bollinger_low_in_uptrend": spyu & v(c > ma200) & v(c < ma20 - 2 * sd20),
     "near_52w_high": spyu & up & v(c >= 0.97 * c.rolling(252).max()),
     "relative_strength": spyu & up & (c.pct_change(63).values - s63 > 0.05),
     "post_earnings_gap_up": spyu & up & pe,
     "high_vol_uptrend": spyu & up2 & v(hvrank > 0.7),
     "low_vol_uptrend": spyu & up2 & v(hv60 < 0.25),
     "ma20_bounce": spyu & v(ma20 > ma50) & v(px.low <= ma20) & v(c > ma20),
     "macd_turn_up": spyu & v(c > ma200) & v((h > 0) & (h.shift(1) <= 0)),
     "downtrend": spyd & v((c < ma50) & (c < ma200)),
     "bounce_in_downtrend": spyd & v(c < ma50) & (r2 > 80),
     "weak_stock_strong_market": spyu & v((c < ma50) & (c < ma200)) & (r2 > 80),
     "range_compression": v((c / ma50 - 1).abs() < 0.03) & v(hv60 < hv252),
    }
def work(sid):
    seg = P[sid]; px = seg["prep"]; out = []
    s = spy.reindex(px.index).ffill(); m = s.rolling(200).mean(); spyu = (s > m).values; spyd = (s < m).values
    S = sigs(px, s.values, spyu, spyd, seg["earn"])
    base = px.in_index.values & ~px.earn_block.values & (px.index >= A_) & (px.index <= B_)
    for N in NS:
        univ = base & (px.dvrank.values <= N)
        for sid_, name, kind, p in GRID:
            busy = -1
            for i in np.flatnonzero(univ & np.asarray(S[name], bool)):
                if i <= busy: continue
                r = SG.trade(px, i, kind, p)
                if r is None: continue
                out.append((N, sid_, sid, px.index[i], r[0], r[1], r[2])); busy = i + r[1]
    return out
if __name__ == "__main__":
    init(); t0 = time.time(); rows = []
    with Pool(2, initializer=init) as pool:
        for k, o in enumerate(pool.imap_unordered(work, list(P), chunksize=8)):
            rows += o
            if k % 150 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
    T = pd.DataFrame(rows, columns=["N", "gi", "seg", "date", "ret", "days", "why"])
    T.to_pickle(f"s20_trades_{PERIOD}{TAG}.pkl"); pickle.dump(GRID, open(f"s20_grid{TAG}.pkl", "wb"))
    print("done", len(T), round(time.time() - t0))
