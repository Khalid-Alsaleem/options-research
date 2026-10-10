"""
Contract-shape study (protocol amendment, path 1). Pre-registered grid, learning period only,
PURE RANDOM entries (no indicator): same entry set for every shape, so shapes are compared fairly.
Question: which contract shape has the least negative (or positive) baseline expectancy?
"""
import sys, time, itertools, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import research_shape as E

root, out = sys.argv[1], sys.argv[2]
P = E.build(root); vrp, spy_r, vix_r = E.market(root)
for s in P.values(): s["prep"] = E.prepare(s, vrp)

# fixed random entry set: 12 eligible days per segment, random side
rng = np.random.default_rng(2026)
entries = []
for sid, seg in P.items():
    if sid in E.EXCLUDE: continue
    px = seg["prep"]
    elig = np.where(px.in_index.values & ~px.earn_block.values &
                    (px.index >= E.LEARN_START) & (px.index <= E.LEARN_LAST_ENTRY))[0]
    if len(elig) == 0: continue
    for i in rng.choice(elig, min(12, len(elig)), replace=False):
        entries.append((sid, int(i), bool(rng.random() < 0.5)))
print("entries:", len(entries), flush=True)

GRID = dict(delta=[0.5, 0.6, 0.7, 0.8], dte=[30, 45, 60, 90], target=[0.20, 0.30], hold=[10, 20])
rows = []
for d, dte, tgt, hold in itertools.product(*GRID.values()):
    t0 = time.time()
    E.DELTA_TGT, E.DTE_TGT, E.DTE_MIN, E.DTE_MAX = d, dte, dte - 10, dte + 35
    E.TARGET, E.MAX_DAYS = tgt, hold
    res = []
    for sid, i, call in entries:
        px = P[sid]["prep"]; neg = np.zeros(len(px), bool)
        r = E.simulate(px, i, call, neg, spy_r, vix_r)
        if r: r["seg"] = sid; res.append(r)
    T = pd.DataFrame(res); yr = T.date.dt.year
    row = dict(delta=d, dte=dte, target=tgt, hold=hold, n=len(T),
               hit20=T.s20_hit.mean(), avg_s20=T.s20_fixed.mean(),
               hit30=T.s30_hit.mean(), avg_s30=T.s30_fixed.mean(),
               avg_call=T.loc[T.call, "s30_fixed"].mean(), avg_put=T.loc[~T.call, "s30_fixed"].mean(),
               cost=T.cost.mean())
    for y in range(2016, 2023): row[f"y{y}"] = T.loc[yr == y, "s30_fixed"].mean()
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith("y")},
          round(time.time() - t0), "s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
