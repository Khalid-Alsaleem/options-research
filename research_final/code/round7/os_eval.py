import pickle, numpy as np, pandas as pd, itertools, gc
R = pickle.load(open("optsearch_raw.pkl", "rb"))
meta = pd.DataFrame({k: [r[k] for r in R] for k in ("t", "date", "side", "dte", "mny", "sch", "src", "flt", "ks")})
L = max(len(r["hb"]) for r in R); N = len(R)
HB = np.full((N, L), np.nan, np.float32); CB = HB.copy(); OB = HB.copy(); ln = np.zeros(N, int)
for i, r in enumerate(R):
    m = len(r["hb"]); ln[i] = m; HB[i, :m] = r["hb"]; CB[i, :m] = r["cb"]; OB[i, :m] = r["ob"]
del R; gc.collect()
ar = np.arange(L)[None, :]; ks = meta.ks.values; last = ln - 1
res = {}
for tp in (0.3, 0.5, 1.0, None):
    hit = (HB >= tp) if tp is not None else np.zeros((N, L), bool)
    first_tp = np.where(hit.any(1), np.argmax(hit, 1), 10**6)
    for mode in ("signal", "hold"):
        stop_i = ks if mode == "signal" else last
        use_tp = first_tp <= stop_i
        sig_val = np.where(stop_i < last, OB[np.arange(N), np.minimum(stop_i, L - 1)], CB[np.arange(N), last])
        r = np.where(use_tp, tp if tp is not None else 0, sig_val)
        res[(tp, mode)] = r.astype(np.float32)
meta["learn"] = meta.date < "2023-01-01"
pickle.dump((meta, res), open("os_eval.pkl", "wb"))
rows = []
keys = ["side", "dte", "mny", "sch", "src", "flt"]
for (tp, mode), r in res.items():
    meta["r"] = r; meta["w"] = r > 0
    g = meta.groupby(keys + ["learn"]).agg(n=("r", "size"), avg=("r", "mean"), win=("w", "mean"))
    g = g.unstack("learn"); g.columns = [f"{a}_{'L' if b else 'T'}" for a, b in g.columns]; g = g.reset_index(); g["tp"] = tp; g["mode"] = mode
    rows.append(g)
G = pd.concat(rows); G.to_pickle("os_grid.pkl"); print(len(G))
