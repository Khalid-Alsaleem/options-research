"""Evaluate trigger x filter x side on the outcome table. LEARNING PERIOD ONLY unless HOLDOUT=1.
Discovery 2016-2019, validation 2020-2022-11. Every combination is logged (trial count)."""
import sys, os, numpy as np, pandas as pd, warnings, itertools, time
warnings.filterwarnings("ignore")
from panel import load, triggers, filters

HOLD = 10
def nonoverlap(sig):
    out = np.zeros_like(sig)
    for j in range(sig.shape[1]):
        last = -10**9
        for i in np.flatnonzero(sig[:, j]):
            if i > last + HOLD: out[i, j] = True; last = i
    return out

def stats(vals, dates):
    if len(vals) < 2: return dict(n=len(vals), mean=np.nan, t=np.nan)
    wk = pd.Series(vals, index=dates).groupby(pd.DatetimeIndex(dates).to_period("W")).mean()
    t = wk.mean() / (wk.std(ddof=1) / np.sqrt(len(wk))) if len(wk) > 2 else np.nan
    return dict(n=len(vals), mean=vals.mean(), t=t, weeks=len(wk))

def run(root, outc, out_csv, period_end="2022-11-30", extra=None):
    D = load(root); C = D["C"]
    O = pd.read_pickle(outc)
    mats = {}
    for side in (True, False):
        o = O[O.call == side]
        for col in ["s30", "s20"]:
            m = o.pivot(index="date", columns="seg", values=col).reindex(index=C.index, columns=C.columns)
            mats[(side, col)] = m.values
    dates = C.index.values
    learn = (C.index >= "2016-01-01") & (C.index <= period_end)
    disc = C.index <= "2019-12-31"
    T, F = triggers(D), filters(D)
    if extra: T, F = extra(T, F, D)
    rows = []; t0 = time.time()
    for (tn, (tl, ts)), (fn, (fl, fs)) in itertools.product(T.items(), F.items()):
        for side, trig, filt in [(True, tl, fl), (False, ts, fs)]:
            if tn == "always" and fn == "none" and not side: pass
            base = mats[(side, "s30")]
            sig = np.asarray(trig, bool) & np.asarray(filt, bool) & ~np.isnan(base) & learn[:, None]
            sig = nonoverlap(sig)
            ii, jj = np.nonzero(sig)
            v30 = base[ii, jj]; v20 = mats[(side, "s20")][ii, jj]; d = dates[ii]
            r = dict(trigger=tn, filter=fn, side="call" if side else "put")
            a = stats(v30, d); r.update(n=a["n"], s30=a["mean"], t=a["t"], s20=np.nanmean(v20) if len(v20) else np.nan)
            dm = disc[ii]
            r["s30_disc"] = v30[dm].mean() if dm.any() else np.nan; r["n_disc"] = int(dm.sum())
            r["s30_val"] = v30[~dm].mean() if (~dm).any() else np.nan; r["n_val"] = int((~dm).sum())
            yrs = pd.DatetimeIndex(d).year
            for y in range(2016, 2023): r[f"y{y}"] = v30[yrs == y].mean() if (yrs == y).any() else np.nan
            r["hit"] = np.mean(v30 >= 0.299) if len(v30) else np.nan
            rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(out_csv, index=False)
    print("combos:", len(R), "time", round(time.time() - t0), "s")
    return R

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
