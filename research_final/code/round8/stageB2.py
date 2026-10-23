"""Price-targeted contracts: for each finalist signal, buy the call whose real price is $2-5 per share ($200-500/contract),
closest to $3.5, for 180- and 365-day expiries. PIT top-100 universe. Raw (unadjusted) price = split-adjusted price x split factor."""
import sys, os, numpy as np, pandas as pd, pickle, time
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"; sys.path.insert(0, A + "/auto"); sys.path.insert(0, A + "/deep")
import single as SG, sigs, eng as E
from stageB import extra
H = 120; DTES = (60, 180, 365); ROOT = "/home/claude/options-research/data"
P = spy = None
def init():
    global P, spy
    os.chdir(A + "/auto"); P, spy = SG.load(ROOT)
def raw_close(sid, t):
    for p in (f"{ROOT}/eod_override/{t}.csv.gz", f"{ROOT}/eod_fixed/{sid}.csv.gz", f"{ROOT}/eod/{t}.csv.gz"):
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["date"]).set_index("date"); return d.close[~d.index.duplicated()]
    return None
def work(sid):
    seg = P[sid]; px = seg["prep"]
    if len(px) < 300: return []
    rc = raw_close(sid, seg["ticker"])
    if rc is None: return []
    fac = (rc.reindex(px.index) / px.close).values
    s = spy.reindex(px.index).ffill(); spyu = (s > s.rolling(200).mean()).values
    L = sigs.library(px, spy, seg["earn"]); S = {k: L[k][0] for k in ("strong_dip", "high52", "rsi2_10")}; X = extra(px); S["gap3"] = X["gap3"]; S["trend_day"] = X["trend_day"]
    ok = px.in_index.values & (px.dvrank.values <= 100) & (px.index >= "2016-01-01") & (px.index <= "2026-08-20")
    eb = px.earn_block.values; o, c, iv, rs = px.open.values, px.close.values, px.iv.values, px.rs.values; idx = px.index; n = len(c); out = []
    for name, sg in S.items():
        last = -99; cd = 20 if name == "trend_day" else 5
        for i in np.flatnonzero(ok & np.asarray(sg, bool)):
            if i - last < cd or i + 2 >= n or eb[i] or not np.isfinite(iv[i]) or not np.isfinite(rs[i]) or not np.isfinite(fac[i]): continue
            e = i + 1; js = np.arange(e, min(e + H, n))
            fw = fac[js]
            if np.nanmax(np.abs(fw / fac[i] - 1)) > 0.02: continue   # split inside the window
            last = i; S0 = o[e]; f0 = fac[i]; Sraw = S0 * f0
            paths = np.full((len(DTES), H), np.nan, np.float32); info = []
            for ci, dte in enumerate(DTES):
                ex = SG.pick_exp(idx[e], int(dte * 0.8), int(dte * 1.5), dte)
                if ex is None: info.append((np.nan, np.nan)); continue
                T0 = (ex - idx[e]).days / 365; st = E.step(Sraw)
                ks = (np.round(Sraw / st) * st + st * np.arange(-30, 80)) / f0; ks = ks[ks > 0]
                pr = SG.bs(S0, ks, T0, iv[i], True) * f0
                okk = (pr >= 2) & (pr <= 5)
                if not okk.any(): info.append((np.nan, np.nan)); continue
                K = ks[okk][np.argmin(np.abs(pr[okk] - 3.5))]
                cost = SG.bs(S0, K, T0, iv[i], True) * (1 + rs[i] / 2) + 0.025 / f0
                dl = (ex - idx[js]).days.values; mm = dl > 7; jj = js[mm]
                val = np.maximum(SG.bs(c[jj], K, dl[mm] / 365, iv[jj], True) * (1 - rs[i] / 2) - 0.025 / f0, 0)
                paths[ci, :len(jj)] = val / cost - 1; info.append((cost * f0 * 100, K / S0 - 1))
            out.append(dict(seg=sid, date=idx[e], sig=name, spy_up=bool(spyu[i]), raw_price=Sraw, paths=paths, info=info))
    return out
if __name__ == "__main__":
    init(); t0 = time.time(); rows = []
    with Pool(2, initializer=init) as pool:
        for k, o in enumerate(pool.imap_unordered(work, list(P), chunksize=4)):
            rows += o
            if k % 150 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
    pickle.dump(rows, open(A + "/deep/stageB2.pkl", "wb")); print("done", len(rows), round(time.time() - t0))
