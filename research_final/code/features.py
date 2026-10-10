"""Feature panel (date x seg) -> long table of eligible stock-days with ~45 technical features,
10-day forward stock return, and the option outcomes (call/put, new shape, calibrated IV)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from panel import load, rsi
root, outc = sys.argv[1], sys.argv[2]
D = load(root); C, H, L, V, I = D["C"], D["H"], D["L"], D["V"], D["INIDX"]; spy, vix = D["spy"], D["vix"]
f = {}
for n in (1, 2, 5, 10, 21, 63, 126, 252): f[f"ret{n}"] = C.pct_change(n)
f["ret252_21"] = C.shift(21) / C.shift(252) - 1
for n in (10, 20, 50, 100, 200): f[f"dma{n}"] = C / C.rolling(n).mean() - 1
f["ma50_200"] = C.rolling(50).mean() / C.rolling(200).mean() - 1
for n in (2, 5, 14): f[f"rsi{n}"] = rsi(C, n)
m = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
f["macdh"] = (m - m.ewm(span=9, adjust=False).mean()) / C
ma20, sd20 = C.rolling(20).mean(), C.rolling(20).std(); f["bbpb"] = (C - ma20) / (2 * sd20); f["bbw"] = 4 * sd20 / ma20
tr = pd.concat([(H - L), (H - C.shift()).abs(), (L - C.shift()).abs()]).groupby(level=0).max()
f["atr14"] = tr.rolling(14).mean() / C
lr = np.log(C).diff(); hv20 = lr.rolling(20).std(); hv60 = lr.rolling(60).std(); hv252 = lr.rolling(252).std()
f["hv20"] = hv20 * np.sqrt(252); f["hv20_60"] = hv20 / hv60; f["hv60_252"] = hv60 / hv252
f["vr5_50"] = V.rolling(5).mean() / V.rolling(50).mean(); f["vr1_50"] = V / V.rolling(50).mean()
f["hi52"] = C / H.rolling(252).max() - 1; f["lo52"] = C / L.rolling(252).min() - 1
f["hi20"] = C / H.rolling(20).max() - 1; f["lo20"] = C / L.rolling(20).min() - 1
f["clv"] = ((C - L) - (H - C)) / (H - L).replace(0, np.nan)
f["updays10"] = (C.diff() > 0).rolling(10).sum()
for n in (5, 21, 63): f[f"rs{n}"] = C.pct_change(n).sub(spy.pct_change(n), axis=0)
for n in (21, 126):
    f[f"xrank{n}"] = C.pct_change(n).where(I).rank(axis=1, pct=True)
mk = {"spy5": spy.pct_change(5), "spy21": spy.pct_change(21), "spy_dma200": spy / spy.rolling(200).mean() - 1,
      "spy_dma50": spy / spy.rolling(50).mean() - 1, "vix": vix, "vix_chg5": vix.pct_change(5),
      "vix_rel": vix / vix.rolling(60).mean() - 1}
fwd = C.shift(-10) / C - 1
O = pd.read_pickle(outc)
oc = O[O.call].pivot(index="date", columns="seg", values="s30").reindex(index=C.index, columns=C.columns)
op = O[~O.call].pivot(index="date", columns="seg", values="s30").reindex(index=C.index, columns=C.columns)
mask = I & oc.notna() & (C.index >= "2016-01-01")[:, None]
ii, jj = np.nonzero(mask.values)
out = pd.DataFrame({"date": C.index.values[ii], "seg": C.columns.values[jj]})
for k, v in f.items(): out[k] = v.values[ii, jj].astype("float32")
for k, v in mk.items(): out[k] = v.values[ii].astype("float32")
out["fwd10"] = fwd.values[ii, jj]; out["opt_call"] = oc.values[ii, jj]; out["opt_put"] = op.values[ii, jj]
out = out.replace([np.inf, -np.inf], np.nan)
out.to_pickle("features.pkl"); print(out.shape, out.date.min(), out.date.max())
