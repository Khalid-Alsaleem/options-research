import sys, numpy as np, pandas as pd, pickle, os
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"; sys.path.insert(0, A + "/auto")
import single as SG
os.chdir(A + "/auto"); P, spy = SG.load("/home/claude/options-research/data")
ab = {}; tot = {}
for k, v in P.items():
    px = v["prep"]; c = px.close; m = (c > c.rolling(200).mean()) & px.in_index
    ab[k] = m.astype(float).where(px.in_index)
B = pd.DataFrame(ab).mean(axis=1)
vix = pd.read_csv("/home/claude/options-research/data/prices/VIX.csv", parse_dates=["Date"]).set_index("Date").Close
R = pd.DataFrame({"breadth": B, "vix": vix.reindex(B.index).ffill(), "spy": spy.reindex(B.index).ffill()})
R["vix_ma50"] = R.vix.rolling(50).mean(); R["spy50"] = R.spy.rolling(50).mean(); R["spy200"] = R.spy.rolling(200).mean(); R["breadth_ma20"] = R.breadth.rolling(20).mean()
R.to_pickle(A + "/deep/regimes.pkl"); print(R.dropna().describe().round(3))
