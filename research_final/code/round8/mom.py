"""Cross-sectional momentum (12-1 month) in the PIT top-100: at each month end rank stocks; forward returns of top/bottom decile vs all."""
import sys, os, numpy as np, pandas as pd
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"; sys.path.insert(0, A + "/auto")
import single as SG
os.chdir(A + "/auto"); P, spy = SG.load("/home/claude/options-research/data")
C = pd.DataFrame({k: v["prep"].close for k, v in P.items()}); O = pd.DataFrame({k: v["prep"].open for k, v in P.items()})
ELIG = pd.DataFrame({k: (v["prep"].in_index & (v["prep"].dvrank <= 100)) for k, v in P.items()}).reindex(C.index).fillna(False)
mom = C.shift(21) / C.shift(252) - 1; mom1 = C / C.shift(21) - 1
ends = C.index[(C.index.to_series().dt.month != pd.Series(C.index, index=C.index).shift(-1).dt.month).values]
ends = ends[(ends >= "2016-01-01") & (ends <= "2026-03-31")]
rows = []
for d in ends:
    i = C.index.get_loc(d); el = ELIG.loc[d] & mom.loc[d].notna()
    if el.sum() < 50: continue
    r = mom.loc[d][el].rank(pct=True)
    for hz in (21, 63, 126):
        if i + hz >= len(C): continue
        fwd = C.iloc[i + hz][el.index[el]] / O.iloc[i + 1][el.index[el]] - 1
        for grp, mask in (("top10", r >= 0.9), ("top20", r >= 0.8), ("bottom10", r <= 0.1), ("all", r >= 0)):
            rows.append(dict(date=d, hz=hz, grp=grp, ret=fwd[mask[mask].index].mean(), n=int(mask.sum())))
R = pd.DataFrame(rows); R["per"] = np.where(R.date < "2023-01-01", "L", "T")
print(R.pivot_table(index=["hz", "grp"], columns="per", values="ret", aggfunc="mean").round(4))
X = R.pivot_table(index=["date", "hz"], columns="grp", values="ret"); X["ex_top10"] = X.top10 - X["all"]; X = X.reset_index(); X["per"] = np.where(X.date < "2023-01-01", "L", "T")
g = X.groupby(["hz", "per"]).ex_top10.agg(["mean", "std", "count"]); g["t"] = g["mean"] / (g["std"] / np.sqrt(g["count"])); print(g.round(4))
