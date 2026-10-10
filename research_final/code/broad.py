import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
X = pd.read_pickle("features.pkl")
L = X[(X.date <= "2022-11-30")].copy()
feats = [c for c in X.columns if c not in ("date", "seg", "fwd10", "opt_call", "opt_put")]
print("learning rows", len(L), "| baseline: fwd10 %.2f%%  call %.2f%%  put %.2f%%" % (L.fwd10.mean()*100, L.opt_call.mean()*100, L.opt_put.mean()*100))
rows = []
for f in feats:
    q = L[f].rank(pct=True)
    for name, m in (("top10", q >= 0.9), ("bot10", q <= 0.1), ("top2", q >= 0.98), ("bot2", q <= 0.02)):
        s = L[m]
        d = s.date.dt.year <= 2019
        rows.append(dict(feat=f, bucket=name, n=len(s), fwd10=s.fwd10.mean(), call=s.opt_call.mean(), put=s.opt_put.mean(),
                         call_disc=s.opt_call[d].mean(), call_val=s.opt_call[~d].mean(),
                         put_disc=s.opt_put[d].mean(), put_val=s.opt_put[~d].mean(),
                         call_yrs=int((s.groupby(s.date.dt.year).opt_call.mean() > 0).sum()),
                         put_yrs=int((s.groupby(s.date.dt.year).opt_put.mean() > 0).sum())))
R = pd.DataFrame(rows); R.to_csv("feature_buckets.csv", index=False)
pd.set_option("display.width", 230)
for side in ("call", "put"):
    print("\nBEST", side); print((R.sort_values(side, ascending=False).head(12).set_index(["feat", "bucket"])[["n", "fwd10", side, f"{side}_disc", f"{side}_val", f"{side}_yrs"]] * [1, 100, 100, 100, 100, 1]).round(2).to_string())
