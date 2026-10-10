"""Walk-forward machine learning over ~45 technical features. Train on all years < Y (labels known),
predict year Y. Learning period only (Y = 2018..2022). Every 10 trading days: buy calls on the top-K
predicted stocks / puts on the top-K predicted for puts. Purge: training labels must end before test year."""
import sys, numpy as np, pandas as pd, warnings, lightgbm as lgb
warnings.filterwarnings("ignore")
X = pd.read_pickle("features.pkl"); X = X[X.date <= "2022-11-30"].reset_index(drop=True)
feats = [c for c in X.columns if c not in ("date", "seg", "fwd10", "opt_call", "opt_put")]
dates = np.sort(X.date.unique()); reb = set(dates[::10])
res = []
for target in ("opt_call", "opt_put"):
    for Y in range(2018, 2023):
        tr = X[(X.date.dt.year < Y) & (X.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=20))].dropna(subset=[target])
        te = X[(X.date.dt.year == Y) & X.date.isin(reb)].dropna(subset=[target])
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
        m.fit(tr[feats], tr[target].clip(-1, 2))
        te = te.assign(pred=m.predict(te[feats]))
        for K in (5, 10, 25):
            pick = te.sort_values("pred", ascending=False).groupby("date").head(K)
            res.append(dict(target=target, year=Y, K=K, n=len(pick), r=pick[target].mean(), base=te[target].mean(),
                            hit=(pick[target] >= .299).mean(), wk_pos=(pick.groupby("date")[target].mean() > 0).mean()))
        print(target, Y, "done", flush=True)
R = pd.DataFrame(res); R.to_csv("ml_walkforward.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string())
print(R.groupby(["target", "K"])[["r", "base", "hit", "wk_pos"]].mean().round(3))
