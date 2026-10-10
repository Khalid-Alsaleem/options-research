import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import eng as E
P = {k: v for k, v in E.build(sys.argv[1]).items() if k not in E.EXCLUDE}
EV = pd.read_pickle("pead_events.pkl")
F = {}
for sid, seg in P.items():
    c = seg["px"].close; F[sid] = (c.shift(-10) / c - 1)
EV["f10"] = [F[s].get(d, np.nan) for s, d in zip(EV.seg, EV.date)]
learn = (EV.date >= "2016-01-01") & (EV.date <= "2022-11-30")
allf = pd.concat([F[s][P[s]["px"].in_index & (F[s].index >= "2016") & (F[s].index <= "2022-11-30")] for s in P])
print("all stock-days mean 10d return %.3f%%, sd %.2f%%" % (allf.mean() * 100, allf.std() * 100))
for lag in (1, 2):
    b = EV[learn & (EV.lag == lag)]
    for thr in (0.03, 0.05, 0.10):
        u = b[b.reac >= thr].f10; d = b[b.reac <= -thr].f10
        print(f"lag{lag} thr{thr}: after UP n={len(u)} mean10d={u.mean()*100:.2f}%  | after DOWN n={len(d)} mean10d={d.mean()*100:.2f}%")
