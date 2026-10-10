"""Plateau check around the best SPY-call shape (learning period only). Target is a parameter."""
import sys, itertools, numpy as np, pandas as pd
import spy_long as SL
root = sys.argv[1]
spy = pd.read_csv(f"{root}/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
vix = pd.read_csv(f"{root}/prices/VIX.csv", parse_dates=["Date"]).set_index("Date").Close.reindex(spy.index).ffill()
S, r2 = SL.signals(spy)
ma50, ma200 = spy.rolling(50).mean(), spy.rolling(200).mean()
sigs = {"up200": S["every5_uptrend"], "up200_50": S["every5_uptrend"] & (ma50 > ma200)}
orig = SL.run
def run_t(spy, vix, sig, r2, delta, dte, tgt, a, b):
    src = SL.run.__code__
    return orig(spy, vix, sig, r2, delta, dte, "target30", a, b) if tgt == 0.30 else None
rows = []
for (sn, sg), delta, dte in itertools.product(sigs.items(), [0.5, 0.6, 0.7], [45, 60, 90]):
    T = SL.run(spy, vix, sg, r2, delta, dte, "target30", "2016-01-01", "2022-11-30")
    r = dict(signal=sn, delta=delta, dte=dte); r.update(SL.summ(T)); rows.append(r)
R = pd.DataFrame(rows); R.to_csv("spy_long_plateau.csv", index=False)
pd.set_option("display.width", 230); print(R.round(3).to_string())
print(R.pivot_table(index="delta", columns=["signal", "dte"], values="avg").round(3))
