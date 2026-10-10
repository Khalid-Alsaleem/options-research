"""HOLDOUT for SPY calls, pre-committed (2026-10-11 ~01:30 Riyadh): SPY > MA200, a new entry every 5 trading days
when flat, call delta 0.6, ~60 DTE, exit +30% / -30% / day 10. Secondary: delta 0.5. Stress: skew x1.5, slippage x2."""
import sys, numpy as np, pandas as pd
import spy_long as SL, credit as C
root = sys.argv[1]
spy = pd.read_csv(f"{root}/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
vix = pd.read_csv(f"{root}/prices/VIX.csv", parse_dates=["Date"]).set_index("Date").Close.reindex(spy.index).ffill()
S, r2 = SL.signals(spy); sg = S["every5_uptrend"]; rows = []
for name, d in (("primary", 0.6), ("secondary", 0.5)):
    for per, (a, b) in {"learn": ("2016-01-01", "2022-11-30"), "holdout": ("2023-01-01", "2026-09-15")}.items():
        T = SL.run(spy, vix, sg, r2, d, 60, "target30", a, b); T.to_csv(f"spycall_trades_{name}_{per}.csv", index=False)
        s = SL.summ(T); s.update(config=name, period=per); rows.append(s)
        if per == "holdout": print(name, T.groupby(T.date.dt.year).ret.agg(["count", "mean", lambda r: (r > 0).mean()]).round(3).to_dict())
# stress tests on primary, both periods
base_iv, base_slip = C.iv, SL.slip
SL.iv = lambda v, S_, K: np.maximum(v / 100 - 0.015, 0.06) + 0.9 * np.maximum(0, np.log(S_ / K))
for per, (a, b) in {"learn": ("2016-01-01", "2022-11-30"), "holdout": ("2023-01-01", "2026-09-15")}.items():
    s = SL.summ(SL.run(spy, vix, sg, r2, 0.6, 60, "target30", a, b)); s.update(config="primary_skew1.5x", period=per); rows.append(s)
SL.iv = base_iv; SL.slip = lambda p: 2 * max(0.03, 0.005 * p)
for per, (a, b) in {"learn": ("2016-01-01", "2022-11-30"), "holdout": ("2023-01-01", "2026-09-15")}.items():
    s = SL.summ(SL.run(spy, vix, sg, r2, 0.6, 60, "target30", a, b)); s.update(config="primary_slip2x", period=per); rows.append(s)
SL.slip = base_slip
R = pd.DataFrame(rows); R.to_csv("spycall_holdout.csv", index=False)
pd.set_option("display.width", 230); print(R.round(3).to_string())
