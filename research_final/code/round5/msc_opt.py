"""MSC signal read on the STOCK chart, executed with a CALL option. Same entry/exit dates as the stock trade.
Pricing: Black-Scholes, IV = 1.2*(0.5*HV60+0.5*HV252) (calibrated), per-stock half-spread per side from the ORATS fit."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import msc as M, single as SG
B = np.load(sys.argv[2] + "/spread_coef.npy")
L, R = 2, 0
TICK = sys.argv[3].split(",")
rows = []
for t in TICK:
    try: px = M.load(t)
    except Exception: continue
    d = pd.read_csv(f"/home/claude/options-research/data/eod/{t}.csv.gz", parse_dates=["date"]).set_index("date")
    lr = np.log(px.c).diff(); iv = (1.2 * (0.5 * lr.rolling(60).std() + 0.5 * lr.rolling(252, min_periods=120).std()) * np.sqrt(252)).clip(0.1, 1.5)
    dv = (d.close * d.volume).rolling(20).mean(); rs = np.exp(B[0] + B[1] * np.log(dv) + B[2] * np.log(d.close)).clip(0.005, 0.25)
    T = M.backtest(px, L, R, False, start="2016-01-01")
    for _, tr in T.iterrows():
        i = px.index.get_loc(tr.entry); k = px.index.get_loc(tr.exit); S0 = px.o.iat[i]; s0 = iv.iat[i - 1]; r = rs.iat[i - 1]
        if not np.isfinite(s0) or not np.isfinite(r): continue
        for delta, dte in ((0.5, 120), (0.7, 120), (0.5, 45)):
            ex = SG.pick_exp(px.index[i], dte - 30, dte + 40, dte)
            if ex is None: continue
            T0 = (ex - px.index[i]).days / 365; K = SG.strike(S0, T0, s0, True, delta)
            cost = SG.bs(S0, K, T0, s0, True) * (1 + r / 2)
            kk = k
            while kk > i and (ex - px.index[kk]).days < 7: kk -= 1          # forced exit 7 days before expiry
            if kk == k and (ex - px.index[k]).days < 7: pass
            Sx = px.o.iat[kk] if kk == k else px.c.iat[kk]; Tx = max((ex - px.index[kk]).days, 0) / 365
            val = SG.bs(Sx, K, Tx, iv.iat[kk - 1], True) * (1 - r / 2)
            rows.append(dict(ticker=t, entry=tr.entry, stock_ret=Sx / S0 - 1, why=tr.why, forced=kk != k, delta=delta, dte=dte, opt_ret=val / cost - 1, days=tr.days))
O = pd.DataFrame(rows); O.to_pickle("msc_opt.pkl")
O["period"] = np.where(O.entry < "2023-01-01", "2016-2022", "2023-2026")
g = O.groupby(["delta", "dte", "period"])
print(pd.DataFrame({"n": g.size(), "stock_avg": g.stock_ret.mean(), "stock_win": g.stock_ret.apply(lambda x: (x > 0).mean()),
    "opt_avg": g.opt_ret.mean(), "opt_med": g.opt_ret.median(), "opt_win": g.opt_ret.apply(lambda x: (x > 0).mean()),
    "opt_avg_win": g.opt_ret.apply(lambda x: x[x > 0].mean()), "opt_avg_loss": g.opt_ret.apply(lambda x: x[x <= 0].mean()),
    "forced": g.forced.mean()}).round(3).to_string())
for t in ["AMZN", "NVDA"]:
    s = O[(O.ticker == t) & (O.delta == 0.7) & (O.dte == 120)]
    print(t, len(s), "stock avg", round(s.stock_ret.mean(), 3), "call avg", round(s.opt_ret.mean(), 3), "call win", round((s.opt_ret > 0).mean(), 2))
