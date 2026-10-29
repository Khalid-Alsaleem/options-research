"""MA50/100/200 breakouts on 4-hour bars with MACD and RSI confirmation. 31 liquid US stocks, Yahoo hourly bars Nov-2023..Oct-2026.
4H bar = 09:30-13:30 and 13:30-16:00 ET (2 bars per day). Signal on a bar's close, entry at next bar open. No look-ahead."""
import numpy as np, pandas as pd, itertools
R = "/home/claude/options-research/data/intraday"
TICK = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "ORCL", "CRM", "INTC", "JPM", "BAC", "GS",
        "V", "MA", "XOM", "CVX", "UNH", "LLY", "JNJ", "PFE", "WMT", "COST", "HD", "KO", "DIS", "BA", "CAT"]
CUT = pd.Timestamp("2025-06-01"); HZ = (5, 10, 20, 40)
def bars4h(t):
    x = pd.read_csv(f"{R}/{t}_1h.csv.gz"); x["dt"] = pd.to_datetime(x.dt, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    x = x[(x.dt.dt.hour >= 9) & (x.dt.dt.hour <= 15)].set_index("dt").sort_index()
    g = x.index.normalize() + pd.to_timedelta(np.where(x.index.hour < 13, 9.5, 13.5), "h")
    return x.groupby(g).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), v=("volume", "sum"))
def rsi(c, n=14):
    d = c.diff(); u = d.clip(lower=0).ewm(alpha=1 / n).mean(); w = (-d.clip(upper=0)).ewm(alpha=1 / n).mean(); return 100 - 100 / (1 + u / w)
rows = []
for t in TICK:
    b = bars4h(t); c = b.c; n = len(c)
    ma = {k: c.rolling(k).mean() for k in (50, 100, 200)}
    m = c.ewm(span=12).mean() - c.ewm(span=26).mean(); sig = m.ewm(span=9).mean(); hist = m - sig; r = rsi(c)
    above_all = (c > ma[50]) & (c > ma[100]) & (c > ma[200]); below_all = (c < ma[50]) & (c < ma[100]) & (c < ma[200])
    ev = {}
    for k in (50, 100, 200):
        ev[f"cross_ma{k}"] = ((c > ma[k]) & (c.shift() <= ma[k].shift()), (c < ma[k]) & (c.shift() >= ma[k].shift()))
    ev["cross_all3"] = (above_all & ~above_all.shift(fill_value=False), below_all & ~below_all.shift(fill_value=False))
    ev["baseline"] = (pd.Series(True, index=c.index), pd.Series(True, index=c.index))
    filt = {"none": (pd.Series(True, index=c.index),) * 2,
            "macd_hist>0": (hist > 0, hist < 0), "macd>0": ((m > 0) & (hist > 0), (m < 0) & (hist < 0)),
            "rsi>50": (r > 50, r < 50), "rsi50-70": ((r > 50) & (r < 70), (r < 50) & (r > 30)),
            "macd_hist>0&rsi50-70": ((hist > 0) & (r > 50) & (r < 70), (hist < 0) & (r < 50) & (r > 30))}
    o = b.o.values; cv = c.values
    fwd = {h: np.r_[cv[h:] , [np.nan] * h] / np.r_[o[1:], np.nan] - 1 for h in HZ}
    # exit-on-MA trade: hold until close falls back below the crossed MA (cross_ma*) or below MA50 (cross_all3), max 40 bars
    for (name, (u, d_)), (fn, (fu, fd)) in itertools.product(ev.items(), filt.items()):
        if name == "baseline" and fn != "none": continue
        for side, s_ in ((1, (u & fu).values), (-1, (d_ & fd).values)):
            last = -99
            for i in np.flatnonzero(s_):
                if i < 200 or i + 1 >= n: continue
                if name == "baseline" and i - last < 10: continue
                last = i
                exma = ma[int(name[8:])] if name.startswith("cross_ma") else ma[50]
                k = i + 1; out = None
                while k < min(i + 41, n):
                    if (cv[k] - exma.values[k]) * side < 0: out = k; break
                    k += 1
                out = out if out is not None else min(i + 40, n - 1)
                tr = side * (cv[out] / o[i + 1] - 1) if i + 1 < n else np.nan
                rows.append(dict(t=t, ts=c.index[i], sig=name, flt=fn, side=side, trade=tr, bars=out - i, **{f"f{h}": side * fwd[h][i] for h in HZ}))
T = pd.DataFrame(rows); T["per"] = np.where(T.ts < CUT, "IS", "OOS"); T.to_pickle("ma4h_events.pkl")
base = T[T.sig == "baseline"].groupby(["side", "per"])[[f"f{h}" for h in HZ]].mean()
out = []
for (sg, fl, sd), g in T[T.sig != "baseline"].groupby(["sig", "flt", "side"]):
    r = dict(sig=sg, flt=fl, side="call/up" if sd == 1 else "put/down")
    for per in ("IS", "OOS"):
        x = g[g.per == per]
        if len(x) < 20: r[per + "_n"] = len(x); continue
        r.update({per + "_n": len(x), per + "_f20": x.f20.mean(), per + "_ex20": x.f20.mean() - base.loc[(sd, per), "f20"], per + "_win20": (x.f20 > 0).mean(),
                  per + "_trade": x.trade.mean(), per + "_trwin": (x.trade > 0).mean()})
        wk = (x.f20 - base.loc[(sd, per), "f20"]).groupby(x.ts.dt.to_period("W")).mean(); r[per + "_t"] = wk.mean() / (wk.std() / np.sqrt(len(wk)))
    out.append(r)
O = pd.DataFrame(out); O.to_csv("ma4h_summary.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
print("baseline f20 (4H bars ~10 trading days):\n", base.f20.round(4))
print(O.round(4).sort_values(["side", "IS_ex20"], ascending=[True, False]).to_string(index=False))
