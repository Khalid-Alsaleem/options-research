import sys, itertools, numpy as np, pandas as pd, pickle
from multiprocessing import Pool
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"; sys.path.insert(0, A + "/smc"); sys.path.insert(0, A + "/auto")
import smc
R = "/home/claude/options-research/data"; MODE = sys.argv[1]   # h1 | h4 | daily
TICK = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "ORCL", "CRM", "INTC", "JPM", "BAC", "GS",
        "V", "MA", "XOM", "CVX", "UNH", "LLY", "JNJ", "PFE", "WMT", "COST", "HD", "KO", "DIS", "BA", "CAT"]
FOMC = pd.to_datetime("""2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14 2017-02-01 2017-03-15 2017-05-03 2017-06-14
2017-07-26 2017-09-20 2017-11-01 2017-12-13 2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19 2019-01-30 2019-03-20
2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11 2020-01-29 2020-03-03 2020-03-16 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05
2020-12-16 2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15 2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27
2022-09-21 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 2024-01-31 2024-03-20 2024-05-01
2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 2026-01-28
2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28""".split())
EARN = pd.read_csv(R + "/earnings/edgar_8k_item202.csv", parse_dates=["filing_date"])
B = np.load(A + "/auto/spread_coef.npy")
GRID = list(itertools.product((2, 3, 5), (3, 5), (20, 50), (3, 4)))   # L_entry, L_htf, max_age, kmin
def daily(t):
    d = pd.read_csv(f"{R}/eod/{t}.csv.gz", parse_dates=["date"]).set_index("date"); d = d[~d.index.duplicated()].sort_index()
    f = d.adjusted_close / d.close
    adj = pd.DataFrame({"o": d.open * f, "h": d.high * f, "l": d.low * f, "c": d.adjusted_close}, index=d.index)
    lr = np.log(adj.c).diff(); iv = (1.2 * (0.5 * lr.rolling(60).std() + 0.5 * lr.rolling(252, min_periods=120).std()) * np.sqrt(252)).clip(0.1, 1.5)
    dv = (d.close * d.volume).rolling(20).mean(); rs = pd.Series(np.exp(B[0] + B[1] * np.log(dv) + B[2] * np.log(d.close)).clip(0.005, 0.25), index=d.index)
    return adj, iv, rs, (d.close / d.adjusted_close), dv
def news_days(t, idx_days):
    ev = pd.DatetimeIndex(list(EARN[EARN.ticker_raw == t].filing_date) + list(FOMC))
    bd = pd.bdate_range("2015-01-01", "2026-12-31"); blk = set()
    for e in ev:
        k = bd.searchsorted(e)
        for q in (k - 1, k, k + 1):
            if 0 <= q < len(bd): blk.add(bd[q])
    return np.array([d in blk for d in idx_days])
def htf_flags(hdf, L, target_index, lag_index):
    S = smc.structure(hdf.h.values, hdf.l.values, hdf.c.values, L)
    up = pd.Series((S["st"] == 1) & S["bull"], index=hdf.index); dn = pd.Series((S["st"] == -1) & S["bear"], index=hdf.index)
    # use the previous COMPLETED higher-frame bar
    u = up.shift(1).reindex(lag_index, method="ffill").fillna(False).values; d_ = dn.shift(1).reindex(lag_index, method="ffill").fillna(False).values
    return u, d_
def work(t):
    adj, iv, rs, fac, dv = daily(t); out = []
    if MODE in ("h1", "h4"):
        x = pd.read_csv(f"{R}/intraday/{t}_1h.csv.gz"); x["dt"] = pd.to_datetime(x.dt, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        x = x[(x.dt.dt.hour >= 9) & (x.dt.dt.hour <= 15)].set_index("dt").sort_index()
        df = x.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})[["o", "h", "l", "c"]]
        if MODE == "h4":
            grp = df.index.normalize() + pd.to_timedelta(np.where(df.index.hour < 13, 9.5, 13.5), "h")
            df = df.groupby(grp).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
        hdf = adj; days = df.index.normalize()
    else:
        df = adj.loc["2015-01-01":].copy(); df.index = df.index + pd.Timedelta(hours=15, minutes=59)
        hdf = adj.resample("W-FRI").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna(); days = df.index.normalize()
    lag_index = days if MODE in ("h1", "h4") else days   # previous completed day/week handled by shift(1)
    news = news_days(t, days); liq = (dv.reindex(days, method="ffill").values >= 5e8)
    if MODE == "daily":
        wk = days.to_period("W-FRI").to_timestamp(how="end").normalize()
    for (Le, Lh, age, k) in GRID:
        if MODE == "daily":
            S = smc.structure(hdf.h.values, hdf.l.values, hdf.c.values, Lh)
            up = pd.Series((S["st"] == 1) & S["bull"], index=hdf.index).shift(1); dn = pd.Series((S["st"] == -1) & S["bear"], index=hdf.index).shift(1)
            hu = up.reindex(wk, method="ffill").fillna(False).values.astype(bool); hd = dn.reindex(wk, method="ffill").fillna(False).values.astype(bool)
        else:
            S = smc.structure(hdf.h.values, hdf.l.values, hdf.c.values, Lh)
            up = pd.Series((S["st"] == 1) & S["bull"], index=hdf.index).shift(1); dn = pd.Series((S["st"] == -1) & S["bear"], index=hdf.index).shift(1)
            hu = up.reindex(days, method="ffill").fillna(False).values.astype(bool); hd = dn.reindex(days, method="ffill").fillna(False).values.astype(bool)
        sets = smc.find_setups(df, hu & liq, hd & liq, Le, age, k, news)
        for s in sets:
            for sm in (0.0, 1.0, 2.0):
                r = smc.price_trade(df, s, iv, rs, fac, spread_mult=sm, comm=0.65)
                if r is None: break
                r.update(ticker=t, Le=Le, Lh=Lh, age=age, kmin=k, cost=sm, fvg=s["fvg"], sweep=s["sweep"], disc=s["disc"]); out.append(r)
    return out
if __name__ == "__main__":
    with Pool(2) as p: rows = sum(p.map(work, TICK), [])
    T = pd.DataFrame(rows); T.to_pickle(f"{A}/smc/trades_{MODE}.pkl"); print(MODE, len(T), T[T.cost == 1].groupby(["Le", "Lh", "age", "kmin"]).size().describe())
