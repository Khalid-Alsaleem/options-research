"""Alternative studied after the indicator search failed: SPY bull-put credit spreads (defined risk).
Implied vol = REAL VIX each day (with a simple skew), so IV spikes/crush in crashes are real.
Pre-registered grid; learning 2016-01..2022-11; holdout 2023-01..2026-08 opened once at the end."""
import sys, numpy as np, pandas as pd, itertools
from scipy.special import ndtr
R_ = 0.02
def bsput(S, K, T, s):
    T = np.maximum(T, 1e-6); v = s * np.sqrt(T); d1 = (np.log(S / K) + (R_ + s * s / 2) * T) / v
    return K * np.exp(-R_ * T) * ndtr(-(d1 - v)) - S * ndtr(-d1)
def putdelta(S, K, T, s):
    v = s * np.sqrt(T); return ndtr((np.log(S / K) + (R_ + s * s / 2) * T) / v) - 1
def iv(vix, S, K):  # ATM ~ VIX - 1.5 pts; put skew +0.6 vol per unit log-moneyness OTM
    return np.maximum(vix / 100 - 0.015, 0.06) + 0.6 * np.maximum(0, np.log(S / K))
SLIP = 0.03   # per leg, per side, $ per share (SPY options trade at 1 cent; conservative incl. commission)

def run(spy, vix, delta, width, manage, mfilter, start, end, step=5):
    ma200 = spy.rolling(200).mean(); idx = spy.index; trades = []
    fridays = pd.date_range("2015-01-02", "2027-12-31", freq="W-FRI")
    for i in range(0, len(idx), step):
        d = idx[i]
        if d < pd.Timestamp(start) or d > pd.Timestamp(end): continue
        if mfilter and not spy.iat[i] > ma200.iat[i]: continue
        S = spy.iat[i]; v = vix.iat[i]
        dd = (fridays - d).days; ok = (dd >= 28) & (dd <= 35)
        if not ok.any(): continue
        ex = fridays[ok][0]; T = (ex - d).days / 365
        ks = np.arange(np.floor(S * 0.6), np.ceil(S), 1.0)
        dl = np.abs(putdelta(S, ks, T, iv(v, S, ks)))
        Ks = ks[np.argmin(np.abs(dl - delta))]; Kl = Ks - max(1.0, round(S * width))
        credit = bsput(S, Ks, T, iv(v, S, Ks)) - bsput(S, Kl, T, iv(v, S, Kl)) - 2 * SLIP
        w = Ks - Kl; risk = w - credit
        if credit <= 0.05 or risk <= 0: continue
        res = None
        for j in range(i + 1, len(idx)):
            dj = idx[j]; Sj = spy.iat[j]; vj = vix.iat[j]; Tj = (ex - dj).days / 365
            if dj >= ex:
                val = max(0, Ks - Sj) - max(0, Kl - Sj); res = ("expiry", credit - val); break
            cost = bsput(Sj, Ks, Tj, iv(vj, Sj, Ks)) - bsput(Sj, Kl, Tj, iv(vj, Sj, Kl)) + 2 * SLIP
            pnl = credit - cost
            if manage == "tp50" and pnl >= 0.5 * credit: res = ("tp50", pnl); break
            if pnl <= -2 * credit: res = ("stop2x", pnl); break
            if (ex - dj).days <= 7: res = ("7dte", pnl); break
        if res is None: continue
        trades.append(dict(date=d, S=S, vix=v, Ks=Ks, Kl=Kl, credit=credit, risk=risk, why=res[0],
                           ret=res[1] / risk, credit_on_risk=credit / risk))
    return pd.DataFrame(trades)

def summary(T):
    if T.empty: return {}
    yr = T.date.dt.year.value_counts().sort_index()
    by = T.groupby(T.date.dt.year).ret.mean()
    eq = (1 + 0.05 * T.ret).cumprod(); dd = (eq / eq.cummax() - 1).min()   # 5% of capital at risk per trade
    return dict(n=len(T), avg=T.ret.mean(), win=(T.ret > 0).mean(), avg_win=T.ret[T.ret > 0].mean(),
                avg_loss=T.ret[T.ret <= 0].mean(), worst=T.ret.min(), credit_on_risk=T.credit_on_risk.mean(),
                yrs_pos=int((by > 0).sum()), yrs=len(by), maxdd_5pct=dd,
                t=T.ret.mean() / (T.ret.std() / np.sqrt(len(T))))

if __name__ == "__main__":
    root = sys.argv[1]; mode = sys.argv[2]
    spy = pd.read_csv(f"{root}/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
    vix = pd.read_csv(f"{root}/prices/VIX.csv", parse_dates=["Date"]).set_index("Date").Close.reindex(spy.index).ffill()
    rows = []
    if mode == "learn":
        for delta, width, manage, mf in itertools.product([0.15, 0.20, 0.30], [0.03, 0.05], ["hold", "tp50"], [False, True]):
            T = run(spy, vix, delta, width, manage, mf, "2016-01-01", "2022-11-30")
            r = dict(delta=delta, width=width, manage=manage, spy_filter=mf); r.update(summary(T)); rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv("credit_learn.csv", index=False)
