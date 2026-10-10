"""Buying SPY calls on technical pullback signals. Index options: tight spreads (unlike single stocks).
IV from REAL daily VIX (+ skew), same pricing as credit.py. Pre-registered grid; learning 2016..2022-11."""
import sys, itertools, numpy as np, pandas as pd
from scipy.special import ndtr
from credit import iv
R_ = 0.02
def bscall(S, K, T, s):
    T = np.maximum(T, 1e-6); v = s * np.sqrt(T); d1 = (np.log(S / K) + (R_ + s * s / 2) * T) / v
    return S * ndtr(d1) - K * np.exp(-R_ * T) * ndtr(d1 - v)
def cdelta(S, K, T, s):
    v = s * np.sqrt(T); return ndtr((np.log(S / K) + (R_ + s * s / 2) * T) / v)
def slip(p): return max(0.03, 0.005 * p)
def rsi(c, n):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def signals(spy):
    ma200 = spy.rolling(200).mean(); up = spy > ma200; r2 = rsi(spy, 2)
    lo = spy.rolling(20).mean() - 2 * spy.rolling(20).std(); dn3 = (spy.diff() < 0).rolling(3).sum() == 3
    return {"rsi2<10": up & (r2 < 10), "rsi2<5": up & (r2 < 5), "below_bb": up & (spy < lo), "3down": up & dn3,
            "every5_uptrend": up & (np.arange(len(spy)) % 5 == 0)}, r2

def run(spy, vix, sig, r2, delta, dte, exit_rule, start, end):
    idx = spy.index; fr = pd.date_range("2015-01-02", "2027-12-31", freq="W-FRI"); T_ = []; busy = -1
    for i in np.flatnonzero(sig.values):
        d = idx[i]
        if d < pd.Timestamp(start) or d > pd.Timestamp(end) or i <= busy: continue
        S, v = spy.iat[i], vix.iat[i]; dd = (fr - d).days; ok = (dd >= dte - 7) & (dd <= dte + 10)
        ex = fr[ok][np.argmin(abs(dd[ok] - dte))]; T = (ex - d).days / 365
        ks = np.arange(np.floor(S * 0.7), np.ceil(S * 1.05), 1.0)
        dl = cdelta(S, ks, T, iv(v, S, ks)); K = ks[np.argmin(abs(dl - delta))]
        mid0 = bscall(S, K, T, iv(v, S, K)); cost = mid0 + slip(mid0)
        out = None
        for j in range(i + 1, min(i + 11, len(idx))):
            Sj, vj = spy.iat[j], vix.iat[j]; Tj = (ex - idx[j]).days / 365
            m = bscall(Sj, K, Tj, iv(vj, Sj, K)); bid = m - slip(m); r = bid / cost - 1
            if r <= -0.30: out = ("stop", r, j); break
            if exit_rule == "target30" and r >= 0.30: out = ("target", r, j); break
            if exit_rule == "rsi70" and r2.iat[j] > 70: out = ("rsi70", r, j); break
        if out is None: out = ("day10", r, j)
        T_.append(dict(date=d, why=out[0], ret=out[1], days=out[2] - i)); busy = out[2]
    return pd.DataFrame(T_)

def summ(T):
    if len(T) < 5: return dict(n=len(T))
    by = T.groupby(T.date.dt.year).ret.mean()
    return dict(n=len(T), avg=T.ret.mean(), win=(T.ret > 0).mean(), avg_win=T.ret[T.ret > 0].mean(),
                avg_loss=T.ret[T.ret <= 0].mean(), worst=T.ret.min(), hit30=(T.ret >= .299).mean(),
                yrs_pos=int((by > 0).sum()), yrs=len(by), t=T.ret.mean() / (T.ret.std() / np.sqrt(len(T))),
                days=T.days.mean())

if __name__ == "__main__":
    root = sys.argv[1]; per = sys.argv[2] if len(sys.argv) > 2 else "learn"
    spy = pd.read_csv(f"{root}/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
    vix = pd.read_csv(f"{root}/prices/VIX.csv", parse_dates=["Date"]).set_index("Date").Close.reindex(spy.index).ffill()
    S, r2 = signals(spy); rows = []
    a, b = ("2016-01-01", "2022-11-30") if per == "learn" else ("2023-01-01", "2026-09-15")
    for (sn, sg), delta, dte, ex in itertools.product(S.items(), [0.6, 0.8], [30, 60], ["target30", "rsi70"]):
        T = run(spy, vix, sg, r2, delta, dte, ex, a, b); r = dict(signal=sn, delta=delta, dte=dte, exit=ex); r.update(summ(T)); rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(f"spy_long_{per}.csv", index=False)
    pd.set_option("display.width", 230); print(R.round(3).sort_values("avg", ascending=False).to_string())
