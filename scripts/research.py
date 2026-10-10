"""
Options indicator research - single-file engine (Phase A, synthetic option prices).
v2 (2026-10-10): contract shape delta 0.8 / ~90 DTE chosen by the pre-registered shape study; results in results/shape_d80_t90/.
Protocol: Khalid's options-indicator research protocol (2026-10-03 + amendments).
Learning period only (entries 2016-01-01 .. 2022-11-30). Every trial is logged in results/trials.csv.
Sections: DATASET -> ENGINE -> INDICATORS -> RUNNER.
"""
# ======================= DATASET =======================

from pathlib import Path
import re
import numpy as np
import pandas as pd

REJECTED_SPLITS = []

def split_adjust(px, splits, name=""):
    px = px.copy(); px["date"] = pd.to_datetime(px.date); px = px.sort_values("date").set_index("date")
    factor = pd.Series(1.0, index=px.index)
    if splits is not None and len(splits):
        for _, s in splits.iterrows():
            a, b = [float(x) for x in str(s.split).split("/")]
            r = a / b
            if r <= 0 or r == 1: continue
            d = pd.Timestamp(s.date)
            before, after = px.close[px.index < d], px.close[px.index >= d]
            if len(before) and len(after):
                jump = before.iloc[-1] / after.iloc[0]
                if not (0.5 * r <= jump <= 2.0 * r):       # vendor split with no matching price jump
                    REJECTED_SPLITS.append((name, s.date, s.split, round(jump, 3)))
                    continue
            factor[px.index < d] *= r
    for c in ["open", "high", "low", "close"]:
        px[c] = px[c] / factor
    px["volume"] = px["volume"] * factor
    px = px[["open", "high", "low", "close", "volume"]]
    med = px.close.rolling(20, min_periods=5).median().shift(1)
    bad = (px.close < 0.05 * med) | (px.close > 20 * med)      # decimal / unit errors
    return px[~bad.fillna(False)]

def build(root):
    root = Path(root)
    U = pd.read_csv(root / "universe/membership.csv", parse_dates=["start", "end"])
    F = pd.read_csv(root / "reports/fix_report.csv", parse_dates=["seg_start", "seg_end"])
    E1 = pd.read_csv(root / "earnings/edgar_8k_item202.csv", parse_dates=["filing_date"])
    E2 = pd.read_csv(root / "earnings/edgar_8k_item202_fixed.csv", parse_dates=["filing_date"])
    fixed = set(F.ticker_raw)
    ov_e = root / "earnings/edgar_8k_item202_override.csv"
    EO = pd.read_csv(ov_e, parse_dates=["filing_date"]) if ov_e.exists() else pd.DataFrame(columns=["ticker_raw", "filing_date"])
    segs = []
    for t, g in U.groupby("ticker_raw"):
        if t in fixed: continue
        p, sp = root / f"eod_override/{t}.csv.gz", root / f"eod_override/{t}__splits.csv"
        if not p.exists():
            p, sp = root / f"eod/{t}.csv.gz", root / f"splits/{t}.csv"
        if not p.exists(): continue
        earn = EO[EO.ticker_raw == t].filing_date if t in set(EO.ticker_raw) else E1[E1.ticker_raw == t].filing_date
        segs.append(dict(seg_id=t, ticker=t, intervals=list(zip(g.start, g.end)), price=p,
                         splits=sp if sp.exists() else None, earn=earn))
    for _, r in F.iterrows():
        p = root / f"eod_fixed/{r.seg_id}.csv.gz"
        if not p.exists(): continue
        sp = root / f"eod_fixed/{r.seg_id}__splits.csv"
        segs.append(dict(seg_id=r.seg_id, ticker=r.ticker_raw, intervals=[(r.seg_start, r.seg_end)], price=p,
                         splits=sp if sp.exists() else None, earn=E2[E2.seg_id == r.seg_id].filing_date))
    out = {}
    for s in segs:
        px = split_adjust(pd.read_csv(s["price"]), pd.read_csv(s["splits"]) if s["splits"] else None, s["seg_id"])
        px = px[px.index >= "2015-01-01"]
        m = pd.Series(False, index=px.index)
        for a, b in s["intervals"]:
            m |= (px.index >= a) & (px.index <= (b if pd.notna(b) else px.index.max()))
        px["in_index"] = m
        out[s["seg_id"]] = dict(ticker=s["ticker"], px=px, earn=pd.DatetimeIndex(s["earn"].drop_duplicates()).sort_values())
    return out


# ======================= ENGINE =======================

import numpy as np, pandas as pd
from math import sqrt
from scipy.special import ndtr

R, SPREAD, TARGET, MAX_DAYS, FORCED_DTE = 0.02, 0.05, 0.30, 10, 10
DTE_MIN, DTE_MAX, DTE_TGT = 80, 125, 90          # shape v2 (2026-10-10): longer-dated contract
DELTA_TGT = 0.8                                  # shape v2: in-the-money, delta 0.70-0.90
SHAPE_TAG = "shape_d80_t90"
LEARN_START, LEARN_LAST_ENTRY = "2016-01-01", "2022-11-30"   # exits stay inside 2022
EARN_BLOCK = 10                                              # trading days ahead
SHOCK_GAP, SHOCK_SPY, SHOCK_VIX = 0.05, 0.02, 0.20
EXCLUDE = {"FRC__53", "SBNY__91", "DISH", "TROW"}            # no reliable earnings dates

def monthly_expiries(a, b):
    out = []
    for m in pd.date_range(a, b, freq="MS"):
        out.append(pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")[2])
    return pd.DatetimeIndex(out)
EXP = monthly_expiries("2015-01-01", "2027-12-31")

def bs(S, K, T, sig, call):
    T = np.maximum(T, 1e-6); v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (R + sig * sig / 2) * T) / v; d2 = d1 - v
    if call: return S * ndtr(d1) - K * np.exp(-R * T) * ndtr(d2)
    return K * np.exp(-R * T) * ndtr(-d2) - S * ndtr(-d1)

def delta(S, K, T, sig, call):
    d1 = (np.log(S / K) + (R + sig * sig / 2) * T) / (sig * np.sqrt(T))
    return ndtr(d1) if call else ndtr(d1) - 1

def step(S): return 0.5 if S < 25 else 1 if S < 100 else 2.5 if S < 250 else 5

def market(root):
    spy = pd.read_csv(f"{root}/prices/SPY.csv", parse_dates=["Date"]).set_index("Date")
    vix = pd.read_csv(f"{root}/prices/VIX.csv", parse_dates=["Date"]).set_index("Date")
    hv = np.log(spy.Close).diff().rolling(30).std() * sqrt(252)
    vrp = (vix.Close / 100 / hv).clip(0.8, 2.0)
    return vrp, spy.Close.pct_change(), vix.Close.pct_change()

def prepare(seg, vrp):
    px = seg["px"].copy()
    hv = np.log(px.close).diff().rolling(30).std() * sqrt(252)
    px["iv"] = (hv * vrp.reindex(px.index).ffill()).clip(0.10, 1.50)
    # earnings in the next EARN_BLOCK trading days (filing date or next day if filed after close)
    e = np.zeros(len(px), bool)
    idx = px.index
    for d in seg["earn"]:
        i = idx.searchsorted(d)
        lo = max(0, i - EARN_BLOCK)
        e[lo:i + 1] = True
    px["earn_block"] = e
    return px

def simulate(px, i, call, ind_neg, spy_r, vix_r):
    """ind_neg: boolean array, True when the entry indicator has turned against the trade."""
    date = px.index[i]; S = px.close.iat[i]; sig = px.iv.iat[i]
    if not np.isfinite(sig): return None
    dd = (EXP - date).days
    ok = (dd >= DTE_MIN) & (dd <= DTE_MAX)
    if not ok.any(): return None
    exp_ = EXP[ok][np.argmin(np.abs(dd[ok] - DTE_TGT))]
    T0 = (exp_ - date).days / 365
    st = step(S); ks = np.round(S / st) * st + st * np.arange(-10, 11); ks = ks[ks > 0]
    dl = np.abs(delta(S, ks, T0, sig, call)); m = (dl >= DELTA_TGT - 0.1) & (dl <= DELTA_TGT + 0.1)
    if not m.any(): return None
    K = ks[m][np.argmin(np.abs(dl[m] - DELTA_TGT))]
    cost = bs(S, K, T0, sig, call) * (1 + SPREAD / 2)
    j = np.arange(i + 1, len(px)); dte = (exp_ - px.index[j]).days.values
    stop = np.argmax(dte <= FORCED_DTE) if (dte <= FORCED_DTE).any() else len(j) - 1
    j = j[:stop + 1]; dte = dte[:stop + 1]
    if len(j) == 0: return None
    S_p = px.close.values[j]; iv_p = px.iv.values[j]
    bid = bs(S_p, K, dte / 365, iv_p, call) * (1 - SPREAD / 2)
    ret = bid / cost - 1
    sgn = 1 if call else -1
    gap = (px.open.values[j] / px.close.values[j - 1] - 1) * sgn
    dts = px.index[j]
    sp = spy_r.reindex(dts).fillna(0).values * sgn; vx = vix_r.reindex(dts).fillna(0).values * sgn
    shock = (gap <= -SHOCK_GAP) | (sp <= -SHOCK_SPY) | ((vx >= SHOCK_VIX) if call else (vx <= -SHOCK_VIX))
    neg = ind_neg[j]
    last = min(MAX_DAYS, len(j)) - 1
    res = {"date": date, "call": call, "dte0": (exp_ - date).days, "cost": cost}
    for sname, kind, lvl in [("s20", "opt", .2), ("s30", "opt", .3), ("sStk", "stock", .3)]:
        hit = None; outcome = None
        for k in range(last + 1):
            if ret[k] <= -lvl or (kind == "stock" and neg[k]):
                outcome = ("stop", ret[k], k); break
            if ret[k] >= TARGET:
                hit = k; break
        if outcome is None and hit is None: outcome = ("day10", ret[last], last)
        if hit is None:
            res[f"{sname}_fixed"] = res[f"{sname}_ind"] = res[f"{sname}_hyb"] = outcome[1]
            res[f"{sname}_hit"] = False
            res[f"{sname}_shockfail"] = bool(outcome[0] == "stop" and shock[:outcome[2] + 1].any())
            continue
        res[f"{sname}_hit"] = True; res[f"{sname}_shockfail"] = False
        res[f"{sname}_fixed"] = ret[hit]
        after = np.arange(hit + 1, len(j))
        ex = after[(neg[after]) | (ret[after] <= 0)] if len(after) else after
        r_ind = ret[ex[0]] if len(ex) else ret[-1]
        res[f"{sname}_ind"] = r_ind; res[f"{sname}_hyb"] = 0.5 * ret[hit] + 0.5 * r_ind
    return res

def run(P, vrp, spy_r, vix_r, signal_fn, label, rng=None, n_random=1):
    """signal_fn(px) -> (long_entry bool array, short_entry bool array, long_neg, short_neg)"""
    trades, rnd = [], []
    for sid, seg in P.items():
        if sid in EXCLUDE: continue
        px = seg["prep"]
        le, se, lneg, sneg = signal_fn(px)
        elig = px.in_index.values & ~px.earn_block.values & \
               (px.index >= LEARN_START) & (px.index <= LEARN_LAST_ENTRY)
        busy = -1
        for i in np.where(elig & (le | se))[0]:
            if i <= busy: continue
            call = bool(le[i])
            r = simulate(px, i, call, lneg if call else sneg, spy_r, vix_r)
            if r is None: continue
            r["seg"] = sid; trades.append(r); busy = i + MAX_DAYS
            if rng is not None:
                pool = np.where(elig)[0]
                for k in rng.choice(pool, n_random):
                    rr = simulate(px, k, call, lneg if call else sneg, spy_r, vix_r)
                    if rr: rr["seg"] = sid; rnd.append(rr)
    return pd.DataFrame(trades), pd.DataFrame(rnd)

def summarize(T, label):
    out = {"config": label, "n": len(T), "calls": int(T.call.sum()) if len(T) else 0}
    for s in ["s20", "s30", "sStk"]:
        out[f"{s}_hit%"] = round(T[f"{s}_hit"].mean() * 100, 1)
        out[f"{s}_avg%"] = round(T[f"{s}_fixed"].mean() * 100, 2)
    out["sStk_ind_avg%"] = round(T.sStk_ind.mean() * 100, 2)
    out["sStk_hyb_avg%"] = round(T.sStk_hyb.mean() * 100, 2)
    return out

# ======================= INDICATORS =======================
import numpy as np, pandas as pd

def _cross_up(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.ndim == 0: a = np.full(b.shape, float(a))
    if b.ndim == 0: b = np.full(a.shape, float(b))
    prev_a = np.r_[np.nan, a[:-1]]; prev_b = np.r_[np.nan, b[:-1]]
    return (a > b) & (prev_a <= prev_b)

def _rsi(c, n):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def _atr(px, n):
    tr = pd.concat([px.high - px.low, (px.high - px.close.shift()).abs(), (px.low - px.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

# Each family: params -> function(px) -> (long_entry, short_entry, long_against, short_against)
def price_ma(n):
    def f(px):
        ma = px.close.rolling(n).mean()
        return _cross_up(px.close, ma), _cross_up(ma, px.close), (px.close < ma).values, (px.close > ma).values
    return f

def ma_cross(fast, slow):
    def f(px):
        a = px.close.rolling(fast).mean(); b = px.close.rolling(slow).mean()
        return _cross_up(a, b), _cross_up(b, a), (a < b).values, (a > b).values
    return f

def rsi_revert(n, L):
    def f(px):
        r = _rsi(px.close, n)
        return _cross_up(r, L), _cross_up(100 - L, r), (r < L).values, (r > 100 - L).values
    return f

def rsi_momo(n, L):
    def f(px):
        r = _rsi(px.close, n)
        return _cross_up(r, L), _cross_up(100 - L, r), (r < 50).values, (r > 50).values
    return f

def macd(fast, slow, sig):
    def f(px):
        m = px.close.ewm(span=fast, adjust=False).mean() - px.close.ewm(span=slow, adjust=False).mean()
        s = m.ewm(span=sig, adjust=False).mean()
        return _cross_up(m, s), _cross_up(s, m), (m < s).values, (m > s).values
    return f

def bb_break(n, k):
    def f(px):
        ma = px.close.rolling(n).mean(); sd = px.close.rolling(n).std()
        up, lo = ma + k * sd, ma - k * sd
        return _cross_up(px.close, up), _cross_up(lo, px.close), (px.close < ma).values, (px.close > ma).values
    return f

def bb_revert(n, k):
    def f(px):
        ma = px.close.rolling(n).mean(); sd = px.close.rolling(n).std()
        up, lo = ma + k * sd, ma - k * sd
        return _cross_up(px.close, lo), _cross_up(up, px.close), (px.close < lo).values, (px.close > up).values
    return f

def donchian(n):
    def f(px):
        hi = px.high.rolling(n).max().shift(); lo = px.low.rolling(n).min().shift()
        mid = (hi + lo) / 2
        return _cross_up(px.close, hi), _cross_up(lo, px.close), (px.close < mid).values, (px.close > mid).values
    return f

def stoch_revert(k, d, L):
    def f(px):
        ll = px.low.rolling(k).min(); hh = px.high.rolling(k).max()
        st = (100 * (px.close - ll) / (hh - ll)).rolling(d).mean()
        return _cross_up(st, L), _cross_up(100 - L, st), (st < L).values, (st > 100 - L).values
    return f

def cci(n, L, mode):
    def f(px):
        tp = (px.high + px.low + px.close) / 3; ma = tp.rolling(n).mean()
        md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        c = (tp - ma) / (0.015 * md)
        if mode == "momo":
            return _cross_up(c, L), _cross_up(-L, c), (c < 0).values, (c > 0).values
        return _cross_up(c, -L), _cross_up(L, c), (c < -L).values, (c > L).values
    return f

def willr(n, L):
    def f(px):
        hh = px.high.rolling(n).max(); ll = px.low.rolling(n).min()
        w = 100 * (px.close - ll) / (hh - ll)
        return _cross_up(w, L), _cross_up(100 - L, w), (w < L).values, (w > 100 - L).values
    return f

def mfi(n, L):
    def f(px):
        tp = (px.high + px.low + px.close) / 3; mf = tp * px.volume
        pos = mf.where(tp > tp.shift(), 0).rolling(n).sum(); neg = mf.where(tp < tp.shift(), 0).rolling(n).sum()
        m = 100 - 100 / (1 + pos / neg)
        return _cross_up(m, L), _cross_up(100 - L, m), (m < L).values, (m > 100 - L).values
    return f

def adx_di(n, amin):
    def f(px):
        upm = px.high.diff(); dnm = -px.low.diff()
        pdm = upm.where((upm > dnm) & (upm > 0), 0); ndm = dnm.where((dnm > upm) & (dnm > 0), 0)
        atr = _atr(px, n)
        pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
        ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / atr
        dx = 100 * (pdi - ndi).abs() / (pdi + ndi); adx = dx.ewm(alpha=1 / n, adjust=False).mean()
        strong = (adx > amin).values
        return _cross_up(pdi, ndi) & strong, _cross_up(ndi, pdi) & strong, (pdi < ndi).values, (pdi > ndi).values
    return f

def aroon(n):
    def f(px):
        up = px.high.rolling(n + 1).apply(lambda x: x.argmax(), raw=True) / n * 100
        dn = px.low.rolling(n + 1).apply(lambda x: x.argmin(), raw=True) / n * 100
        return _cross_up(up, dn), _cross_up(dn, up), (up < dn).values, (up > dn).values
    return f

def roc(n, thr):
    def f(px):
        r = px.close.pct_change(n)
        return _cross_up(r, thr), _cross_up(-thr, r), (r < 0).values, (r > 0).values
    return f

def keltner(n, mult):
    def f(px):
        ma = px.close.ewm(span=n, adjust=False).mean(); a = _atr(px, n)
        return _cross_up(px.close, ma + mult * a), _cross_up(ma - mult * a, px.close), (px.close < ma).values, (px.close > ma).values
    return f

def obv_ma(n):
    def f(px):
        obv = (np.sign(px.close.diff()).fillna(0) * px.volume).cumsum(); ma = obv.rolling(n).mean()
        return _cross_up(obv, ma), _cross_up(ma, obv), (obv < ma).values, (obv > ma).values
    return f

def cmf(n, thr):
    def f(px):
        mfm = ((px.close - px.low) - (px.high - px.close)) / (px.high - px.low).replace(0, np.nan)
        c = (mfm * px.volume).rolling(n).sum() / px.volume.rolling(n).sum()
        return _cross_up(c, thr), _cross_up(-thr, c), (c < 0).values, (c > 0).values
    return f

def grid():
    G = []
    G += [("price_ma", (n,), price_ma(n)) for n in range(10, 251, 2)]
    for a in [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        for b in [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]:
            if a < b: G.append(("ma_cross", (a, b), ma_cross(a, b)))
    for n in [2, 3, 4, 5, 7, 9, 11, 14, 17, 21, 25]:
        for L in [10, 15, 20, 25, 30, 35, 40]: G.append(("rsi_revert", (n, L), rsi_revert(n, L)))
    for n in [5, 7, 9, 14, 21, 30]:
        for L in [55, 60, 65, 70, 75]: G.append(("rsi_momo", (n, L), rsi_momo(n, L)))
    for a in [5, 8, 12, 16, 20]:
        for b in [17, 21, 26, 30, 35, 40]:
            for s in [5, 7, 9, 12]:
                if a < b: G.append(("macd", (a, b, s), macd(a, b, s)))
    for n in [10, 15, 20, 25, 30, 40, 50]:
        for k in [1.5, 2.0, 2.5, 3.0]:
            G.append(("bb_break", (n, k), bb_break(n, k))); G.append(("bb_revert", (n, k), bb_revert(n, k)))
    G += [("donchian", (n,), donchian(n)) for n in range(10, 101, 5)]
    for k in [5, 9, 14, 21]:
        for d in [3, 5]:
            for L in [10, 20, 30]: G.append(("stoch_revert", (k, d, L), stoch_revert(k, d, L)))
    for n in [10, 14, 20, 30, 40]:
        for L in [100, 150, 200]:
            G.append(("cci_momo", (n, L), cci(n, L, "momo"))); G.append(("cci_revert", (n, L), cci(n, L, "revert")))
    for n in [7, 10, 14, 21, 28]:
        for L in [10, 20]: G.append(("willr", (n, L), willr(n, L)))
    for n in [7, 10, 14, 21]:
        for L in [10, 20, 30]: G.append(("mfi", (n, L), mfi(n, L)))
    for n in [7, 10, 14, 20, 28]:
        for a in [15, 20, 25, 30]: G.append(("adx_di", (n, a), adx_di(n, a)))
    G += [("aroon", (n,), aroon(n)) for n in [10, 14, 25, 50]]
    for n in [5, 10, 20, 40, 60]:
        for t in [0.05, 0.10, 0.15]: G.append(("roc", (n, t), roc(n, t)))
    for n in [10, 20, 30]:
        for m in [1.5, 2.0, 2.5]: G.append(("keltner", (n, m), keltner(n, m)))
    G += [("obv_ma", (n,), obv_ma(n)) for n in [10, 20, 50, 100]]
    for n in [10, 20, 40]:
        for t in [0.05, 0.10, 0.20]: G.append(("cmf", (n, t), cmf(n, t)))
    return G


# ======================= RUNNER =======================
import os, sys, time, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def trial_row(fam, params, T):
    row = {"trial_id": f"{fam}{params}", "family": fam, "params": json.dumps(list(params)), "n": len(T)}
    if len(T) == 0: return row
    row["calls"] = int(T.call.sum()); row["puts"] = int((~T.call).sum())
    for s in ["s20", "s30", "sStk"]:
        row[f"{s}_hit"] = round(T[f"{s}_hit"].mean(), 4)
        row[f"{s}_fixed"] = round(T[f"{s}_fixed"].mean(), 4)
        row[f"{s}_ind"] = round(T[f"{s}_ind"].mean(), 4)
        row[f"{s}_hyb"] = round(T[f"{s}_hyb"].mean(), 4)
        sf = T[f"{s}_shockfail"].sum()
        row[f"{s}_hit_exshock"] = round(T[f"{s}_hit"].sum() / max(len(T) - sf, 1), 4)
    for side, m in [("call", T.call), ("put", ~T.call)]:
        row[f"s30_fixed_{side}"] = round(T.loc[m, "s30_fixed"].mean(), 4) if m.any() else None
    yr = T.date.dt.year
    for y in range(2016, 2023):
        m = yr == y
        row[f"n_{y}"] = int(m.sum()); row[f"s30_fixed_{y}"] = round(T.loc[m, "s30_fixed"].mean(), 4) if m.any() else None
    row["n_stocks"] = T.seg.nunique()
    return row

def main(root, out_dir, minutes):
    t0 = time.time(); out = Path(out_dir) / SHAPE_TAG; out.mkdir(parents=True, exist_ok=True)
    P = build(root); vrp, spy_r, vix_r = market(root)
    for s in P.values(): s["prep"] = prepare(s, vrp)
    print("prepared", len(P), "segments in", round(time.time() - t0), "s", flush=True)
    base_f = out / "random_baseline.csv"
    if not base_f.exists():
        _, RB = run(P, vrp, spy_r, vix_r, price_ma(200), "random", rng=np.random.default_rng(1), n_random=1)
        pd.DataFrame([trial_row("random_baseline", (), RB)]).to_csv(base_f, index=False)
        print("random baseline saved", flush=True)
    trials_f = out / "trials.csv"
    done = set(pd.read_csv(trials_f).trial_id) if trials_f.exists() else set()
    G = [g for g in grid() if f"{g[0]}{g[1]}" not in done]
    print("remaining trials:", len(G), flush=True)
    for fam, params, fn in G:
        if time.time() - t0 > minutes * 60:
            print("time budget reached", flush=True); break
        T, _ = run(P, vrp, spy_r, vix_r, fn, fam)
        row = trial_row(fam, params, T)
        pd.DataFrame([row]).to_csv(trials_f, mode="a", header=not trials_f.exists(), index=False)
        print(row["trial_id"], row.get("n"), row.get("s30_hit"), row.get("s30_fixed"), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(os.environ.get("MAX_MINUTES", "300")))
