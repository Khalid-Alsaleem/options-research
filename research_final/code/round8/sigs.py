"""Signal library (daily bars, no look-ahead: everything uses data up to the signal day's close)."""
import numpy as np, pandas as pd
def rsi(c, n):
    d = c.diff(); u = d.clip(lower=0).ewm(alpha=1 / n).mean(); v = (-d.clip(upper=0)).ewm(alpha=1 / n).mean(); return 100 - 100 / (1 + u / v)
def xup(a, b): return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False).values
def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1); return tr.ewm(alpha=1 / n).mean()
def msc(h, l, c, L):
    hv, lv, cv = h.values, l.values, c.values; n = len(cv); sh = sl = np.nan; st = 0; up = np.zeros(n, bool); dn = np.zeros(n, bool)
    for i in range(n):
        j = i - L
        if j - L >= 0:
            if hv[j] == hv[j - L:i + 1].max(): sh = hv[j]
            if lv[j] == lv[j - L:i + 1].min(): sl = lv[j]
        if cv[i] > sh and st <= 0: up[i] = st == -1; st = 1
        elif cv[i] < sl and st >= 0: dn[i] = st == 1; st = -1
    return up, dn
def pivots(h, l, L):
    n = len(h); P = []
    for k in range(L, n):
        j = k - L
        if j - L < 0: continue
        if h[j] == h[j - L:k + 1].max():
            if P and P[-1][0] == "H":
                if h[j] > P[-1][2]: P[-1] = ("H", j, h[j], k)
            else: P.append(("H", j, h[j], k))
        if l[j] == l[j - L:k + 1].min():
            if P and P[-1][0] == "L":
                if l[j] < P[-1][2]: P[-1] = ("L", j, l[j], k)
            else: P.append(("L", j, l[j], k))
    return P
def elliott(h, l, c, L):
    P = pivots(h.values, l.values, L); cv = c.values; n = len(cv); up = np.zeros(n, bool); dn = np.zeros(n, bool)
    for q in range(2, len(P)):
        (k1, _, A, _), (k2, _, B, _), (k3, _, C, known) = P[q - 2], P[q - 1], P[q]
        side = 1 if (k1, k2, k3) == ("L", "H", "L") else -1 if (k1, k2, k3) == ("H", "L", "H") else 0
        if not side: continue
        w1 = (B - A) * side
        if w1 <= 0 or not (0.382 <= (B - C) * side / w1 <= 0.786): continue
        nxt = P[q + 1][3] if q + 1 < len(P) else n
        for i in range(known, min(known + 40, nxt, n)):
            if (cv[i] - B) * side > 0: (up if side == 1 else dn)[i] = True; break
    return up, dn
def library(px, spy, earn):
    """px: open/high/low/close/volume (split-adjusted). returns {name: (up, dn)}"""
    o, h, l, c, v = px.open, px.high, px.low, px.close, px.volume; n = len(c); Z = np.zeros(n, bool)
    ma = {k: c.rolling(k).mean() for k in (5, 10, 20, 30, 50, 100, 200)}; up200 = (c > ma[200]).values; dn200 = (c < ma[200]).values
    r2 = rsi(c, 2).values; r14 = rsi(c, 14)
    S = {}
    for L in (2, 3, 5, 8): S[f"msc{L}"] = msc(h, l, c, L)
    for L in (3, 5, 8): S[f"elliott{L}"] = elliott(h, l, c, L)
    for N in (10, 20, 55):
        hi, lo = h.rolling(N).max().shift(), l.rolling(N).min().shift()
        S[f"donch{N}"] = (xup(c, hi), xup(-c, -lo))
    for a, b in ((5, 20), (10, 30), (20, 50), (50, 200)): S[f"ma{a}_{b}"] = (xup(ma[a], ma[b]), xup(ma[b], ma[a]))
    m = c.ewm(span=12).mean() - c.ewm(span=26).mean(); hs = m - m.ewm(span=9).mean(); z = pd.Series(0.0, index=c.index)
    S["macd"] = (xup(hs, z), xup(z, hs)); S["macd_trend"] = (xup(hs, z) & up200, xup(z, hs) & dn200)
    for t in (5, 10, 20): S[f"rsi2_{t}"] = ((r2 < t) & up200, (r2 > 100 - t) & dn200)
    sd = c.rolling(20).std(); S["boll_revert"] = (((c < ma[20] - 2 * sd).values & up200), ((c > ma[20] + 2 * sd).values & dn200))
    bbw = 4 * sd / ma[20]; sq = ((bbw <= bbw.rolling(252, min_periods=120).quantile(0.1)).rolling(5).max() > 0).values
    S["squeeze"] = (sq & xup(c, ma[20] + 2 * sd), sq & xup(-c, -(ma[20] - 2 * sd)))
    inside = ((h < h.shift()) & (l > l.shift())).shift().fillna(False).values
    S["inside_break"] = (inside & (c > h.shift()).values, inside & (c < l.shift()).values)
    gap = (o / c.shift() - 1).values; S["gap2"] = ((gap > 0.02) & (c > o).values, (gap < -0.02) & (c < o).values)
    S["high52"] = (xup(c, c.rolling(252).max().shift()), xup(-c, -c.rolling(252).min().shift()))
    vr = (v / v.rolling(50).mean()).values; rt = c.pct_change().values
    S["vol_surge"] = ((vr > 2) & (rt > 0.02), (vr > 2) & (rt < -0.02))
    A_ = atr(h, l, c); upm = h.diff(); dnm = -l.diff(); pdm = upm.where((upm > dnm) & (upm > 0), 0.0); mdm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1 / 14).mean() / A_; mdi = 100 * mdm.ewm(alpha=1 / 14).mean() / A_; dx = 100 * (pdi - mdi).abs() / (pdi + mdi); adx = dx.ewm(alpha=1 / 14).mean()
    S["adx_di"] = (xup(pdi, mdi) & (adx > 20).values, xup(mdi, pdi) & (adx > 20).values)
    ll, hh = l.rolling(14).min(), h.rolling(14).max(); k = 100 * (c - ll) / (hh - ll); dk = k.rolling(3).mean()
    S["stoch"] = (xup(k, dk) & (dk < 20).values & up200, xup(dk, k) & (dk > 80).values & dn200)
    tp_ = (h + l + c) / 3; cci = (tp_ - tp_.rolling(20).mean()) / (0.015 * tp_.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True))
    S["cci"] = (xup(cci, pd.Series(-100.0, index=c.index)) & up200, xup(pd.Series(100.0, index=c.index), cci) & dn200)
    ten = (h.rolling(9).max() + l.rolling(9).min()) / 2; kij = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa = ((ten + kij) / 2).shift(26); sb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    above = (c > np.maximum(sa, sb)).values; below = (c < np.minimum(sa, sb)).values
    S["ichimoku"] = (xup(ten, kij) & above, xup(kij, ten) & below)
    ema20 = c.ewm(span=20).mean(); S["keltner"] = (xup(c, ema20 + 2 * A_), xup(-c, -(ema20 - 2 * A_)))
    idx = c.index; me = pd.Series(idx.month, index=idx); last_day = (me != me.shift(-1)).values
    S["turn_of_month"] = (last_day & up200, Z.copy())
    pre = np.zeros(n, bool); pe_u = Z.copy(); pe_d = Z.copy(); cv = c.values
    for d in earn:
        kk = idx.searchsorted(d)
        if kk - 10 >= 0: pre[kk - 10] = True
        if 1 <= kk < n - 1:
            r = cv[kk + 1] / cv[kk - 1] - 1
            if r > 0.04: pe_u[kk + 1] = True
            if r < -0.04: pe_d[kk + 1] = True
    S["pre_earnings"] = (pre & up200, pre & dn200); S["pead"] = (pe_u, pe_d)
    s = spy.reindex(idx).ffill(); rs63 = (c.pct_change(63) - s.pct_change(63)).values
    S["strong_dip"] = (((c > ma[50]) & (ma[50] > ma[200])).values & (rs63 > 0) & (r2 < 10), ((c < ma[50]) & (ma[50] < ma[200])).values & (rs63 < 0) & (r2 > 90))
    S["baseline"] = (np.ones(n, bool), np.ones(n, bool))
    return S
