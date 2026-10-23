"""Six analysis 'schools' on daily stock charts, executed by BUYING a single call (up signal) or put (down signal).
Signal at close of day i -> enter next open. Exits at next open after the exit condition (no look-ahead).
Option: delta 0.6, expiry ~75 calendar days (50-110), forced exit when <=7 days left.
Pricing: Black-Scholes, IV = 1.2*(0.5*HV60+0.5*HV252) (calibrated on real chains), per-stock half-spread each side."""
import numpy as np, pandas as pd, sys
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad/auto"
sys.path.insert(0, A)
import single as SG
B = np.load(A + "/spread_coef.npy"); COST_STK = 0.0005; MAXBARS = 60
def load(t):
    d = pd.read_csv(f"/home/claude/options-research/data/eod/{t}.csv.gz", parse_dates=["date"]).set_index("date")
    d = d[~d.index.duplicated()].sort_index(); f = d.adjusted_close / d.close
    px = pd.DataFrame({"o": d.open * f, "h": d.high * f, "l": d.low * f, "c": d.adjusted_close})
    lr = np.log(px.c).diff()
    px["iv"] = (1.2 * (0.5 * lr.rolling(60).std() + 0.5 * lr.rolling(252, min_periods=120).std()) * np.sqrt(252)).clip(0.1, 1.5)
    dv = (d.close * d.volume).rolling(20).mean()
    px["rs"] = np.exp(B[0] + B[1] * np.log(dv) + B[2] * np.log(d.close)).clip(0.005, 0.25) * float(__import__("os").environ.get("SPREAD_MULT", "1"))
    return px.dropna(subset=["c"])
def rsi(c, n):
    d = c.diff(); u = d.clip(lower=0).ewm(alpha=1 / n).mean(); v = (-d.clip(upper=0)).ewm(alpha=1 / n).mean(); return 100 - 100 / (1 + u / v)
def cross_up(a, b): return (a > b) & (a.shift(1) <= b.shift(1))
def pivots(h, l, L):
    """confirmed alternating pivots: list of (kind, idx, price, known_at)"""
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
def setups(px, fam, p):
    """returns dict side -> (entry bool array, exit bool array, stop array, target array)"""
    o, h, l, c = (px[k] for k in "ohlc"); n = len(c); Z = np.zeros(n, bool); N = np.full(n, np.nan)
    ma200 = c.rolling(200).mean()
    if fam == "ma":
        f, s = c.rolling(p[0]).mean(), c.rolling(p[1]).mean(); up, dn = cross_up(f, s).values, cross_up(s, f).values
        return {1: (up, dn, N, N), -1: (dn, up, N, N)}
    if fam == "msc":
        hv, lv, cv = h.values, l.values, c.values; L = p[0]; sh = sl = np.nan; st = 0; up = Z.copy(); dn = Z.copy()
        for i in range(n):
            j = i - L
            if j - L >= 0:
                if hv[j] == hv[j - L:i + 1].max(): sh = hv[j]
                if lv[j] == lv[j - L:i + 1].min(): sl = lv[j]
            if cv[i] > sh and st <= 0: up[i] = st == -1; st = 1
            elif cv[i] < sl and st >= 0: dn[i] = st == 1; st = -1
        return {1: (up, dn, N, N), -1: (dn, up, N, N)}
    if fam == "donch":
        hi, lo = h.rolling(p[0]).max().shift(), l.rolling(p[0]).min().shift(); xl, xh = l.rolling(p[1]).min().shift(), h.rolling(p[1]).max().shift()
        return {1: ((c > hi).values, (c < xl).values, N, N), -1: ((c < lo).values, (c > xh).values, N, N)}
    if fam == "pullback":
        r = rsi(c, 2); a, b = p
        return {1: (((r < a) & (c > ma200)).values, (r > 70).values, N, N), -1: (((r > b) & (c < ma200)).values, (r < 30).values, N, N)}
    if fam == "macd":
        m = c.ewm(span=12).mean() - c.ewm(span=26).mean(); hs = m - m.ewm(span=9).mean(); z = pd.Series(0.0, index=c.index)
        up, dn = cross_up(hs, z), cross_up(z, hs)
        return {1: ((up & (c > ma200)).values, dn.values, N, N), -1: ((dn & (c < ma200)).values, up.values, N, N)}
    if fam == "elliott":
        P = pivots(h.values, l.values, p[0]); cv = c.values
        out = {1: [Z.copy(), Z.copy(), N.copy(), N.copy()], -1: [Z.copy(), Z.copy(), N.copy(), N.copy()]}
        for q in range(2, len(P)):
            (k1, i1, A_, _), (k2, i2, B_, _), (k3, i3, C_, known) = P[q - 2], P[q - 1], P[q]
            side = 1 if (k1, k2, k3) == ("L", "H", "L") else -1 if (k1, k2, k3) == ("H", "L", "H") else 0
            if side == 0: continue
            wave1 = (B_ - A_) * side; retr = (B_ - C_) * side / wave1 if wave1 > 0 else 0
            if not (0.382 <= retr <= 0.786): continue
            nxt = P[q + 1][3] if q + 1 < len(P) else n
            for i in range(known, min(known + 40, nxt, n)):
                if (cv[i] - B_) * side > 0:
                    out[side][0][i] = True; out[side][2][i] = C_; out[side][3][i] = C_ + side * 1.618 * wave1; break
        return {s: tuple(v) for s, v in out.items()}
FAMS = [("ma", (10, 30)), ("ma", (20, 50)), ("ma", (50, 200)), ("msc", (2,)), ("msc", (3,)), ("msc", (5,)),
        ("donch", (20, 10)), ("donch", (55, 20)), ("pullback", (10, 90)), ("pullback", (5, 95)), ("macd", ()),
        ("elliott", (3,)), ("elliott", (5,)), ("elliott", (8,))]
TP = [None, 0.5, 1.0]
def trades(px, fam, p, side):
    ent, ext, stp, tgt = setups(px, fam, p)[side]
    o, c = px.o.values, px.c.values; iv, rs = px.iv.values, px.rs.values; idx = px.index; n = len(c); res = []; i = 200
    while i < n - 2:
        if not ent[i] or not np.isfinite(iv[i]) or not np.isfinite(rs[i]): i += 1; continue
        e = i + 1; S0 = o[e]; ex = SG.pick_exp(idx[e], 50, 110, 75)
        if ex is None: i += 1; continue
        T0 = (ex - idx[e]).days / 365; call = side == 1; K = SG.strike(S0, T0, iv[i], call, 0.6)
        cost = SG.bs(S0, K, T0, SG.ivk(iv[i], S0, K, call), call) * (1 + rs[i] / 2)
        if cost <= 0: i += 1; continue
        stop, target = stp[i], tgt[i]; k = e; why = "time"
        # stock exit day (exit at open of day k+1 after condition at close k)
        while k < n - 1 and k - e < MAXBARS:
            if ext[k]: why = "signal"; break
            if np.isfinite(stop) and (c[k] - stop) * side < 0: why = "stop"; break
            if np.isfinite(target) and (c[k] - target) * side >= 0: why = "target"; break
            if (ex - idx[k + 1]).days <= 7: why = "expiry"; break
            k += 1
        xk = k + 1 if k + 1 < n else k
        # option value path on closes e..k, final at open xk
        js = np.arange(e, k + 1); Tj = np.maximum((ex - idx[js]).days.values, 0) / 365
        Kiv = SG.ivk(iv[js - 1], c[js], K, call)
        path = SG.bs(c[js], K, Tj, Kiv, call) * (1 - rs[i] / 2) / cost - 1
        fin = SG.bs(o[xk], K, max((ex - idx[xk]).days, 0) / 365, SG.ivk(iv[xk - 1], o[xk], K, call), call) * (1 - rs[i] / 2) / cost - 1
        stock = side * (o[xk] / S0 - 1) - 2 * COST_STK
        row = dict(entry=idx[e], exit=idx[xk], side=side, days=xk - e, why=why, stock=stock, opt=fin)
        for tp in TP[1:]:
            r = fin
            for q, v in enumerate(path):
                if v >= tp: r = v; break       # take profit at that close
                if v <= -0.5: r = v; break     # option stop -50%
            row[f"opt_tp{int(tp*100)}"] = r
        res.append(row); i = xk
    return pd.DataFrame(res)
