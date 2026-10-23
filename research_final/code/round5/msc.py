"""MSC (market structure change) engine, daily bars, no look-ahead.
Swing high at bar j = high[j] is the max of high[j-L .. j+L]; it is only KNOWN at bar j+L.
Structure: bullish MSC = close above the last confirmed swing high while structure was bearish
(last break was downward); bearish MSC = mirror. Entry next day's open.
Long trade: stop = last confirmed swing low at signal; exit at target (R multiple) / stop / opposite MSC."""
import numpy as np, pandas as pd, itertools
COST = 0.0005  # per side
def load(t, root="/home/claude/options-research/data/eod"):
    d = pd.read_csv(f"{root}/{t}.csv.gz", parse_dates=["date"]).set_index("date")
    f = d.adjusted_close / d.close
    return pd.DataFrame({"o": d.open * f, "h": d.high * f, "l": d.low * f, "c": d.adjusted_close})
def signals(px, L):
    h, l, c = px.h.values, px.l.values, px.c.values; n = len(c)
    sh = sl = np.nan; state = 0; sig = np.zeros(n, int); SH = np.full(n, np.nan); SL = np.full(n, np.nan)
    for i in range(n):
        j = i - L  # bar whose swing status becomes known today
        if j - L >= 0:
            if h[j] == h[j - L:i + 1].max(): sh = h[j]
            if l[j] == l[j - L:i + 1].min(): sl = l[j]
        if not np.isnan(sh) and c[i] > sh and state <= 0:
            sig[i] = 1 if state == -1 else 0; state = 1
        elif not np.isnan(sl) and c[i] < sl and state >= 0:
            sig[i] = -1 if state == 1 else 0; state = -1
        SH[i], SL[i] = sh, sl
    return sig, SH, SL
def backtest(px, L, R, trend, side="long", start=None, end=None):
    sig, SH, SL = signals(px, L); o, h, l, c = (px[k].values for k in "ohlc")
    ma = px.c.rolling(200).mean().values; idx = px.index; out = []; i = 0; n = len(c)
    lo = idx.searchsorted(pd.Timestamp(start)) if start else 0; hi = idx.searchsorted(pd.Timestamp(end)) if end else n
    want = 1 if side == "long" else -1
    i = max(lo, 1)
    while i < min(hi, n - 1):
        if sig[i] == want and (not trend or (c[i] > ma[i] if want == 1 else c[i] < ma[i])):
            e = o[i + 1]; stop = SL[i] if want == 1 else SH[i]
            risk = (e - stop) * want
            if not np.isfinite(risk) or risk <= 0 or risk / e > 0.25: i += 1; continue
            tgt = e + want * R * risk if R else None
            k = i + 1; ex = None
            while k < n:
                if want == 1 and l[k] <= stop: ex = (min(o[k], stop), "stop"); break
                if want == -1 and h[k] >= stop: ex = (max(o[k], stop), "stop"); break
                if tgt and want == 1 and h[k] >= tgt: ex = (max(o[k], tgt), "target"); break
                if tgt and want == -1 and l[k] <= tgt: ex = (min(o[k], tgt), "target"); break
                if sig[k] == -want and k + 1 < n: ex = (o[k + 1], "msc"); k += 1; break
                k += 1
            if ex is None: break
            ret = want * (ex[0] / e - 1) - 2 * COST
            out.append(dict(entry=idx[i + 1], exit=idx[min(k, n - 1)], ret=ret, R=ret * e / risk, why=ex[1], days=k - i))
            i = k
        else: i += 1
    return pd.DataFrame(out)
GRID = dict(L=[2, 3, 5, 8, 12], R=[1.5, 2, 3, 0], trend=[False, True])
def stats(T, yrs):
    if len(T) == 0: return dict(n=0, per_year=0, win=np.nan, avg=np.nan, avgR=np.nan, cagr=0.0)
    eq = np.prod(1 + T.ret.values)
    return dict(n=len(T), per_year=len(T) / yrs, win=(T.ret > 0).mean(), avg=T.ret.mean(), avg_win=T.ret[T.ret > 0].mean(),
                avg_loss=T.ret[T.ret <= 0].mean(), avgR=T.R.mean(), total=eq - 1, cagr=eq ** (1 / yrs) - 1,
                maxdd=(np.cumprod(1 + T.ret.values) / np.maximum.accumulate(np.cumprod(1 + T.ret.values)) - 1).min())
