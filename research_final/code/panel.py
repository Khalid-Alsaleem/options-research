"""Panel (date x segment) matrices + indicator library computed on whole panels at once."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import eng as E

def load(root):
    P = E.build(root)
    P = {k: v for k, v in P.items() if k not in E.EXCLUDE}
    C = pd.DataFrame({k: v["px"].close for k, v in P.items()}).sort_index()
    H = pd.DataFrame({k: v["px"].high for k, v in P.items()}).reindex(C.index)
    L = pd.DataFrame({k: v["px"].low for k, v in P.items()}).reindex(C.index)
    V = pd.DataFrame({k: v["px"].volume for k, v in P.items()}).reindex(C.index)
    INIDX = pd.DataFrame({k: v["px"].in_index for k, v in P.items()}).reindex(C.index).fillna(False).astype(bool)
    spy = pd.read_csv(f"{root}/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close.reindex(C.index).ffill()
    vix = pd.read_csv(f"{root}/prices/VIX.csv", parse_dates=["Date"]).set_index("Date").Close.reindex(C.index).ffill()
    return dict(C=C, H=H, L=L, V=V, INIDX=INIDX, spy=spy, vix=vix)

def xup(a, b):
    a = np.asarray(a.values if hasattr(a, "values") else a, float)
    b = np.asarray(b.values if hasattr(b, "values") else b, float)
    if b.ndim == 1: b = b[:, None]
    shape = np.broadcast_shapes(a.shape, b.shape)
    a = np.broadcast_to(a, shape); b = np.broadcast_to(b, shape)
    nanrow = np.full((1, shape[1]), np.nan)
    pa = np.vstack([nanrow, a[:-1]]); pb = np.vstack([nanrow, b[:-1]])
    return (a > b) & (pa <= pb)

def rsi(C, n):
    d = C.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def triggers(D):
    """name -> (long_trigger, short_trigger) boolean arrays (date x seg)"""
    C, H, L = D["C"], D["H"], D["L"]; T = {}
    ma = {n: C.rolling(n).mean() for n in [5, 10, 20, 30, 50, 60, 100, 150, 200]}
    for a, b in [(5, 20), (10, 50), (20, 50), (20, 60), (30, 50), (30, 60), (50, 200)]:
        T[f"macross_{a}_{b}"] = (xup(ma[a], ma[b]), xup(ma[b], ma[a]))
    for n in [20, 50, 100, 200]:
        T[f"pricema_{n}"] = (xup(C, ma[n]), xup(ma[n], C))
    for f, s, g in [(12, 26, 9), (8, 17, 9), (5, 35, 5)]:
        m = C.ewm(span=f, adjust=False).mean() - C.ewm(span=s, adjust=False).mean(); sg = m.ewm(span=g, adjust=False).mean()
        T[f"macd_{f}_{s}_{g}"] = (xup(m, sg), xup(sg, m))
    for n, lv in [(2, 10), (2, 5), (3, 15), (5, 20), (14, 30)]:
        r = rsi(C, n); T[f"rsirev_{n}_{lv}"] = (xup(r, lv), xup(100 - lv, r))
    # oversold/overbought STATE entries (enter on the day the extreme happens, not the cross back)
    for n, lv in [(2, 10), (2, 5), (3, 10)]:
        r = rsi(C, n).values; T[f"rsiext_{n}_{lv}"] = (r < lv, r > 100 - lv)
    for n in [20, 55]:
        hi = H.rolling(n).max().shift(); lo = L.rolling(n).min().shift()
        T[f"donchian_{n}"] = (xup(C, hi), xup(lo, C))
    for n, k in [(20, 2.0)]:
        m_ = C.rolling(n).mean(); sd = C.rolling(n).std()
        T[f"bbrev_{n}_{k}"] = (xup(C, m_ - k * sd), xup(m_ + k * sd, C))
        T[f"bbbrk_{n}_{k}"] = (xup(C, m_ + k * sd), xup(m_ - k * sd, C))
    # short-term reversal: 5-day drop / rise of more than x%
    for n, x in [(5, 0.07), (5, 0.10), (3, 0.06)]:
        r = C.pct_change(n).values; T[f"rev_{n}_{int(x*100)}"] = (r < -x, r > x)
    # every eligible day (pure filter strategies)
    allv = np.ones(C.shape, bool); T["always"] = (allv, allv)
    return T

def filters(D):
    """name -> (long_ok, short_ok) boolean arrays. Side-aware states."""
    C = D["C"]; spy = D["spy"]; vix = D["vix"]; F = {"none": (np.ones(C.shape, bool),) * 2}
    for n in [50, 200]:
        m = C.rolling(n).mean(); F[f"stk>ma{n}"] = ((C > m).values, (C < m).values)
        sm = spy.rolling(n).mean(); up = (spy > sm).values[:, None] & np.ones(C.shape, bool)
        F[f"spy>ma{n}"] = (up, ~up)
    v = vix.values[:, None] & np.ones(C.shape, bool) if False else np.repeat(vix.values[:, None], C.shape[1], 1)
    F["vix<20"] = (v < 20, v < 20); F["vix>25"] = (v > 25, v > 25)
    ret126 = C.pct_change(126); rk = ret126.where(D["INIDX"]).rank(axis=1, pct=True).values
    F["mom_top20"] = (rk > 0.8, rk < 0.2)        # calls on winners, puts on losers
    F["mom_bot20"] = (rk < 0.2, rk > 0.8)        # contrarian
    ret21 = C.pct_change(21); rk21 = ret21.where(D["INIDX"]).rank(axis=1, pct=True).values
    F["st_loser20"] = (rk21 < 0.2, rk21 > 0.8)   # 1-month reversal
    hv = np.log(C).diff().rolling(20).std() * np.sqrt(252)
    hvr = hv.where(D["INIDX"]).rank(axis=1, pct=True).values
    F["lowvol_half"] = (hvr < 0.5, hvr < 0.5)
    rs = (C.pct_change(60).sub(spy.pct_change(60), axis=0)).values
    F["relstr60"] = (rs > 0, rs < 0)
    return F
