"""Single-stock option strategies with ~10% profit target and MANY trades.
Costs: real per-stock bid-ask spread model (fit on ORATS 2024-01-03: log spread ~ log $volume + log price),
charged as half-spread per leg on entry and on exit. IV: calibrated model (1.2 x blend HV60/HV252) + put skew.
Universe: point-in-time S&P 500 members ranked by 20-day dollar volume (top N that day). Earnings excluded."""
import sys, numpy as np, pandas as pd, warnings, itertools, pickle
from scipy.special import ndtr
warnings.filterwarnings("ignore")
import eng as E
R_ = 0.02; B = np.load("spread_coef.npy")
EXP = E.EXP
def bs(S, K, T, s, call):
    T = np.maximum(T, 1e-6); v = s * np.sqrt(T); d1 = (np.log(S / K) + (R_ + s * s / 2) * T) / v; d2 = d1 - v
    return S * ndtr(d1) - K * np.exp(-R_ * T) * ndtr(d2) if call else K * np.exp(-R_ * T) * ndtr(-d2) - S * ndtr(-d1)
def dlt(S, K, T, s, call):
    d1 = (np.log(S / K) + (R_ + s * s / 2) * T) / (s * np.sqrt(T)); return ndtr(d1) if call else ndtr(d1) - 1
def ivk(iv, S, K, call):  # put-side skew for strikes below spot
    return iv + 0.4 * np.maximum(0, np.log(S / K))

def load(root):
    P = {k: v for k, v in E.build(root).items() if k not in E.EXCLUDE}
    vrp, _, _ = E.market(root)
    spy = pd.read_csv(f"{root}/prices/SPY.csv", parse_dates=["Date"]).set_index("Date").Close
    for s in P.values():
        px = E.prepare(s, vrp); px["dv20"] = (px.close * px.volume).rolling(20).mean()
        px["rs"] = np.exp(B[0] + B[1] * np.log(px.dv20) + B[2] * np.log(px.close)).clip(0.005, 0.25) * float(__import__("os").environ.get("SPREAD_MULT", "1"))
        s["prep"] = px
    dv = pd.DataFrame({k: v["prep"].dv20.where(v["prep"].in_index) for k, v in P.items()})
    rank = dv.rank(axis=1, ascending=False)
    for k, v in P.items(): v["prep"]["dvrank"] = rank[k].reindex(v["prep"].index)
    return P, spy

def pick_exp(d, lo, hi, tgt):
    dd = (EXP - d).days.values; ok = (dd >= lo) & (dd <= hi)
    return None if not ok.any() else EXP[ok][np.argmin(np.abs(dd[ok] - tgt))]

def strike(S, T, iv, call, d):
    st = E.step(S); ks = np.round(S / st) * st + st * np.arange(-20, 21); ks = ks[ks > 0]
    dl = np.abs(dlt(S, ks, T, ivk(iv, S, ks, call), call)); return ks[np.argmin(np.abs(dl - d))]

def trade(px, i, kind, p):
    """kind: long_call, long_put, put_credit, call_debit. Returns return on capital at risk."""
    d = px.index[i]; S = px.close.iat[i]; iv = px.iv.iat[i]; rs = px.rs.iat[i]
    if not np.isfinite(iv) or not np.isfinite(rs): return None
    ex = pick_exp(d, p["dte"] - 10, p["dte"] + 20, p["dte"])
    if ex is None: return None
    T = (ex - d).days / 365
    if kind in ("long_call", "long_put"):
        call = kind == "long_call"; K = strike(S, T, iv, call, p["delta"])
        legs = [(K, call, +1)]
    elif kind == "put_credit":
        Ks = strike(S, T, iv, False, p["delta"]); Kl = Ks - max(E.step(S), np.round(S * p["width"] / E.step(S)) * E.step(S))
        legs = [(Ks, False, -1), (Kl, False, +1)]
    else:  # call_debit
        K1 = strike(S, T, iv, True, p["delta"]); K2 = strike(S, T, iv, True, p["delta2"])
        if K2 <= K1: return None
        legs = [(K1, True, +1), (K2, True, -1)]
    def value(Sx, Tx, ivx, side_cost):
        v = 0.0; c = 0.0
        for K, call, q in legs:
            m = bs(Sx, K, Tx, ivk(ivx, Sx, K, call), call); v += q * m; c += rs / 2 * m
        return v, c
    v0, c0 = value(S, T, iv, 1)
    if kind == "put_credit":
        credit = -v0 - c0; width = legs[0][0] - legs[1][0]; risk = width - credit
        if credit <= 0.02 * width or risk <= 0: return None
    else:
        cost = v0 + c0; risk = cost
        if cost <= 0: return None
    for k in range(1, p["hold"] + 1):
        j = i + k
        if j >= len(px): return None
        Tj = (ex - px.index[j]).days / 365
        vj, cj = value(px.close.iat[j], Tj, px.iv.iat[j], -1)
        pnl = (credit - (-vj + cj)) if kind == "put_credit" else (vj - cj - cost)
        r = pnl / risk
        if r >= p["target"]: return r, k, "target"
        if r <= -p["stop"]: return r, k, "stop"
        if (ex - px.index[j]).days <= 7: return r, k, "dte"
    return r, p["hold"], "time"
