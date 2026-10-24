"""SMC options backtest engine (buy calls/puts only).
Trend: higher frame (Daily for H1/H4 entries; Weekly for the 10-year Daily-entry proxy) AND entry frame must agree:
  bullish = last break of structure (BOS) was up AND last two confirmed swing highs/lows are higher (HH+HL); bearish mirrors.
Entry: first touch of an unmitigated Order Block created by the latest BOS in trend direction (limit order at the OB edge),
  confluence >= k of 4: OB (always true at an OB touch), FVG in the displacement leg, liquidity sweep before entry, discount/premium (<50% / >50% of the leg).
Exit (stock-level, then priced on the option): structural stop = entry-frame CLOSE beyond the OB far edge;
  50% off at the nearest opposing liquidity (last confirmed swing high for longs) and stop to breakeven (close back through entry);
  forced exit when 10 calendar days remain to expiry.
Contract: 30-45 days to expiry (closest to 37), delta closest to 0.55 with premium <= $20/share; if needed move OTM down to delta 0.30, else skip.
No look-ahead: swings are known only L bars after the pivot; higher-frame trend uses the previous COMPLETED higher-frame bar."""
import numpy as np, pandas as pd, sys
A = "/tmp/claude-0/-home-claude/8b9291c0-456b-5b05-ba74-483766727c6b/scratchpad"; sys.path.insert(0, A + "/auto")
import single as SG
FRI = pd.date_range("2015-01-02", "2027-12-31", freq="W-FRI")
def structure(h, l, c, L):
    """per bar: state (+1/-1 after BOS), hhhl (bull pattern), lhll (bear pattern), last swing high/low values+idx known at bar"""
    n = len(c); sh = np.full(n, np.nan); sl = np.full(n, np.nan); shi = np.full(n, -1); sli = np.full(n, -1)
    st = np.zeros(n, int); bull = np.zeros(n, bool); bear = np.zeros(n, bool); bos_up = np.zeros(n, bool); bos_dn = np.zeros(n, bool)
    H = []; Lo = []; cur = 0; lh = np.nan; ll = np.nan; lhi = -1; lli = -1
    for k in range(n):
        j = k - L
        if j - L >= 0:
            if h[j] == h[j - L:k + 1].max(): H.append(h[j]); lh, lhi = h[j], j
            if l[j] == l[j - L:k + 1].min(): Lo.append(l[j]); ll, lli = l[j], j
        if not np.isnan(lh) and c[k] > lh and cur <= 0: cur = 1; bos_up[k] = True
        elif not np.isnan(ll) and c[k] < ll and cur >= 0: cur = -1; bos_dn[k] = True
        st[k] = cur; sh[k], sl[k], shi[k], sli[k] = lh, ll, lhi, lli
        if len(H) >= 2 and len(Lo) >= 2:
            bull[k] = H[-1] > H[-2] and Lo[-1] > Lo[-2]; bear[k] = H[-1] < H[-2] and Lo[-1] < Lo[-2]
    return dict(st=st, bull=bull, bear=bear, sh=sh, sl=sl, shi=shi, sli=sli, bos_up=bos_up, bos_dn=bos_dn)
def find_setups(df, htf_up, htf_dn, L, max_age, kmin, news_block):
    """df: entry-frame bars (o,h,l,c). returns list of dict setups with fill bar/price, stop, tp, side, confluence flags"""
    o, h, l, c = (df[x].values for x in ("o", "h", "l", "c")); n = len(c); S = structure(h, l, c, L); out = []
    for side in (1, -1):
        bos = S["bos_up"] if side == 1 else S["bos_dn"]
        for b in np.flatnonzero(bos):
            # displacement leg starts at the last confirmed opposite swing before the break
            start = S["sli"][b] if side == 1 else S["shi"][b]
            if start < 0: continue
            seg = np.arange(start, b + 1)
            opp = seg[(c[seg] < o[seg])] if side == 1 else seg[(c[seg] > o[seg])]
            if not len(opp): continue
            ob = opp[-1]; ob_hi, ob_lo = h[ob], l[ob]
            edge, far = (ob_hi, ob_lo) if side == 1 else (ob_lo, ob_hi)
            # FVG inside the displacement after the OB
            fvg = any((l[i] > h[i - 2]) if side == 1 else (h[i] < l[i - 2]) for i in range(ob + 2, min(b + 3, n)))
            leg_lo = l[start:b + 1].min() if side == 1 else None
            for e in range(b + 1, min(b + 1 + max_age, n)):
                # trend must still agree (entry frame state and pattern, higher frame)
                okt = (S["st"][e - 1] == side) and (S["bull"][e - 1] if side == 1 else S["bear"][e - 1]) and (htf_up[e] if side == 1 else htf_dn[e])
                touched = (l[e] <= edge) if side == 1 else (h[e] >= edge)
                if touched:
                    if not okt or news_block[e]: break
                    if (side == 1 and o[e] < far) or (side == -1 and o[e] > far): break      # gapped through the whole OB
                    fill = min(o[e], edge) if side == 1 else max(o[e], edge)
                    ext = h[b:e].max() if side == 1 else l[b:e].min()
                    lo_ = l[start:e].min() if side == 1 else None; hi_ = h[start:e].max() if side == -1 else None
                    mid = (lo_ + ext) / 2 if side == 1 else (hi_ + ext) / 2
                    disc = (fill < mid) if side == 1 else (fill > mid)
                    w0 = max(0, ob - 5)
                    ref = S["sl"][w0] if side == 1 else S["sh"][w0]
                    sweep = False
                    for q in range(w0, e + 1):
                        r = S["sl"][q - 1] if side == 1 else S["sh"][q - 1]
                        if np.isfinite(r) and ((l[q] < r and c[q] > r) if side == 1 else (h[q] > r and c[q] < r)): sweep = True; break
                    score = 1 + int(fvg) + int(sweep) + int(disc)
                    tp = S["sh"][e - 1] if side == 1 else S["sl"][e - 1]
                    if not np.isfinite(tp) or (tp - fill) * side <= 0: tp = ext
                    if score >= kmin and (tp - fill) * side > 0:
                        out.append(dict(side=side, e=e, fill=fill, far=far, tp=tp, fvg=fvg, sweep=sweep, disc=disc, score=score, b=b))
                    break
    return out
def manage(df, s):
    """stock-level path: returns (tp_bar or -1, exit_bar, exit_price) without the expiry rule (applied in pricing)."""
    h, l, c = df.h.values, df.l.values, df.c.values; side = s["side"]; n = len(c); tpb = -1
    for k in range(s["e"], n):
        if tpb < 0:
            if k > s["e"] and (c[k] - s["far"]) * side < 0: return tpb, k, c[k], "stop"
            if (h[k] >= s["tp"]) if side == 1 else (l[k] <= s["tp"]): tpb = k; continue
        else:
            if (c[k] - s["fill"]) * side <= 0: return tpb, k, c[k], "breakeven"
    return tpb, n - 1, c[n - 1], "end"
def price_trade(df, s, iv_d, rs_d, fac_d, spread_mult=1.0, comm=0.65, dte_rng=(30, 45)):
    """price the option leg; df index = timestamps (America/New_York). returns dict of gross/net returns."""
    side = s["side"]; call = side == 1; t0 = df.index[s["e"]]; day = t0.normalize()
    ex = FRI[(FRI >= day + pd.Timedelta(days=dte_rng[0])) & (FRI <= day + pd.Timedelta(days=dte_rng[1]))]
    if not len(ex): return None
    ex = ex[np.argmin(np.abs((ex - day).days - 37))] + pd.Timedelta(hours=16)
    iv = iv_d.asof(day - pd.Timedelta(days=1)); rs = rs_d.asof(day - pd.Timedelta(days=1)); f = fac_d.asof(day)
    if not (np.isfinite(iv) and np.isfinite(rs) and np.isfinite(f)): return None
    S0 = s["fill"]; Sraw = S0 * f; T = lambda ts: max((ex - ts).total_seconds(), 3600) / (365 * 86400)
    st = 0.5 if Sraw < 25 else 1 if Sraw < 200 else 2.5 if Sraw < 500 else 5
    ks = (np.round(Sraw / st) * st + side * st * np.arange(-15, 40)) / f
    dl = np.abs(SG.dlt(S0, ks, T(t0), SG.ivk(iv, S0, ks, call), call)); pr = SG.bs(S0, ks, T(t0), SG.ivk(iv, S0, ks, call), call) * f
    ok = (pr <= 20) & (dl >= 0.30) & (dl <= 0.75)
    if not ok.any(): return None
    K = ks[ok][np.argmin(np.abs(dl[ok] - 0.55))]
    V = lambda Sx, ts: SG.bs(Sx, K, T(ts), SG.ivk(iv, Sx, K, call), call) * f
    p0 = V(S0, t0); half = lambda p: max(rs * p, 0.05) / 2 * spread_mult; cm = comm / 100 * spread_mult
    tpb, xb, xp, why = manage(df, s)
    # forced exit 10 days before expiry
    fx = np.flatnonzero(df.index[s["e"]:] >= ex - pd.Timedelta(days=10))
    if len(fx):
        fb = s["e"] + fx[0]
        if fb < xb: xb, xp, why = fb, df.c.values[fb], "expiry10d"
        if tpb >= fb: tpb = -1
    legs = []
    if tpb >= 0: legs.append((0.5, V(s["tp"], df.index[tpb]))); legs.append((0.5, V(xp, df.index[xb])))
    else: legs.append((1.0, V(xp, df.index[xb])))
    gross = sum(w * (v / p0 - 1) for w, v in legs)
    cost = p0 + half(p0) + cm; net = sum(w * ((v - half(v) - cm) / cost - 1) for w, v in legs)
    return dict(t=t0, side=side, premium=round(p0 * 100), delta=float(dl[ks == K][0]), gross=gross, net=net, why=why, tp_hit=tpb >= 0,
                bars=xb - s["e"], score=s["score"])
