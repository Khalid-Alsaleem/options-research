import numpy as np
def exit_ret(p, Hh, tp, sl, trail):
    p = p[:, :Hh]; N, L = p.shape; valid = ~np.isnan(p)
    last = np.where(valid.any(1), L - 1 - np.argmax(valid[:, ::-1], 1), -1)
    hit = np.zeros_like(valid)
    if tp is not None: hit |= valid & (p >= tp)
    if sl is not None: hit |= valid & (p <= sl)
    if trail:
        pk = np.fmax.accumulate(np.nan_to_num(p, nan=-9), axis=1)
        hit |= valid & (pk >= 0.3) & (p <= pk * 0.5)
    first = np.where(hit.any(1), np.argmax(hit, 1), last)
    out = p[np.arange(N), np.clip(first, 0, L - 1)]; out[last < 0] = np.nan; return out
