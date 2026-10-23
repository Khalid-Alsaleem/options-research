"""Daily call-option scanner (research result of round 8). Runs on GitHub Actions, never on a personal/work computer.
Signals (calls only, market filter SPY 50-day MA above 200-day MA):
  F4 high52    : close breaks above its prior 52-week closing high           -> 1-year call ~10% OTM, take profit +200%, else exit after 6 months
  F1 strong_dip: close>MA50>MA200, 3-month return beats SPY, RSI(2) < 10   -> 1-year call ~10% OTM, take profit +200%, else exit after 6 months
  F2 gap3      : opens >3% above prior close and closes above the open     -> 1-year call ~10% ITM, exit after 6 months
Universe: current S&P 500 members, top 100 by 20-day dollar volume. No entry within 10 trading days before earnings (when the date is known).
Secrets: EODHD_API_TOKEN (required); TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (optional). Nothing is printed from secrets."""
import os, sys, datetime as dt, numpy as np, pandas as pd, requests
TOKEN = os.environ.get("EODHD_API_TOKEN"); OUT = "scanner/signals.md"
def eod(sym, start):
    r = requests.get(f"https://eodhd.com/api/eod/{sym}.US", params={"api_token": TOKEN, "fmt": "json", "from": start}, timeout=60)
    if r.status_code != 200: return None
    d = pd.DataFrame(r.json())
    if d.empty: return None
    d["date"] = pd.to_datetime(d["date"]); d = d.set_index("date"); f = d.adjusted_close / d.close
    return pd.DataFrame({"open": d.open * f, "high": d.high * f, "low": d.low * f, "close": d.adjusted_close, "volume": d.volume, "raw_close": d.close})
def rsi(c, n):
    x = c.diff(); u = x.clip(lower=0).ewm(alpha=1 / n).mean(); v = (-x.clip(upper=0)).ewm(alpha=1 / n).mean(); return 100 - 100 / (1 + u / v)
def signals(px, spy):
    c, o = px.close, px.open; ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    hi = c.rolling(252).max().shift(); s = spy.reindex(c.index).ffill(); rs63 = c.pct_change(63) - s.pct_change(63)
    gap = o / c.shift() - 1
    return {"F4 high52": bool(c.iloc[-1] > hi.iloc[-1] and c.iloc[-2] <= hi.iloc[-2]),
            "F1 strong_dip": bool(c.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1] and rs63.iloc[-1] > 0 and rsi(c, 2).iloc[-1] < 10),
            "F2 gap3": bool(gap.iloc[-1] > 0.03 and c.iloc[-1] > o.iloc[-1])}
CONTRACT = {"F4 high52": (0.10, "+200% أو 6 أشهر"), "F1 strong_dip": (0.10, "+200% أو 6 أشهر"), "F2 gap3": (-0.10, "6 أشهر")}
def earnings_soon(sym):
    try:
        today = dt.date.today(); r = requests.get("https://eodhd.com/api/calendar/earnings", params={"api_token": TOKEN, "fmt": "json", "symbols": f"{sym}.US",
                     "from": str(today), "to": str(today + dt.timedelta(days=15))}, timeout=60)
        return bool(r.json().get("earnings")) if r.status_code == 200 else None
    except Exception: return None
def universe():
    u = pd.read_csv("data/universe/membership.csv", parse_dates=["start", "end"]); cur = u[u.end.isna()].ticker.unique()
    return [t for t in cur if "-" not in t or True]
if __name__ == "__main__":
    if not TOKEN: sys.exit("EODHD_API_TOKEN missing")
    start = str(dt.date.today() - dt.timedelta(days=500))
    spy = eod("SPY", start).close; regime = spy.rolling(50).mean().iloc[-1] > spy.rolling(200).mean().iloc[-1]
    data = {}
    for t in universe():
        px = eod(t.replace(".", "-"), start)
        if px is not None and len(px) > 260: data[t] = px
    dv = pd.Series({t: (p.raw_close * p.volume).tail(20).mean() for t, p in data.items()}).sort_values(ascending=False)
    top = list(dv.index[:100]); rows = []
    for t in top:
        sg = signals(data[t], spy)
        for k, on in sg.items():
            if not on: continue
            soon = earnings_soon(t.replace(".", "-")); mny, ex = CONTRACT[k]; price = data[t].raw_close.iloc[-1]
            rows.append(f"| {t} | {k} | {price:.2f} | {price * (1 + mny):.0f} | ~سنة | {ex} | {'⚠️ أرباح قريبة — تجاهل' if soon else ('غير معروف' if soon is None else 'لا')} |")
    asof = spy.index[-1].date()
    head = f"# إشارات {asof}\n\nفلتر السوق (SPY متوسط 50 فوق 200): {'✅ مفعّل' if regime else '❌ متوقف — لا صفقات جديدة'}\n\n"
    body = "| السهم | الإشارة | السعر | السترايك المقترح | الانتهاء | الخروج | أرباح خلال 10 أيام |\n| --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows) if (rows and regime) else "لا توجد إشارات اليوم."
    open(OUT, "w").write(head + body + "\n"); print(head + body)
    bt, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if bt and chat:
        requests.post(f"https://api.telegram.org/bot{bt}/sendMessage", data={"chat_id": chat, "text": head + body}, timeout=60)
