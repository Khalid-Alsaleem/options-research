"""HOLDOUT, opened once. Pre-committed before running (2026-10-11 00:xx Riyadh):
primary = delta 0.30, width 5%, hold to 7 DTE, stop 2x credit, only when SPY > MA200; secondary = same with delta 0.35."""
import sys, pandas as pd
import credit as C
root=sys.argv[1]
spy=pd.read_csv(f"{root}/prices/SPY.csv",parse_dates=["Date"]).set_index("Date").Close
vix=pd.read_csv(f"{root}/prices/VIX.csv",parse_dates=["Date"]).set_index("Date").Close.reindex(spy.index).ffill()
out=[]
for name,d in [("primary",0.30),("secondary",0.35)]:
    for per,(a,b) in {"learn":("2016-01-01","2022-11-30"),"holdout":("2023-01-01","2026-08-28")}.items():
        T=C.run(spy,vix,d,0.05,"hold",True,a,b); s=C.summary(T); s.update(config=name,period=per); out.append(s)
        T.to_csv(f"credit_trades_{name}_{per}.csv",index=False)
        if per=="holdout": print(name, T.groupby(T.date.dt.year).ret.agg(["count","mean"]).round(3).to_dict())
R=pd.DataFrame(out); pd.set_option("display.width",250); print(R.round(3).to_string()); R.to_csv("credit_holdout.csv",index=False)
