import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import eng as E
root=sys.argv[1]; P=E.build(root); vrp,_,_=E.market(root)
from math import sqrt
res=[]
def hvs(px,D):
    lr=np.log(px.close).diff()
    return {n: lr.rolling(n).std().loc[D]*sqrt(252) for n in (10,20,30,60,120,252)}
# date 1: ORATS smooth IV at ~90d, delta~0.8 call strike
D1=pd.Timestamp("2024-01-03")
o=pd.read_csv(sys.argv[2],usecols=["ticker","expirDate","delta","smoothSmvVol","cMidIv","strike","stkPx"])
o["dte"]=(pd.to_datetime(o.expirDate)-D1).dt.days; o=o[(o.dte>=70)&(o.dte<=110)]
o=o.iloc[(o.delta-0.8).abs().argsort()].drop_duplicates("ticker")
iv1=dict(zip(o.ticker,o.smoothSmvVol))
# date 2: HOD IV, calls, ~90d, delta~0.8 (use previous close data date 2026-03-18)
D2=pd.Timestamp("2026-03-18")
h=pd.read_csv(sys.argv[3]); h.columns=[c.strip() for c in h.columns]
h=h[(h.Type=="call")&(h.IV>0)&(h.Bid>0)]
h["dte"]=(pd.to_datetime(h.Expiration)-pd.Timestamp("2026-03-19")).dt.days; h=h[(h.dte>=70)&(h.dte<=110)]
h=h.iloc[(h.Delta-0.8).abs().argsort()].drop_duplicates("UnderlyingSymbol")
iv2=dict(zip(h.UnderlyingSymbol,h.IV))
for sid,seg in P.items():
    px=seg["px"]; t=sid.split("__")[0]
    for D,ivm in ((D1,iv1),(D2,iv2)):
        if D in px.index and px.in_index.get(D,False) and t in ivm and ivm[t]>0:
            r=dict(t=t,date=D,iv=ivm[t],vrp=vrp.loc[D]); r.update({f"hv{n}":v for n,v in hvs(px,D).items()}); res.append(r)
R=pd.DataFrame(res).dropna()
for D,g in R.groupby("date"):
    print(D.date(), len(g), "vrp", round(g.vrp.iloc[0],2), {c: round((g.iv/g[c]).median(),2) for c in ["hv20","hv30","hv60","hv120","hv252"]})
    for blend in ["hv30","hv60","hv252"]:
        pass
# candidate model: IV = k * (0.5*hv60 + 0.5*hv252); find k per date and error dispersion
for name,f in {"hv30*vrp":lambda g:g.hv30*g.vrp,"hv30":lambda g:g.hv30,"0.5hv60+0.5hv252":lambda g:0.5*g.hv60+0.5*g.hv252,"(hv20+hv60+hv252)/3":lambda g:(g.hv20+g.hv60+g.hv252)/3}.items():
    for D,g in R.groupby("date"):
        r=g.iv/f(g); print(f"{name:22s}",D.date(),"median",round(r.median(),3),"IQR",round(r.quantile(.25),2),round(r.quantile(.75),2),"MAPE",round((abs(np.log(r))).median(),3))
R.to_csv("iv_calib.csv",index=False)
