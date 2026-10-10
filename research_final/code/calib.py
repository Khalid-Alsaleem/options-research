import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import eng as E
root=sys.argv[1]; P=E.build(root); vrp,_,_=E.market(root)
D=pd.Timestamp("2024-01-03")
o=pd.read_csv(sys.argv[2], usecols=["ticker","stkPx","expirDate","strike","cBidPx","cAskPx","pBidPx","pAskPx","cOi","pOi","delta","smoothSmvVol","cMidIv","pMidIv"])
o["exp"]=pd.to_datetime(o.expirDate); o["dte"]=(o.exp-D).dt.days
mon=(o.exp.dt.weekday==4)&(o.exp.dt.day>=15)&(o.exp.dt.day<=21)
o=o[mon&(o.dte>=80)&(o.dte<=125)]
rows=[]
for sid,seg in P.items():
    px=seg["px"]
    if D not in px.index or not px.in_index.get(D,False): continue
    t=sid.split("__")[0]
    q=o[o.ticker==t.replace("-",".")]
    if q.empty: q=o[o.ticker==t]
    if q.empty: continue
    pr=E.prepare(seg,vrp); iv=pr.iv.loc[D]
    q=q[q.dte==q.dte.iloc[(q.dte-90).abs().argmin()]]
    for call in (True,False):
        dl=q.delta if call else 1-q.delta
        k=q.iloc[(dl-0.8).abs().argmin()]
        bid,ask=(k.cBidPx,k.cAskPx) if call else (k.pBidPx,k.pAskPx)
        if bid<=0: continue
        mid=(bid+ask)/2; S=k.stkPx
        model=E.bs(S,k.strike,k.dte/365,iv,call)
        rows.append(dict(t=t,call=call,S=S,K=k.strike,dte=k.dte,real_mid=mid,model=model,real_spread=(ask-bid)/mid,
                         model_iv=iv,real_iv=k.cMidIv if call else k.pMidIv,smv=k.smoothSmvVol,oi=k.cOi if call else k.pOi))
R=pd.DataFrame(rows); R["ratio"]=R.model/R.real_mid; R["iv_ratio"]=R.model_iv/R.smv
print(len(R)); print(R.groupby("call")[["ratio","iv_ratio","real_spread"]].median())
print(R[["ratio","iv_ratio","real_spread"]].describe(percentiles=[.1,.25,.5,.75,.9]).round(3))
print("vrp that day", round(vrp.loc[D],3))
R.to_csv("calib_20240103.csv",index=False)
