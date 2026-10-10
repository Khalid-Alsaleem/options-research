import sys, itertools, pandas as pd, numpy as np
import credit as C
root=sys.argv[1]
spy=pd.read_csv(f"{root}/prices/SPY.csv",parse_dates=["Date"]).set_index("Date").Close
vix=pd.read_csv(f"{root}/prices/VIX.csv",parse_dates=["Date"]).set_index("Date").Close.reindex(spy.index).ffill()
rows=[]
for d,w,mf in itertools.product([0.25,0.35,0.40],[0.03,0.05],[False,True]):
    T=C.run(spy,vix,d,w,"hold",mf,"2016-01-01","2022-11-30"); r=dict(delta=d,width=w,manage="hold",spy_filter=mf); r.update(C.summary(T)); rows.append(r)
# sensitivity of the pre-chosen config (0.30, 5%, hold, SPY>MA200) to skew and slippage
base_iv=C.iv
for sk in [0.3,0.9]:
    C.iv=lambda v,S,K,sk=sk: np.maximum(v/100-0.015,0.06)+sk*np.maximum(0,np.log(S/K))
    T=C.run(spy,vix,0.30,0.05,"hold",True,"2016-01-01","2022-11-30"); r=dict(delta=0.30,width=0.05,manage=f"skew{sk}",spy_filter=True); r.update(C.summary(T)); rows.append(r)
C.iv=base_iv
for sl in [0.01,0.06]:
    C.SLIP=sl; T=C.run(spy,vix,0.30,0.05,"hold",True,"2016-01-01","2022-11-30"); r=dict(delta=0.30,width=0.05,manage=f"slip{sl}",spy_filter=True); r.update(C.summary(T)); rows.append(r)
C.SLIP=0.03
R=pd.DataFrame(rows); R.to_csv("credit_learn2.csv",index=False)
pd.set_option("display.width",250); print(R.round(3).to_string())
