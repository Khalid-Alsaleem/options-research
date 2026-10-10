import pandas as pd, numpy as np, sys
names={0:'A uptrend d.20 w5% t10%',1:'B pullback d.20 w5% t10%',2:'C uptrend d.25 w8% t15%'}
for f in sys.argv[1:]:
    d=pd.read_pickle(f); d=d[(d.N==30)&d.ret.notna()]; print(f, 'nan dropped', int(pd.read_pickle(f).ret.isna().sum())); print('==',f)
    for gi,g in d.groupby('gi'):
        g=g.copy(); g['exit']=g.date+pd.to_timedelta(np.ceil(g.days*1.45),'D')
        days=pd.bdate_range(g.date.min(), g.exit.max())
        conc=np.array([((g.date<=x)&(g.exit>x)).sum() for x in days[::3]])
        m=g.set_index('date').ret.resample('ME').mean()
        # portfolio: risk 1% of equity per trade, P&L booked at exit
        eq=1.0; peak=1; dd=0; yr={}
        for _,r in g.sort_values('exit').iterrows():
            eq*=1+0.01*r.ret; peak=max(peak,eq); dd=min(dd,eq/peak-1); yr[r.exit.year]=eq
        ys=pd.Series(yr); ann=ys.pct_change().fillna(ys.iloc[0]-1)
        print(names[gi],'| conc max',conc.max(),'med',int(np.median(conc)),
              '| worst month',round(m.min(),3),m.idxmin().strftime('%Y-%m'),'| neg months',f"{(m<0).mean():.0%}",
              '| p1 trade',round(g.ret.quantile(.01),2),'| stops',f"{(g.why=='stop').mean():.1%}",
              '| port@1%risk/yr',dict(ann.round(2)),'maxDD',round(dd,3))
