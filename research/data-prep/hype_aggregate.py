"""Aggregate tweet hype to the coin-day, then ask: does aggregate hype move the coin?"""
import os
import numpy as np, pandas as pd
from scipy import stats
ROOT="/home/ubuntu/Projects/hypecheck"
M=pd.read_csv("/tmp/hype_meter_rows.csv")
M["avail_day"]=pd.to_datetime(M.avail_day); M["trading_day"]=pd.to_datetime(M.trading_day)

# same-day performance for the hype day, from the price files
def prices(coin):
    p=f"/tmp/meme-prices/{coin}-prices.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p, parse_dates=["date"]).assign(day=lambda x: x.date.dt.date)
    return d.set_index("day").sort_index()
PX={c:prices(c) for c in M.coin.unique()}
def bar(coin, day):
    d=PX.get(coin)
    if d is None: return None
    try: return d.loc[day]
    except KeyError: return None

rows=[]
for r in M.itertuples():
    b=bar(r.coin, r.avail_day.date())
    if b is None: continue
    rows.append(dict(tweet_id=r.tweet_id, coin=r.coin, day=r.avail_day.date(),
                     hype=r.hype, hype_abs=r.hype_abs, ev_bull=r.ev_bull,
                     same_oc=float(b.close/b.open-1.0), same_abs=abs(float(b.close/b.open-1.0))))
R=pd.DataFrame(rows)

# --- the coin-day aggregate: total hype about a coin on a day ---
A=R.groupby(["coin","day"]).agg(
    n_tweets=("tweet_id","size"),
    hype_sum=("hype","sum"),        # net signed hype for the coin that day
    hype_mean=("hype","mean"),
    hype_max=("hype","max"), hype_min=("hype","min"),
    hype_volume=("hype_abs","sum"), # total hype regardless of direction
    bull_terms=("ev_bull","sum"),
    same_oc=("same_oc","first"), same_abs=("same_abs","first")).reset_index()
# forward performance of the coin after the hype day
fwd=[]
for r in A.itertuples():
    d=PX.get(r.coin);  b=bar(r.coin, r.day)
    if d is None or b is None: continue
    idx=list(d.index); 
    try: k=idx.index(r.day)
    except ValueError: continue
    rec=dict(coin=r.coin, day=r.day)
    for h in (1,3,7):
        if k+h < len(idx):
            c0=d.iloc[k].close; rec[f"ret_{h}d"]=float(d.iloc[k+h].close/c0-1.0)
    fwd.append(rec)
F=pd.DataFrame(fwd)
A=A.merge(F, on=["coin","day"], how="left")
A.to_csv("/tmp/coin_day_hype.csv", index=False)
print(f"coin-days: {len(A):,} over {A.coin.nunique()} coins")
print(f"  tweets per coin-day: median {A.n_tweets.median():.0f}  mean {A.n_tweets.mean():.2f}  max {A.n_tweets.max()}")
print(f"  hype_sum per day: mean {A.hype_sum.mean():+.3f}  sd {A.hype_sum.std():.3f}  min {A.hype_sum.min():+.2f}  max {A.hype_sum.max():+.2f}")
print(f"  hype_volume per day: mean {A.hype_volume.mean():.3f}  max {A.hype_volume.max():.2f}")

print("\n" + "="*92)
print("DOES AGGREGATE HYPE MOVE THE COIN?  (Spearman across coin-days)")
print("="*92)
pairs=[("hype_sum","same_oc","same day, open->close  (IMPACT)"),
       ("hype_mean","same_oc","same day (mean hype)"),
       ("hype_volume","same_abs","same day, |move|  (does hype drive VOLATILITY?)"),
       ("n_tweets","same_abs","same day |move| vs tweet count"),
       ("hype_sum","ret_1d","+1 day (PREDICTIVE)"),
       ("hype_sum","ret_3d","+3 days"),
       ("hype_sum","ret_7d","+7 days"),
       ("hype_volume","ret_1d","+1 day, hype volume"),
       ("n_tweets","ret_1d","+1 day, tweet count")]
for x,y,lbl in pairs:
    s=A[[x,y]].dropna()
    if len(s)<30: print(f"  {lbl:44s} n={len(s)} too few"); continue
    rho,p=stats.spearmanr(s[x], s[y])
    bar_="**" if p<0.05 else "  "
    print(f"  {lbl:44s} n={len(s):5,d}  rho {rho:+.4f}  p={p:.2e} {bar_}")
