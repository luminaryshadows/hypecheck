"""The decay curve: what does a memecoin tweet do to price at 1m, 5m, 15m, 1h, 4h, 1d, 3d, 7d?
Market-adjusted against BTC, with a same-day placebo at every horizon."""
import os
import numpy as np, pandas as pd
from scipy import stats

ROOT="/home/ubuntu/Projects/hypecheck"
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv")
M["available_at"]=pd.to_datetime(M.available_at); M["avail_day"]=pd.to_datetime(M.avail_day)
M["call"]=np.sign(M.lex_net).astype(int)
INTRA=[1,5,15,60,240]          # minutes, from the 1m bars
DAILY=[1,3,7]                  # days, from daily bars

btc_d=pd.read_csv("/tmp/newsrc/BTC-1d.csv", parse_dates=["date"])
btc_d["day"]=btc_d.date.dt.date
btc_dmap=dict(zip(btc_d.day, btc_d.close))
def dmap(coin):
    p=f"/tmp/meme-prices/{coin}-prices.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p, parse_dates=["date"]); d["day"]=d.date.dt.date
    return dict(zip(d.day, d.close))

def day_bars(path):
    if not os.path.exists(path): return None
    d=pd.read_csv(path); d["t"]=pd.to_datetime(d.ot, unit="ms", utc=True)
    d["close"]=d["close"].astype(float)
    return d.sort_values("t").reset_index(drop=True)

rows=[]; dmaps={}
for r in M.itertuples():
    day=str(r.avail_day.date())
    ci=day_bars(f"/tmp/m1/{r.coin}_{day}.csv"); bi=day_bars(f"/tmp/m1btc/{day}.csv")
    if ci is None or bi is None: continue
    i=ci.t.searchsorted(r.available_at, side="right")-1
    j=bi.t.searchsorted(r.available_at, side="right")-1
    if i<0 or j<0 or i+max(INTRA)>=len(ci) or j+max(INTRA)>=len(bi): continue
    if r.coin not in dmaps: dmaps[r.coin]=dmap(r.coin)
    cd=dmaps[r.coin]
    if cd is None: continue
    days=sorted(cd); didx={d:k for k,d in enumerate(days)}
    if r.avail_day.date() not in didx: continue
    k=didx[r.avail_day.date()]
    rec=dict(tweet_id=r.tweet_id, coin=r.coin, call=r.call, hype_meter=r.hype_meter, lex_net=r.lex_net)
    for h in INTRA:
        rc=ci.close.iloc[i+h]/ci.close.iloc[i]-1.0
        rb=bi.close.iloc[j+h]/bi.close.iloc[j]-1.0
        rec[f"r_{h}m"]=rc; rec[f"a_{h}m"]=(1+rc)/(1+rb)-1.0
    for h in DAILY:
        if k+h>=len(days): continue
        rc=cd[days[k+h]]/cd[days[k]]-1.0
        rb=btc_dmap.get(days[k+h], np.nan)/btc_dmap.get(days[k], np.nan)-1.0
        if not np.isfinite(rb): continue
        rec[f"r_{h}d"]=rc; rec[f"a_{h}d"]=(1+rc)/(1+rb)-1.0
    # placebo: same computation from a random minute of the same day
    rng=np.random.default_rng(abs(hash(str(r.tweet_id)))%2**32)
    p=int(rng.integers(0, len(ci)-max(INTRA)-1))
    q=int(rng.integers(0, len(bi)-max(INTRA)-1))
    for h in INTRA:
        rc=ci.close.iloc[p+h]/ci.close.iloc[p]-1.0
        rb=bi.close.iloc[q+h]/bi.close.iloc[q]-1.0
        rec[f"pa_{h}m"]=(1+rc)/(1+rb)-1.0
    rows.append(rec)
E=pd.DataFrame(rows); E.to_csv("/tmp/decay_curve.csv", index=False)
print(f"decay curve built on {len(E):,} tweets (market-adjusted vs BTC)\n")
print(f"{'horizon':>8s} {'n':>6s} {'abnormal':>12s} {'placebo':>11s} {'effect':>11s} {'t':>7s} {'p':>7s}  verdict")
for h in INTRA+DAILY:
    k=f"a_{h}m" if h in INTRA else f"a_{h}d"
    pk=f"pa_{h}m"
    if k not in E.columns: continue
    a=E[k].dropna(); b=E[pk].dropna() if pk in E.columns else pd.Series(dtype=float)
    lbl=f"{h}m" if h in INTRA else f"{h}d"
    if len(b)==0:
        t,p=stats.ttest_1samp(a,0); print(f"{lbl:>8s} {len(a):>6,d} {a.mean():>+11.4%} {'—':>11s} {'—':>11s} {t:>7.2f} {p:>7.3f}")
    else:
        eff=a.mean()-b.mean(); t,p=stats.ttest_ind(a,b,equal_var=False)
        v="EFFECT" if p<0.05 else "no"
        print(f"{lbl:>8s} {len(a):>6,d} {a.mean():>+11.4%} {b.mean():>+10.4%} {eff:>+10.4%} {t:>7.2f} {p:>7.3f}  {v}")
print("\n=== the same curve for BULLISH tweets only ===")
for h in INTRA+DAILY:
    k=f"a_{h}m" if h in INTRA else f"a_{h}d"
    if k not in E.columns: continue
    a=E.loc[E.call==1,k].dropna(); lbl=f"{h}m" if h in INTRA else f"{h}d"
    t,p=stats.ttest_1samp(a,0) if len(a)>2 else (np.nan,np.nan)
    print(f"  {lbl:>5s}  n={len(a):5,d}  abnormal {a.mean():+.4%}  t={t:6.2f}  p={p:.3f}")
