"""Same-day event study: what actually happens in the minutes after a memecoin tweet?
Placebo = the same computation at random minutes in the same day, so a null means something."""
import glob, os, json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

ROOT="/home/ubuntu/Projects/hypecheck"
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv")
M["available_at"]=pd.to_datetime(M.available_at); M["avail_day"]=pd.to_datetime(M.avail_day)
posts=pd.read_parquet(f"{ROOT}/data/posts.parquet")[["tweet_id","user_name"]]
M["tweet_id"]=M["tweet_id"].astype(str); posts["tweet_id"]=posts["tweet_id"].astype(str)
posts=posts.drop_duplicates("tweet_id")
M=M.merge(posts, on="tweet_id", how="left")
# author weight = how often that account appears in this dataset (the only prominence proxy we have)
aw=M.user_name.value_counts()
M["author_n"]=M.user_name.map(aw).fillna(1)
M["call"]=np.sign(M.lex_net).astype(int)

HORIZONS=[1,5,15,60]
def load_day(coin, day):
    p=f"/tmp/m1/{coin}_{day}.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); d["t"]=pd.to_datetime(d.ot, unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]: d[c]=d[c].astype(float)
    return d.sort_values("t").reset_index(drop=True)

rows=[]; miss=0
for r in M.itertuples():
    d=load_day(r.coin, str(r.avail_day.date()))
    if d is None: miss+=1; continue
    idx=d.t.searchsorted(r.available_at, side="right")-1     # bar containing the tweet
    if idx<0 or idx+max(HORIZONS)>=len(d): miss+=1; continue
    base=d.close.iloc[idx]
    if not base>0: miss+=1; continue
    rec=dict(tweet_id=r.tweet_id, coin=r.coin, call=r.call, hype_meter=r.hype_meter,
             n_hype=r.n_hype, lex_net=r.lex_net, author_n=r.author_n,
             own_move=d.close.iloc[idx]/d.open.iloc[idx]-1.0,
             vol_ratio=d.volume.iloc[idx]/max(d.volume.median(),1e-9))
    for h in HORIZONS:
        rec[f"fwd_{h}m"]=d.close.iloc[idx+h]/base-1.0
    # placebo: a uniformly random minute in the same day, same forward computation
    j=int(np.random.default_rng(abs(hash(r.tweet_id))%2**32).integers(0, len(d)-max(HORIZONS)-1))
    rec["pl_own_move"]=d.close.iloc[j]/d.open.iloc[j]-1.0
    for h in HORIZONS:
        rec[f"pl_fwd_{h}m"]=d.close.iloc[j+h]/d.close.iloc[j]-1.0
    rows.append(rec)
E=pd.DataFrame(rows)
E.to_csv("/tmp/event_study.csv", index=False)
print(f"event study: {len(E):,} tweets matched to minute bars ({miss:,} unmatched)")
print(f"  authors: {M.user_name.notna().sum():,} of {len(M):,} tweets have one ({M.user_name.notna().mean():.0%}); {M.user_name.nunique()} distinct accounts")

print(f"\n=== the minute the tweet lands ===")
print(f"  mean move in the tweet's own minute: {E.own_move.mean():+.4%}   median {E.own_move.median():+.4%}")
print(f"  placebo (random minute, same day):   {E.pl_own_move.mean():+.4%}   median {E.pl_own_move.median():+.4%}")
print(f"  median volume ratio vs the day's typical minute: {E.vol_ratio.median():.2f}x  (mean {E.vol_ratio.mean():.1f}x)")

print(f"\n=== forward returns after the tweet, vs placebo ===")
print(f"  {'horizon':>8s} {'after tweet':>13s} {'placebo':>11s} {'difference':>12s}   p (Mann-Whitney)")
for h in HORIZONS:
    a=E[f"fwd_{h}m"]; b=E[f"pl_fwd_{h}m"]
    u=mannwhitneyu(a, b, alternative="two-sided")
    print(f"  {str(h)+'m':>8s} {a.mean():>+12.4%} {b.mean():>+10.4%} {a.mean()-b.mean():>+11.4%}   p={u.pvalue:.3f}")

print(f"\n=== does the tweet's DIRECTION predict the next hour? ===")
for label, sub in [("bullish tweets", E[E.call==1]), ("bearish tweets", E[E.call==-1]), ("no call", E[E.call==0])]:
    if len(sub)<5: print(f"  {label:16s} n={len(sub)} (too few)"); continue
    print(f"  {label:16s} n={len(sub):5,d}   fwd_60m mean {sub.fwd_60m.mean():+.4%}  median {sub.fwd_60m.median():+.4%}")
if (E.call==1).any() and (E.call==-1).any():
    u=mannwhitneyu(E.loc[E.call==1,"fwd_60m"], E.loc[E.call==-1,"fwd_60m"], alternative="two-sided")
    print(f"  bullish vs bearish at 60m: p={u.pvalue:.3f}")

print(f"\n=== weighted by author prominence (accounts with more tweets count more) ===")
w=E.author_n.astype(float)
for h in HORIZONS:
    wa=np.average(E[f"fwd_{h}m"], weights=w); wp=np.average(E[f"pl_fwd_{h}m"], weights=w)
    print(f"  {str(h)+'m':>5s}  weighted after-tweet {wa:+.4%}   weighted placebo {wp:+.4%}   diff {wa-wp:+.4%}")
print(f"\n  top accounts by tweet count: {aw.head(6).to_dict()}")
print(f"\n=== direct test: bullish tweets vs tweets with no call, 60m ===")
from scipy.stats import mannwhitneyu as mw
a=E.loc[E.call==1,"fwd_60m"]; b=E.loc[E.call==0,"fwd_60m"]
print(f"  bullish n={len(a)} mean {a.mean():+.4%} | no-call n={len(b)} mean {b.mean():+.4%} | diff {a.mean()-b.mean():+.4%}  p={mw(a,b).pvalue:.3f}")
print(f"  (4 horizons were tested, so a Bonferroni-corrected threshold is p<0.0125)")
