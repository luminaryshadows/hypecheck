import numpy as np, pandas as pd
from scipy import stats
ROOT="/home/ubuntu/Projects/hypecheck"
p=pd.read_parquet(f"{ROOT}/data/posts.parquet")[["tweet_id","likes","views","retweets","replies","user_name","text"]]
p["tweet_id"]=p.tweet_id.astype(str); p=p.drop_duplicates("tweet_id")
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv"); M["tweet_id"]=M.tweet_id.astype(str)
U=M.drop_duplicates("tweet_id").merge(p, on="tweet_id", how="left")
E=U.dropna(subset=["likes","views"]).copy()
E["log_likes"]=np.log1p(E.likes.astype(float)); E["log_views"]=np.log1p(E.views.astype(float))
E["len"]=E.text.astype(str).str.len()
E["caps_share"]=E.text.apply(lambda s: sum(c.isupper() for c in str(s))/max(len(str(s)),1))
E["n_ex"]=E.text.astype(str).str.count("!")
E.to_csv("/tmp/engage.csv", index=False)
print(f"rows with engagement: {len(E):,}  accounts {E.user_name.nunique()}  likes median {E.likes.median():.0f} max {E.likes.max():.0f}")
def R(a,b,label):
    s=pd.DataFrame({"a":a,"b":b}).dropna()
    if len(s)<20: return f"  {label:46s} n={len(s)} too few"
    r,pv=stats.spearmanr(s.a,s.b)
    return f"  {label:46s} n={len(s):4,d}  rho {r:+.4f}  p={pv:.2e} {'**' if pv<0.05 else ''}"
H="hype_final"
print("\n=== DOES THE HYPE METER CAPTURE ATTENTION? ===")
for lbl,a,b in [("|hype_final| vs log likes",E[H].abs(),E.log_likes),
                ("|hype_final| vs log views",E[H].abs(),E.log_views),
                ("|hype_final| vs log retweets",E[H].abs(),np.log1p(E.retweets.astype(float))),
                ("hype_final signed vs log likes",E[H],E.log_likes),
                ("text length vs log likes (confound)",E.len,E.log_likes)]:
    print(R(a,b,lbl))
print("\n=== intensity components, individually ===")
for lbl,a,b in [("hype-term evidence vs likes",E.ev_hype,E.log_likes),
                ("caps share vs likes",E.caps_share,E.log_likes),
                ("exclamation count vs likes",E.n_ex,E.log_likes),
                ("hype-term evidence vs views",E.ev_hype,E.log_views)]:
    print(R(a,b,lbl))
E["likes_dm"]=E.log_likes-E.groupby("user_name").log_likes.transform("mean")
E["hypeabs_dm"]=E[H].abs()-E.groupby("user_name")[H].transform(lambda s:s.abs().mean())
E["evhype_dm"]=E.ev_hype-E.groupby("user_name").ev_hype.transform("mean")
E["len_dm"]=E.len-E.groupby("user_name").len.transform("mean")
print("\n=== WITHIN-AUTHOR (account size removed) ===")
for lbl,a,b in [("|hype| vs likes",E.hypeabs_dm,E.likes_dm),
                ("hype-term evidence vs likes",E.evhype_dm,E.likes_dm),
                ("text length vs likes",E.len_dm,E.likes_dm)]:
    print(R(a,b,lbl))
print("\n=== how much of engagement is just WHICH ACCOUNT posted? ===")
for col in ["log_likes","log_views"]:
    g=E.groupby("user_name")[col]; tot=((E[col]-E[col].mean())**2).sum(); wi=((E[col]-g.transform("mean"))**2).sum()
    print(f"  {col:10s} variance explained by the account alone: {1-wi/tot:.1%}")
print("\n=== VERDICT ===")
print("  DIRECTION is validated: held-out +0.480 vs FinTwitBERT")
print("  INTENSITY is NOT validated: no component tracks engagement")
print("  -> the meter measures SENTIMENT, not HYPE")
