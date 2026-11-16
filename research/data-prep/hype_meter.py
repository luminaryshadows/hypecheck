"""Build a -1..+1 hype meter per tweet, validate it, aggregate to the coin-day,
and correlate aggregate hype with the coin's performance."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
ROOT="/home/ubuntu/Projects/hypecheck"
LEX=pd.read_csv(f"{ROOT}/data/hype-lexicon.csv")
def terms(kind, statuses):
    sel=np.logical_and(LEX.curated_as.eq(kind).values, LEX.status.isin(statuses).values)
    return LEX.loc[sel,["term","status"]].assign(term=lambda d: d.term.astype(str).str.lower())
BULL=terms("bullish",["validated","weak"]); BEAR=terms("bearish",["validated","weak"])
HYP=LEX.loc[np.logical_and(LEX.curated_as.eq("hype").values, LEX.hype_meter.notna().values),["term","hype_meter"]]
BW={r.term:(1.0 if r.status=="validated" else 0.6) for r in BULL.itertuples()}
RW={r.term:(1.0 if r.status=="validated" else 0.6) for r in BEAR.itertuples()}
HW={str(r.term).lower(): float(r.hype_meter)/100.0 for r in HYP.itertuples()}

posts=pd.read_parquet(f"{ROOT}/data/posts.parquet")[["tweet_id","text","created_at","first_posted_at"]]
posts["tweet_id"]=posts.tweet_id.astype(str); posts=posts.drop_duplicates("tweet_id")
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv"); M["tweet_id"]=M.tweet_id.astype(str)
M=M.merge(posts[["tweet_id","text"]], on="tweet_id", how="left")
print(f"rows: {len(M):,}  with text: {M.text.notna().sum():,}")

def meter(t):
    s=str(t).lower()
    if not s or s=="nan": return 0.0, 0.0, 0.0, 0.0
    B=sum(w for k,w in BW.items() if k in s)
    R=sum(w for k,w in RW.items() if k in s)
    H=sum(w for k,w in HW.items() if k in s)
    caps=sum(c.isupper() for c in str(t))/max(len(str(t)),1)
    ex=str(t).count("!")
    F=1.0 + (0.5 if caps>=0.30 else 0.0) + 0.25*min(ex,4)
    net=B-R
    h=float(np.tanh(net*(1.0+H)*F/3.0))
    return h, B, R, H
vals=[meter(t) for t in M.text]
M["hype"]=[v[0] for v in vals]; M["ev_bull"]=[v[1] for v in vals]
M["ev_bear"]=[v[2] for v in vals]; M["ev_hype"]=[v[3] for v in vals]
M["hype_abs"]=M.hype.abs()
print(f"\n=== the hype meter (-1 to +1) ===")
print(f"  mean {M.hype.mean():+.4f}  sd {M.hype.std():.4f}  min {M.hype.min():+.3f}  max {M.hype.max():+.3f}")
print(f"  exactly 0 (no evidence either way): {(M.hype==0).mean():.1%}")
for lo,hi,lbl in [(-1.01,-0.3,"strongly bearish (<-0.3)"),(-0.3,-0.05,"mildly bearish"),
                  (-0.05,0.05,"neutral"),(0.05,0.3,"mildly bullish"),(0.3,1.01,"strongly bullish (>0.3)")]:
    print(f"  {lbl:28s} {((M.hype>lo).values & (M.hype<=hi).values).mean():6.1%}")

# ---- VALIDATE against FinTwitBERT (independent neural sentiment) ----
sent=json.load(open(f"{ROOT}/data/sentiment.json"))["scores"]
M["ftb"]=[ (sent.get(i,{}).get("bullish",np.nan)-sent.get(i,{}).get("bearish",np.nan)) if i in sent else np.nan
           for i in M.tweet_id]
v=M.dropna(subset=["ftb"])
rho,p=stats.spearmanr(v.hype, v.ftb)
print(f"\n=== validation vs FinTwitBERT (n={len(v):,}) ===")
print(f"  Spearman(hype meter, FinTwitBERT bull-minus-bear) = {rho:+.4f}  p={p:.2e}")
print(f"  -> {'the text meter tracks an independent neural sentiment model' if p<0.05 else 'NO agreement'}")
print(f"\n  most bullish tweets by our meter:")
for t in M.nlargest(3,"hype").itertuples():
    print(f"    {t.hype:+.3f}  {str(t.text)[:105]}")
print(f"  most bearish:")
for t in M.nsmallest(3,"hype").itertuples():
    print(f"    {t.hype:+.3f}  {str(t.text)[:105]}")
M.drop(columns=["text"]).to_csv("/tmp/hype_meter_rows.csv", index=False)
print(f"\nwrote /tmp/hype_meter_rows.csv ({len(M):,} rows)")
