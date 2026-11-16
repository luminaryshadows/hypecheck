import json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy import stats
ROOT="/home/ubuntu/Projects/hypecheck"
posts=pd.read_parquet(f"{ROOT}/data/posts.parquet")[["tweet_id","text"]]
posts["tweet_id"]=posts.tweet_id.astype(str); posts=posts.drop_duplicates("tweet_id")
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv"); M["tweet_id"]=M.tweet_id.astype(str)
U=M.drop_duplicates("tweet_id")[["tweet_id"]].merge(posts,on="tweet_id",how="left").dropna(subset=["text"]).reset_index(drop=True)
sent=json.load(open(f"{ROOT}/data/sentiment.json"))["scores"]
U["ftb"]=[(sent[i]["bullish"]-sent[i]["bearish"]) if i in sent else np.nan for i in U.tweet_id]
H=pd.read_csv("/tmp/hype_meter_rows.csv")[["tweet_id","ev_hype","caps_share","n_exclaim"]]
H["tweet_id"]=H.tweet_id.astype(str); H=H.drop_duplicates("tweet_id")
U=U.merge(H,on="tweet_id",how="left")
def intensity(d):   # gentler, and capped
    return (1.0 + 0.5*d.ev_hype.fillna(0) + 0.25*(d.caps_share.fillna(0)>=0.30)
            + 0.10*d.n_exclaim.fillna(0).clip(0,4)).clip(1.0, 2.0)
def make(dir_c, conf, inten, k=1.5):
    # intensity amplifies the evidence BEFORE the squash, so tanh does the saturating
    return np.clip(np.tanh(dir_c*inten*k)*conf, -1, 1)
def fit_predict(tr_t, tr_y, te_t, alpha=1.0):
    v=TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    Xtr=v.fit_transform(tr_t); Xte=v.transform(te_t)
    m=Ridge(alpha=alpha, solver="sparse_cg").fit(Xtr, tr_y)
    nz=np.asarray((Xte>0).sum(axis=1)).ravel()
    return m.predict(Xte), m.predict(Xtr), np.minimum(1.0, nz/4.0), v, m
lab=U.dropna(subset=["ftb"]).reset_index(drop=True)
kf=KFold(n_splits=5, shuffle=True, random_state=0)
d=np.full(len(lab),np.nan); c=np.full(len(lab),np.nan)
for tr,te in kf.split(lab):
    p,ptr,conf,v,m=fit_predict(lab.text.iloc[tr], lab.ftb.iloc[tr], lab.text.iloc[te])
    d[te]=p-ptr.mean(); c[te]=conf
lab["dir"]=d; lab["conf"]=c; lab["inten"]=intensity(lab)
lab["meter"]=make(lab["dir"],lab["conf"],lab["inten"])
rho,pv=stats.spearmanr(lab.ftb, lab.meter)
print(f"HELD-OUT (5-fold): Spearman {rho:+.4f} p={pv:.2e}  mean {lab.meter.mean():+.4f} sd {lab.meter.std():.4f}")
print(f"  saturated at exactly +-1: {(lab.meter.abs()==1).mean():.1%}")
mask=U.tweet_id.isin(set(lab.tweet_id))
pred,ptr,conf,v,m=fit_predict(U.loc[mask,"text"], U.loc[mask,"ftb"], U.text)
U["dir"]=pred-ptr.mean(); U["conf"]=conf; U["inten"]=intensity(U)
U["hype_v2"]=make(U["dir"],U["conf"],U["inten"])
print(f"\nACROSS ALL {len(U):,} TWEETS: coverage {(U.hype_v2.abs()>0).mean():.1%}  mean {U.hype_v2.mean():+.4f}  sd {U.hype_v2.std():.4f}")
print(f"  saturated at +-1: {(U.hype_v2.abs()==1).mean():.1%}")
for lo,hi,lbl in [(-1.01,-0.3,"strongly bearish"),(-0.3,-0.05,"mildly bearish"),(-0.05,0.05,"neutral"),(0.05,0.3,"mildly bullish"),(0.3,1.01,"strongly bullish")]:
    sel=np.logical_and(U.hype_v2>lo,U.hype_v2<=hi); print(f"  {lbl:18s} {sel.mean():6.1%}")
print("\n  most bullish:"); [print(f"    {r.hype_v2:+.3f}  {' '.join(str(r.text).split())[:92]}") for r in U.nlargest(4,"hype_v2").itertuples()]
print("  most bearish:"); [print(f"    {r.hype_v2:+.3f}  {' '.join(str(r.text).split())[:92]}") for r in U.nsmallest(4,"hype_v2").itertuples()]
U[["tweet_id","dir","conf","hype_v2"]].to_csv("/tmp/learned_hype.csv", index=False)
print("\nwrote /tmp/learned_hype.csv")
