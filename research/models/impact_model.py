import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
ROOT="/home/ubuntu/Projects/hypecheck"
D=pd.read_csv("/tmp/decay_curve.csv")
posts=pd.read_parquet(f"{ROOT}/data/posts.parquet")[["tweet_id","text","created_at","user_name"]]
posts["tweet_id"]=posts.tweet_id.astype(str); posts=posts.drop_duplicates("tweet_id")
D["tweet_id"]=D.tweet_id.astype(str)
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv"); M["tweet_id"]=M.tweet_id.astype(str)
meta=M.drop_duplicates("tweet_id")[["tweet_id","len_chars","caps_share","n_exclaim","n_cashtag","n_links","hype_final","ev_bull","ev_bear","ev_hype"]]
X=D.merge(posts,on="tweet_id",how="left").merge(meta,on="tweet_id",how="left").dropna(subset=["text"]).reset_index(drop=True)
X["created_at"]=pd.to_datetime(X.created_at,utc=True,errors="coerce")
X["hour"]=X.created_at.dt.hour; X["dow"]=X.created_at.dt.dayofweek; X["month"]=X.created_at.dt.month
X["acct"]=X.user_name.fillna("UNKNOWN")
TEXT=["len_chars","caps_share","n_exclaim","n_cashtag","n_links","hype_final","ev_bull","ev_bear","ev_hype"]
META=["hour","dow","month"]
X[TEXT+META]=X[TEXT+META].fillna(0)
def s1(v):
    v=np.asarray(v,float); k=np.nanpercentile(np.abs(v),75); return np.tanh(v/max(k,1e-9))
for h,src in [("15m","a_15m"),("60m","a_60m"),("1d","a_1d")]:
    if src in X.columns: X["y_"+h]=s1(X[src])
print(f"tweets: {len(X):,}  authors {X.acct.nunique()}  coins {X.coin.nunique()}")
for h in ["15m","60m","1d"]:
    s=X["y_"+h].dropna(); print(f"  label y_{h}: n={len(s):,} mean {s.mean():+.4f} sd {s.std():.4f} range {s.min():+.3f}..{s.max():+.3f}")
Y=X.dropna(subset=["y_60m"]).sort_values("created_at").reset_index(drop=True)
coin_d=pd.get_dummies(Y.coin,prefix="c"); acct_d=pd.get_dummies(Y.acct,prefix="a")
def build(tr_i, te_i, blocks):
    tr,te=Y.iloc[tr_i],Y.iloc[te_i]; A=[]; B=[]
    if "text" in blocks:
        v=TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True,strip_accents="unicode")
        A.append(v.fit_transform(tr.text)); B.append(v.transform(te.text))
    if "meta" in blocks:
        sc=StandardScaler().fit(tr[TEXT+META])
        A.append(sc.transform(tr[TEXT+META].to_numpy(float))); B.append(sc.transform(te[TEXT+META].to_numpy(float)))
    if "coin" in blocks: A.append(coin_d.iloc[tr_i].to_numpy(float)); B.append(coin_d.iloc[te_i].to_numpy(float))
    if "acct" in blocks: A.append(acct_d.iloc[tr_i].to_numpy(float)); B.append(acct_d.iloc[te_i].to_numpy(float))
    f=lambda L: np.hstack([x.toarray() if hasattr(x,"toarray") else x for x in L])
    return f(A), f(B)
def wf(blocks):
    qs=np.linspace(0.4,0.8,5); pred=np.full(len(Y),np.nan)
    for i in range(4):
        a,b=int(len(Y)*qs[i]),int(len(Y)*qs[i+1])
        if b-a<20 or a<60: continue
        Xtr,Xte=build(list(range(a)),list(range(a,b)),blocks)
        m=Ridge(alpha=3.0,solver="sparse_cg").fit(Xtr,Y.y_60m.to_numpy(float)[:a])
        pred[a:b]=m.predict(Xte)
    return pred
print("\n"+"="*96)
print(f"CAN THE TWEET PREDICT THE MOVE ATTRIBUTED TO IT?  label = 60m abnormal return, scaled -1..1  (n={len(Y):,})")
print("="*96)
print(f"  {'feature blocks':40s} {'n_oos':>6s} {'Spearman':>10s} {'p':>10s}")
for blocks,lbl in [(["text"],"tweet TEXT only"),
                   (["meta"],"text META only (length, caps, hype terms)"),
                   (["coin"],"COIN identity only"),
                   (["acct"],"AUTHOR identity only"),
                   (["text","meta"],"tweet CONTENT (text + meta)"),
                   (["coin","acct","meta"],"CONTEXT only (coin + author + meta)"),
                   (["text","meta","coin","acct"],"EVERYTHING")]:
    p=wf(blocks); m=~np.isnan(p)
    if m.sum()<20: print(f"  {lbl:40s} {m.sum():>6d}  too few"); continue
    r,pv=stats.spearmanr(Y.y_60m[m],p[m])
    print(f"  {lbl:40s} {m.sum():>6d} {r:>+10.4f} {pv:>10.2e} {'**' if pv<0.05 else ''}")
print("\n  how much of the attributed move is just WHICH COIN / WHICH AUTHOR?")
for c in ["coin","acct"]:
    g=Y.groupby(c).y_60m; tot=((Y.y_60m-Y.y_60m.mean())**2).sum(); wi=((Y.y_60m-g.transform("mean"))**2).sum()
    print(f"    variance explained by {c:7s}: {1-wi/tot:.1%}")
print("\n  controls on 'tweet CONTENT':")
rng=np.random.default_rng(0); real=stats.spearmanr(Y.y_60m[~np.isnan(wf(['text','meta']))], wf(['text','meta'])[~np.isnan(wf(['text','meta']))])[0]
Ys=Y.copy(); Ys["y_60m"]=rng.permutation(Ys.y_60m.to_numpy())
p=wf(['text','meta']); m=~np.isnan(p); null=stats.spearmanr(Ys.y_60m[m],p[m])[0]
print(f"    real {real:+.4f}   label-permuted {null:+.4f}  -> {'real clears' if abs(real)>abs(null) else 'no'}")
