"""Model the coin-day: can aggregate hype predict the coin's next move, or its size?"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from scipy import stats
A=pd.read_csv("/tmp/coin_day_hype.csv"); A["day"]=pd.to_datetime(A.day)
A=A.sort_values("day").reset_index(drop=True)
HYPE_F=["hype_sum","hype_mean","hype_max","hype_min","hype_volume","n_tweets","bull_terms"]
# market context known before the day
A["prior_ret"]=A.groupby("coin").same_oc.shift(1)
A["prior_abs"]=A.groupby("coin").same_abs.shift(1)
A["prior_n"]=A.groupby("coin").n_tweets.shift(1)
CTX=["prior_ret","prior_abs","prior_n"]
A[HYPE_F+CTX]=A[HYPE_F+CTX].fillna(0)
print(f"coin-days: {len(A):,}  coins {A.coin.nunique()}  {A.day.min().date()} -> {A.day.max().date()}")

def split(df, frac=0.8):
    n=int(len(df)*frac); return df.iloc[:n], df.iloc[n:]

def clf(tr,te,cols,model="logit"):
    Xtr,Xte=tr[cols].to_numpy(float),te[cols].to_numpy(float)
    ytr=tr.target.astype(int).to_numpy(); yte=te.target.astype(int).to_numpy()
    if model=="logit":
        sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
        m=LogisticRegression(max_iter=3000).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    else:
        m=GradientBoostingClassifier(random_state=0).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    pred=(p>0.5).astype(int); base=max(yte.mean(),1-yte.mean())
    return accuracy_score(yte,pred), base, roc_auc_score(yte,p) if len(set(yte))>1 else np.nan

print("\n"+"="*94); print("MODEL A - predict the NEXT DAY's direction (up/down)"); print("="*94)
A["target"]=(A.ret_1d>0).astype(float)
D=A.dropna(subset=["ret_1d"]).reset_index(drop=True)
print(f"  rows with a next-day label: {len(D):,}  up-rate {D.target.mean():.1%}")
for cols,lbl in [(HYPE_F,"hype only"),(CTX,"market context only"),(HYPE_F+CTX,"hype + context")]:
    for mdl in ("logit","gbm"):
        tr,te=split(D)
        acc,base,auc=clf(tr,te,cols,mdl)
        print(f"  {'chronological':14s} {lbl:20s} {mdl:6s} acc {acc:.4f}  base {base:.4f}  edge {acc-base:+.4f}  AUC {auc:.4f}")
rng=np.random.default_rng(0); D2=D.sample(frac=1,random_state=0).reset_index(drop=True)
tr,te=split(D2)
acc,base,auc=clf(tr,te,HYPE_F+CTX,"gbm")
print(f"  {'RANDOM (flatters)':14s} {'hype + context':20s} {'gbm':6s} acc {acc:.4f}  base {base:.4f}  edge {acc-base:+.4f}  AUC {auc:.4f}")

print("\n"+"="*94); print("MODEL B - predict the NEXT DAY's MOVE SIZE (|return|) - not direction"); print("="*94)
D["target"]=D.same_abs.shift(0)  # placeholder
A2=A.dropna(subset=["ret_1d"]).copy(); A2["abs_next"]=A2.ret_1d.abs()
A2=A2.dropna(subset=["abs_next"]).reset_index(drop=True)
tr,te=split(A2)
for cols,lbl in [(HYPE_F,"hype only"),(CTX,"market context only"),(HYPE_F+CTX,"hype + context")]:
    Xtr,Xte=tr[cols].to_numpy(float),te[cols].to_numpy(float)
    sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
    m=Ridge(alpha=1.0).fit(Xtr,tr.abs_next); p=m.predict(Xte)
    # baseline: predict the training mean
    b=np.full(len(te), tr.abs_next.mean())
    print(f"  {lbl:20s} R2 {r2_score(te.abs_next,p):+.4f}   (predict-the-mean R2 = {r2_score(te.abs_next,b):+.4f})")
rho,p=stats.spearmanr(A2.n_tweets, A2.abs_next)
print(f"  raw check: Spearman(today's tweet count, |next-day return|) = {rho:+.4f}  p={p:.2e}")

print("\n"+"="*94); print("CONTEMPORANEOUS: same-day |move| from today's hype (the relationship that IS real)"); print("="*94)
tr,te=split(A)
for cols,lbl in [(HYPE_F,"hype only"),(HYPE_F+CTX,"hype + context")]:
    Xtr,Xte=tr[cols].to_numpy(float),te[cols].to_numpy(float)
    sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
    m=Ridge(alpha=1.0).fit(Xtr,tr.same_abs); p=m.predict(Xte); b=np.full(len(te), tr.same_abs.mean())
    print(f"  {lbl:20s} R2 {r2_score(te.same_abs,p):+.4f}   (mean baseline {r2_score(te.same_abs,b):+.4f})")
rho,p=stats.spearmanr(A.n_tweets, A.same_abs)
print(f"  Spearman(tweet count, same-day |move|) = {rho:+.4f}  p={p:.2e}")
