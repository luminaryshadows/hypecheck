import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
exec(open("/tmp/impact_model.py").read().split("print(f\"tweets:")[0])
TEXT=["len_chars","caps_share","n_exclaim","n_cashtag","n_links","hype_final","ev_bull","ev_bear","ev_hype"]
META=["hour","dow","month"]
X["day"]=X.created_at.dt.strftime("%Y-%m-%d").fillna("NA")
tgt="y_60m"; Y=X.dropna(subset=[tgt]).reset_index(drop=True)
def mats(tr,te,blocks):
    A=[];B=[]
    if "text" in blocks:
        v=TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True,strip_accents="unicode")
        A.append(v.fit_transform(tr.text)); B.append(v.transform(te.text))
    if "meta" in blocks:
        sc=StandardScaler().fit(tr[TEXT+META]); A.append(sc.transform(tr[TEXT+META].to_numpy(float))); B.append(sc.transform(te[TEXT+META].to_numpy(float)))
    for nm in [c for c in ["coin","acct"] if c in blocks]:
        d=pd.get_dummies(pd.concat([tr[nm],te[nm]])); A.append(d.iloc[:len(tr)].to_numpy(float)); B.append(d.iloc[len(tr):].to_numpy(float))
    f=lambda L: np.hstack([x.toarray() if hasattr(x,"toarray") else x for x in L])
    return f(A),f(B)
# ---------- TEST 1: strip the market out, then ask if the tweet explains what remains ----------
Y["resid"]=Y[tgt]-Y.groupby("day")[tgt].transform("mean")
print("="*98)
print("TEST 1 - 'tweets move the coin ONLY': remove the market entirely, predict the residual")
print("="*98)
print(f"  full label sd {Y[tgt].std():.4f}   ->  after removing the day/market: {Y.resid.std():.4f}  ({Y.resid.std()/Y[tgt].std():.1%} remains)")
print(f"  variance removed by the market (the day): {1-Y.resid.var()/Y[tgt].var():.1%}")
def wf(target,blocks):
    qs=np.linspace(0.4,0.8,5); pred=np.full(len(Y),np.nan)
    for i in range(4):
        a,b=int(len(Y)*qs[i]),int(len(Y)*qs[i+1])
        if b-a<20 or a<60: continue
        Xtr,Xte=mats(Y.iloc[:a],Y.iloc[a:b],blocks)
        pred[a:b]=Ridge(alpha=3.0,solver="sparse_cg").fit(Xtr,Y[target].to_numpy(float)[:a]).predict(Xte)
    return pred
print(f"\n  {'feature block':34s} {'n_oos':>6s} {'Spearman':>10s} {'p':>10s}")
for blocks,lbl in [(["text"],"tweet TEXT"),(["meta"],"text META"),(["acct"],"AUTHOR"),(["coin"],"COIN"),
                   (["text","meta","coin","acct"],"EVERYTHING about the tweet")]:
    p=wf("resid",blocks); m=~np.isnan(p)
    r,pv=stats.spearmanr(Y.resid[m],p[m])
    print(f"  {lbl:34s} {m.sum():>6d} {r:>+10.4f} {pv:>10.2e} {'**' if pv<0.05 else ''}")
# ---------- TEST 2: literal classification into up / flat / down ----------
print("\n"+"="*98)
print("TEST 2 - classify each tweet's impact into up / flat / down, score against its base rate")
print("="*98)
Y["cls"]=np.where(Y[tgt]>0.05,"up",np.where(Y[tgt]<-0.05,"down","flat"))
vc=Y.cls.value_counts(); base=vc.max()/len(Y)
print(f"  classes: {dict(vc)}   majority-class base rate {base:.4f}")
for blocks,lbl in [(["text"],"tweet TEXT"),(["meta"],"text META"),(["text","meta","coin","acct"],"EVERYTHING")]:
    qs=np.linspace(0.4,0.8,5); pred=[]; ytrue=[]
    for i in range(4):
        a,b=int(len(Y)*qs[i]),int(len(Y)*qs[i+1])
        if b-a<20 or a<60: continue
        Xtr,Xte=mats(Y.iloc[:a],Y.iloc[a:b],blocks)
        m=LogisticRegression(max_iter=2000,C=1.0).fit(Xtr,Y.cls.to_numpy()[:a])
        pred+=list(m.predict(Xte)); ytrue+=list(Y.cls.to_numpy()[a:b])
    acc=np.mean(np.array(pred)==np.array(ytrue))
    print(f"  {lbl:34s} n={len(pred):5d}  accuracy {acc:.4f}  vs base {base:.4f}  edge {acc-base:+.4f}")
