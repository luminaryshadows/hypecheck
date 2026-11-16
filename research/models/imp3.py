import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
exec(open("/tmp/impact_model.py").read().split("print(f\"tweets:")[0])
TEXT=["len_chars","caps_share","n_exclaim","n_cashtag","n_links","hype_final","ev_bull","ev_bear","ev_hype"]
META=["hour","dow","month"]
X["day"]=X.created_at.dt.strftime("%Y-%m-%d").fillna("NA")
H="60m"; tgt="y_"+H
Y=X.dropna(subset=[tgt]).reset_index(drop=True)          # keep chronological order
print(f"tweets {len(Y):,}  authors {Y.acct.nunique()}  coins {Y.coin.nunique()}  distinct DAYS {Y.day.nunique()}")
def mats(tr,te,blocks,fit_text=True):
    A=[];B=[]
    if "text" in blocks:
        v=TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True,strip_accents="unicode")
        A.append(v.fit_transform(tr.text)); B.append(v.transform(te.text))
    if "meta" in blocks:
        sc=StandardScaler().fit(tr[TEXT+META]); A.append(sc.transform(tr[TEXT+META].to_numpy(float))); B.append(sc.transform(te[TEXT+META].to_numpy(float)))
    for nm,col in [("coin","coin"),("acct","acct"),("day","day")]:
        if nm in blocks:
            d=pd.get_dummies(pd.concat([tr[col],te[col]]))
            A.append(d.iloc[:len(tr)].to_numpy(float)); B.append(d.iloc[len(tr):].to_numpy(float))
    f=lambda L: np.hstack([x.toarray() if hasattr(x,"toarray") else x for x in L])
    return f(A),f(B)
def run(blocks, mode, seed=0):
    if mode=="random":
        rng=np.random.default_rng(seed); idx=rng.permutation(len(Y)); tr_i,te_i=idx[:int(.8*len(Y))],idx[int(.8*len(Y)):]
        tr,te=Y.iloc[tr_i],Y.iloc[te_i]
        Xtr,Xte=mats(tr,te,blocks)
        m=Ridge(alpha=3.0,solver="sparse_cg").fit(Xtr,tr[tgt].to_numpy(float))
        p=m.predict(Xte); y=te[tgt].to_numpy(float)
    else:
        cut=int(.8*len(Y)); tr,te=Y.iloc[:cut],Y.iloc[cut:]
        Xtr,Xte=mats(tr,te,blocks)
        m=Ridge(alpha=3.0,solver="sparse_cg").fit(Xtr,tr[tgt].to_numpy(float))
        p=m.predict(Xte); y=te[tgt].to_numpy(float)
    r=stats.spearmanr(y,p)[0]; r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    return r,r2,len(y)
print("\n"+"="*100)
print(f"'TWEETS MOVE THE COIN' - the most favourable version.  label = {H} attributed impact, scaled -1..1")
print("="*100)
print(f"  {'feature blocks':38s} | {'RANDOM 80/20':>22s} | {'HONEST time 80/20':>22s}")
print(f"  {'':38s} | {'Spearman':>10s}{'R2':>12s} | {'Spearman':>10s}{'R2':>12s}")
for blocks,lbl in [(["text"],"tweet TEXT"),
                   (["meta"],"text META"),
                   (["coin"],"COIN identity"),
                   (["acct"],"AUTHOR identity"),
                   (["day"],"the DATE (the market that day)"),
                   (["text","meta"],"tweet CONTENT (text+meta)"),
                   (["text","meta","coin","acct"],"everything about the tweet"),
                   (["text","meta","coin","acct","day"],"everything + the date")]:
    r1,R1,n1=run(blocks,"random"); r2,R2,n2=run(blocks,"time")
    print(f"  {lbl:38s} | {r1:>+10.4f}{R1:>12.4f} | {r2:>+10.4f}{R2:>12.4f}")
print("\n  the decisive decomposition: what IS this attributed move a property of?")
for c in ["day","coin","acct"]:
    g=Y.groupby(c)[tgt]; tot=((Y[tgt]-Y[tgt].mean())**2).sum(); wi=((Y[tgt]-g.transform("mean"))**2).sum()
    print(f"    variance of the attributed impact explained by {c:5s}: {1-wi/tot:5.1%}")
print(f"\n  if tweets drove the price, TEXT would explain it. It explains: ", end="")
r,R,n=run(["text"],"time"); print(f"{R:.4%} (R2), Spearman {r:+.4f}")
