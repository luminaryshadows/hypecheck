import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
exec(open("/tmp/impact_model.py").read().split("print(f\"tweets:")[0])   # reuse the data build
TEXT=["len_chars","caps_share","n_exclaim","n_cashtag","n_links","hype_final","ev_bull","ev_bear","ev_hype"]
META=["hour","dow","month"]
print(f"tweets {len(X):,}  authors {X.acct.nunique()}  coins {X.coin.nunique()}")
for H in ["15m","60m","1d"]:
    tgt="y_"+H
    Y=X.dropna(subset=[tgt]).sort_values("created_at").reset_index(drop=True)
    coin_d=pd.get_dummies(Y.coin,prefix="c"); acct_d=pd.get_dummies(Y.acct,prefix="a")
    def build(tr_i,te_i,blocks):
        tr,te=Y.iloc[tr_i],Y.iloc[te_i]; A=[];B=[]
        if "text" in blocks:
            v=TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True,strip_accents="unicode")
            A.append(v.fit_transform(tr.text)); B.append(v.transform(te.text))
        if "meta" in blocks:
            sc=StandardScaler().fit(tr[TEXT+META])
            A.append(sc.transform(tr[TEXT+META].to_numpy(float))); B.append(sc.transform(te[TEXT+META].to_numpy(float)))
        if "coin" in blocks: A.append(coin_d.iloc[tr_i].to_numpy(float)); B.append(coin_d.iloc[te_i].to_numpy(float))
        if "acct" in blocks: A.append(acct_d.iloc[tr_i].to_numpy(float)); B.append(acct_d.iloc[te_i].to_numpy(float))
        f=lambda L: np.hstack([x.toarray() if hasattr(x,"toarray") else x for x in L])
        return f(A),f(B)
    def wf(blocks):
        qs=np.linspace(0.4,0.8,5); pred=np.full(len(Y),np.nan)
        for i in range(4):
            a,b=int(len(Y)*qs[i]),int(len(Y)*qs[i+1])
            if b-a<20 or a<60: continue
            Xtr,Xte=build(list(range(a)),list(range(a,b)),blocks)
            pred[a:b]=Ridge(alpha=3.0,solver="sparse_cg").fit(Xtr,Y[tgt].to_numpy(float)[:a]).predict(Xte)
        return pred
    print("\n"+"="*88); print(f"label = {H} abnormal return scaled -1..1   (n={len(Y):,})"); print("="*88)
    for blocks,lbl in [(["text"],"tweet TEXT only"),(["meta"],"text META only"),
                       (["coin"],"COIN identity only"),(["acct"],"AUTHOR identity only"),
                       (["text","meta","coin","acct"],"EVERYTHING")]:
        p=wf(blocks); m=~np.isnan(p)
        if m.sum()<20: continue
        r,pv=stats.spearmanr(Y[tgt][m],p[m])
        print(f"  {lbl:26s} n_oos={m.sum():4d}  Spearman {r:+.4f}  p={pv:.2e} {'**' if pv<0.05 else ''}")
    g=Y.groupby("coin")[tgt]; tot=((Y[tgt]-Y[tgt].mean())**2).sum(); wi=((Y[tgt]-g.transform("mean"))**2).sum()
    g2=Y.groupby("acct")[tgt]; wi2=((Y[tgt]-g2.transform("mean"))**2).sum()
    print(f"  variance explained by coin {1-wi/tot:.1%} | by author {1-wi2/tot:.1%}")
