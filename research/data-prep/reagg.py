"""Re-aggregate with the learned meter, and re-test whether aggregate hype moves the coin."""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
ROOT="/home/ubuntu/Projects/hypecheck"
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv"); M["tweet_id"]=M.tweet_id.astype(str)
L=pd.read_csv("/tmp/learned_hype.csv"); L["tweet_id"]=L.tweet_id.astype(str)
M["avail_day"]=pd.to_datetime(M.avail_day); M["trading_day"]=pd.to_datetime(M.trading_day)
M=M.merge(L, on="tweet_id", how="left")
print(f"rows {len(M):,}  learned-meter coverage {M.hype_v2.notna().mean():.1%}")
M["hype_lexicon"]=M["hype"] if "hype" in M.columns else np.nan
M["hype_final"]=M.hype_v2.fillna(M.get("hype_lexicon", 0)).fillna(0)
M["hype_abs2"]=M.hype_final.abs()
M.to_csv(f"{ROOT}/data/memecoin-hype-dataset.csv", index=False)
print(f"published dataset: {len(M):,} rows x {M.shape[1]} cols (hype_final = the learned -1..+1 meter)")

def prices(c):
    p=f"/tmp/meme-prices/{c}-prices.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p, parse_dates=["date"]).assign(day=lambda x: x.date.dt.date)
    return d.set_index("day").sort_index()
PX={c:prices(c) for c in M.coin.unique()}
rows=[]
for r in M.itertuples():
    d=PX.get(r.coin)
    if d is None or r.avail_day.date() not in d.index: continue
    b=d.loc[r.avail_day.date()]
    rows.append(dict(coin=r.coin, day=r.avail_day.date(), hype=r.hype_final, hype_abs=r.hype_abs2,
                     same_oc=float(b.close/b.open-1.0), same_abs=abs(float(b.close/b.open-1.0))))
R=pd.DataFrame(rows)
A=R.groupby(["coin","day"]).agg(n_tweets=("hype","size"), hype_sum=("hype","sum"),
    hype_mean=("hype","mean"), hype_abs_sum=("hype_abs","sum"),
    bull_share=("hype", lambda s:(s>0.05).mean()),
    same_oc=("same_oc","first"), same_abs=("same_abs","first")).reset_index()
fwd=[]
for r in A.itertuples():
    d=PX.get(r.coin)
    if d is None: continue
    idx=list(d.index)
    if r.day not in idx: continue
    k=idx.index(r.day); rec=dict(coin=r.coin, day=r.day)
    for h in (1,3,7):
        if k+h<len(idx): rec[f"ret_{h}d"]=float(d.iloc[k+h].close/d.iloc[k].close-1.0)
    fwd.append(rec)
A=A.merge(pd.DataFrame(fwd), on=["coin","day"], how="left")
A.to_csv(f"{ROOT}/data/coin-day-hype.csv", index=False)
print(f"\ncoin-days: {len(A):,} over {A.coin.nunique()} coins   tweets/day median {A.n_tweets.median():.0f} max {A.n_tweets.max()}")
print(f"  hype_sum/day: mean {A.hype_sum.mean():+.3f} sd {A.hype_sum.std():.3f} range {A.hype_sum.min():+.2f}..{A.hype_sum.max():+.2f}")

print("\n" + "="*94)
print("AGGREGATE HYPE vs COIN PERFORMANCE  (Spearman, learned meter)")
print("="*94)
for x,y,lbl in [("hype_sum","same_oc","net hype -> same-day open->close (IMPACT)"),
                ("hype_mean","same_oc","mean hype -> same day"),
                ("hype_abs_sum","same_abs","hype magnitude -> same-day |move|"),
                ("n_tweets","same_abs","tweet count -> same-day |move|"),
                ("bull_share","same_oc","share of bullish tweets -> same day"),
                ("hype_sum","ret_1d","net hype -> +1 day (PREDICTIVE)"),
                ("hype_sum","ret_3d","net hype -> +3 days"),
                ("hype_sum","ret_7d","net hype -> +7 days"),
                ("n_tweets","ret_1d","tweet count -> +1 day"),
                ("hype_abs_sum","ret_1d","hype magnitude -> +1 day |direction|")]:
    s=A[[x,y]].dropna()
    if len(s)<30: continue
    rho,p=stats.spearmanr(s[x],s[y]); flag="**" if p<0.05 else "  "
    print(f"  {lbl:46s} n={len(s):5,d}  rho {rho:+.4f}  p={p:.2e} {flag}")

print("\n" + "="*94); print("MODEL: next-day direction from aggregate hype"); print("="*94)
A["day"]=pd.to_datetime(A.day); A=A.sort_values("day").reset_index(drop=True)
F=["hype_sum","hype_mean","hype_abs_sum","n_tweets","bull_share"]
A["prior_ret"]=A.groupby("coin").same_oc.shift(1); A["prior_abs"]=A.groupby("coin").same_abs.shift(1)
C=["prior_ret","prior_abs"]; A[F+C]=A[F+C].fillna(0)
D=A.dropna(subset=["ret_1d"]).reset_index(drop=True); D["y"]=(D.ret_1d>0).astype(int)
n=int(len(D)*0.8); tr,te=D.iloc[:n],D.iloc[n:]
print(f"  rows {len(D):,}  up-rate {D.y.mean():.1%}  base {max(te.y.mean(),1-te.y.mean()):.4f}")
for cols,lbl in [(F,"hype only"),(C,"context only"),(F+C,"hype + context")]:
    for mdl in ("logit","gbm"):
        Xtr,Xte=tr[cols].to_numpy(float),te[cols].to_numpy(float)
        ytr,yte=tr.y.to_numpy(),te.y.to_numpy()
        if mdl=="logit":
            sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
            m=LogisticRegression(max_iter=3000).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
        else:
            m=GradientBoostingClassifier(random_state=0).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
        pr=(p>0.5).astype(int); acc=accuracy_score(yte,pr); base=max(yte.mean(),1-yte.mean())
        print(f"  chrono  {lbl:16s} {mdl:6s} acc {acc:.4f} base {base:.4f} edge {acc-base:+.4f} AUC {roc_auc_score(yte,p):.4f}")
print("\n" + "="*94); print("MODEL: next-day MOVE SIZE (|return|) - the relationship that is real"); print("="*94)
D2=D.copy(); D2["abs_next"]=D2.ret_1d.abs()
tr,te=D2.iloc[:n],D2.iloc[n:]
for cols,lbl in [(F,"hype only"),(F+C,"hype + context")]:
    Xtr,Xte=tr[cols].to_numpy(float),te[cols].to_numpy(float)
    sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
    m=Ridge(alpha=1.0).fit(Xtr,tr.abs_next); p=m.predict(Xte)
    rho,pv=stats.spearmanr(te.abs_next, p)
    print(f"  {lbl:16s} Spearman(pred, actual) = {rho:+.4f}  p={pv:.2e}  (n={len(te)})")
