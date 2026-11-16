"""Can hype predict whether a memecoin goes up tomorrow? Same evaluation discipline
as the first HypeCheck: baselines, a feature ladder, an honest time split, two controls."""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
ROOT="/home/ubuntu/Projects/hypecheck"
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv")
M["trading_day"]=pd.to_datetime(M.trading_day)
Y="went_up"
TWEET=["n_bull","n_bear","n_hype","hype_meter","lex_net","lex_hit","len_chars","caps_share","n_exclaim","n_cashtag","n_links"]
CROWD=["day_tweets","day_hype_sum","day_hype_mean","day_hype_max","day_bull_sum","day_bear_sum","day_net_sum","day_lex_hits","day_hype_intensity"]
M[TWEET+CROWD]=M[TWEET+CROWD].fillna(0)

def fit_eval(tr,te,cols,model="logit"):
    Xtr,Xte=tr[cols].to_numpy(float),te[cols].to_numpy(float)
    ytr,yte=tr[Y].to_numpy(int),te[Y].to_numpy(int)
    if model=="logit":
        sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
        m=LogisticRegression(max_iter=3000,C=1.0).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    else:
        m=GradientBoostingClassifier(random_state=0).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    pred=(p>0.5).astype(int); base=max(yte.mean(),1-yte.mean())
    return dict(acc=accuracy_score(yte,pred),base=base,edge=accuracy_score(yte,pred)-base,
                auc=roc_auc_score(yte,p) if len(set(yte))>1 else float("nan"),n=len(yte))
def show(t,r): print(f"  {t:40s} acc {r['acc']:.4f}  base {r['base']:.4f}  edge {r['edge']:+.4f}  AUC {r['auc']:.4f}")

print(f"dataset: {len(M):,} rows, {M.coin.nunique()} coins, up-rate {M[Y].mean():.1%}")
print(f"effective independent coin-days: {M.groupby(['coin','trading_day']).ngroups:,}")
out={}
o=M.sort_values("trading_day").reset_index(drop=True); cut=int(len(o)*0.8)
tr,te=o.iloc[:cut],o.iloc[cut:]
print(f"\n=== baselines (chronological test, n={len(te)}) ===")
print(f"  always majority                            acc {max(te[Y].mean(),1-te[Y].mean()):.4f}")
print(f"  always 'up'                                acc {te[Y].mean():.4f}")

print("\n=== A. RANDOM 80/20 (flattering) ===")
rng=np.random.default_rng(0); idx=rng.permutation(len(M))
rtr,rte=M.iloc[idx[:int(len(M)*.8)]],M.iloc[idx[int(len(M)*.8):]]
for t,c,m in [("tweet hype only",TWEET,"logit"),("crowd hype only",CROWD,"logit"),
              ("tweet + crowd",TWEET+CROWD,"logit"),("tweet + crowd (GBM)",TWEET+CROWD,"gbm")]:
    r=fit_eval(rtr,rte,c,m); show(t,r); out[f"random_{t}"]=r

print("\n=== B. CHRONOLOGICAL 80/20 (honest) ===")
print(f"  train to {o.trading_day.iloc[cut-1].date()}, test from {o.trading_day.iloc[cut].date()}")
for t,c,m in [("tweet hype only",TWEET,"logit"),("crowd hype only",CROWD,"logit"),
              ("tweet + crowd",TWEET+CROWD,"logit"),("tweet + crowd (GBM)",TWEET+CROWD,"gbm")]:
    r=fit_eval(tr,te,c,m); show(t,r); out[f"chrono_{t}"]=r

print("\n=== controls on the honest GBM ===")
real=out["chrono_tweet + crowd (GBM)"]["auc"]; rng=np.random.default_rng(1)
nl=[]; pl=[]
for _ in range(20):
    s=tr.copy(); s[Y]=rng.permutation(s[Y].to_numpy()); nl.append(fit_eval(s,te,TWEET+CROWD,"gbm")["auc"])
for _ in range(20):
    s=tr.copy()
    for c in TWEET+CROWD: s[c]=rng.permutation(s[c].to_numpy())
    pl.append(fit_eval(s,te,TWEET+CROWD,"gbm")["auc"])
nl=np.array(nl); pl=np.array(pl)
print(f"  real AUC {real:.4f}")
print(f"  label-permutation null {nl.mean():.4f} +/- {nl.std():.4f}  max {nl.max():.4f}  -> {'clears' if real>nl.max() else 'does NOT clear'}")
print(f"  feature-shuffle placebo {pl.mean():.4f} +/- {pl.std():.4f}  max {pl.max():.4f}  -> {'clears' if real>pl.max() else 'does NOT clear'}")
out["controls"]={"real_auc":float(real),"null_mean":float(nl.mean()),"null_sd":float(nl.std()),"null_max":float(nl.max()),
                 "placebo_mean":float(pl.mean()),"placebo_sd":float(pl.std()),"placebo_max":float(pl.max())}

print("\n=== C. WALK-FORWARD (4 windows) ===")
qs=np.linspace(.4,.8,5); wins=[]
for i in range(4):
    a,b=int(len(o)*qs[i]),int(len(o)*qs[i+1]); wtr,wte=o.iloc[:a],o.iloc[a:b]
    if len(wte)<20 or wtr[Y].nunique()<2: continue
    r=fit_eval(wtr,wte,TWEET+CROWD,"gbm"); r["w"]=i+1; r["beats"]=bool(r["edge"]>0); wins.append(r)
    print(f"  window {r['w']}: n {r['n']:4d}  acc {r['acc']:.4f}  base {r['base']:.4f}  edge {r['edge']:+.4f}  AUC {r['auc']:.4f}  {'beats base' if r['beats'] else 'does NOT beat base'}")
if wins: print(f"  -> beats its base rate in {sum(w['beats'] for w in wins)} of {len(wins)} windows")
out["walkforward"]=wins
json.dump(out, open(f"{ROOT}/data/meme-model.json","w"), indent=1, default=str)
print("\nwrote data/meme-model.json")
