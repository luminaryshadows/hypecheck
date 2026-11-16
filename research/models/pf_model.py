"""Does a tweet's wording tell you whether buying the memecoin paid off?
Same evaluation discipline as the first HypeCheck: baselines, a feature ladder,
an honest time split, and two controls that try to destroy the result."""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT="/home/ubuntu/Projects/hypecheck"
M=pd.read_csv(f"{ROOT}/data/memecoin-tweet-trades.csv")
M["trading_day"]=pd.to_datetime(M.trading_day)

MARKET=["prior_1d_ret","prior_3d_ret","prior_7d_vol","drawdown_20d","prior_day_up","n_history"]
TEXT=["n_bull","n_bear","n_hype","hype_meter","len_chars","caps_share","n_exclaim","n_cashtag","n_links","lex_net","lex_hit"]
Y="worth_it"
M[MARKET]=M[MARKET].fillna(M[MARKET].median())

def fit_eval(tr, te, cols, model="logit"):
    Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    ytr, yte = tr[Y].to_numpy(int), te[Y].to_numpy(int)
    if model=="logit":
        sc=StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m=LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr); p=m.predict_proba(Xte)[:,1]
    else:
        m=GradientBoostingClassifier(random_state=0).fit(Xtr, ytr); p=m.predict_proba(Xte)[:,1]
    pred=(p>0.5).astype(int)
    base=max(yte.mean(), 1-yte.mean())
    return dict(acc=accuracy_score(yte,pred), auc=roc_auc_score(yte,p) if len(set(yte))>1 else np.nan,
                base=base, edge=accuracy_score(yte,pred)-base, n=len(yte),
                sel_ret=te.loc[pred==1,"net_return"].mean() if (pred==1).any() else np.nan,
                all_ret=te.net_return.mean(), hit=pred.mean())

print(f"dataset: {len(M):,} tweet-rows over {M.coin.nunique()} pump.fun-era memecoins")
print(f"  target worth_it rate {M[Y].mean():.1%}   mean net return {M.net_return.mean():+.3%}")
_w=M.net_return[M.net_return>0].mean(); _l=-M.net_return[M.net_return<0].mean()
print(f"  mean winning trade {_w:+.2%}, mean losing trade {-_l:+.2%}")
print(f"  BREAK-EVEN hit rate = loss/(loss+win) = {_l/(_l+_w):.1%}  (must beat this to make money)")

def show(tag, r):
    print(f"  {tag:44s} acc {r['acc']:.4f}  base {r['base']:.4f}  edge {r['edge']:+.4f}  "
          f"AUC {r['auc']:.4f}  selected-trade mean net {r['sel_ret']:+.3%}  (all {r['all_ret']:+.3%})")

out={}
# ---------- baselines ----------
print("\n=== baselines (on the chronological test set) ===")
ordered=M.sort_values("trading_day").reset_index(drop=True)
cut=int(len(ordered)*0.8)
tr, te = ordered.iloc[:cut], ordered.iloc[cut:]
maj=int(tr[Y].mean()>0.5)
print(f"  always '{'worth it' if maj else 'not worth it'}'".ljust(46)+f" acc {max(te[Y].mean(),1-te[Y].mean()):.4f}")
print(f"  always buy (predict every tweet worth it)".ljust(46)+f" acc {te[Y].mean():.4f}  mean net {te.net_return.mean():+.3%}")
out["baselines"]={"majority":float(max(te[Y].mean(),1-te[Y].mean())),"always_buy_acc":float(te[Y].mean()),
                  "test_mean_net_return":float(te.net_return.mean()),"test_n":int(len(te))}

# ---------- the ladder: random 80/20 (flattering) then chronological (honest) ----------
print("\n=== A. RANDOM 80/20 (the split to distrust) ===")
rng=np.random.default_rng(0); idx=rng.permutation(len(M))
rtr, rte = M.iloc[idx[:int(len(M)*0.8)]], M.iloc[idx[int(len(M)*0.8):]]
for tag, cols, mdl in [("market only", MARKET, "logit"), ("text only", TEXT, "logit"),
                       ("market + text", MARKET+TEXT, "logit"), ("market + text (GBM)", MARKET+TEXT, "gbm")]:
    r=fit_eval(rtr, rte, cols, mdl); show(tag, r); out[f"random_{tag}"]=r

print("\n=== B. CHRONOLOGICAL 80/20 (train on the past, test on the future) ===")
print(f"  train to {ordered.trading_day.iloc[cut-1].date()}, test from {ordered.trading_day.iloc[cut].date()}")
for tag, cols, mdl in [("market only", MARKET, "logit"), ("text only", TEXT, "logit"),
                       ("market + text", MARKET+TEXT, "logit"), ("market + text (GBM)", MARKET+TEXT, "gbm")]:
    r=fit_eval(tr, te, cols, mdl); show(tag, r); out[f"chrono_{tag}"]=r

# ---------- controls ----------
print("\n=== controls: can we destroy the best honest result? ===")
best=out["chrono_market + text (GBM)"]
real=best["auc"]
rng=np.random.default_rng(1); nulls=[]
for _ in range(20):
    sh=tr.copy(); sh[Y]=rng.permutation(sh[Y].to_numpy())
    nulls.append(fit_eval(sh, te, MARKET+TEXT, "gbm")["auc"])
nulls=np.array(nulls)
placebo=[]
for _ in range(20):
    sh=tr.copy()
    for c in TEXT: sh[c]=rng.permutation(sh[c].to_numpy())
    placebo.append(fit_eval(sh, te, MARKET+TEXT, "gbm")["auc"])
placebo=np.array(placebo)
print(f"  real AUC {real:.4f}")
print(f"  label-permutation null: {nulls.mean():.4f} +/- {nulls.std():.4f}  max {nulls.max():.4f}  "
      f"-> {'real clears every draw' if real>nulls.max() else 'real does NOT clear the null'}")
print(f"  text-shuffle placebo:   {placebo.mean():.4f} +/- {placebo.std():.4f}  max {placebo.max():.4f}  "
      f"-> {'real clears every draw' if real>placebo.max() else 'real does NOT clear the placebo'}")
out["controls"]={"real_auc":float(real),"null_mean":float(nulls.mean()),"null_sd":float(nulls.std()),
                 "null_max":float(nulls.max()),"placebo_mean":float(placebo.mean()),
                 "placebo_sd":float(placebo.std()),"placebo_max":float(placebo.max())}

# ---------- walk-forward ----------
print("\n=== C. WALK-FORWARD (4 windows, the most honest test) ===")
qs=np.linspace(0.4,0.8,5); wins=[]; oos=pd.DataFrame()
for i in range(4):
    a,b=int(len(ordered)*qs[i]), int(len(ordered)*qs[i+1])
    wtr, wte = ordered.iloc[:a], ordered.iloc[a:b]
    if len(wte)<20 or wtr[Y].nunique()<2: continue
    r=fit_eval(wtr, wte, MARKET+TEXT, "gbm"); r["window"]=i+1
    r["beats_base"]=bool(r["edge"]>0); wins.append(r)
    oos=pd.concat([oos, wte.assign(_p=None)])
for w in wins:
    print(f"  window {w['window']}: n {w['n']:4d}  acc {w['acc']:.4f}  base {w['base']:.4f}  "
          f"edge {w['edge']:+.4f}  AUC {w['auc']:.4f}  {'beats base' if w['beats_base'] else 'does NOT beat base'}")
if wins:
    print(f"  -> beats its base rate in {sum(w['beats_base'] for w in wins)} of {len(wins)} windows")
out["walkforward"]=wins
json.dump(out, open(f"{ROOT}/data/pf-model.json","w"), indent=1, default=str)
print(f"\nwrote data/pf-model.json")
