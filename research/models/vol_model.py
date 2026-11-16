import os
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
ROOT="/home/ubuntu/Projects/hypecheck"
M=pd.read_csv(f"{ROOT}/data/memecoin-hype-dataset.csv"); M["avail_day"]=pd.to_datetime(M.avail_day)
H="hype_final" if "hype_final" in M.columns else "hype"
M["hype_abs"]=M[H].abs()
A=M.groupby(["coin","avail_day"]).agg(n_tweets=(H,"size"), hype_sum=(H,"sum"),
      hype_abs_sum=("hype_abs","sum"), bull_share=(H, lambda s:float((s>0.05).mean()))).reset_index()
A=A.sort_values(["coin","avail_day"])
for w in (3,7):
    A[f"hype_sum_{w}d"]=A.groupby("coin").hype_sum.rolling(w,min_periods=1).sum().reset_index(drop=True)
    A[f"tweets_{w}d"]=A.groupby("coin").n_tweets.rolling(w,min_periods=1).sum().reset_index(drop=True)
def px(c):
    p=f"/tmp/meme-prices/{c}-prices.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p, parse_dates=["date"]); d["day"]=d.date.dt.date
    d["range_pct"]=(d.high-d.low)/d.open; d["abs_oc"]=(d.close/d.open-1).abs()
    d["next_range"]=d.range_pct.shift(-1); d["next_up"]=(d.close.shift(-1)>d.open.shift(-1)).astype(float)
    d.loc[d.next_range.isna(),"next_up"]=np.nan
    return d.set_index("day")
PX={c:px(c) for c in A.coin.unique()}
rows=[]
for r in A.itertuples():
    d=PX.get(r.coin)
    if d is None or r.avail_day.date() not in d.index: continue
    b=d.loc[r.avail_day.date()]
    if pd.isna(b.next_range): continue
    rows.append(dict(coin=r.coin, day=r.avail_day.date(), n_tweets=r.n_tweets, hype_sum=r.hype_sum,
        hype_abs_sum=r.hype_abs_sum, bull_share=r.bull_share, hype_sum_3d=r.hype_sum_3d,
        tweets_3d=r.tweets_3d, hype_sum_7d=r.hype_sum_7d, tweets_7d=r.tweets_7d,
        today_range=float(b.range_pct), today_abs=float(b.abs_oc),
        y_range=float(b.next_range), y_dir=float(b.next_up)))
D=pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
print(f"coin-days with a next-day target: {len(D):,}  {D.day.min()} -> {D.day.max()}")
print(f"  next-day range: mean {D.y_range.mean():.4f} sd {D.y_range.std():.4f} | up-rate {D.y_dir.mean():.1%}")
HF=["n_tweets","hype_sum","hype_abs_sum","bull_share","hype_sum_3d","tweets_3d","hype_sum_7d","tweets_7d"]
CF=["today_range","today_abs"]; D[HF+CF]=D[HF+CF].fillna(0)
def wf_reg(df, cols, target, model="ridge"):
    qs=np.linspace(0.4,0.8,5); pred=np.full(len(df), np.nan)
    for i in range(4):
        a,b=int(len(df)*qs[i]),int(len(df)*qs[i+1]); tr,te=df.iloc[:a],df.iloc[a:b]
        if len(te)<20 or len(tr)<50: continue
        Xtr,Xte=tr[cols].to_numpy(float),te[cols].to_numpy(float)
        sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
        m=(Ridge(alpha=1.0) if model=="ridge" else GradientBoostingRegressor(random_state=0)).fit(Xtr,tr[target])
        pred[df.index.get_indexer(te.index)]=m.predict(Xte)
    return pred
print("\n=== WALK-FORWARD: next-day VOLATILITY (daily range) from aggregate hype ===")
for cols,lbl in [(HF,"hype only"),(CF,"context only"),(HF+CF,"hype + context")]:
    p=wf_reg(D,cols,"y_range"); m=~np.isnan(p)
    rho,pv=stats.spearmanr(D.y_range[m],p[m]); print(f"  {lbl:16s} n_oos={m.sum():4d}  Spearman {rho:+.4f}  p={pv:.2e} {'**' if pv<0.05 else ''}")
print("\n=== CONTROLS on 'hype + context' ===")
p=wf_reg(D,HF+CF,"y_range"); m=~np.isnan(p); real=stats.spearmanr(D.y_range[m],p[m])[0]
rng=np.random.default_rng(0); nl=[]; pl=[]
for _ in range(20):
    Dp=D.copy(); Dp["y_range"]=rng.permutation(Dp.y_range.to_numpy())
    nl.append(stats.spearmanr(Dp.y_range[m], wf_reg(Dp,HF+CF,"y_range")[m])[0])
for _ in range(20):
    Ds=D.copy()
    for c in HF: Ds[c]=rng.permutation(Ds[c].to_numpy())
    pl.append(stats.spearmanr(Ds.y_range[m], wf_reg(Ds,HF+CF,"y_range")[m])[0])
nl=np.array(nl); pl=np.array(pl)
print(f"  real {real:+.4f} | label-perm null {nl.mean():+.4f}+/-{nl.std():.4f} max {nl.max():+.4f} -> {'CLEARS' if real>nl.max() else 'does NOT clear'}")
print(f"  hype-shuffle placebo {pl.mean():+.4f}+/-{pl.std():.4f} max {pl.max():+.4f} -> {'CLEARS' if real>pl.max() else 'does NOT clear'}")
print("\n=== CONTRAST: same walk-forward on DIRECTION ===")
qs=np.linspace(0.4,0.8,5); pred=np.full(len(D),np.nan)
for i in range(4):
    a,b=int(len(D)*qs[i]),int(len(D)*qs[i+1]); tr,te=D.iloc[:a],D.iloc[a:b]
    if len(te)<20: continue
    Xtr,Xte=tr[HF+CF].to_numpy(float),te[HF+CF].to_numpy(float)
    sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
    mdl=LogisticRegression(max_iter=3000).fit(Xtr,tr.y_dir); pred[D.index.get_indexer(te.index)]=mdl.predict(Xte)
m=~np.isnan(pred); acc=accuracy_score(D.y_dir[m],pred[m]); base=max(D.y_dir[m].mean(),1-D.y_dir[m].mean())
print(f"  n_oos={m.sum():4d}  acc {acc:.4f}  base {base:.4f}  edge {acc-base:+.4f}")
D.to_csv("/tmp/vol_panel.csv", index=False); print("\nwrote /tmp/vol_panel.csv")
