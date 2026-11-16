"""Build the merged dataset: tweets about pump.fun-era memecoins + prices + whether the trade paid."""
import json, glob, os, re
import numpy as np, pandas as pd

ROOT="/home/ubuntu/Projects/hypecheck"
FEE=0.001          # 0.1% round-trip
LEX=pd.read_csv(f"{ROOT}/data/hype-lexicon.csv")

def terms_for(kind, statuses):
    sel=np.logical_and(LEX.curated_as.eq(kind).values, LEX.status.isin(statuses).values)
    return sorted(LEX.loc[sel,"term"].astype(str).str.lower())
BULL=terms_for("bullish",["validated","weak"])
BEAR=terms_for("bearish",["validated","weak"])
hyp=LEX.loc[np.logical_and(LEX.curated_as.eq("hype").values, LEX.hype_meter.notna().values)]
HYPE={str(t).lower(): float(m) for t,m in zip(hyp.term,hyp.hype_meter)}

# ---------- price series ----------
def closes_for(sym):
    b=f"{ROOT}/data/{sym}-prices.csv"; c=f"/tmp/pf-prices/{sym}-cg.csv"
    if os.path.exists(b):
        d=pd.read_csv(b, parse_dates=["date"]); d["close"]=d["close"].astype(float)
    elif os.path.exists(c):
        d=pd.read_csv(c, parse_dates=["date"]); d["close"]=d["close_usd"].astype(float)
    else:
        return None, None
    d=d.assign(day=d.date.dt.date)[["day","close"]].dropna().drop_duplicates("day").sort_values("day")
    src="binance" if os.path.exists(b) else "coingecko"
    return d.reset_index(drop=True), src

# ---------- tweets ----------
posts=pd.read_parquet(f"{ROOT}/data/posts.parquet")
posts=posts.loc[posts.tweet_type.eq("tweet")]
posts=posts.loc[posts.text.fillna("").astype(str).str.len().gt(0)].drop_duplicates("tweet_id")
ca=pd.to_datetime(posts.created_at,utc=True,errors="coerce")
fp=pd.to_datetime(posts.first_posted_at,utc=True,errors="coerce")
posts=posts.assign(available_at=pd.concat([ca,fp],axis=1).max(axis=1)).dropna(subset=["available_at"])
print(f"eligible original posts: {len(posts):,}")

PUMPFUN=["WIF","FWOG","POPCAT","ACT","GOAT","FARTCOIN","PNUT","MOG","GIGA","MOODENG","CHILLGUY","SLERF","BOME"]
rows=[]; sources={}
for sym in PUMPFUN:
    px,src=closes_for(sym)
    if px is None:
        print(f"  {sym:9s} no price series yet"); continue
    sources[sym]=src
    cmap=dict(zip(px.day, px.close)); days=list(px.day)
    idx={d:i for i,d in enumerate(days)}
    tagged=posts.tickers.apply(lambda x: x is not None and any(str(v).upper()==sym for v in x))
    sub=posts.loc[tagged].copy()
    if not len(sub): continue
    sub["trading_day"]=(sub.available_at.dt.normalize()+pd.Timedelta(days=1)).dt.date
    n0=len(sub)
    recs=[]
    for _,r in sub.iterrows():
        td=r.trading_day
        if td not in idx: continue
        i=idx[td]
        if i<1 or i+1>=len(days): continue           # need history before, and an exit after
        entry=cmap[days[i]]; exitp=cmap[days[i+1]]
        if not (entry and exitp and entry>0): continue
        gross=exitp/entry-1.0; net=gross-FEE
        # market context strictly BEFORE the trading day
        hist=np.array([cmap[d] for d in days[max(0,i-21):i]], dtype=float)
        rets=np.diff(hist)/hist[:-1] if len(hist)>1 else np.array([0.0])
        prior1=rets[-1] if len(rets) else 0.0
        prior3=np.prod(1+rets[-3:])-1 if len(rets)>=3 else np.nan
        prior7=rets[-7:] if len(rets)>=7 else rets
        recs.append(dict(tweet_id=r.tweet_id, coin=sym, price_source=src,
            available_at=r.available_at, trading_day=td, entry_day=days[i], exit_day=days[i+1],
            entry_close=entry, exit_close=exitp, gross_return=gross, net_return=net,
            worth_it=int(net>0),
            prior_1d_ret=prior1, prior_3d_ret=prior3, prior_7d_vol=float(np.std(prior7)) if len(prior7) else np.nan,
            drawdown_20d=float(hist[-1]/np.max(hist)-1.0) if len(hist) else np.nan,
            prior_day_up=int(prior1>0), n_history=len(hist), text=str(r.text)))
    if recs:
        rows.append(pd.DataFrame(recs))
        print(f"  {sym:9s} {len(recs):4d} tradeable tweet-rows of {n0} tagged  (source {src}, "
              f"{str(days[0])} -> {str(days[-1])})")
    else:
        print(f"  {sym:9s} 0 tradeable rows of {n0} tagged (price window does not overlap the tweets)")

M=pd.concat(rows, ignore_index=True)
# ---------- text features ----------
low=M.text.str.lower()
M["n_bull"]=low.apply(lambda s: sum(t in s for t in BULL))
M["n_bear"]=low.apply(lambda s: sum(t in s for t in BEAR))
M["n_hype"]=low.apply(lambda s: sum(t in s for t in HYPE))
M["hype_meter"]=low.apply(lambda s: float(np.mean([HYPE[t] for t in HYPE if t in s])) if any(t in s for t in HYPE) else 0.0)
M["len_chars"]=M.text.str.len()
M["caps_share"]=M.text.apply(lambda s: sum(ch.isupper() for ch in s)/max(len(s),1))
M["n_exclaim"]=M.text.str.count("!")
M["n_cashtag"]=M.text.str.count(r"\$")
M["n_links"]=M.text.str.count(r"http")
M["lex_net"]=M.n_bull-M.n_bear
M["lex_hit"]=(M.n_bull+M.n_bear+M.n_hype>0).astype(int)
M=M.drop(columns=["text"])
M.to_csv(f"{ROOT}/data/memecoin-tweet-trades.csv", index=False)
print(f"\nMERGED DATASET: {len(M):,} rows, {M.coin.nunique()} coins")
print(f"  worth_it rate: {M.worth_it.mean():.1%}   mean net return {M.net_return.mean():+.2%}")
print(f"  date range: {str(M.trading_day.min())} -> {str(M.trading_day.max())}")
print(f"  price sources: {json.dumps({k:v for k,v in sorted(sources.items())})}")
print(f"  columns: {list(M.columns)}")
print(f"\n  per coin:")
for c,g in M.groupby("coin"):
    print(f"    {c:9s} {len(g):4d} rows  worth_it {g.worth_it.mean():5.1%}  mean net {g.net_return.mean():+7.2%}")
