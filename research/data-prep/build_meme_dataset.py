"""Dataset: every tweet about a memecoin, its hype, how much the crowd hyped that coin that
day, and whether the coin went up. Exclusions for symbol collisions are explicit."""
import json, os, re
import numpy as np, pandas as pd
ROOT="/home/ubuntu/Projects/hypecheck"
PRICES="/tmp/meme-prices"
LEX=pd.read_csv(f"{ROOT}/data/hype-lexicon.csv")

def terms_for(kind, statuses):
    sel=np.logical_and(LEX.curated_as.eq(kind).values, LEX.status.isin(statuses).values)
    return sorted(LEX.loc[sel,"term"].astype(str).str.lower())
BULL=terms_for("bullish",["validated","weak"]); BEAR=terms_for("bearish",["validated","weak"])
hyp=LEX.loc[np.logical_and(LEX.curated_as.eq("hype").values, LEX.hype_meter.notna().values)]
HYPE={str(t).lower(): float(m) for t,m in zip(hyp.term, hyp.hype_meter)}

# symbols where the Binance asset of that ticker is NOT the memecoin CoinGecko means:
# the ticker collides with a non-memecoin. Excluded, with the reason recorded.
EXCLUDE={
 "LUNA":"Binance LUNA is Terra; the corpus means Terra, not the memecoin of that ticker",
 "MASK":"Binance MASK is Mask Network, a middleware token, not a memecoin",
 "AI":"ambiguous ticker; Binance AI is not the memecoin CoinGecko lists under that symbol",
 "JOE":"Binance JOE is Trader Joe, a DEX token, not a memecoin",
 "BABY":"ambiguous ticker; the Binance asset is not the memecoin of that symbol",
 "MARSCOIN":"ambiguous ticker, 72 bars from 2026-09 only",
 "SC":"Binance SC is Siacoin, a storage chain, not Shark Cat",
}
meme_all=json.load(open("/tmp/memecoins.json"))["meme"]
COINS=[s for s in meme_all if s not in EXCLUDE]

posts=pd.read_parquet(f"{ROOT}/data/posts.parquet")
posts=posts.loc[posts.tweet_type.eq("tweet")]
posts=posts.loc[posts.text.fillna("").astype(str).str.len().gt(0)].drop_duplicates("tweet_id")
ca=pd.to_datetime(posts.created_at,utc=True,errors="coerce")
fp=pd.to_datetime(posts.first_posted_at,utc=True,errors="coerce")
posts=posts.assign(available_at=pd.concat([ca,fp],axis=1).max(axis=1)).dropna(subset=["available_at"])
posts=posts.assign(avail_day=posts.available_at.dt.normalize().dt.date)
print(f"corpus: {len(posts):,} eligible original posts")
print(f"coins included: {len(COINS)}   excluded for symbol collisions: {len(EXCLUDE)}")

rows=[]; skipped={}
for sym in COINS:
    p=f"{PRICES}/{sym}-prices.csv"
    if not os.path.exists(p): skipped[sym]="no price file"; continue
    px=pd.read_csv(p, parse_dates=["date"]).assign(day=lambda d: d.date.dt.date)
    px["up"]=(px.close>px.open).astype(int)
    px["next_close"]=px.close.shift(-1); px["next_day"]=px.day.shift(-1)
    pmap={r.day:r for r in px.itertuples()}
    tagged=posts.tickers.apply(lambda x: x is not None and any(str(v).upper()==sym for v in x))
    sub=posts.loc[tagged].copy()
    if not len(sub): skipped[sym]="no tweets"; continue
    sub["trading_day"]=(sub.available_at.dt.normalize()+pd.Timedelta(days=1)).dt.date
    keep=[]
    for r in sub.itertuples():
        bar=pmap.get(r.trading_day)
        if bar is None or pd.isna(bar.next_close): continue
        keep.append((r.tweet_id, sym, r.available_at, r.avail_day, r.trading_day,
                     bar.open, bar.close, bar.next_close, int(bar.up),
                     (bar.close/bar.open-1.0), (bar.next_close/bar.close-1.0), str(r.text)))
    if keep:
        rows.append(pd.DataFrame(keep, columns=["tweet_id","coin","available_at","avail_day","trading_day",
                    "open","close","next_close","went_up","open_to_close","close_to_next","text"]))
        print(f"  {sym:9s} {len(keep):5d} rows of {len(sub):5d} tagged tweets")
    else:
        skipped[sym]=f"{len(sub)} tweets, none with a matching bar"
M=pd.concat(rows, ignore_index=True)
print(f"\nraw rows: {len(M):,}")

# ---- per-tweet hype features ----
low=M.text.str.lower()
M["n_bull"]=low.apply(lambda s: sum(t in s for t in BULL))
M["n_bear"]=low.apply(lambda s: sum(t in s for t in BEAR))
M["n_hype"]=low.apply(lambda s: sum(t in s for t in HYPE))
M["hype_meter"]=low.apply(lambda s: float(np.mean([HYPE[t] for t in HYPE if t in s])) if any(t in s for t in HYPE) else 0.0)
M["lex_net"]=M.n_bull-M.n_bear
M["lex_hit"]=(M.n_bull+M.n_bear+M.n_hype>0).astype(int)
M["len_chars"]=M.text.str.len()
M["caps_share"]=M.text.apply(lambda s: sum(c.isupper() for c in s)/max(len(s),1))
M["n_exclaim"]=M.text.str.count("!")
M["n_cashtag"]=M.text.str.count(r"\$")
M["n_links"]=M.text.str.count("http")

# ---- how much the crowd hyped this coin that day (known before the trading day) ----
g=M.groupby(["coin","avail_day"])
agg=g.agg(day_tweets=("tweet_id","size"), day_hype_sum=("hype_meter","sum"),
          day_hype_mean=("hype_meter","mean"), day_hype_max=("hype_meter","max"),
          day_bull_sum=("n_bull","sum"), day_bear_sum=("n_bear","sum"),
          day_net_sum=("lex_net","sum"), day_lex_hits=("lex_hit","sum")).reset_index()
M=M.merge(agg, on=["coin","avail_day"], how="left")
M["day_hype_intensity"]=M.day_hype_sum/ (M.day_tweets*100)      # 0..1 scaled crowd hype
M=M.drop(columns=["text"])
M.to_csv(f"{ROOT}/data/memecoin-hype-dataset.csv", index=False)

print(f"\n=== DATASET: {len(M):,} rows x {M.shape[1]} cols, {M.coin.nunique()} coins ===")
print(f"  went_up rate: {M.went_up.mean():.1%}")
print(f"  date range:   {M.trading_day.min()} -> {M.trading_day.max()}")
print(f"  distinct (coin, trading_day): {M.groupby(['coin','trading_day']).ngroups:,}")
print(f"  coins: {sorted(M.coin.unique())}")
if skipped: print(f"  skipped: {json.dumps(skipped, indent=1)}")
print(f"\n  per coin:")
for c,gg in M.groupby("coin"):
    print(f"    {c:9s} {len(gg):5d} rows  up {gg.went_up.mean():5.1%}  mean O->C {gg.open_to_close.mean():+7.2%}")
print(f"\n  crowd-hype columns: day_tweets mean {M.day_tweets.mean():.2f} max {M.day_tweets.max()}, "
      f"day_hype_sum mean {M.day_hype_sum.mean():.1f} max {M.day_hype_sum.max():.1f}")
json.dump({"rows":int(len(M)),"coins":sorted(M.coin.unique()),"excluded":EXCLUDE,"skipped":skipped},
          open(f"{ROOT}/data/memecoin-hype-dataset.meta.json","w"), indent=1)
