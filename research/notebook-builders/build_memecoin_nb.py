import json, hashlib
from pathlib import Path
ROOT = Path("/home/ubuntu/Projects/hypecheck")
REPO = "adeviwon/hypecheck"
RAW  = f"https://raw.githubusercontent.com/{REPO}/main/data"
FILES = ["DOGE-prices.csv","SHIB-prices.csv","PEPE-prices.csv","BONK-prices.csv",
         "FLOKI-prices.csv","WIF-prices.csv","ACT-prices.csv","PNUT-prices.csv","BOME-prices.csv",
         "hype-lexicon.csv"]
SHA = {n: hashlib.sha256((ROOT/"data"/n).read_bytes()).hexdigest() for n in FILES}
SHA_JSON = "{\n" + "".join(f'    "{n}": "{SHA[n]}",\n' for n in FILES) + "}"

CELLS = []
def md(s):   CELLS.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(keepends=True)})
def code(s): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                           "source":s.strip("\n").splitlines(keepends=True)})

md(r'''
# HypeCheck: can tweet hype predict next-day memecoin moves?

**Veritas AI International High School Hackathon 2026**

**Track:** AI + Finance (built on suggested project 4.2.4, *News headlines and the stock market*)
**Team:** Nathaniel Wong (model) · Aaden Yick (data) · Alastair Chu (demo and write-up) — Malvern College Hong Kong

**One-sentence pitch:** HypeCheck reads a memecoin tweet and tells a beginner investor whether the hype in it carries any predictive signal about tomorrow's price — and reports honestly that it does not.

**How to run:** Runtime → Run all. It takes about two minutes, needs no API keys, and downloads two public inputs which it verifies by SHA-256 before training on them.

*This project is a prototype and is not financial advice.*
''')

md(r'''
## 1. Load the data

Two public sources:

| # | Source | What we take | Licence |
|---|---|---|---|
| 1 | [`StephanAkkerman/financial-tweets-crypto`](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto) | 104,379 X posts with text, ticker tags and a timestamp for when the archive first saw them | MIT |
| 2 | [Binance spot daily bars](https://github.com/binance/binance-spot-api-docs) | open, high, low, close and volume for nine memecoins (USDT pairs) | Public market data |

The tweet file is fetched from Hugging Face at a pinned revision. The price files are fetched from our [repository](https://github.com/REPLACE_REPO) rather than a live Binance call, because Binance blocks some regions and cloud IP ranges — a submission that only runs in certain countries is not a submission. Every file is checked against a SHA-256 digest before anything is trained, so bad bytes stop the run instead of quietly changing a number.
''')

code(r'''
# 1 · Fetch both inputs and verify them.
import hashlib
from pathlib import Path

import requests

ROOT = Path.cwd()
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
RAW = "RAWURL"
FILES = SHA_JSON
# The tweet file is 20 MB of source data, so we take it from Hugging Face at a pinned revision
# rather than republishing a second copy in our own repository.
HF_REVISION = "c4b17a94a667af3da1c9ed5144f998e3a4f29966"
POSTS_URL = ("https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/resolve/"
             + HF_REVISION + "/data/train.parquet")
POSTS_SHA = "81b2f7563a44c1a103e9468a83d52319dfeae0fc616422b4a4df90d8d684b960"

def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

sources = [("posts.parquet", POSTS_URL, POSTS_SHA)] + [(n, f"{RAW}/{n}", s) for n, s in FILES.items()]

for name, url, expected in sources:
    dest = DATA / name
    if dest.exists() and sha256(dest) == expected:
        print(f"  ok  {name:20s} {dest.stat().st_size / 1e6:8.2f} MB  already present, digest matches")
        continue
    response = requests.get(url, timeout=(10, 300))
    response.raise_for_status()
    dest.write_bytes(response.content)
    if sha256(dest) != expected:
        raise ValueError(f"{name} does not match its pinned SHA-256; refusing to continue")
    print(f"  ok  {name:20s} {dest.stat().st_size / 1e6:8.2f} MB  downloaded and verified")

print("\nEvery input verified by SHA-256. Nothing here is invented.")
''')

md(r'''
## 2. Look at the data before training on it

We join each tweet to the next day's price move of the coin it tags.

**The rules, stated up front** (they matter, so we are not hiding them):

- **Original posts only.** Retweets and quote tweets are dropped, so we are not counting the same opinion twice.
- **When information was available.** A post is only usable from `max(written_at, first_seen_by_archive)`. We take the *later* of the two, which is the conservative choice, and only ever use a post to predict a day that starts *after* it was available.
- **The target.** Did the coin's close finish above its open on the next UTC day? A flat day counts as *not up*.
- **The coin must have traded.** A tweet is dropped if there is no daily bar for the day it would predict.

We split the coins into two groups. The **large-cap group** is the five best-known memecoins. The **pump.fun-era group** is four Solana memecoins from the launchpad era of late 2023 onwards. We train only on the large-cap group and use the other as an unseen test set in section 7.
''')

code(r'''
# 2 · Build the labelled tweet panel.
import numpy as np
import pandas as pd

LARGE = ["DOGE", "SHIB", "PEPE", "BONK", "FLOKI"]
NEW   = ["WIF", "ACT", "PNUT", "BOME"]

posts = pd.read_parquet("data/posts.parquet")
print(f"Source posts: {len(posts):,}   columns include: {list(posts.columns)[:6]} ...")

posts = posts.loc[posts.tweet_type.eq("tweet")]
posts = posts.loc[posts.text.fillna("").astype(str).str.len().gt(0)]
posts = posts.drop_duplicates(subset=["tweet_id"])
written = pd.to_datetime(posts.created_at, utc=True, errors="coerce")
seen    = pd.to_datetime(posts.first_posted_at, utc=True, errors="coerce")
posts = posts.assign(available_at=pd.concat([written, seen], axis=1).max(axis=1))
posts = posts.dropna(subset=["available_at"])
print(f"Original posts with usable text and a timestamp: {len(posts):,}")

def build_panel(coins):
    frames = []
    for coin in coins:
        prices = pd.read_csv(f"data/{coin}-prices.csv", parse_dates=["date"])
        prices = prices.assign(day=prices.date.dt.date, up=(prices.close > prices.open).astype(int))
        tagged = posts.tickers.apply(lambda x: x is not None and any(str(v).upper() == coin for v in x))
        sub = posts.loc[tagged].copy()
        if not len(sub):
            continue
        # the post predicts the day AFTER it became available
        sub["trading_day"] = (sub.available_at.dt.normalize() + pd.Timedelta(days=1)).dt.date
        sub["coin"] = coin
        joined = sub.merge(prices[["day", "up"]], left_on="trading_day", right_on="day", how="inner")
        frames.append(joined[["tweet_id", "coin", "available_at", "trading_day", "text", "up"]])
    return pd.concat(frames, ignore_index=True)

big = build_panel(LARGE)
new = build_panel(NEW)

for label, frame, coins in [("large-cap", big, LARGE), ("pump.fun-era", new, NEW)]:
    print(f"\n{label} group: {len(frame):,} labelled tweets, next-day up rate {frame.up.mean():.1%}")
    for coin in coins:
        c = frame.loc[frame.coin.eq(coin)]
        if len(c):
            print(f"    {coin:6s} {len(c):5,d} tweets   up rate {c.up.mean():5.1%}   "
                  f"{str(c.available_at.min())[:10]} to {str(c.available_at.max())[:10]}")
''')

md(r'''
Two things to notice immediately.

**The up rate is close to a coin flip** in both groups. That is the first hint that this will be a hard problem: if roughly half of all days go up, a model that always says "up" is already half right, so any accuracy figure has to be read against that base rate rather than against zero.

**The sample is small and lopsided.** One coin (DOGE) supplies most of the large-cap rows, and the pump.fun-era group is under 300 tweets in total. We flag this now because it bounds everything that follows.
''')

code(r'''
# 2b · Read some of the tweets ourselves, as the brief's own advice suggests.
sample = big.sample(6, random_state=7)
for _, row in sample.iterrows():
    text = " ".join(str(row.text).split())[:150]
    print(f"[{str(row.available_at)[:10]}] next day {'UP  ' if row.up else 'down'}  {row.coin:5s}  {text}")

print(f"\nText length: median {big.text.str.len().median():.0f} characters, "
      f"longest {big.text.str.len().max():,}")
''')

md(r'''
Most of these posts are technical-analysis commentary rather than shilling: they talk about support, breakouts and cycle positions. That is an important observation about *this* dataset, and it comes back in section 9 — the vocabulary people expect in memecoin hype ("to the moon", "100x") turns out to be almost entirely absent from it.
''')

md(r'''
## 3. Split into training and test sets

We hold back 20% of the tweets and never train on them, exactly as the brief asks. `stratify` keeps the same share of "up" days in both halves. `random_state=42` makes the split reproducible.
''')

code(r'''
# 3 · 80/20 split.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    big.text, big.up, test_size=0.2, stratify=big.up, random_state=42
)
print(f"Training: {len(X_train):,} tweets   Test: {len(X_test):,} tweets")
print(f"Up rate — training {y_train.mean():.1%}, test {y_test.mean():.1%}")
''')
print("part 1 cells:", len(CELLS))
Path("/tmp/_cells_part1.json").write_text(json.dumps({"cells":CELLS,"sha":SHA_JSON,"raw":RAW}))
